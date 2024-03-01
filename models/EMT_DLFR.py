import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.subnets.BertTextEncoder import BertTextEncoder
from models.subnets.EMT import EMT


__all__ = ['EMT_DLFR']

class EMT_DLFR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.aligned = config.need_data_aligned
        # unimodal encoders
        ## text encoder
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert)

        ## audio-vision encoders
        audio_in, video_in = config.feature_dims[1:]
        self.audio_model = AuViSubNet(audio_in, config.a_lstm_hidden_size, config.audio_out, \
                            num_layers=config.a_lstm_layers, dropout=config.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, config.v_lstm_hidden_size, config.video_out, \
                            num_layers=config.v_lstm_layers, dropout=config.v_lstm_dropout)

        # equalization
        self.proj_audio = nn.Linear(config.audio_out, config.d_model, bias=False) if config.audio_out != config.d_model else nn.Identity()
        self.proj_video = nn.Linear(config.video_out, config.d_model, bias=False) if config.video_out != config.d_model else nn.Identity()
        self.proj_text = nn.Linear(config.text_out, config.d_model, bias=False) if config.text_out != config.d_model else nn.Identity()

        # fusion: emt
        num_modality = 3
        # self.fusion_method = config.fusion_method

        self.fusion = EMT(dim=config.d_model, depth=config.fusion_layers, heads=config.heads, num_modality=num_modality,
                          learnable_pos_emb=config.learnable_pos_emb, emb_dropout=config.emb_dropout, attn_dropout=config.attn_dropout,
                          ff_dropout=config.ff_dropout, ff_expansion=config.ff_expansion, mpu_share=config.mpu_share,
                          modality_share=config.modality_share, layer_share=config.layer_share,
                          attn_act_fn=config.attn_act_fn)

        # high-level feature attraction via SimSiam
        ## projector
        ## gmc_tokens: global multimodal context
        gmc_tokens_dim = num_modality * config.d_model
        self.gmc_tokens_projector = Projector(gmc_tokens_dim, gmc_tokens_dim)
        self.text_projector = Projector(config.text_out, config.text_out)
        self.audio_projector = Projector(config.audio_out, config.audio_out)
        self.video_projector = Projector(config.video_out, config.video_out)

        ## predictor
        self.gmc_tokens_predictor = Predictor(gmc_tokens_dim, config.gmc_tokens_pred_dim, gmc_tokens_dim)
        self.text_predictor = Predictor(config.text_out, config.text_pred_dim, config.text_out)
        self.audio_predictor = Predictor(config.audio_out, config.audio_pred_dim, config.audio_out)
        self.video_predictor = Predictor(config.video_out, config.video_pred_dim, config.video_out)

        # low-level feature reconstruction
        self.recon_text = nn.Linear(config.d_model, config.feature_dims[0])
        self.recon_audio = nn.Linear(config.d_model, config.feature_dims[1])
        self.recon_video = nn.Linear(config.d_model, config.feature_dims[2])

        # final prediction module
        self.post_fusion_dropout = nn.Dropout(p=config.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(config.text_out + config.video_out + config.audio_out + gmc_tokens_dim, config.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(config.post_fusion_dim, config.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(config.post_fusion_dim, config.output_dim)

    def forward_once(self, text, text_lengths, audio, audio_lengths, video, video_lengths, missing):
        # unimodal encoders
        text = self.text_model(text)
        text_utt, text = text[:,0], text[:, 1:] # (B, 1, D), (B, T, D)
        text_for_recon = text.detach()

        audio, audio_utt = self.audio_model(audio, audio_lengths, return_temporal=True)
        video, video_utt = self.video_model(video, video_lengths, return_temporal=True)

        # projection
        ## gmc_tokens: global multimodal context, (B, 3, D)
        gmc_tokens = torch.stack([self.proj_text(text_utt), self.proj_audio(audio_utt), self.proj_video(video_utt)], dim=1)
        ## local unimodal features, (B, T, D)
        text, audio, video = self.proj_text(text), self.proj_audio(audio), self.proj_video(video)

        # get attention mask
        modality_masks = [length_to_mask(seq_len, max_len=max_len)
                          for seq_len, max_len in zip([text_lengths, audio_lengths, video_lengths],
                                                      [text.shape[1], audio.shape[1], video.shape[1]])]

        # fusion
        gmc_tokens, modality_ouputs = self.fusion(gmc_tokens,[text, audio, video], modality_masks)
        gmc_tokens = gmc_tokens.reshape(gmc_tokens.shape[0], -1) # (B, 3*D)

        # high-level feature attraction via SimSiam
        ## projector
        z_gmc_tokens = self.gmc_tokens_projector(gmc_tokens)
        z_text = self.text_projector(text_utt)
        z_audio = self.audio_projector(audio_utt)
        z_video = self.video_projector(video_utt)

        ## predictor
        p_gmc_tokens = self.gmc_tokens_predictor(z_gmc_tokens)
        p_text = self.text_predictor(z_text)
        p_audio = self.audio_predictor(z_audio)
        p_video = self.video_predictor(z_video)

        # final prediction module
        fusion_h = torch.cat([text_utt, audio_utt, video_utt, gmc_tokens], dim=-1)
        fusion_h = self.post_fusion_dropout(fusion_h)
        fusion_h = F.relu(self.post_fusion_layer_1(fusion_h), inplace=False)
        x_f = F.relu(self.post_fusion_layer_2(fusion_h), inplace=False)
        output_fusion = self.post_fusion_layer_3(x_f)

        suffix = '_m' if missing else ''
        res = {
            f'pred{suffix}': output_fusion,
            f'z_gmc_tokens{suffix}': z_gmc_tokens.detach(),
            f'p_gmc_tokens{suffix}': p_gmc_tokens,
            f'z_text{suffix}': z_text.detach(),
            f'p_text{suffix}': p_text,
            f'z_audio{suffix}': z_audio.detach(),
            f'p_audio{suffix}': p_audio,
            f'z_video{suffix}': z_video.detach(),
            f'p_video{suffix}': p_video,
        }

        # low-level feature reconstruction
        if missing:
            text_recon = self.recon_text(modality_ouputs[0])
            audio_recon = self.recon_audio(modality_ouputs[1])
            video_recon = self.recon_video(modality_ouputs[2])
            res.update(
                {
                    'text_recon': text_recon,
                    'audio_recon': audio_recon,
                    'video_recon': video_recon,
                }
            )
        else:
            res.update({'text_for_recon': text_for_recon})

        return res

    def forward(self, text, audio, video, mode='train'):
        text, text_m = text
        audio, audio_m, audio_lengths = audio
        video, video_m, video_lengths = video

        mask_len = torch.sum(text[:,1,:], dim=1, keepdim=True)
        text_lengths = mask_len.squeeze().int().detach() - 2 # -2 for CLS and SEP

        # complete view
        res = self.forward_once(text, text_lengths, audio, audio_lengths, video, video_lengths, missing=False)
        # incomplete view
        if not mode.startswith('Robustness Test'):
            res_m = self.forward_once(text_m, text_lengths, audio_m, audio_lengths, video_m, video_lengths, missing=True)
        
        return res if mode.startswith('Robustness Test') else {**res, **res_m}



class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size=None, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        config:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super().__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        feature_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear_1 = nn.Linear(feature_size, out_size) if feature_size != out_size and out_size is not None else nn.Identity()

    def forward(self, x, lengths, return_temporal=False):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        # for pytorch1.2
        # packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # for pytorch1.7
        packed_sequence = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_last_hidden_state, final_states = self.rnn(packed_sequence)

        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        if not return_temporal:
            return y_1
        else:
            unpacked_last_hidden_state, _ = pad_packed_sequence(packed_last_hidden_state, batch_first=True)
            last_hidden_state = self.linear_1(unpacked_last_hidden_state)
            return last_hidden_state, y_1


class Projector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(input_dim, output_dim),
                                 nn.BatchNorm1d(output_dim, affine=False))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_dim, pred_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, pred_dim, bias=False),
                                 nn.BatchNorm1d(pred_dim),
                                 nn.ReLU(inplace=True),  # hidden layer
                                 nn.Linear(pred_dim, output_dim))  # output layer

    def forward(self, x):
        return self.net(x)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask
