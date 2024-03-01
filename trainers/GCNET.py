import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from numpy.random import randint
from torch import optim
from sklearn.preprocessing import OneHotEncoder

from models.subnets.AlignSubNet import AlignSubNet
from models.subnets.BertTextEncoder import BertTextEncoder
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer

def random_mask(view_num, input_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix


class GCNET_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.alignment_network = AlignSubNet(config, mode='avg_pool').to(config.device)
        self.text_model = BertTextEncoder(pretrained=config.pretrained_bert_model, finetune=config.finetune_bert).to(config.device)
        self.rec_loss = MaskedReconLoss()
        
    def do_train(self, model, dataloader):
        # OPTIMIZER: finetune Bert Parameters.
        optimizer = optim.Adam(list(model.parameters()), lr=self.config.learning_rate, weight_decay=self.config.l2)

        # loop util earlystop
        while True: 
            self.epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    text = batch_data['text'].to(self.config.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                     # Align audio/vision modality to spoken words.
                    text, audio, vision = self.alignment_network(text, audio, vision, 
                                                           audio_lengths, vision_lengths)
                    # The umask, seq_lengths is corresponding to the text mask (after alignments)
                    umask = text[:,1,:].to(self.config.device)
                    qmask = torch.zeros_like(umask).to(self.config.device) # This framework considers one speaker scenario only.
                    seq_lengths = torch.sum(umask, dim=-1).to(self.config.device)

                    text = self.text_model(text) # Bert Text Encoder.
                    batch, seqlen = audio.size(0), audio.size(1)
                    audio, text, vision = audio.transpose(0, 1), text.transpose(0, 1), vision.transpose(0, 1)
                    inputfeats = torch.cat([audio, text, vision], dim=-1)
                
                    
                    matrix = random_mask(view_num=3, input_len=seqlen*batch, missing_rate=0.2) # [seqlen*batch, view_num]

                    audio_host_mask = torch.LongTensor(np.reshape(matrix[:, 0], (seqlen, batch, 1))).to(self.config.device)
                    text_host_mask = torch.LongTensor(np.reshape(matrix[:, 1], (seqlen, batch, 1))).to(self.config.device)
                    vision_host_mask = torch.LongTensor(np.reshape(matrix[:, 2], (seqlen, batch, 1))).to(self.config.device)
                    masked_audio_host = (audio * audio_host_mask).to(self.config.device)
                    masked_text_host = (text * text_host_mask).to(self.config.device)
                    masked_vision_host = (vision * vision_host_mask).to(self.config.device)
                    
                    masked_inputfeats = torch.cat([masked_audio_host, masked_text_host, masked_vision_host], dim=-1)
                    input_features_mask = torch.cat([audio_host_mask, text_host_mask, vision_host_mask], dim=-1)

                    prediction, recon_input_features, hidden = model(masked_inputfeats, qmask, umask, seq_lengths)
                    
                    ## calculate loss
                    clf_loss = self.criterion(prediction, labels)
                    rec_loss = self.rec_loss(recon_input_features, inputfeats, input_features_mask, umask,
                                            self.config.feature_dims[1], self.config.feature_dims[0], self.config.feature_dims[2],
                                            self.config.a_rec_weight, self.config.t_rec_weight, self.config.v_rec_weight)
                    if self.config.loss_recon: loss = clf_loss + rec_loss
                    if not self.config.loss_recon: loss = clf_loss
                    # clear gradient
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
                    
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            self.logger.info(
                f"TRAIN-({self.config.model}) [{self.epochs - self.best_epoch}/{self.epochs}/{self.config.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_valid(model, dataloader['valid'])
            # save best model
            cur_valid = val_results[self.config.KeyEval]
            isBetter = cur_valid <= (self.best_valid - 1e-6)
            # save best model
            if isBetter:
                self.best_valid, self.best_epoch = cur_valid, self.epochs
                # save model
                torch.save(model.cpu().state_dict(), self.config.model_save_path)
                model.to(self.config.device)
            if self.epochs - self.best_epoch >= self.config.early_stop:
                return None


    def do_valid(self, model, dataloader, mode='EVAL'):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    text = batch_data['text'].to(self.config.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                     # Align audio/vision modality to spoken words.
                    text, audio, vision = self.alignment_network(text, audio, vision, 
                                                           audio_lengths, vision_lengths)
                    # The umask, seq_lengths is corresponding to the text mask (after alignments)
                    umask = text[:,1,:].to(self.config.device)
                    qmask = torch.zeros_like(umask).to(self.config.device) # This framework considers one speaker scenario only.
                    seq_lengths = torch.sum(umask, dim=-1).to(self.config.device)
                    
                    text = self.text_model(text) # Bert Text Encoder.
                    audio, text, vision = audio.transpose(0, 1), text.transpose(0, 1), vision.transpose(0, 1)
                    inputfeats = torch.cat([audio, text, vision], dim=-1)

                    prediction, _, _ = model(inputfeats, qmask, umask, seq_lengths)
                    prediction = torch.where(
                    torch.isnan(prediction),
                    torch.full_like(prediction, 0),
                    prediction)
                    loss = self.criterion(prediction, labels)
                    eval_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        self.logger.info(f"{mode}-({self.config.model}) >> {dict_to_str(eval_results)}")

        return eval_results

## for reconstruction [only recon loss on miss part]
class MaskedReconLoss(nn.Module):

    def __init__(self):
        super(MaskedReconLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, recon_input, target_input, input_mask, umask, adim, tdim, vdim, aw, tw, vw):
        """ ? => refer to spk and modality
        recon_input  -> ? * [seqlen, batch, dim]
        target_input -> ? * [seqlen, batch, dim]
        input_mask   -> ? * [seqlen, batch, dim]
        umask        -> [batch, seqlen]
        """
        assert len(recon_input) == 1
        max_len = recon_input[0].size(0)
        recon = recon_input[0] # [seqlen, batch, dim]
        target = target_input[:max_len] # [seqlen, batch, dim]
        mask = input_mask[:max_len] # [seqlen, batch, 3]

        recon  = torch.reshape(recon, (-1, recon.size(2)))   # [seqlen*batch, dim]
        target = torch.reshape(target, (-1, target.size(2))) # [seqlen*batch, dim]
        mask   = torch.reshape(mask, (-1, mask.size(2)))     # [seqlen*batch, 3] 1(exist); 0(mask)
        umask = torch.reshape(umask[:, :max_len], (-1, 1)) # [seqlen*batch, 1]

        A_rec = recon[:, :adim]
        L_rec = recon[:, adim:adim+tdim]
        V_rec = recon[:, adim+tdim:]
        A_full = target[:, :adim]
        L_full = target[:, adim:adim+tdim]
        V_full = target[:, adim+tdim:]
        A_miss_index = torch.reshape(mask[:, 0], (-1, 1))
        L_miss_index = torch.reshape(mask[:, 1], (-1, 1))
        V_miss_index = torch.reshape(mask[:, 2], (-1, 1))

        loss_recon1 = self.loss(A_rec*umask, A_full*umask) * -1 * (A_miss_index - 1)
        loss_recon2 = self.loss(L_rec*umask, L_full*umask) * -1 * (L_miss_index - 1)
        loss_recon3 = self.loss(V_rec*umask, V_full*umask) * -1 * (V_miss_index - 1)
        loss_recon1 = torch.sum(loss_recon1) / adim * aw
        loss_recon2 = torch.sum(loss_recon2) / tdim * tw
        loss_recon3 = torch.sum(loss_recon3) / vdim * vw
        loss_recon = (loss_recon1 + loss_recon2 + loss_recon3) / torch.sum(umask)

        return loss_recon
