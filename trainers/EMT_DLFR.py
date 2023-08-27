""" EMT_DLFR Trainer is partially adapted from https://github.com/sunlicai/EMT-DLFR.
    NOTE: Modification is made for fair comparison under the situation where the missing 
    position is unknown in the noisy instances during the training periods.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer


class EMT_DLFR_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.criterion_attra = nn.CosineSimilarity(dim=1).to(self.config.device)
        self.criterion_recon = ReconLoss(self.config.recon_loss)
        self.config.KeyEval = 'Loss'
        
    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.text_model.named_parameters())
        audio_params = list(model.audio_model.named_parameters())
        video_params = list(model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for _, p in audio_params]
        video_params = [p for _, p in video_params]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.config.weight_decay_audio, 'lr': self.config.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.config.weight_decay_video, 'lr': self.config.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        # loop util earlystop
        while True: 
            self.epochs += 1
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    text = batch_data['text'].to(self.config.device)
                    text_m = batch_data['text_m'].to(self.config.device) # Paired Noisy data
                    audio = batch_data['audio'].to(self.config.device)
                    audio_m = batch_data['audio_m'].to(self.config.device) # Paired Noisy data
                    vision = batch_data['vision'].to(self.config.device)
                    vision_m = batch_data['vision_m'].to(self.config.device) # Paired Noisy data
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)
                    audio_mask = torch.arange(audio.shape[1]).expand(audio.shape[0], audio.shape[1]).to(audio.device) < audio_lengths.unsqueeze(1)
                    vision_mask = torch.arange(vision.shape[1]).expand(vision.shape[0], vision.shape[1]).to(vision.device) < vision_lengths.unsqueeze(1)
                    
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                    # clear gradient
                    optimizer.zero_grad()
                    # compute loss, backward, update.
                    outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths))
                    # compute loss
                    ## prediction loss
                    loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                    loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))
                    ## attraction loss (high-level)
                    loss_attra_gmc_tokens = -(self.criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                          self.criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                    loss_attra_text = -(self.criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                          self.criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                    loss_attra_audio = -(self.criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                          self.criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                    loss_attra_video = -(self.criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                          self.criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
                    loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred
                    ## reconstruction loss (low-level)
                    mask = text[:, 1, 1:] # '1:' for excluding CLS
                    loss_recon_text = self.criterion_recon(outputs['text_recon'], outputs['text_for_recon'], mask)
                    loss_recon_audio = self.criterion_recon(outputs['audio_recon'], audio[:,: batch_data['audio_lengths'].max()], audio_mask[:,: batch_data['audio_lengths'].max()])
                    loss_recon_video = self.criterion_recon(outputs['video_recon'], vision[:,: batch_data['vision_lengths'].max()], vision_mask[:,: batch_data['vision_lengths'].max()])
                    loss_recon = loss_recon_text + loss_recon_audio * self.config.recon_loss_wa + loss_recon_video * self.config.recon_loss_wv
                    ## total loss
                    loss = loss_pred_m + self.config.loss_attra_weight * loss_attra + self.config.loss_recon_weight * loss_recon
                    loss.backward()
                    optimizer.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(outputs['pred_m'].view(-1).cpu())
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
                    text = batch_data['text'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    vision = batch_data['vision'].to(self.config.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.config.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.config.device)
                    audio_mask = torch.arange(audio.shape[1]).expand(audio.shape[0], audio.shape[1]).to(audio.device) < audio_lengths.unsqueeze(1)
                    vision_mask = torch.arange(vision.shape[1]).expand(vision.shape[0], vision.shape[1]).to(vision.device) < vision_lengths.unsqueeze(1)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                    if mode == 'EVAL':
                        text_m = batch_data['text_m'].to(self.config.device) # Paired Noisy data
                        audio_m = batch_data['audio_m'].to(self.config.device) # Paired Noisy data
                        vision_m = batch_data['vision_m'].to(self.config.device) # Paired Noisy data
                        outputs = model((text, text_m), (audio, audio_m, audio_lengths), (vision, vision_m, vision_lengths), mode=mode)
                        loss_pred_m = torch.mean(torch.abs(outputs['pred_m'].view(-1) - labels.view(-1)))
                        ## attraction loss (high-level)
                        loss_attra_gmc_tokens = -(self.criterion_attra(outputs['p_gmc_tokens_m'], outputs['z_gmc_tokens']).mean() +
                                            self.criterion_attra(outputs['p_gmc_tokens'], outputs['z_gmc_tokens_m']).mean()) * 0.5
                        loss_attra_text = -(self.criterion_attra(outputs['p_text_m'], outputs['z_text']).mean() +
                                            self.criterion_attra(outputs['p_text'], outputs['z_text_m']).mean()) * 0.5
                        loss_attra_audio = -(self.criterion_attra(outputs['p_audio_m'], outputs['z_audio']).mean() +
                                            self.criterion_attra(outputs['p_audio'], outputs['z_audio_m']).mean()) * 0.5
                        loss_attra_video = -(self.criterion_attra(outputs['p_video_m'], outputs['z_video']).mean() +
                                            self.criterion_attra(outputs['p_video'], outputs['z_video_m']).mean()) * 0.5
                        loss_pred = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1))) # prediction loss of complete view
                        loss_attra = loss_attra_gmc_tokens + loss_attra_text + loss_attra_audio + loss_attra_video + loss_pred
                        ## reconstruction loss (low-level)
                        
                        loss_recon_text = self.criterion_recon(outputs['text_recon'], outputs['text_for_recon'],  text[:, 1, 1:])
                        loss_recon_audio = self.criterion_recon(outputs['audio_recon'], audio[:,: batch_data['audio_lengths'].max()], audio_mask[:,: batch_data['audio_lengths'].max()])
                        loss_recon_video = self.criterion_recon(outputs['video_recon'], vision[:,: batch_data['vision_lengths'].max()], vision_mask[:,: batch_data['vision_lengths'].max()])
                        loss_recon = loss_recon_text + loss_recon_audio * self.config.recon_loss_wa + loss_recon_video * self.config.recon_loss_wv
                        ## total loss
                        loss = loss_pred_m + self.config.loss_attra_weight * loss_attra + self.config.loss_recon_weight * loss_recon
                        
                        y_pred.append(outputs['pred_m'].cpu())
                    else: # In Test Phase no paired clean data is available.
                        outputs = model((text, None), (audio, None, audio_lengths), (vision, None, vision_lengths), mode=mode)
                        loss = torch.mean(torch.abs(outputs['pred'].view(-1) - labels.view(-1)))
                        y_pred.append(outputs['pred'].cpu())

                    eval_loss += loss.item()
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        self.logger.info(f"{mode}-({self.config.model}) >> Eval loss: {round(eval_loss, 4)} {dict_to_str(eval_results)}")

        return eval_results


class ReconLoss(nn.Module):
    def __init__(self, type):
        super().__init__()
        self.eps = 1e-6
        self.type = type
        if type == 'L1Loss':
            self.loss = nn.L1Loss(reduction='sum')
        elif type == 'SmoothL1Loss':
            self.loss = nn.SmoothL1Loss(reduction='sum')
        elif type == 'MSELoss':
            self.loss = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

    def forward(self, pred, target, mask):
        """
            pred, target -> batch, seq_len, d
            mask -> batch, seq_len
        """
        mask = mask.unsqueeze(-1).expand(pred.shape[0], pred.shape[1], pred.shape[2]).float()

        loss = self.loss(pred*mask, target*mask) / (torch.sum(mask) + self.eps)

        return loss