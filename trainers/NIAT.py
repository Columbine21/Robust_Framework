import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer


class NIAT_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.adversarial_loss = nn.BCELoss()
        self.pixelwise_loss = nn.L1Loss()
        
    def do_train(self, model, dataloader):
        # OPTIMIZER: finetune Bert Parameters.
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.fusion.text_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.config.weight_decay_bert, 'lr': self.config.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.config.learning_rate_bert},
            {'params': model_params_other, 'weight_decay': self.config.weight_decay_other, 'lr': self.config.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)
        optimizer_D = optim.Adam(model.discriminator.parameters(), lr=self.config.learning_rate_other, weight_decay=self.config.weight_decay_other)
        # loop util earlystop
        while True: 
            self.epochs += 1
            # train
            y_pred, y_true = [], []
            avg_rloss, avg_closs, avg_dloss = [], [], []
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

                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                    # encoder-decoder
                    optimizer.zero_grad()

                    text, audio, vision = model.alignment_network(text, audio, vision, 
                                                            audio_lengths, vision_lengths)
                    text_m, audio_m, vision_m = model.alignment_network(text_m, audio_m, vision_m,
                                                            audio_lengths, vision_lengths)

                    fusion_feature_x = model.fusion(text, audio, vision)
                    fusion_feature_m = model.fusion(text_m, audio_m, vision_m)
                    
                    recon_fusion_f = model.reconstruction(fusion_feature_m)
                    rl1 = self.pixelwise_loss(fusion_feature_x, recon_fusion_f)
                    avg_rloss.append(rl1.item())

                    valid = torch.ones(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.config.device)
                    fake = torch.zeros(size=[labels.shape[0], 1], requires_grad=False).type_as(audio).to(self.config.device)

                    t = model.discriminator(fusion_feature_m)
                    advl1 = self.adversarial_loss(t, valid)
                    g_loss = 0.2 * (self.config.alpha * (advl1) + (1-self.config.alpha) * (rl1)) 
                    g_loss.backward(retain_graph=True)

                    optimizer_D.zero_grad()

                    output_x = model.classifier(fusion_feature_x)
                    y_pred.append(output_x.cpu())
                    y_true.append(labels.cpu())
                    output_m = model.classifier(fusion_feature_m)
                    y_pred.append(output_m.cpu())
                    y_true.append(labels.cpu())
                    c_loss = (self.criterion(output_x, labels)+self.criterion(output_m, labels) * self.config.beta) / (1 + self.config.beta)
                    avg_closs.append(c_loss.item())
                    c_loss.backward()

                    real_loss = self.adversarial_loss(model.discriminator(fusion_feature_x.clone().detach()), valid)
                    fake_loss = self.adversarial_loss(model.discriminator(fusion_feature_m.clone().detach()), fake)
                    
                    d_loss = 0.1 * (real_loss + fake_loss)
                    avg_dloss.append(d_loss.item())
                
                    if self.config.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.config.grad_clip)
                    optimizer.step()
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            self.logger.info("TRAIN-(%s) (%d/%d/%d)>> rloss: %.4f closs: %.4f dloss: %.4f %s" % (self.config.model, \
                        self.epochs - self.best_epoch, self.epochs, self.config.cur_seed, np.mean(avg_rloss), np.mean(avg_closs), np.mean(avg_dloss), dict_to_str(train_results)))
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
                    text, audio, vision = model.alignment_network(text, audio, vision, 
                                                            audio_lengths, vision_lengths)
                    fusion_feature = model.fusion(text, audio, vision)
                    prediction = model.classifier(fusion_feature)
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
