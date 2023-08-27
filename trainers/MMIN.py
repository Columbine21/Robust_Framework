import copy
import torch
import torch.nn as nn

from tqdm import tqdm
from torch import optim
from torch.optim import lr_scheduler
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer

class MMIN_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.recon_loss = nn.MSELoss()

    def do_train(self, model, dataloader):
        # Phase One. Training.
        parameters_p1 = [{'params': getattr(model, 'net'+net).parameters()} for net in self.config.module_names_p1]
        optimizer_p1 = optim.Adam(parameters_p1, lr=self.config.lr, betas=(self.config.beta1, 0.999))
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - self.config.niter) / float(self.config.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer_p1, lr_lambda=lambda_rule)
        # loop util earlystop
        for epoch in range(0, self.config.niter + self.config.niter_decay):
            # train
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    text = batch_data['text'].to(self.config.device)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                    # clear gradient
                    optimizer_p1.zero_grad()
                    
                    feat_A = model.netA(audio)
                    feat_T = model.netL(model.text_model(text))
                    feat_V = model.netV(vision)
                    prediction = model.netC1(torch.cat([feat_A, feat_T, feat_V], dim=-1))
                    loss = self.criterion(prediction, labels)
                    loss.backward()

                    for module in self.config.module_names_p1:
                        torch.nn.utils.clip_grad_norm_(getattr(model, 'net'+module).parameters(), 0.5)

                    optimizer_p1.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())

            # update learning rates.
            scheduler.step()
            lr = optimizer_p1.param_groups[0]['lr']
            self.logger.info('learning rate = %.4f' % lr)
                    
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            self.logger.info(
                f"TRAIN-({self.config.model})-Phase one (Utter Fusion)  [{epoch + 1}/{self.config.niter + self.config.niter_decay}/{self.config.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_valid(model, dataloader['valid'], phase='phase_one')
            # save best model
            cur_valid = val_results[self.config.KeyEval]
            isBetter = cur_valid <= (self.best_valid - 1e-6)
            # save best model
            if isBetter:
                self.best_valid, self.best_epoch = cur_valid, epoch
                # save model
                torch.save(model.cpu().state_dict(), self.config.model_save_path)
                model.to(self.config.device)
            
        # Phase Two. Training.
        parameters_p2 = [{'params': getattr(model, 'net'+net).parameters()} for net in self.config.module_names_p2]
        optimizer_p2 = optim.Adam(parameters_p2, lr=self.config.lr, betas=(self.config.beta1, 0.999))

        # Load Previous Best Parameters.
        model.load_state_dict(torch.load(self.config.model_save_path))
        model.to(self.config.device)

        # Reset self.best_valid, best_epoch
        self.epochs, self.best_epoch, self.best_valid = 0, 0, 1e8

        pretrained_model = copy.deepcopy(model)

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
                    
                    batch_size = vision.shape[0]

                    vision = torch.tile(vision, dims=(6,1,1))
                    audio = torch.tile(audio, dims=(6,1,1))
                    text = torch.tile(model.text_model(text), dims=(6,1,1))

                    missing_index = torch.tile(torch.Tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1]]), (batch_size, 1)).to(self.config.device)
                    audio_miss_index = missing_index[:, 0].unsqueeze(1).unsqueeze(2)
                    audio_miss = audio * audio_miss_index
                    audio_reverse = audio * -1 * (audio_miss_index - 1)

                    text_miss_index = missing_index[:, 2].unsqueeze(1).unsqueeze(2)
                    text_miss = text * text_miss_index
                    text_reverse = text * -1 * (text_miss_index - 1)
                    # V modality
                    vision_miss_index = missing_index[:, 1].unsqueeze(1).unsqueeze(2)
                    vision_miss = vision * vision_miss_index
                    vision_reverse = vision * -1 * (vision_miss_index - 1)

                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)

                    labels = torch.tile(labels, (6, 1))
                    # clear gradient
                    optimizer_p2.zero_grad()
                    # compute loss, backward, update.
                    feat_A = model.netA(audio_miss)
                    feat_T = model.netL(text_miss)
                    feat_V = model.netV(vision_miss)
                    feat_fusion = torch.cat([feat_A, feat_T, feat_V], dim=-1)
                    recon_fusion, latent = model.netAE(feat_fusion)
                    recon_cycle, _ = model.netAE_cycle(recon_fusion)
                    prediction = model.netC2(latent)

                    with torch.no_grad():
                        T_embd_A = pretrained_model.netA(audio_reverse)
                        T_embd_L = pretrained_model.netL(text_reverse)
                        T_embd_V = pretrained_model.netV(vision_reverse)
                        T_embds = torch.cat([T_embd_A, T_embd_L, T_embd_V], dim=-1)

                    loss_mse = self.recon_loss(T_embds, recon_fusion) 
                    loss_cycle = self.recon_loss(feat_fusion, recon_cycle)

                    loss = self.config.ce_weight * self.criterion(prediction, labels) \
                        + self.config.mse_weight * loss_mse + self.config.cycle_weight * loss_cycle
                    
                    loss.backward()
                    optimizer_p2.step()
                    # store results
                    train_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
                    
            train_loss = train_loss / len(dataloader['train'])
            
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            self.logger.info(
                f"TRAIN-({self.config.model})-Phase two (MMIN) [{self.epochs - self.best_epoch}/{self.epochs}/{self.config.cur_seed}] >> loss: {round(train_loss, 4)} {dict_to_str(train_results)}"
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


    def do_valid(self, model, dataloader, phase='phase_two', mode='EVAL'):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.config.device)
                    audio = batch_data['audio'].to(self.config.device)
                    text = batch_data['text'].to(self.config.device)
                    labels = batch_data['labels']['M'].to(self.config.device)
                    if self.config.task == 'MIR': # Classification Task.
                        labels = labels.view(-1).long()
                    else: # Regression Task.
                        labels = labels.view(-1, 1)
                    
                    if phase == 'phase_one':
                        feat_A = model.netA(audio)
                        feat_T = model.netL(model.text_model(text))
                        feat_V = model.netV(vision)
                        prediction = model.netC1(torch.cat([feat_A, feat_T, feat_V], dim=-1))
                    else:
                        feat_A = model.netA(audio)
                        feat_T = model.netL(model.text_model(text))
                        feat_V = model.netV(vision)
                        _, latent = model.netAE(torch.cat([feat_A, feat_T, feat_V], dim=-1))
                        prediction = model.netC2(latent)
                    
                    loss = self.criterion(prediction, labels)
                    eval_loss += loss.item()
                    y_pred.append(prediction.cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        self.logger.info(f"{mode}-({self.config.model})-({phase}) >> {dict_to_str(eval_results)}")

        return eval_results
