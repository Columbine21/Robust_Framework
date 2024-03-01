import torch
from tqdm import tqdm
from torch import optim
from utils.functions import dict_to_str
from trainers.base_trainer import BaseTrainer


class CTFN_Trainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        
    def do_train(self, model, dataloader):
        optimizer_a2t = optim.Adam(model.a2t_model.parameters(), self.config.trans_lr, (self.config.beta1, self.config.beta2))
        optimizer_a2v = optim.Adam(model.a2v_model.parameters(), self.config.trans_lr, (self.config.beta1, self.config.beta2))
        optimizer_v2t = optim.Adam(model.v2t_model.parameters(), self.config.trans_lr, (self.config.beta1, self.config.beta2))
        optimizer_clf = optim.Adam(model.sa_model.parameters(), lr=self.config.lr)
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
                    text, audio, vision = model.align_subnet(text, audio, vision, 
                                                           audio_lengths, vision_lengths)
                    with torch.no_grad():
                        text = model.text_model(text)
                    # Training Translation Network.
                    self.train_transnet(model.a2t_model, optimizer_a2t, audio, text)
                    self.train_transnet(model.a2v_model, optimizer_a2v, audio, vision)
                    self.train_transnet(model.v2t_model, optimizer_v2t, vision, text)
                    
                    # Overall pipeline updating.
                    optimizer_clf.zero_grad()
                    optimizer_a2t.zero_grad()
                    optimizer_a2v.zero_grad()
                    optimizer_v2t.zero_grad()

                    a2t_fake_a, a2t_fake_t, bimodal_at, bimodal_ta = self.trans_fusion(model.a2t_model, optimizer_a2t, audio, text, need_grad=True)
                    audio_a2t = self.specific_modal_fusion(audio, a2t_fake_a, bimodal_ta)
                    text_a2t = self.specific_modal_fusion(text, a2t_fake_t, bimodal_at)
                    a2v_fake_a, a2v_fake_v, bimodal_av, bimodal_va = self.trans_fusion(model.a2v_model, optimizer_a2v, audio, vision, need_grad=True)
                    audio_a2v = self.specific_modal_fusion(audio, a2v_fake_a, bimodal_va)
                    vision_a2v = self.specific_modal_fusion(vision, a2v_fake_v, bimodal_av)
                    v2t_fake_v, v2t_fake_t, bimodal_vt, bimodal_tv = self.trans_fusion(model.v2t_model, optimizer_v2t, vision, text, need_grad=True)
                    vision_v2t = self.specific_modal_fusion(vision, v2t_fake_v, bimodal_tv)
                    text_v2t = self.specific_modal_fusion(text, v2t_fake_t, bimodal_vt)
                    prediction = model.sa_model(audio, text, vision, audio_a2t, text_a2t, vision_v2t, text_v2t, audio_a2v, vision_a2v)

                    loss = self.criterion(prediction, labels)
                    loss.backward()
                    optimizer_clf.step()
                    optimizer_a2t.step()
                    optimizer_a2v.step()
                    optimizer_v2t.step()
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
                    text, audio, vision = model.align_subnet(text, audio, vision, 
                                                           audio_lengths, vision_lengths)
                    
                    text = model.text_model(text)

                    a2t_fake_a, a2t_fake_t, bimodal_at, bimodal_ta = self.trans_fusion(model.a2t_model, None, audio, text, need_grad=False)
                    audio_a2t = self.specific_modal_fusion(audio, a2t_fake_a, bimodal_ta)
                    text_a2t = self.specific_modal_fusion(text, a2t_fake_t, bimodal_at)
                    a2v_fake_a, a2v_fake_v, bimodal_av, bimodal_va = self.trans_fusion(model.a2v_model, None, audio, vision, need_grad=False)
                    audio_a2v = self.specific_modal_fusion(audio, a2v_fake_a, bimodal_va)
                    vision_a2v = self.specific_modal_fusion(vision, a2v_fake_v, bimodal_av)
                    v2t_fake_v, v2t_fake_t, bimodal_vt, bimodal_tv = self.trans_fusion(model.v2t_model, None, vision, text, need_grad=False)
                    vision_v2t = self.specific_modal_fusion(vision, v2t_fake_v, bimodal_tv)
                    text_v2t = self.specific_modal_fusion(text, v2t_fake_t, bimodal_vt)

                    prediction = model.sa_model(audio, text, vision, audio_a2t, text_a2t, vision_v2t, text_v2t, audio_a2v, vision_a2v)
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
    
    def train_transnet(self, trans_net, optimizer, source, target):
        optimizer.zero_grad()
        fake_target, _ = trans_net[0](source)
        recon_source, _ = trans_net[1](fake_target)
        g_loss = torch.mean((source-recon_source)**2)
        g_loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        fake_target, _ = trans_net[1](target)
        recon_source, _ = trans_net[0](fake_target)
        g_loss = torch.mean((target-recon_source)**2)
        g_loss.backward()
        optimizer.step()

    def trans_fusion(self, trans_net, optimizer, source, target, need_grad=True):
        if need_grad:
            optimizer.zero_grad()
            fake_target, bimodal_12 = trans_net[0](source)
            fake_source, bimodal_21 = trans_net[1](target)

        else:
            trans_net.eval()
            with torch.no_grad():
                fake_target, bimodal_12 = trans_net[0](source)
                fake_source, bimodal_21 = trans_net[1](target)

        return fake_source, fake_target, bimodal_12, bimodal_21

    def specific_modal_fusion(self, true_data, fake_data, mid_data):
        alphas = torch.sum(torch.abs(true_data - fake_data), (1, 2))
        alphas_sum = torch.sum(alphas)
        alphas = torch.div(alphas, alphas_sum).unsqueeze(-1).unsqueeze(-1)
        return torch.mul(alphas, mid_data[-1])