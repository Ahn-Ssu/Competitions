import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl

# from torchmetrics.functional import dice, f1_score
from sklearn.metrics import f1_score
import numpy as np
from functools import partial
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)  # CE(pt) = -log(pt) --> -ce_loss = log(pt) --> exp(log(pt)) --> pt
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss = FocalLoss()
        self.args = args

        # self.mixup_fn = Mixup(**mixup_args)
        # self.rand_erase = RandomE

        # val archieve
        self.preds = []
        self.true_labels = []
        self.loss_log = []


    def configure_optimizers(self):
        # optimizer = Apollo(params=self.parameters(), lr=self.args.init_lr, beta=0.9, eps=1e-4, rebound='constant', warmup=10, init_lr=None, weight_decay=0, weight_decay_type=None)
        optimizer = optim.AdamW(params=self.parameters(), lr=self.args.init_lr,betas=[0.9, 0.999], weight_decay=0.05)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epochs, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epochs//10, gamma=0.8)
        return [optimizer], [lr_scheduler]
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f'in training x.y shape: {x.shape}, {y.shape}')
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    # def training_step_end(self, step_output):
    #     ret = torch.mean(step_output)
    #     return torch.mean(step_output)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    # def validation_step_end(self, batch_parts):
    #     avg_dice = np.mean(batch_parts['avg_dice'])
    #     dice_tc  = np.mean(batch_parts['dice_tc'])
    #     dice_wt  = np.mean(batch_parts['dice_wt'])
    #     dice_et  = np.mean(batch_parts['dice_et'])
    #     return {'avg_dice':avg_dice, 'dice_tc':dice_tc, 'dice_wt':dice_wt, 'dice_et':dice_et}

    def on_validation_epoch_end(self) -> None:
        # avg_dice = np.mean(np.stack([output['avg_dice'] for output in outputs]))
        # dice_tc  = np.mean(np.stack([output['dice_tc'] for output in outputs]))
        # dice_wt  = np.mean(np.stack([output['dice_wt'] for output in outputs]))
        # dice_et  = np.mean(np.stack([output['dice_et'] for output in outputs]))

        val_loss = np.mean(self.loss_log)
        avg_f1 = f1_score(self.true_labels, self.preds, average='weighted')
        # print(val_loss, avg_f1)
        self.log_dict({'val_loss':val_loss, 'avg_f1':avg_f1}, prog_bar=True, sync_dist=True)

        self.loss_log = []
        self.true_labels = []
        self.preds = []

    def _shared_eval_step(self, batch, batch_idx):
        imgs,  labels = batch

        y_hat = self.model(imgs)
        loss = self.loss(y_hat, labels)

        self.preds += y_hat.argmax(1).detach().cpu().numpy().tolist()
        self.true_labels += labels.detach().cpu().numpy().tolist()
        self.loss_log.append(loss.item())

        # y_hat = y_hat.argmax(1).detach().cpu().numpy().tolist()
        # loss = loss.detach().cpu().numpy().tolist()
        # labels = labels.detach().cpu().numpy().tolist()
        
        # f1_ = f1_score(y_hat, labels, average='weighted')



