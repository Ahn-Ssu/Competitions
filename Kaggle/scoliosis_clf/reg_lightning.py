from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from monai.losses import FocalLoss

from model.cosineLR import CosineAnnealingWarmupRestarts
from adamp import AdamP
from adabelief_pytorch import AdaBelief

import pytorch_lightning as pl

from torchmetrics.functional import auroc, f1, r2_score
from sklearn import metrics

class Regression_Network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.loss = nn.L1Loss() 
        
        ###########################
        
        self.preds = []
        self.reg_targets = []
        self.clf_targets = []

    
    def configure_optimizers(self) -> Any:
        optimizer = AdaBelief(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, _, cobbs = batch['image'], batch['y'], batch['cobbs']
        y_hat = self.model(x)
        loss = self.loss(y_hat, cobbs)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        
        preds = torch.cat(self.preds, dim=0)
        reg_targets = torch.cat(self.reg_targets, dim=0)
        clf_targets = torch.cat(self.clf_targets, dim=0)
        
        
        val_loss = self.loss(preds, reg_targets).item()
        val_r2   = r2_score(preds, reg_targets).item()
        
        clf_preds = torch.where(preds > 0.2, 1, 0)
        clf_targets = torch.where(clf_targets > 0 , 1 , 0)
        val_f1 = f1(clf_preds, clf_targets).item()
        val_auroc = auroc(clf_preds, clf_targets, pos_label=1).item()
        

        self.log_dict({'val_loss':val_loss,
                       'val_r2':val_r2,
                       'val_auroc':val_auroc,
                       'val_f1score':val_f1,},
                      prog_bar=True, sync_dist=True)
        
        self.preds.clear()
        self.reg_targets.clear()
        self.clf_targets.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y, cobbs = batch['image'], batch['y'], batch['cobbs']
        y_hat = self.model(x)
        
        self.preds.append(y_hat)
        self.reg_targets.append(cobbs)
        self.clf_targets.append(y)
        
