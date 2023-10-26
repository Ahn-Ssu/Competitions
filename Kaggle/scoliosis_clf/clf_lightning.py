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

from torchmetrics.functional import auroc, f1
from sklearn import metrics

class Classification_network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.loss = FocalLoss()
        
        ###########################
        
        self.preds = []
        self.clf_targets = []
    
    def configure_optimizers(self) -> Any:
        optimizer = AdaBelief(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        x, _, targets = batch['image'], batch['y'], batch['clss_onehot']
        y_hat = self.model(x)
        loss = self.loss(y_hat, targets)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        
        preds = torch.stack(self.preds, dim=0)
        clf_targets = torch.stack(self.clf_targets, dim=0)
        
        val_loss = self.loss(preds, clf_targets).item()
        
        logits = preds.argmax(dim=1)
        classes = clf_targets.argmax(dim=1)
        
        val_f1_multi = f1(logits, classes, num_classes=4).item()
        val_auroc_multi = auroc(logits, classes, num_classes=4).item()
        
        binary_preds = torch.where(logits > 0, 1, 0)
        binary_class = torch.where(clf_targets.argmax(dim=1) > 0, 1, 0)
        
        val_f1_binary = f1(binary_preds, binary_class, num_classes=2).item()
        val_auroc_binary = auroc(binary_preds, binary_class, pos_label=1).item()

        self.log_dict({'val_loss':val_loss,
                       'val_auroc_multi':val_auroc_multi,
                       'val_f1_multi':val_f1_multi,
                       'val_f1_binary':val_f1_binary,
                       'val_auroc_binary':val_auroc_binary}, prog_bar=True, sync_dist=True)
        
        self.preds.clear()
        self.clf_targets.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        x, _, targets = batch['image'], batch['y'], batch['clss_onehot']
        y_hat = self.model(x)
        
        self.preds.append(y_hat)
        self.clf_targets.append(targets)