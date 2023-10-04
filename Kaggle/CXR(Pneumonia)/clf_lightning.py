from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adamp import AdamP
from adabelief_pytorch import AdaBelief

import pytorch_lightning as pl

from torchmetrics.classification import BinaryAUROC, BinaryF1Score

class Classification_network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.loss = nn.CrossEntropyLoss()
        
        ###########################
        self.auroc = BinaryAUROC() # task='multiclass', num_classes=n
        self.f1score= BinaryF1Score()
        
        self.auroc_log = []
        self.f1score_log = []
        self.loss_log = []
    
    def configure_optimizers(self) -> Any:
        optimizer = AdaBelief(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch['image'], batch['y']
        device = x.device
        y = y.type(torch.LongTensor).to(device)
        y_hat = self.model(x)
        
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:

        val_loss = np.mean(self.loss_log)
        val_auroc = np.mean(self.auroc_log)
        val_f1score = np.mean(self.f1score_log)

        self.log_dict({'val_loss':val_loss,
                       'val_auroc':val_auroc,
                       'val_f1score':val_f1score}, prog_bar=True, sync_dist=True)
        
        self.loss_log.clear()
        self.auroc_log.clear()
        self.f1score_log.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch['image'], batch['y']
        device = x.device
        y = y.type(torch.LongTensor).to(device)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        
        # preds: Tensor, target: Tensor
        y_hat = F.softmax(y_hat, dim=1).argmax(dim=1)
        auroc = self.auroc(y_hat, y)
        f1score = self.f1score(y_hat, y)
        
        self.loss_log.append(loss.detach().cpu())
        self.auroc_log.append(auroc.detach().cpu())
        self.f1score_log.append(f1score.detach().cpu())
