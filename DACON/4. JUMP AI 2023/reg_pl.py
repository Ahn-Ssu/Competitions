from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn

import numpy as np

from torchmetrics.regression import MeanSquaredError
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adamp import AdamP


class Regression_network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        
        self.loss = nn.MSELoss()
        self.metric = MeanSquaredError(squared=False) # If True returns MSE value, if False returns RMSE value.
        
        self.rmse_log = [] 
        
    def configure_optimizers(self) -> Any:
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, **kwargs:Any) -> STEP_OUTPUT:
        MLM, HLM = batch.MLM, batch.HLM
        
        if self.args.MLM:
            y_hat = self.model(batch)
            loss = torch.sqrt(self.loss(y_hat, MLM))
        else:
            y_hat = self.model(batch)
            loss = torch.sqrt(self.loss(y_hat, HLM))
            
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch, batch_idx)
        
        
    def on_validation_batch_end(self, *args, **kwargs) -> None:
        val_loss = np.mean(self.rmse_log)
        self.log_dict({f'val_loss': val_loss}, prog_bar=True)
        
        self.rmse_log.clear()
        
    def _shared_eval_step(self, batch, batch_idx):
        
        MLM, HLM = batch.MLM, batch.HLM
        
        if self.args.MLM:
            y_hat = self.model(batch)
            loss = torch.sqrt(self.loss(y_hat, MLM))
        else:
            y_hat = self.model(batch)
            loss = torch.sqrt(self.loss(y_hat, HLM))
            
        self.rmse_log.append(loss.item())
        
