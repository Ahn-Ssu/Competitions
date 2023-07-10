from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adamp import AdamP

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from modelgenesis_utils import get_pair
from monai.losses.ssim_loss import SSIMLoss


class Modelgenesis_network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.loss = nn.MSELoss()
        # self.loss = SSIMLoss() # Since the author said MSELoss is enough
        self.loss_log = []
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.0001, warmup_steps=self.args.epoch//20, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, **kwargs: Any) -> STEP_OUTPUT:
        ct, pet, seg_y, clf_y = batch['ct'], batch['pet'], batch['label'], batch['diagnosis']
        x, y = get_pair(img=ct, batch_size=self.args.batch_size,
                        config=self.args.genesis_args)
        
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        val_loss = np.mean(self.loss_log)

        self.log_dict({'val_loss':val_loss},
                       prog_bar=True, sync_dist=True)
        
        self.loss_log.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        ct, pet, seg_y, clf_y = batch['ct'], batch['pet'], batch['label'], batch['diagnosis']
        y_hat = self.model(ct)
        loss = self.loss(y_hat, ct)
        self.loss_log.append(loss.item())
        

    def log_img_on_TB(self, ct, pet, seg_y, pred, batch_idx) -> None:
         
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
       
        # Log the images (Give them different names)
        # [400, 400, D] -> [~250, ~250, D]
        # print(ct.shape)    # torch.Size([1, 1, 347, 347, 232])
        # print(pet.shape)   # torch.Size([1, 1, 347, 347, 232]) 
        # print(seg_y.shape) # torch.Size([1, 1, 347, 347, 232])
        # print(pred.size()) # torch.Size([1, 347, 347, 232])

        ct = ct.squeeze(0)
        pet = pet.squeeze(0)
        seg_y = seg_y.squeeze(0)

        C, W, H, D = ct.size()
        target_idx = [idx for idx in range(H//5, H, H//5)]

        for vol_idx in target_idx:
            # add_images('title', data', dataformats='NCHW)
            tb_logger.add_image(f"CT/BZ[{batch_idx}]_{vol_idx}", ct[..., vol_idx, :], dataformats='CHW')
            tb_logger.add_image(f"PET/BZ[{batch_idx}]_{vol_idx}", pet[..., vol_idx, :], dataformats='CHW')
            tb_logger.add_image(f"GroundTruth/BZ[{batch_idx}]_{vol_idx}", torch.where(seg_y[..., vol_idx, :] ==1, 255, 0), dataformats='CHW')
            tb_logger.add_image(f"Prediction/BZ[{batch_idx}]_{vol_idx}", torch.where(pred[..., vol_idx, :]==1, 255, 0), dataformats='CHW')




