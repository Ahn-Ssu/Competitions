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


class Generation_networks(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.CE_loss = nn.MultiLabelSoftMarginLoss(reduction='sum')
        self.MSE_loss = nn.MSELoss(reduction='sum')
        
        ###########################
        
        self.cat_fs = []
        self.num_fs = []
        self.recons = []
        self.muz = []
        self.log_varz = []
        
    def loss_calc(self, cat_f, num_f, recon_x, mu, log_var):
        #batch_size = x.size(0)
        #MSE_loss = MSE(x, recon_x.view(batch_size, 1, 28, 28))
        cat_size = cat_f.size(1)
        
        recon_cat = recon_x[..., :cat_size]
        recon_num = recon_x[..., cat_size:]

        CE_loss = self.CE_loss(F.sigmoid(recon_cat), cat_f)
        MSE_loss = self.MSE_loss(recon_num, num_f)
        KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE_loss + MSE_loss + KLD_loss, CE_loss, MSE_loss, KLD_loss
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        cat_f, num_f = batch['cat_f'], batch['num_f']
        outputs, mu, log_var = self.model(cat_f, num_f)
        total_loss, CE_loss, MSE_loss, KLD_loss = self.loss_calc(cat_f, num_f, outputs, mu, log_var)
        self.log("train_loss", total_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        self.log("train_CE", CE_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        self.log("train_MSE", MSE_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        self.log("train_KL", KLD_loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        
        cat_fs = torch.concat(self.cat_fs , dim=0)
        num_fs = torch.concat(self.num_fs, dim=0)
        recons = torch.concat(self.recons, dim=0)
        muz = torch.concat(self.muz, dim=0)
        log_varz = torch.concat(self.log_varz, dim=0)
        
        
        total_loss, CE_loss, MSE_loss, KLD_loss = self.loss_calc(cat_fs, num_fs, recons, muz, log_varz)
        
        epc = self.current_epoch
        
        
        recon_cat = recons[..., :4]
        recon_num = recons[..., 4:]
        
        recon_cat = F.sigmoid(recon_cat)
        recon_cat = torch.where(recon_cat > 0.5, 1, 0)
        recons = torch.concat([recon_cat, recon_num], dim=1)
        recons = recons.detach().cpu().numpy()
        
        GT = torch.concat([cat_fs, num_fs], dim=1).detach().cpu().numpy()
        
        np.savetxt(f'/root/Competitions/Kaggle/MRI-SNSB/out/VAE-Recons[epoch={epc}].csv', recons, delimiter=',')
        np.savetxt(f'/root/Competitions/Kaggle/MRI-SNSB/out/GT.csv', GT, delimiter=',')
        

        self.log_dict({'val_loss':total_loss.item(),
                       'val_CE':CE_loss.item(),
                       'val_MSE':MSE_loss.item(),
                       'val_KL':KLD_loss.item()}, prog_bar=True, sync_dist=True)
        
        self.cat_fs.clear()
        self.num_fs.clear()
        self.recons.clear()
        self.muz.clear()
        self.log_varz.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        cat_f, num_f = batch['cat_f'], batch['num_f']
        outputs, mu, log_var = self.model(cat_f, num_f)
        
        self.cat_fs.append(cat_f)
        self.num_fs.append(num_f)
        self.recons.append(outputs)
        self.muz.append(mu)
        self.log_varz.append(log_var)