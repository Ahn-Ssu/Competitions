import torch
import torch.nn as nn
import torch.optim

import pytorch_lightning as pl

from monai.losses.dice import DiceLoss, DiceFocalLoss


from models import apollo

class LightningRunner(pl.LightningModule):
    def __init__(self, network, loss, optimizer, args) -> None:
        super().__init__()

        self.model = network(args)
        self.loss  = loss
        self.optimizer = optimizer
        self.loss = DiceFocalLoss()
        self.args = args

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)
        # return apollo(params=self.parameters, lr=self.args.lr,
        #                beta=0.9, eps=1e-4, rebound='constant', warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None)
    
    
    def tarining_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return 