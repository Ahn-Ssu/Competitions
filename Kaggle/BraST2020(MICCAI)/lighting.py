import torch
import torch.nn as nn
import torch.optim

import pytorch_lightning as pl

from monai.losses.dice import DiceLoss, DiceFocalLoss
from torchmetrics.functional import dice, f1_score
from monai.transforms import (
                        Activations,
                        Compose,
                        AsDiscrete,
                    )

from models.apollo import Apollo


class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss  = DiceFocalLoss(smooth_nr=1e-5)
        self.args = args
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.args.init_lr)

        # lr scheduler_config = {}
        return Apollo(params=self.parameters(), lr=self.args.init_lr, beta=0.9, eps=1e-4, rebound='constant', warmup=10, init_lr=None, weight_decay=0, weight_decay_type=None)
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)

        return loss
    
    def training_step_end(self, step_output):
        return torch.mean(step_output)

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def validation_step_end(self, batch_parts):
        print(batch_parts)
        losses = batch_parts['loss']
        dice_score = batch_parts['dice']
        return torch.mean(dice_score)

    def validation_epoch_end(self, outputs) -> None:
        dice_scores= torch.mean(torch.stack(outputs))
        self.log_dict({'val_dice':dice_scores}, sync_dist=True)
        return 

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # pred = torch.argmax(y_hat, dim=1)
        dice_scores = dice(y_hat, y.int())
        return {'loss':loss,'dice':dice_scores}





# def training_step(self, batch, batch_idx):
#     x, y = batch
#     y_hat = self.model(x)
#     loss = F.cross_entropy(y_hat, y)
#     pred = ...
#     return {"loss": loss, "pred": pred}


# def training_step_end(self, batch_parts):
#     # predictions from each GPU
#     predictions = batch_parts["pred"]
#     # losses from each GPU
#     losses = batch_parts["loss"]

#     gpu_0_prediction = predictions[0]
#     gpu_1_prediction = predictions[1]

#     # do something with both outputs
#     return (losses[0] + losses[1]) / 2


# def training_epoch_end(self, training_step_outputs):
#     for out in training_step_outputs:
#         ...