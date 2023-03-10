import torch
import torch.nn as nn
import torch.optim

import pytorch_lightning as pl

from monai.losses.dice import DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.data.utils import decollate_batch
from torchmetrics.functional import dice, f1_score
from monai.transforms import (
                        Activations,
                        Compose,
                        AsDiscrete,
                    )

from models.apollo import Apollo
from models.cosine_anealing_warmup import CosineAnnealingWarmupRestarts
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss  = DiceFocalLoss(sigmoid=True)
        self.args = args
        self.post_trans = Compose([Activations(sigmoid=True), AsDiscrete(argmax=False,threshold=0.5)])
        self.dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.confusion = ConfusionMatrixMetric(include_background=True,metric_name=['sensitivity', 'specificity','precision'], get_not_nans=True, reduction=MetricReduction.MEAN_BATCH)
        self.hausdorff = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True, percentile=95.0)
        self.run_acc = AverageMeter()
        

    def configure_optimizers(self):
        optimizer = Apollo(params=self.parameters(), lr=self.args.init_lr, beta=0.9, eps=1e-4, rebound='constant', warmup=10, init_lr=None, weight_decay=0, weight_decay_type=None)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=100, max_lr=0.001, min_lr=1e-6, warmup_steps=20, gamma=0.9)
        return [optimizer], [lr_scheduler]
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        y_hat = self.model(x)

        if isinstance(y_hat, list):
            y_hat = y_hat[0]

        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def training_step_end(self, step_output):
        return torch.mean(step_output)

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval_step(batch, batch_idx)
        return metrics

    def validation_step_end(self, batch_parts):
        avg_dice = np.mean(batch_parts['avg_dice'])
        dice_tc  = np.mean(batch_parts['dice_tc'])
        dice_wt  = np.mean(batch_parts['dice_wt'])
        dice_et  = np.mean(batch_parts['dice_et'])
        return {'avg_dice':avg_dice, 'dice_tc':dice_tc, 'dice_wt':dice_wt, 'dice_et':dice_et}

    def validation_epoch_end(self, outputs) -> None:
        avg_dice = np.mean(np.stack([output['avg_dice'] for output in outputs]))
        dice_tc  = np.mean(np.stack([output['dice_tc'] for output in outputs]))
        dice_wt  = np.mean(np.stack([output['dice_wt'] for output in outputs]))
        dice_et  = np.mean(np.stack([output['dice_et'] for output in outputs]))
        self.log_dict({'avg_dice':avg_dice, 'dice_tc':dice_tc, 'dice_wt':dice_wt, 'dice_et':dice_et}, prog_bar=True, sync_dist=True, logger=True)
        return 

    def _shared_eval_step(self, batch, batch_idx):
        # Dice Similarity Coefficient
        # Hausdorff distance (95%)
        # Sensitivity
        # Specificity
        # Precision

        x, y = batch['image'], batch['label']
        y_hat = self.model(x)

        if isinstance(y_hat, list): # for UNet++
            y_hat = y_hat[0]
        

        labels_list = [one_label for one_label in y]
        preds_list = [one_pred for one_pred in y_hat]
        preds_converted = [self.post_trans(pred) for pred in preds_list]

        self.dice_acc.reset()
        self.dice_acc(y_pred=preds_converted, y=labels_list)
        acc, not_nans = self.dice_acc.aggregate()

        self.run_acc.reset()
        self.run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

        dice_tc = self.run_acc.avg[0]
        dice_wt = self.run_acc.avg[1]
        dice_et = self.run_acc.avg[2]
        avg_dice= np.average([dice_tc,dice_wt,dice_et])

        # self.hausdorff.reset()
        # self.hausdorff(y_pred=preds_converted, y=labels_list)
        # acc, not_nans = self.hausdorff.aggregate()

        # self.run_acc.reset()
        # self.run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

        # hausdorff_tc = self.run_acc.avg[0]
        # hausdorff_wt = self.run_acc.avg[1]
        # hausdorff_et = self.run_acc.avg[2]
        # avg_hausdorff= np.average([hausdorff_tc,hausdorff_wt,hausdorff_et])

        return {'avg_dice':avg_dice, 'dice_tc':dice_tc, 'dice_wt':dice_wt, 'dice_et':dice_et}



