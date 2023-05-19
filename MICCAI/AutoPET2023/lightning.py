from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn
import torch.optim as optim 

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adamp import AdamP

import pytorch_lightning as pl

from monai.losses.dice import DiceFocalLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils.enums import MetricReduction
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete # Compose, Activations

class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        
        self.model = network
        self.args = args
        ###########################
        self.loss = DiceFocalLoss(sigmoid=True)
        ###########################
        self.dice_M = DiceMetric()
        self.confusion_M = ConfusionMatrixMetric()
        self.discreter = AsDiscrete(threshold=0.5)
        ###########################

        self.loss_log = []
        self.dice = []
        self.fpr = []
        self.fnr = []

        # self.seg_GT = []
        # self.cls_GT = []
        # self.seg_pred = []
        # self.cls_pred = []
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, **kwargs: Any) -> STEP_OUTPUT:
        image, seg_y = batch['image'], batch['label']
        # image, seg_y, cls_y = batch['image'], batch['label'], batch['diagnosis']

        y_hat = self.model(image)
        loss = self.loss(y_hat, seg_y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch)

    def on_validation_epoch_end(self) -> None:

        val_loss = np.mean(self.loss_log)
        val_dice = np.mean(self.dice)
        val_fpr  = np.mean(self.fpr)
        val_fnr  = np.mean(self.fnr)

        self.log_dict({'val_loss':val_loss,
                       'val_dice':val_dice,
                       'val_False_Positive':val_fpr,
                       'val_False_Negative':val_fnr},
                       prog_bar=True, sync_dist=True)
        
        self.loss_log.clear()
        self.dice.clear()
        self.fpr.clear()
        self.fnr.clear()
    
    def _shared_eval_step(self, batch):
        # image, seg_y, cls_y = batch['image'], batch['label'], batch['diagnosis']
        image, seg_y = batch['image'], batch['label']


        y_hat = sliding_window_inference(
                                    inputs=image,
                                    roi_size=self.args.img_size, # (128, 128, 128)
                                    sw_batch_size=8, # number of the multiprocessor
                                    predictor= self.model,
                                    overlap=0.5,
                                    mode= "constant" # GAUSSIAN = "gaussian" 
                                )
        
        loss = self.loss(y_hat, seg_y)
        logit = self.discreter(y_hat)
        dice = self.dice_M(logit, seg_y)
        confusion_ret = self.confusion_M(logit, seg_y)
        tp, fp, tn, fn = confusion_ret[..., 0], confusion_ret[..., 1], confusion_ret[..., 2], confusion_ret[..., 3]

        fpr = fp / (fp+tn+1e-8)
        fnr = fn / (fn+tp+1e-8)

        self.loss_log.append(loss.detach().cpu().numpy())
        self.dice.append(dice.detach().cpu().numpy())
        self.fpr.append(fpr.detach().cpu().numpy())
        self.fnr.append(fnr.detach().cpu().numpy())
        ######################################################################
        ## 이렇게 metric을 기록하는 방법이랑 tensor들을 보관한다음에 전체를 processing하는 방법 중에 메모리 관리 생각하면서 확인 
        ######################################################################

        # self.seg_GT.append(seg_y)
        # self.cls_GT.append(cls_y)
        # self.seg_pred.append(y_hat)
        # self.cls_pred.append(None)



