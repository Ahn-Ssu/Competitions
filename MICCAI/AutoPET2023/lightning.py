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

from monai.losses.dice import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils.enums import MetricReduction
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose, Activations, EnsureType

class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        
        self.model = network
        self.args = args
        self.loss = DiceFocalLoss(to_onehot_y=True, softmax=True)
        # self.loss =DiceLoss(sigmoid=True)
        ###########################
        self.dice_M = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.confusion_M = ConfusionMatrixMetric(metric_name=['fpr','fnr'])
        
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        ###########################

        self.loss_log = []
        self.fpr_log = []
        self.fnr_log = []
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, **kwargs: Any) -> STEP_OUTPUT:
        ct, pet, seg_y, clf_y = batch['ct'], batch['pet'], batch['label'], batch['diagnosis']
        images = torch.concat([ct,pet], dim=1)
        y_hat = self.model(images)
        loss = self.loss(y_hat, seg_y)
        ##### 이 이외엔 필요가 없지만.. 내가 모니터 해야함

        
        # pred = torch.argmax(y_hat, dim=1)
        # pred = torch.where(F.sigmoid(pred) > 0.5, 1., 0.)
        # print(torch.sum(seg_y))
        # print(f'\t IN training || loss  = {loss}, {loss.requires_grad=}')
        # print(f'\t IN training || label = {torch.sum(seg_y).item()}, {torch.unique(seg_y)}')
        # print(f'\t IN training || PRED  = {torch.sum(pred).item()}, {torch.unique(pred)}')
        # print(f'\t IN training || y_hat = {torch.unique(y_hat)}')
        #### Since the tarinig step losses are too oscilated, remove the logging code line
        # self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch)

    def on_validation_epoch_end(self) -> None:

        val_loss = np.mean(self.loss_log)
        val_dice = self.dice_M.aggregate().item()
        val_fpr, val_fnr = np.mean(self.fpr_log), np.mean(self.fnr_log)

        self.log_dict({'val_loss':val_loss,
                       'val_dice':val_dice,
                       'val_False_Positive':val_fpr,
                       'val_False_Negative':val_fnr},
                       prog_bar=True, sync_dist=True)
        
        self.dice_M.reset()
        self.loss_log.clear()
        self.fpr_log.clear()
        self.fnr_log.clear()
    
    def _shared_eval_step(self, batch):
        ct, pet, seg_y, clf_y = batch['ct'], batch['pet'], batch['label'], batch['diagnosis']
        image = torch.concat([ct,pet],dim=1)

        y_hat = sliding_window_inference(
                                    inputs=image,
                                    roi_size= (self.args.img_size,self.args.img_size,self.args.img_size), # (128, 128, 128)
                                    sw_batch_size=4, # number of the multiprocessor
                                    predictor= self.model,
                                    overlap=0.5,
                                    mode= "constant" # GAUSSIAN = "gaussian" 
                                )
        loss = self.loss(y_hat, seg_y)
        
        # for using the following code, you should use track_meta=False of EnsureTyped
        outputs = [self.post_pred(i) for i in decollate_batch(y_hat)]
        labels = [self.post_label(i) for i in decollate_batch(seg_y)]

        pred = torch.argmax(y_hat, dim=1)
        pred = torch.where(F.sigmoid(pred) > 0.5, 1., 0.)
        confusion_vector = pred / seg_y
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        fpr = false_positives / (false_positives + true_negatives+ 1e-6)
        fnr = false_negatives / (false_negatives + true_positives+ 1e-6)

        self.dice_M(y_pred=outputs, y=labels)
        # self.confusion_M(y_pred=pred.detach().cpu(), y=seg_y.squeeze(0).detach().cpu())
        self.loss_log.append(loss.item())
        self.fpr_log.append(fpr)
        self.fnr_log.append(fnr)


        print('**per prediction monitor**')
        print(f'\t in VALIDATION || input shape: {image.shape}')
        print(f'\t in VALIDATION || loss  = {loss}, {loss.requires_grad=}')
        print(f'\t in VALIDATION || label = {torch.sum(seg_y).item()}, {torch.unique(seg_y)}')
        print(f'\t in VALIDATION || PRED  = {torch.sum(pred).item()}, {torch.unique(pred)}')
        print(f'\t True  (pos, neg) || {true_positives=}, {true_negatives=}')
        print(f'\t False (pos, neg) || {false_positives=}, {false_negatives=}')
        print(f'\t TPR, TNR || {fpr=}, {fnr=}')
        


        # self.dice.append(dice.detach().cpu().numpy())
        # self.fpr.append(fpr.detach().cpu().numpy())
        # self.fnr.append(fnr.detach().cpu().numpy())


        # print(image.shape, seg_y.shape, y_hat.shape)
        # self.log_dict(
        #     {
        #         'monitor/input_CT': image[0,0, :, 200, :].detach().cpu().numpy(),
        #         'monitor/input_PET': image[0,1, :, 200, :].detach().cpu().numpy(),
        #         'monitor/seg_label': seg_y[0,:, 200, :].detach().cpu().numpy(),
        #         'monitor/seg_Pred': y_hat[0,:, 200, :].detach().cpu().numpy(),
        #     }
        # )
        # self.log('monitor/input_CT', image[0,0, :, 200, :].detach().cpu().numpy())
        # self.log('monitor/input_PET', image[0,1, :, 200, :].detach().cpu().numpy())
        # self.log('monitor/seg_label', seg_y[0,:, 200, :].detach().cpu().numpy())
        # self.log('monitor/seg_Pred', y_hat[0,:, 200, :].detach().cpu().numpy())
        ######################################################################
        ## 이렇게 metric을 기록하는 방법이랑 tensor들을 보관한다음에 전체를 processing하는 방법 중에 메모리 관리 생각하면서 확인 
        ######################################################################

        # self.seg_GT.append(seg_y)
        # self.cls_GT.append(cls_y)
        # self.seg_pred.append(y_hat)
        # self.cls_pred.append(None)



