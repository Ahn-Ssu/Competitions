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
import pytorch_lightning.loggers as pl_loggers

from monai.losses.dice import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.utils.enums import MetricReduction
from monai.data.utils import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose, Activations, EnsureType

class Segmentation_network(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.model = network
        self.args = args
        self.loss = DiceFocalLoss(to_onehot_y=False, softmax=True)
        self.dice_M = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
        
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=False, threshold=0.5)])
        self.post_label = Compose([EnsureType("tensor", device="cpu")])
        self.one_hot = AsDiscrete(to_onehot=3)
        ###########################

        self.loss_log = np.array([])
    
    def configure_optimizers(self) -> Any:
        optimizer = AdaBelief(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=self.args.weight_decay)
        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epoch, max_lr=self.args.init_lr, min_lr=self.args.init_lr*self.args.lr_dec_rate, warmup_steps=self.args.epoch//10, gamma=0.8)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, **kwargs: Any) -> STEP_OUTPUT:
        img, seg_y = batch['image'], batch['label']
        *_, y_hat = self.model(img)
        loss = self.loss(y_hat, seg_y)
        
        # self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        print(f'{self.loss_log=}')
        val_loss = np.mean(self.loss_log)
        print(f'{val_loss=}')
        # val_dice = self.dice_M.aggregate().item()
        val_dice = self.dice_M.aggregate()
        print(val_dice)
        dice0, dice1, dice2 = val_dice
        dealloc = lambda x: x.detach().cpu().item()
        dice0 = dealloc(dice0)
        dice1 = dealloc(dice1)
        dice2 = dealloc(dice2)
        val_dice = val_dice.mean().detach().cpu().item()
        # val_fpr, val_fnr = np.mean(self.fpr_log), np.mean(self.fnr_log)
        

        self.log_dict({'val_loss':val_loss,
                       'val_dice':val_dice,
                       'dice0':dice0,
                       'dice1':dice1,
                       'dice2':dice2
                       },
                       prog_bar=True, sync_dist=True)
        
        self.dice_M.reset()
        self.loss_log = np.array([])
        # self.fpr_log.clear()
        # self.fnr_log.clear()
    
    def _shared_eval_step(self, batch, batch_idx):
        img, seg_y = batch['image'], batch['label']
        *_, y_hat = self.model(img)
        loss = self.loss(y_hat, seg_y)
        # y_hat = sliding_window_inference(
        #                             inputs=image,
        #                             roi_size= (self.args.img_size,self.args.img_size,self.args.img_size), # (128, 128, 128)
        #                             sw_batch_size=2, # number of the multiprocessor
        #                             predictor= self.model,
        #                             overlap=0.5,
        #                             mode= "constant" # GAUSSIAN = "gaussian" 
        #                         )


        
        # for using the following code, you should use track_meta=False of EnsureTyped
        outputs = [self.post_pred(i) for i in decollate_batch(y_hat)]
        labels = [self.post_label(i) for i in decollate_batch(seg_y)]


        ## Softmax
        logit = F.softmax(y_hat, dim=1)
        logit = torch.argmax(logit, dim=1)
        self.dice_M(y_pred=outputs, y=labels)
        print(f'{loss.item()=}')
        print(f'before append {self.loss_log}')
        self.loss_log = np.append(self.loss_log, loss.item())
        print(f'After append {self.loss_log}')
        
        # self.fpr_log.append(fpr)
        # self.fnr_log.append(fnr)

        # [400, 400, D] -> [~250, ~250, D]
        
        # one_hoted = self.one_hot(logit)
        # print(one_hoted.shape)
        # unsqz_oh = self.one_hot(logit.unsqueeze(0))
        # print(unsqz_oh.shape)
        if batch_idx % 5 == 0:
            self.log_img_on_TB(img, seg_y, self.one_hot(logit).unsqueeze(0), batch_idx)


 

    def log_img_on_TB(self, img, seg_y, pred, batch_idx) -> None:
         
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')
        fn_tonumpy = lambda x: torch.as_tensor(x, dtype=torch.half, device='cpu').detach().numpy().transpose(0, 2, 3, 4, 1)
        seg_y = fn_tonumpy(seg_y)
        pred = fn_tonumpy(pred)
        img = fn_tonumpy(img)
        print(f'{pred.shape=}, {seg_y.shape=}, {np.unique(pred[..., 0])=}, {np.unique(pred[..., 1])=}, {np.unique(pred[..., 2])=}')
        
        
        
        tb_logger.add_images(f"img/{batch_idx}", img[0:2,:,:,80], self.current_epoch + 1, dataformats="NHWC")
        tb_logger.add_images(f"label/{batch_idx}", seg_y[0:2,:,:,80], self.current_epoch + 1, dataformats="NHWC")
        tb_logger.add_images(f"y_hat/{batch_idx}", pred[0:2,:,:,80], self.current_epoch + 1, dataformats="NHWC")
            
            
            
            



# confusion_vector = pred / seg_y
# true_positives = torch.sum(confusion_vector == 1).item()
# false_positives = torch.sum(confusion_vector == float('inf')).item()
# true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
# false_negatives = torch.sum(confusion_vector == 0).item()

# fpr = false_positives / (false_positives + true_negatives+ 1e-6)
# fnr = false_negatives / (false_negatives + true_positives+ 1e-6)