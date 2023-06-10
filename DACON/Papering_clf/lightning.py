import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from timm.loss import AsymmetricLossSingleLabel
from timm.data import Mixup
from timm.data.random_erasing import RandomErasing

from sklearn.metrics import classification_report, f1_score
import numpy as np
from functools import partial
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from model.apollo import Apollo
from adamp import AdamP

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
    
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)  # CE(pt) = -log(pt) --> -ce_loss = log(pt) --> exp(log(pt)) --> pt
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss = FocalLoss()
        # self.loss = LADELoss(args.num_cls, args.prior)
        self.args = args
        self.traget_names=[f'cls{idx}' for idx in range(19)]

        mixup_args = {
            'mixup_alpha': 0.3,
            'cutmix_alpha': 0.5,
            'cutmix_minmax': None,
            'prob': 1.0,
            'switch_prob': 0.5,
            'mode': 'batch',
            'label_smoothing': 0.1,
            'num_classes': 19}

        self.mixup_fn = Mixup(**mixup_args)
        # self.rand_erase = RandomErasing(probability=0.5)

        # val archieve
        self.preds = []
        self.true_labels = []
        self.loss_log = []


    def configure_optimizers(self):
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=0.05)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epochs, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epochs//10, gamma=0.8)
        return [optimizer], [lr_scheduler]
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = self.rand_erase(x)

        if (x.shape[0] % 2 == 0):
                x, y = self.mixup_fn(x, y)

        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    def on_validation_epoch_end(self) -> None:

        val_loss = np.mean(self.loss_log)
        
        avg_f1 = f1_score(self.true_labels, self.preds, average='weighted')
        self.log_dict({'val_loss':val_loss, 'avg_f1':avg_f1}, prog_bar=True, sync_dist=True)

        if len(np.unique(self.true_labels)) == 19:
            report = classification_report(self.true_labels, self.preds, target_names=self.traget_names)
            report_lines = report.split('\n')

            for line in report_lines[2: 2+len(self.traget_names)]:
                cls_name, *cls_metrics, support = line.split()
                self.log_dict({
                    f'precision/{cls_name}': torch.Tensor([float(cls_metrics[0])]),
                    f'recall/{cls_name}': torch.Tensor([float(cls_metrics[1])]),
                    f'f1-score/{cls_name}': torch.Tensor([float(cls_metrics[2])]),
                    f'support/{cls_name}': torch.Tensor([float(support)]),
                })
            

        self.loss_log.clear()
        self.true_labels.clear()
        self.preds.clear()

    def _shared_eval_step(self, batch, batch_idx):
        imgs,  labels = batch

        y_hat = self.model(imgs)
        loss = self.loss(y_hat, labels)

        self.preds += y_hat.argmax(1).detach().cpu().numpy().tolist()
        self.true_labels += labels.detach().cpu().numpy().tolist()
        self.loss_log.append(loss.item())
