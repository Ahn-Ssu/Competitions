from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
import torch.nn as nn

import pytorch_lightning as pl

from sklearn.metrics import classification_report, accuracy_score

from adamp import AdamP
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss = nn.CrossEntropyLoss()
        self.args = args

        # validation archieve
        self.preds = []
        self.true_labels = []
        self.losses = [] 
        self.traget_names=[f'cls{idx}' for idx in range(8)]

    def configure_optimizers(self):
        optimizer = AdamP(params=self.parameters(), lr=self.args.init_lr, betas=[0.9, 0.999], weight_decay=0.05)
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epochs, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epochs//10, gamma=0.8)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y = batch 

        logits = self.model(x)
        loss = self.loss(logits, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch)
    
    # def on_validation_batch_end(self, outputs: STEP_OUTPUT | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
    #     return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    def on_validation_batch_end(self) -> None:

        val_loss = np.mean(self.losses)
        val_acc = accuracy_score(self.true_labels, self.preds)

        self.log_dict({'val_loss':val_loss, 'val_acc':val_acc}, prog_bar=True, sync_dist=True)

        if len(np.unique(self.true_labels)) == 8:
            report = classification_report(self.true_labels, self.preds, self.traget_names)
            report_lines = report.split('\n')

            for line in report_lines[2: 2+len(self.traget_names)]:
                cls_name, *cls_metrics, support = line.split()
                self.log_dict({
                    f'precision/{cls_name}': torch.Tensor([float(cls_metrics[0])]),
                    f'recall/{cls_name}': torch.Tensor([float(cls_metrics[1])]),
                    f'f1-score/{cls_name}': torch.Tensor([float(cls_metrics[2])]),
                    f'support/{cls_name}': torch.Tensor([float(support)]),
                })

        self.losses.clear()
        self.true_labels.clear()
        self.preds.clear()

    def _shared_eval_step(self, batch):
        x,  labels = batch

        y_hat = self.model(x)
        loss = self.loss(y_hat, labels)

        self.preds += y_hat.argmax(1).detach().cpu().numpy().tolist()
        self.true_labels += labels.detach().cpu().numpy().tolist()
        self.losses.append(loss.item())