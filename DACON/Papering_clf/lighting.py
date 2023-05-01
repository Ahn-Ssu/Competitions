import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl
from timm.loss import AsymmetricLossSingleLabel
from timm.data import Mixup

# from torchmetrics.functional import dice, f1_score
from sklearn.metrics import classification_report, f1_score
import numpy as np
from functools import partial
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

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


import functools

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from utils import *


class LADELoss(nn.Module):
    def __init__(self, num_classes=10, prior=None, remine_lambda=0.1):
        super().__init__()
        if prior is not None:
            self.img_num_per_cls = torch.Tensor(prior).squeeze().cuda()
            self.prior = self.img_num_per_cls / self.img_num_per_cls.sum()
        else:
            self.prior = None

        self.balanced_prior = torch.tensor(1. / num_classes).float().cuda()
        self.remine_lambda = remine_lambda

        self.num_classes = num_classes
        self.cls_weight = (self.img_num_per_cls.float() / torch.sum(self.img_num_per_cls.float())).cuda()

    def mine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        N = x_p.size(-1)
        first_term = torch.sum(x_p, -1) / (num_samples_per_cls + 1e-8)
        second_term = torch.logsumexp(x_q, -1) - np.log(N)

        return first_term - second_term, first_term, second_term

    def remine_lower_bound(self, x_p, x_q, num_samples_per_cls):
        loss, first_term, second_term = self.mine_lower_bound(x_p, x_q, num_samples_per_cls)
        reg = (second_term ** 2) * self.remine_lambda
        return loss - reg, first_term, second_term

    def forward(self, y_pred, target, q_pred=None):
        """
        y_pred: N x C
        target: N
        """
        per_cls_pred_spread = y_pred.T * (target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target))  # C x N
        # print(self.prior.device, self.balanced_prior.device)
        pred_spread = (y_pred - torch.log(self.prior + 1e-9) + torch.log(self.balanced_prior + 1e-9)).T  # C x N

        num_samples_per_cls = torch.sum(target == torch.arange(0, self.num_classes).view(-1, 1).type_as(target), -1).float()  # C
        estim_loss, first_term, second_term = self.remine_lower_bound(per_cls_pred_spread, pred_spread, num_samples_per_cls)

        loss = -torch.sum(estim_loss * self.cls_weight)
        return loss
    
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
 
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
 
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
 
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
 
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
 
        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
 
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
 
        return -loss.sum()
class LightningRunner(pl.LightningModule):
    def __init__(self, network, args) -> None:
        super().__init__()

        self.model = network
        self.loss = FocalLoss()
        # self.loss = LADELoss(args.num_cls, args.prior)
        self.args = args
        self.traget_names=[f'cls{idx}' for idx in range(19)]

        # mixup_args = {
        #     'mixup_alpha': 0.8,
        #     'cutmix_alpha': 1,
        #     'cutmix_minmax': None,
        #     'prob': 1.0,
        #     'switch_prob': 0.5,
        #     'mode': 'elem',
        #     'label_smoothing': 0.1,
        #     'num_classes': 19}

        # self.mixup_fn = Mixup(**mixup_args)
        # self.rand_erase = RandomE

        # val archieve
        self.preds = []
        self.true_labels = []
        self.loss_log = []


    def configure_optimizers(self):
        # optimizer = Apollo(params=self.parameters(), lr=self.args.init_lr, beta=0.9, eps=1e-4, rebound='constant', warmup=10, init_lr=None, weight_decay=0, weight_decay_type=None)
        optimizer = optim.AdamW(params=self.parameters(), lr=self.args.init_lr,betas=[0.9, 0.999], weight_decay=0.05)
        # optimizer = AsymmetricLoss()
        lr_scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=self.args.epochs, max_lr=self.args.init_lr, min_lr=self.args.init_lr*0.001, warmup_steps=self.args.epochs//10, gamma=0.8)
        return [optimizer], [lr_scheduler]
    
    
    def training_step(self, batch, batch_idx):

        x, y = batch
        # print(batch)


        # if (x.shape[0] % 2 == 0):
        #         x, y = self.mixup_fn(x, y)
        
        # print(f'in training x.y shape: {x.shape}, {y.shape}')
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size, sync_dist=True)
        return loss
    
    # def training_step_end(self, step_output):
    #     ret = torch.mean(step_output)
    #     return torch.mean(step_output)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx)

    # def validation_step_end(self, batch_parts):
    #     avg_dice = np.mean(batch_parts['avg_dice'])
    #     dice_tc  = np.mean(batch_parts['dice_tc'])
    #     dice_wt  = np.mean(batch_parts['dice_wt'])
    #     dice_et  = np.mean(batch_parts['dice_et'])
    #     return {'avg_dice':avg_dice, 'dice_tc':dice_tc, 'dice_wt':dice_wt, 'dice_et':dice_et}
    

    def on_validation_epoch_end(self) -> None:
        # avg_dice = np.mean(np.stack([output['avg_dice'] for output in outputs]))
        # dice_tc  = np.mean(np.stack([output['dice_tc'] for output in outputs]))
        # dice_wt  = np.mean(np.stack([output['dice_wt'] for output in outputs]))
        # dice_et  = np.mean(np.stack([output['dice_et'] for output in outputs]))

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
        # print(y_hat.shape, labels.shape)
        # print(y_hat.device, labels.device)
        loss = self.loss(y_hat, labels)

        self.preds += y_hat.argmax(1).detach().cpu().numpy().tolist()
        self.true_labels += labels.detach().cpu().numpy().tolist()
        self.loss_log.append(loss.item())

        # y_hat = y_hat.argmax(1).detach().cpu().numpy().tolist()
        # loss = loss.detach().cpu().numpy().tolist()
        # labels = labels.detach().cpu().numpy().tolist()
        
        # f1_ = f1_score(y_hat, labels, average='weighted')



