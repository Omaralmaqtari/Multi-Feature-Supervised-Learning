# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:52:09 2026

@author: Omar Al-maqtari
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# R: Results
# GT: Ground Truth

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, r, gt):
        r = r.float().view(r.shape[0], -1)
        gt = gt.float().view(gt.shape[0], -1)
        inter = (r * gt).sum(dim=1)
        union = r.sum(dim=1) + gt.sum(dim=1)
        Dc = (2.0 * inter + self.eps) / (union + self.eps)

        return 1.0 - Dc.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, r, gt):
        gt = gt.type(r.type())
        r = r.float().view(-1)
        gt = gt.float().view(-1)
        Focal = 0.

        bce = F.binary_cross_entropy(r, gt, reduction="none")
        pt = torch.exp(-bce)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * bce
        
        if self.reduction == "mean":
            Focal = loss.mean()
        if self.reduction == "sum":
            Focal = loss.sum()
            
        return Focal

class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, gt):
        return 1.0 - (r * gt).sum(dim=1).mean()

class DistillLoss(nn.Module):
    def __init__(self, w_cos = 1.):
        super().__init__()
        self.w_cos = w_cos
        self.cosine_loss = CosineLoss()

    def forward(self, r_patch, gt_patch):
        cosine_loss = self.cosine_loss(r_patch, gt_patch)
        total_loss = self.w_cos*cosine_loss
        return total_loss