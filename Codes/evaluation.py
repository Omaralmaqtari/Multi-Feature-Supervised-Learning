# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 23:51:06 2026

@author: Omar al-maqtari
"""

import copy
import torch
import numpy as np

# R : Model Result
# GT : Ground Truth

class Evaluation(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.thresholdlist = np.linspace(0, 1, 51)
        self.loss = 0.
        self.Acc = 0.	# Accuracy
        self.Rc = 0.	# Recall (Sensitivity)
        self.Pr = 0. 	# Precision
        self.F1 = 0.    # F1-score
        self.IoU = 0.   # Intersection over Union (Jaccard Index)
        self.mIoU = 0.	# mean of Intersection over Union (mIoU)
        self.OIS = 0.   # 
        self.AIU = 0.   #
        self.Dc = 0.    # Dice Coefficient
        self.cs = 0.    # Cosine Similarity
        self.RC_all = 0
        self.PC_all = 0
        self.length = 0
        
    @torch.no_grad()
    def get_results(self, R, GT):
        Acc = 0.
        Rc = 0.
        Rc_all = []
        Pr = 0.
        Pr_all = []
        F1 = 0.
        OIS = 0.
        IoU = 0.
        AIU = 0.
        Dc = 0.
        GT_copy = copy.deepcopy(GT)
        
        for threshold in self.thresholdlist:
            R_copy = copy.deepcopy(R)
            R_copy[R_copy < threshold] = 0.
            R_copy[R_copy >= threshold] = 1.
            
            tp = torch.sum((R_copy==1.)&(GT_copy==1.)).item()
            tn = torch.sum((R_copy==0.)&(GT_copy==0.)).item()
            fp = torch.sum((R_copy==1.)&(GT_copy==0.)).item()
            fn = torch.sum((R_copy==0.)&(GT_copy==1.)).item()

            Acc_copy = ((tp + tn) / (tp + fp + fn + tn + 1e-12))
            Rc_copy = (tp / (tp + fn + 1e-12))
            Rc_all.append(Rc_copy)
            Pr_copy = (tp / (tp + fp + 1e-12))
            Pr_all.append(Pr_copy)
            F1_copy = ((2. * Rc_copy * Pr_copy) / (Rc_copy + Pr_copy + 1e-12))
            IoU_copy = (tp / (tp + fp + fn + 1e-12))
            Dc_copy = ((2. * tp) / (tp + tp + fp + fn + 1e-12))
            
            if threshold == 0.5:
                Acc = copy.deepcopy(Acc_copy)
                Rc = copy.deepcopy(Rc_copy)
                Pr = copy.deepcopy(Pr_copy)
                F1 = copy.deepcopy(F1_copy)
                IoU = copy.deepcopy(IoU_copy)
                Dc = copy.deepcopy(Dc_copy)
                
            if F1_copy > OIS:    
                OIS = copy.deepcopy(F1_copy)
                
            if IoU_copy > AIU:
                AIU = copy.deepcopy(IoU_copy)
                
        return [Acc, Rc, Rc_all, Pr, Pr_all, F1, OIS, IoU, AIU, Dc]
    
    def metrics(self, r, gt, total_loss):
        if self.cfg.learning_mode == 'Dist':
            r = r.detach()
            gt = gt.detach()
            self.loss += total_loss
            self.cs += (r * gt).sum(dim=1).mean()
            self.length += 1

            return [self.loss, self.cs, self.length]

        elif self.cfg.learning_mode in ['Seed', 'Seg']:
            r = r.detach().view(-1)
            gt = gt.detach().view(-1)
            results = self.get_results(r, gt)
            self.loss += total_loss
            self.Acc += results[0]
            self.Rc += results[1]
            self.Rc_all = results[2]
            self.Pr += results[3]
            self.Pr_all = results[4]
            self.F1 += results[5]
            self.OIS += results[6]
            self.IoU += results[7]
            self.AIU += results[8]
            self.Dc += results[9]
            self.length += 1

            return [self.loss, self.Acc, self.Rc, self.Rc_all, self.Pr, self.Pr_all, self.F1, self.OIS, self.IoU,
                    self.AIU, self.Dc, self.length]
            
    def metrics_avg(self, metric):
        if self.cfg.learning_mode == 'Dist':
            loss = (metric[0]/metric[-1]).item()
            cs = ((metric[1]/metric[-1])*100).item()

            return [loss, cs]

        elif self.cfg.learning_mode in ['Seed', 'Seg']:
            loss = (metric[0]/metric[-1]).item()
            Acc = (metric[1]/metric[-1])*100
            Rc = (metric[2]/metric[-1])*100
            Pr = (metric[4]/metric[-1])*100
            F1 = (metric[6]/metric[-1])*100
            OIS = (metric[7]/metric[-1])*100
            IoU = (metric[8]/metric[-1])*100
            AIU = (metric[9]/metric[-1])*100
            Dc = (metric[10]/metric[-1])*100

            return [loss, Acc, Rc, Pr, F1, OIS, IoU, AIU, Dc]
