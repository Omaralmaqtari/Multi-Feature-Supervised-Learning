# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:41:44 2026

@author: Omar Al-maqtari
"""

import numpy as np
from model import M
from swin_transformer_moe import init_SwinTransformerMoE
from swin_transformer_v2 import init_SwinTransformerV2
from mobilevitv3_v2 import MobileViTv3_v2
from torch import optim
from scheduler import LinearWarmupCosineDecay
from hybrid_optimizer import (
    build_muon_adam_optimizer,
    CompositeReduceLROnPlateau,
    CompositeLinearWarmupCosineDecay,
    CompositeCosineAnnealingWarmRestarts)
import matplotlib.pyplot as plt

class initialize_model(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.optimizer_type = cfg.optimizer_type
        self.lr_sch_type = cfg.lr_sch_type
        self.model = None
        self.optimizer = None
        self.lr_sch = None
        self.optimizer_stats = None

    def get_model(self, model_type):
        if model_type == 'M':
            self.model = M(self.cfg, self.cfg.img_in, self.cfg.ch_out)

        elif model_type == 'Swin-MoE':
            self.model = None # Replace with SwinTransformerMoE() call

        elif model_type == 'Swin-v2':
            self.model = None # Replace with SwinTransformerV2() call

        elif model_type == 'MobileViTv3':
            self.model = None # Replace with MobileViTv3_v2(image_size=(256, 256), width_multiplier=1, patch_size=(2, 2)) call

        if model_type == 'M' and self.cfg.learning_mode in ['Seed','Seg']:
            for p in self.model.E.parameters():
                p.requires_grad = False
                
        return self.model

    def get_optimizer(self, steps):
        if self.optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), self.cfg.lr_adam)

        elif self.optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), self.cfg.lr_adam)

        elif self.optimizer_type == 'HybridMuonAdam':
            self.optimizer, self.optimizer_stats = build_muon_adam_optimizer(model=self.model, lr_muon=self.cfg.lr_muon, lr_adam=self.cfg.lr_adam)

        if self.lr_sch_type == 'ReduceLROnPlateau':
            if self.optimizer_type == 'HybridMuonAdam':
                self.lr_sch = CompositeReduceLROnPlateau(self.optimizer, factor_muon=0.89, factor_adam=0.89, patience=self.cfg.epochs_decay)
            else:
                self.lr_sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=0.89, patience=self.cfg.epochs_decay)

        elif self.lr_sch_type == 'LinearWarmupCosineDecay':
            total_steps = self.cfg.epochs * steps
            warmup_steps = int(total_steps * (self.cfg.epochs_decay/100))

            if self.optimizer_type == 'HybridMuonAdam':
                self.lr_sch = CompositeLinearWarmupCosineDecay(self.optimizer, total_steps=total_steps, peak_lr_muon=self.cfg.lr_muon, peak_lr_adam=self.cfg.lr_adam,
                                                               warmup_steps=warmup_steps)
            else:
                self.lr_sch = LinearWarmupCosineDecay(self.optimizer, total_steps, self.cfg.lr_adam, warmup_steps, 0, 1e-6)

        elif self.lr_sch_type == 'CosineAnnealingWarmRestarts':
            if self.optimizer_type == 'HybridMuonAdam':
                self.lr_sch = CompositeCosineAnnealingWarmRestarts(self.optimizer, T_0=self.cfg.epochs // 4, max_lr_muon=self.cfg.lr_muon, max_lr_adam=self.cfg.lr_adam,
                                                                   T_mult=1.0)
            else:
                self.lr_sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.cfg.epochs // 4, eta_min=1e-6)

        return self.optimizer, self.lr_sch

def displayfigures(results, result_path, report_name):
    for i in range(len(results)):
        plt.Figure()
        plt.plot(results[i][1], marker='o', markersize=3, label="Train "+results[i][0])
        plt.plot(results[i][2], marker='s', markersize=3, label="Val "+results[i][0])
        plt.legend(loc="lower right")
        plt.xlabel("Epochs")
        plt.ylabel(results[i][0]+"%")
        if results[i][0] != "Loss":
            plt.ylim(0,100)
        plt.savefig(result_path+report_name+'_'+results[i][0]+'_results.png')
        plt.close()

def PRC(Pr, Rc, result_path, report_name):
    Rc1 = []
    Pr1 = []
    Rc = list(map(list, zip(*Rc)))
    Pr = list(map(list, zip(*Pr)))
    
    for i in range(len(Rc)):
        Rc1.append(np.sum(Rc[i])/len(Rc[i]))
        
    for i in range(len(Pr)):
        Pr1.append(np.sum(Pr[i])/len(Pr[i]))
    
    Pr = np.fliplr([Pr1])[0]  #to avoid getting negative AUC
    Rc = np.fliplr([Rc1])[0]  #to avoid getting negative AUC
    AUC_Pr_Rc = np.trapz(Pr,Rc)
    print("\nArea under Precision-Recall curve: " +str(AUC_Pr_Rc))
    plt.figure()
    plt.plot(Rc,Pr,'-',label='Area Under the Curve (AUC = %0.4f)' % AUC_Pr_Rc)
    plt.title('Precision - Recall curve')
    plt.legend(loc="lower right")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(result_path+report_name+'_PRC.png')
    
    return Rc, Pr