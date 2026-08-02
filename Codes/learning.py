# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 23:19:49 2026

@author: Omar Al-maqtari
"""

import torch
from torch import nn
import kornia.morphology as morph
from mask_generator import Mask_Generator

class Learner(object):
    def __init__(self, cfg, model, teacher, optimizer, grad_scaler, loss_eq, steps):
        self.cfg = cfg
        self.steps = steps
        self.counter = 0
        self.model = model.to(self.cfg.device)
        self.teacher = teacher.to(self.cfg.device) if teacher is not None else None
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler
        self.DLoss = loss_eq[0].to(self.cfg.device)
        self.FocalLoss = loss_eq[1].to(self.cfg.device)
        self.DiceLoss = loss_eq[2].to(self.cfg.device)
        self.BCELoss = loss_eq[3].to(self.cfg.device)

        self.generator = Mask_Generator(self.cfg).to(self.cfg.device)

    def _get_muon_lr(self):
        for group in self.optimizer.param_groups:
            if group.get("use_muon", False):
                return float(group["lr"])
        return float(self.optimizer.param_groups[0]["lr"])
    
    def loss(self, r_patch, gt_patch):
        total_loss = 0.
        if self.cfg.learning_mode == 'Dist':
            total_loss = self.DLoss(r_patch, gt_patch)
        elif self.cfg.learning_mode == 'Seed':
            loss1 = self.FocalLoss(r_patch, gt_patch)
            loss2 = self.DiceLoss(r_patch, gt_patch)
            total_loss = (0.15*loss1) + (0.85*loss2)
        elif self.cfg.learning_mode == 'Seg':
            loss1 = self.DiceLoss(r_patch, gt_patch)
            loss2 = self.BCELoss(r_patch, gt_patch)
            total_loss = (0.9*loss1) + (0.1*loss2)
        return total_loss

    def MSL_masks(self, images, sr1, sr2, feats, mgt, enable_memory_match=False, memory_selected_index=None, sample_ids=None):
        sgt = torch.tensor([])
        for b in range(images.shape[0]):
            mgt[b] = sr1[b] if mgt[b].sum() < 1. else mgt[b]

        # Original MSL generated soft mask.
        mask = self.generator.Generate(images, sr1, sr2, feats, mgt, visualize=False)

        memory_info = None
        if (self.cfg.learning_mode == "Seg" and hasattr(self.model, "mask_memory")):
            mask, memory_info = self.model.mask_memory.refine_selected_from_batch(gen_masks=mask.detach(),
                                                                                  enable_memory_match=enable_memory_match,
                                                                                  selected_index=memory_selected_index,
                                                                                  sample_ids=sample_ids)

        if self.counter >= 0 and self.counter <= self.steps*2:
            sgt = ((0.5*sr1.detach()) + (0.3*mgt) + (0.2*mask))
            self.counter += 1
        elif self.counter > self.steps*2 and self.counter <= self.steps*4:
            sgt = ((0.3*sr1.detach()) + (0.5*mgt) + (0.2*mask))
            self.counter += 1
        elif self.counter > self.steps*4:
            sgt = ((0.3*sr1.detach()) + (0.2*mgt) + (0.5*mask))
            self.counter = 0

        sgt[sgt < 0.5] = 0.0
        sgt[sgt >= 0.5] = 1.0

        return sgt, memory_info

    def learn(self, image, gt, train='test', enable_memory_match=False, memory_selected_index=None, sample_ids=None):
        loss = None
        loads = None
        r = None
        memory_info = None
        hebb_lr = self._get_muon_lr() * 10.0
        image1 = image[0].to(self.cfg.device)
        image2 = image[1].to(self.cfg.device)

        if train == 'train':
            self.model.train(True)
            if self.cfg.learning_mode == 'Dist':
                r_patch, loads = self.model(image1)
                gt_patch = self.teacher(image2)

                r = torch.cat(r_patch, 0)
                gt = torch.cat(gt_patch, 0)
                loss = self.loss(r, gt)

            elif self.cfg.learning_mode == 'Seed':
                gt = gt[0].to(self.cfg.device)
                gt = morph.dilation(gt, torch.ones(9,9).to(self.cfg.device))
                r, _, loads = self.model(image1, seed=gt, hebb_lr=hebb_lr, update_hebbian=True)

                loss = self.loss(r, gt)

            elif self.cfg.learning_mode == 'Seg':
                self.model.E.eval()
                self.teacher.eval()
                mgt = gt[0].to(self.cfg.device)
                with torch.no_grad():
                    seg_logits, _, _ = self.teacher(image1)

                r, feats, loads = self.model(image2, seed=None, update_hebbian=False)
                gt, memory_info = self.MSL_masks(image2, r, seg_logits, feats, mgt, enable_memory_match=enable_memory_match, memory_selected_index=memory_selected_index,
                                                 sample_ids=sample_ids)

                loss = self.loss(r, gt)

            if self.cfg.learning_mode == 'Dist' and loads is not None:
                if self.cfg.model_type == 'Swin-MoE':
                    loss = loss + loads.to(loss.device)

            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            if self.cfg.learning_mode == 'Dist' and loads is not None:
                if self.cfg.model_type == 'M':
                    self.model.update_expert_bias(loads)

        else:
            self.model.train(False)
            with torch.no_grad():
                if self.cfg.learning_mode == 'Dist':
                    r_patch, loads = self.model(image1)
                    gt_patch = self.teacher(image2)

                    r = torch.cat(r_patch, 0)
                    gt = torch.cat(gt_patch, 0)
                    loss = self.loss(r, gt)
                    
                elif self.cfg.learning_mode in ['Seed', 'Seg']:
                    gt = gt[0].to(self.cfg.device)
                    gt = morph.dilation(gt, torch.ones(9,9).to(self.cfg.device)) if self.cfg.learning_mode == 'Seed' else gt

                    r, _, loads = self.model(image2, update_hebbian=False)

                    loss = self.loss(r, gt)

        return [r, gt, loss, memory_info]