# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:44:51 2026

@author: Omar Al-maqtari
"""

import math

import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def linear_warmup_cosine_decay(start, peak, end, warmup_steps, total_steps):
    linear = np.linspace(start, peak, warmup_steps, endpoint=False, dtype=np.float64)

    cosine_steps = total_steps - warmup_steps
    cosine = np.cos(np.linspace(0, np.pi, cosine_steps, dtype=np.float64))
    cosine = (cosine + 1.) / 2.
    cosine = (peak - end) * cosine + end

    schedule = np.concatenate((linear, cosine), axis=0)

    return schedule

class LinearWarmupCosineDecay(_LRScheduler):
    def __init__(self, optimizer, total_steps, peak_lr, warmup_steps, start_warmup_lr, final_lr, last_epoch=-1):
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.start_warmup_lr = float(start_warmup_lr)
        self.final_lr = float(final_lr)

        self.schedule = linear_warmup_cosine_decay(self.start_warmup_lr, self.peak_lr, self.final_lr, self.warmup_steps, self.total_steps)

        super().__init__(optimizer, last_epoch)

        init_lr = float(self.schedule[0]) if self.total_steps > 0 else self.final_lr
        for group in self.optimizer.param_groups:
            group['lr'] = init_lr

    def get_lr(self):
        idx = min(max(self.last_epoch, 0), self.total_steps - 1)
        lr = float(self.schedule[idx])
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)

        idx = min(max(self.last_epoch, 0), self.total_steps - 1)
        lr = float(self.schedule[idx])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr_value(self):
        idx = min(max(self.last_epoch, 0), self.total_steps - 1)
        return float(self.schedule[idx])