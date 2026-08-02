from __future__ import annotations

import math
import torch.nn as nn
from muon import SingleDeviceMuonWithAuxAdam
from scheduler import linear_warmup_cosine_decay

def _split_group_indices(optimizer):
    muon_idx = []
    adam_idx = []
    for i, group in enumerate(optimizer.param_groups):
        if group.get("use_muon", False):
            muon_idx.append(i)
        else:
            adam_idx.append(i)
    return muon_idx, adam_idx

def _iter_named_module_params(model):
    for module_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            yield full_name, module, local_name, param

class _GroupLRSchedulerBase:
    def __init__(self, optimizer, group_indices):
        self.optimizer = optimizer
        self.group_indices = group_indices

    def _get_lrs(self):
        return [float(self.optimizer.param_groups[i]["lr"]) for i in self.group_indices]

    def _set_lrs(self, lrs):
        if isinstance(lrs, (int, float)):
            lrs = [float(lrs)] * len(self.group_indices)
        for i, lr in zip(self.group_indices, lrs):
            self.optimizer.param_groups[i]['lr'] = float(lr)

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class GroupReduceLROnPlateau(_GroupLRSchedulerBase):
    def __init__(self, optimizer, group_indices, mode="max", factor=0.89, patience=4, min_lr=1e-6):
        super().__init__(optimizer, group_indices)
        self.mode = mode
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.best = -float("inf") if mode == "max" else float("inf")
        self.num_bad_epochs = 0

    def step(self, metric):
        metric = float(metric)
        improved = metric > self.best if self.mode == "max" else metric < self.best

        if improved:
            self.best = metric
            self.num_bad_epochs = 0
            return

        self.num_bad_epochs += 1
        if self.num_bad_epochs > self.patience:
            new_lrs = [max(lr * self.factor, self.min_lr) for lr in self._get_lrs()]
            self._set_lrs(new_lrs)
            self.num_bad_epochs = 0

class GroupLinearWarmupCosineDecay(_GroupLRSchedulerBase):
    def __init__(self, optimizer, group_indices, total_steps, peak_lr, warmup_steps, start_warmup_lr, final_lr):
        super().__init__(optimizer, group_indices)
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.start_warmup_lr = float(start_warmup_lr)
        self.final_lr = float(final_lr)
        self.last_step = -1

        self.schedule = linear_warmup_cosine_decay(self.start_warmup_lr, self.peak_lr, self.final_lr, self.warmup_steps, self.total_steps)

        init_lr = float(self.schedule[0]) if self.total_steps > 0 else self.final_lr
        self._set_lrs(init_lr)

    def step(self):
        self.last_step += 1
        idx = min(max(self.last_step, 0), self.total_steps - 1)
        self._set_lrs(float(self.schedule[idx]))

class GroupCosineAnnealingWarmRestarts(_GroupLRSchedulerBase):
    def __init__(self, optimizer, group_indices, T_0, max_lr, min_lr=1e-6, T_mult=1.0):
        super().__init__(optimizer, group_indices)
        self.T_0 = int(T_0)
        self.T_mult = float(T_mult)
        self.base_max_lr = float(max_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)

        self.cur_cycle_steps = self.T_0
        self.cycle = 0
        self.step_in_cycle = -1
        self.last_epoch = -1

        self._set_lrs(self.min_lr)

    def _get_lr(self):
        if self.step_in_cycle == -1:
            return self.min_lr
        return self.min_lr + (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * self.step_in_cycle / self.cur_cycle_steps)) / 2.0

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.T_mult)
        else:
            epoch = int(epoch)
            if epoch >= self.T_0:
                if self.T_mult == 1.0:
                    self.cycle = epoch // self.T_0
                    self.step_in_cycle = epoch % self.T_0
                    self.cur_cycle_steps = self.T_0
                else:
                    n = int(math.log(epoch / self.T_0 * (self.T_mult - 1.0) + 1.0, self.T_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.T_0 * (self.T_mult ** n - 1.0) / (self.T_mult - 1.0))
                    self.cur_cycle_steps = int(self.T_0 * (self.T_mult ** n))
            else:
                self.cycle = 0
                self.cur_cycle_steps = self.T_0
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr
        self.last_epoch = epoch
        self._set_lrs(self._get_lr())

class CompositeReduceLROnPlateau:
    def __init__(self, optimizer, factor_muon=0.89, factor_adam=0.89, patience=4, min_lr_muon=1e-6, min_lr_adam=1e-6):
        muon_idx, adam_idx = _split_group_indices(optimizer)
        self.schedulers = []

        if muon_idx:
            self.schedulers.append(GroupReduceLROnPlateau(optimizer, muon_idx, mode="max", factor=factor_muon, patience=patience, min_lr=min_lr_muon))
        if adam_idx:
            self.schedulers.append(GroupReduceLROnPlateau(optimizer, adam_idx, mode="max", factor=factor_adam, patience=patience, min_lr=min_lr_adam))

    def step(self, metric):
        for sch in self.schedulers:
            sch.step(metric)

    def state_dict(self):
        return [sch.state_dict() for sch in self.schedulers]

    def load_state_dict(self, state_dict):
        for sch, sd in zip(self.schedulers, state_dict):
            sch.load_state_dict(sd)

class CompositeLinearWarmupCosineDecay:
    def __init__(self, optimizer, total_steps, peak_lr_muon, peak_lr_adam, warmup_steps, start_warmup_lr_muon=1e-6,
        start_warmup_lr_adam=1e-6, final_lr_muon=1e-6, final_lr_adam=1e-6):

        muon_idx, adam_idx = _split_group_indices(optimizer)
        self.schedulers = []

        if muon_idx:
            self.schedulers.append(GroupLinearWarmupCosineDecay(optimizer, muon_idx, total_steps, peak_lr_muon, warmup_steps, start_warmup_lr_muon, final_lr_muon))
        if adam_idx:
            self.schedulers.append(GroupLinearWarmupCosineDecay(optimizer, adam_idx, total_steps, peak_lr_adam, warmup_steps, start_warmup_lr_adam, final_lr_adam))

    def step(self):
        for sch in self.schedulers:
            sch.step()

    def state_dict(self):
        return [sch.state_dict() for sch in self.schedulers]

    def load_state_dict(self, state_dict):
        for sch, sd in zip(self.schedulers, state_dict):
            sch.load_state_dict(sd)

class CompositeCosineAnnealingWarmRestarts:
    def __init__(self, optimizer, T_0, max_lr_muon, max_lr_adam, min_lr_muon=1e-6, min_lr_adam=1e-6, T_mult=1.0):
        muon_idx, adam_idx = _split_group_indices(optimizer)
        self.schedulers = []

        if muon_idx:
            self.schedulers.append(GroupCosineAnnealingWarmRestarts(optimizer, muon_idx, T_0=T_0, max_lr=max_lr_muon, min_lr=min_lr_muon, T_mult=T_mult))
        if adam_idx:
            self.schedulers.append(GroupCosineAnnealingWarmRestarts(optimizer, adam_idx, T_0=T_0, max_lr=max_lr_adam, min_lr=min_lr_adam, T_mult=T_mult))

    def step(self, epoch=None):
        for sch in self.schedulers:
            sch.step(epoch)

    def state_dict(self):
        return [sch.state_dict() for sch in self.schedulers]

    def load_state_dict(self, state_dict):
        for sch, sd in zip(self.schedulers, state_dict):
            sch.load_state_dict(sd)

def hybrid_muon_adam_param_groups(model, lr_muon, lr_adam, weight_decay_muon, weight_decay_adam):
    muon_params = []
    adam_params = []
    muon_names = []
    adam_names = []

    for full_name, module, local_name, param in _iter_named_module_params(model):
        if not param.requires_grad:
            continue

        is_muon_weight = (local_name == "weight" and param.ndim >= 2 and isinstance(module, (nn.Conv2d, nn.Linear)))
        if is_muon_weight:
            muon_params.append(param)
            muon_names.append(full_name)
        else:
            adam_params.append(param)
            adam_names.append(full_name)

    param_groups = []
    if len(muon_params) > 0:
        param_groups.append(dict(params=muon_params, use_muon=True, lr=lr_muon, weight_decay=weight_decay_muon))

    if len(adam_params) > 0:
        param_groups.append(dict(params=adam_params, use_muon=False, lr=lr_adam, betas=(0.9, 0.99), weight_decay=weight_decay_adam))

    stats = {
        "muon_param_tensors": len(muon_params),
        "adam_param_tensors": len(adam_params),
        "muon_param_count": sum(p.numel() for p in muon_params),
        "adam_param_count": sum(p.numel() for p in adam_params),
        "muon_names": muon_names,
        "adam_names": adam_names,
        }
    return param_groups, stats

def build_muon_adam_optimizer(model, lr_muon=1e-4, lr_adam=1e-3, weight_decay_muon=1e-1, weight_decay_adam=1e-6):

    param_groups, stats = hybrid_muon_adam_param_groups(model=model, lr_muon=lr_muon, lr_adam=lr_adam, weight_decay_muon=weight_decay_muon, weight_decay_adam=weight_decay_adam)

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
    return optimizer, stats
