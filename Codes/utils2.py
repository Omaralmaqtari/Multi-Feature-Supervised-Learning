# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 16:37:09 2026

@author: Omar al-maqtari
"""

import math
import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import gaussian_filter, laplace

def gaussiankernel(ch_out, ch_in, kernelsize, sigma, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:, :, int((kernelsize - 1) / 2), int((kernelsize - 1) / 2)] = kernelvalue
    g = gaussian_filter(n, sigma)
    gaussiankernel = torch.from_numpy(g)

    return gaussiankernel.float()

def laplaceiankernel(ch_out, ch_in, kernelsize, kernelvalue):
    n = np.zeros((ch_out, ch_in, kernelsize, kernelsize))
    n[:, :, int((kernelsize - 1) / 2), int((kernelsize - 1) / 2)] = kernelvalue
    l = laplace(n)
    laplacekernel = torch.from_numpy(l)

    return laplacekernel.float()

class SEM(nn.Module):
    def __init__(self, ch_out, reduction=4):
        super(SEM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(ch_out, reduction, kernel_size=1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(reduction, ch_out, kernel_size=1, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y.expand_as(x)

# Edge Extraction Attention
class EEA(nn.Module):
    def __init__(self, ch, kernel=3):
        super(EEA, self).__init__()

        self.groups = ch
        self.gk = gaussiankernel(ch, int(ch / ch), kernel, kernel - 2, 0.9)
        self.lk = laplaceiankernel(ch, int(ch / ch), kernel, 0.9)
        self.gk = nn.Parameter(self.gk, requires_grad=False)
        self.lk = nn.Parameter(self.lk, requires_grad=False)

        self.conv = nn.Sequential(nn.InstanceNorm2d(ch),
                                  nn.MaxPool2d(3, stride=1, padding=1),
                                  nn.Conv2d(ch, ch, 1, bias=True, groups=ch // 4))
        self.sem = SEM(ch)

    def forward(self, x):
        DoG = F.conv2d(x, self.gk.to(x.device), padding='same', groups=self.groups)
        LoG = F.conv2d(DoG, self.lk.to(x.device), padding='same', groups=self.groups)
        DoG = DoG - x
        x1 = self.conv((DoG + LoG) / 2)

        return x + self.sem(x1)

class FRCM(nn.Module):
    def __init__(self, ch_ins, ch_out):
        super(FRCM, self).__init__()
        n_sides = len(ch_ins)

        self.reducers = nn.ModuleList([])
        for i in range(n_sides):
            self.reducers.append(nn.Conv2d(ch_ins[i], ch_out, kernel_size=1))

        self.gn = nn.GroupNorm(1, ch_out)
        self.prelu = nn.PReLU(num_parameters=ch_out, init=0.05)

        self.fused = nn.Sequential(nn.Conv2d(ch_out * n_sides, ch_out, kernel_size=1),
                                   nn.PReLU(num_parameters=ch_out, init=0.05))

    def forward_sides(self, sides, img_shape):
        late_sides = []
        for x, conv in zip(sides, self.reducers):
            x = self.prelu(self.gn(conv(x)))
            x = F.interpolate(x, size=img_shape, mode='bilinear', align_corners=False)
            late_sides.append(x)

        return late_sides

    def forward(self, sides, img_shape):
        late_sides = self.forward_sides(sides, img_shape)

        fused = self.fused(torch.cat(late_sides, 1))
        late_sides.append(fused)

        return torch.cat(late_sides, 1)

class RoPE(nn.Module):
    def __init__(self, cfg, h, d, heads):
        super().__init__()
        self.cfg = cfg
        self.H = h
        self.W = h
        self.d = d
        self.D = d // heads
        self.base = 100.0
        self.dtype = torch.bfloat16
        self.register_buffer("periods", torch.empty(self.D // 4, device=self.cfg.device, dtype=self.dtype), persistent=True)

        self._init_weights()

    def _init_weights(self):
        periods = self.base ** (2 * torch.arange(self.D // 4, device=self.cfg.device, dtype=self.dtype) / (self.D // 2))
        self.periods.data = periods

    def _forward(self):
        dd = {"device": self.cfg.device, "dtype": self.dtype}
        coords_h = torch.arange(0.5, self.H, **dd) / self.H  # [H]
        coords_w = torch.arange(0.5, self.W, **dd) / self.W  # [W]
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = (2.0 * coords) - 1.0  # Shift range [0, 1] to [-1, +1]

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return [sin, cos]

class FrequencyFilter(nn.Module):
    def __init__(self, h, sigma=60):
        super().__init__()
        self.sigma = sigma
        x = torch.arange(h) - (h // 2)
        y = torch.arange(h) - (h // 2)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        dist_sq = grid_x ** 2 + grid_y ** 2
        self.mask = torch.exp(-dist_sq / (2 * sigma ** 2))

    def forward(self, x):
        x = torch.nan_to_num(x)
        x_ = fft.fft2(x)
        x_ = fft.fftshift(x_)
        x_ = x_ * self.mask.to(x.device)
        x_ = fft.ifftshift(x_)
        x_ = fft.ifft2(x_)
        return x + x_.abs()

# Fourier Series
class FourierSeriesMapping(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ch = ch
        self.n_terms = 5

        # base frequencies: [1, 2, 3, 4, 5]
        freq = torch.arange(1, self.n_terms + 1, dtype=torch.float32).view(1, self.n_terms).repeat(ch, 1)
        self.omega = nn.Parameter(freq, requires_grad=True)
        self.c = nn.Conv2d(ch, ch, 1, bias=True, groups=ch//4)

    def forward(self, x):
        x = torch.nan_to_num(x)
        b, c, h, w = x.shape

        x_ = x.unsqueeze(2)
        omega = self.omega.to(x.device, dtype=x.dtype).view(1, c, self.n_terms, 1, 1)
        theta = 2.0 * math.pi * omega * x_

        fourier = torch.cos(theta) + torch.sin(theta)
        x_ = self.c(fourier.sum(dim=2))
        return x + x_