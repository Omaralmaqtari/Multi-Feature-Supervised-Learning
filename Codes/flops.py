# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:48:55 2026

@author: Omar Al-maqtari
"""

import argparse
import torch
import time
from model import M
from ptflops import get_model_complexity_info
import re

import torch
import torch.nn as nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_in', type=int, default=3)
    parser.add_argument('--ch_out', type=int, default=1)
    parser.add_argument('--image_height', type=int, default=256)
    parser.add_argument('--image_width', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--teacher_type', type=str, default='large', help='7B, large')
    parser.add_argument('--learning_mode', type=str, default='Seg', help='Dist, Seed, Seg')
    parser.add_argument("--memory_selected_batches", type=int, default=40)
    parser.add_argument("--memory_sim_threshold", type=float, default=0.85)
    parser.add_argument("--memory_unique_threshold", type=float, default=0.65)
    parser.add_argument("--memory_capacity", type=int, default=262)
    parser.add_argument("--memory_grid_size", type=int, default=8)
    parser.add_argument("--memory_metric", type=str, default="iou", help="dice, iou")
    cfg = parser.parse_args()

    model = M(cfg, cfg.img_in, cfg.ch_out).to(cfg.device)
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=False, verbose=False)

    # Extract the numerical value
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    # Extract the unit
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))

    x = torch.randn(1, 3, 256, 256).to(cfg.device)
    t1 = 0
    for i in range(100):
        t = time.time()
        y = model(x)
        t1 += (time.time() - t)
    print(t1 / 100.0)