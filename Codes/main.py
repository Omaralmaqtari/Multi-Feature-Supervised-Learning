# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:42:49 2026

@author: Omar Al-maqtari
"""

import os
import argparse
from training import Trainer
import torch
from torch.backends import cudnn

def main(cfg):
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.enable_math_sdp(True)
    if cfg.model_type not in ['M', 'Swin-MoE', 'Swin-v2', 'MobileViTv3']:
        print('ERROR!! model_type should be selected in M')
        print('Your input for model_type was %s'%cfg.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    if not os.path.exists(cfg.sr_path):
        os.makedirs(cfg.sr_path)

    print(cfg)
    trainer = Trainer(cfg)
    if cfg.mode == 'train':
        trainer.train()
    elif cfg.mode == 'test':
        trainer.test()


if __name__ == '__main__':
    lr_muon = 0.
    lr_adam = 0.
    epochs = 0.
    batch_size = 0.
    aug_prob = 0.
    memory_capacity = 0
    experiment_num = 1
    learning_mode = 'Seg'    # 'Dist', 'Seed', 'Seg'
    teacher_type = 'large'      # 'large, '7B'
    for dataset in ['VOC 2012', 'COCO 2017']: # 'VOC 2012', 'COCO 2017'
        subset = '/coco20k/' if dataset == 'COCO 2017' else '/'
        for model_type in ['M']: # 'M', 'MobileViTv3', 'Swin-v2', 'Swin-MoE'
            for mode in ['train']:# 'train', 'test'
                if learning_mode == 'Dist':
                    lr_muon = 1e-3
                    lr_adam = 8e-4
                    epochs = 250
                    batch_size = 16
                    aug_prob = 0.2
                elif learning_mode == 'Seed':
                    lr_muon = 8e-4
                    lr_adam = 7e-4
                    epochs = 300
                    batch_size = 16
                    aug_prob = 0.3
                elif learning_mode == 'Seg':
                    lr_muon = 8e-4
                    lr_adam = 7e-4
                    epochs = 60
                    batch_size = 16
                    aug_prob = 0.1
                    memory_selected_batches = 40 if dataset == 'VOC 2012' else 312
                    memory_capacity = 262 if dataset == 'VOC 2012' else 2000

                parser = argparse.ArgumentParser()
                # model hyper-parameters
                parser.add_argument('--img_in', type=int, default=3)
                parser.add_argument('--ch_out', type=int, default=1)
                parser.add_argument('--image_height', type=int, default=256)
                parser.add_argument('--image_width', type=int, default=256)
                parser.add_argument('--device', type=str, default='cuda:0')

                # training hyper-parameters
                parser.add_argument('--lr_muon', type=float, default=lr_muon)
                parser.add_argument('--lr_adam', type=float, default=lr_adam)
                parser.add_argument('--epochs', type=int, default=epochs)
                parser.add_argument('--epochs_decay', type=int, default=5)
                parser.add_argument('--batch_size', type=int, default=batch_size)
                parser.add_argument('--num_workers', type=int, default=2)
                parser.add_argument('--aug_prob', type=float, default=aug_prob)
                parser.add_argument('--parameters', type=int, default=0)
                parser.add_argument('--optimizer_type', type=str, default='HybridMuonAdam', help='Adam, AdamW, HybridMuonAdam')
                parser.add_argument('--lr_sch_type', type=str, default='LinearWarmupCosineDecay',
                                    help='ReduceLROnPlateau, CosineAnnealingWarmRestarts, LinearWarmupCosineDecay')

                # misc
                parser.add_argument('--mode', type=str, default=mode, help='train, test')
                parser.add_argument('--model_type', type=str, default=model_type, help='M')
                parser.add_argument('--dataset', type=str, default=dataset, help='VOC 2012, COCO 2017')
                parser.add_argument('--teacher_type', type=str, default=teacher_type, help='7B, large')
                parser.add_argument('--learning_mode', type=str, default=learning_mode, help='Dist, Seed, Seg')
                parser.add_argument('--experiment_num', type=int, default=experiment_num, help='1, 2, ...')
                parser.add_argument('--report_name', type=str,
                                    default=model_type + '_' + dataset + '_' + teacher_type + '_' + learning_mode + '_' + str(
                                        experiment_num))

                # memory parameters
                parser.add_argument("--memory_selected_batches", type=int, default=memory_selected_batches)
                parser.add_argument("--memory_sim_threshold", type=float, default=0.85)
                parser.add_argument("--memory_unique_threshold", type=float, default=0.65)
                parser.add_argument("--memory_capacity", type=int, default=memory_capacity)
                parser.add_argument("--memory_grid_size", type=int, default=8)
                parser.add_argument("--memory_metric", type=str, default="iou", help="dice, iou")

                # paths
                parser.add_argument('--model_path', type=str, default='/...')
                parser.add_argument('--result_path', type=str, default='/...')
                parser.add_argument('--sr_path', type=str, default='/...')
                parser.add_argument('--dataset_path', type=str,
                                    default='/.../' + dataset + subset)

                cfg = parser.parse_args()
                main(cfg)