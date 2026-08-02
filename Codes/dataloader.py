# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 22:13:16 2026

@author: Omar Al-maqtari
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F


class ImageFolder(Dataset):
    def __init__(self, cfg, mode, aug_prob):
        """Initializes image paths and preprocessing module."""
        self.cfg = cfg
        self.mode = mode
        self.root = cfg.dataset_path
        self.image_height = cfg.image_height
        self.image_width = cfg.image_width
        self.learning_mode = cfg.learning_mode
        self.teacher_type = cfg.teacher_type
        self.aug_prob = aug_prob

        if self.learning_mode == 'Dist':
            self.submode = mode + '_mask' if self.teacher_type != '7B' else mode + '_dist_7B'
        elif self.learning_mode == 'Seed':
            self.mode = mode + '_t'
            self.submode = mode + '_t_seed'
        elif self.learning_mode == 'Seg':
            self.submode = mode + '_mask'

        self.image_paths = sorted(list(os.listdir(self.root + self.mode +'/')))
        self.sgt_paths = sorted(list(os.listdir(self.root + self.submode +'/')))
        print(f"image count in {self.submode} path {len(self.image_paths)}")
    
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        sgt_path = self.sgt_paths[index]
        image1 = Image.open(self.root+self.mode+'/'+image_path).resize((self.image_width, self.image_width), resample=Image.BICUBIC)
        image2 = Image.open(self.root+self.mode+'/'+image_path).resize((self.image_width, self.image_width), resample=Image.BICUBIC)
        sgt = torch.empty([0])
        read_sgt = self.learning_mode != 'Dist' or self.teacher_type != 'large'
        if read_sgt:
            sgt = Image.open(self.root+self.submode+'/'+sgt_path).resize((self.image_width, self.image_width), resample=Image.NEAREST)

        Transform = []
        Transform.append(T.Resize((self.image_width,self.image_height)))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image1 = Transform(image1).to(dtype=torch.float32)
        image2 = Transform(image2).to(dtype=torch.float32)
        if read_sgt:
            sgt = Transform(sgt).to(dtype=torch.float32)

        if self.mode == 'train':
            Transform = []
            kernel_size = random.choice([3, 5, 7])
            Transform.append(T.GaussianBlur(kernel_size, sigma=(0.1, 2.0)))
            Transform.append(T.RandomInvert(p=0.01))
            Transform.append(T.ColorJitter(brightness=0.3, contrast=0.2, hue=0.015))
            Transform = T.Compose(Transform)

            if random.random() < self.aug_prob:
                image1 = Transform(image1)

            if random.random() < self.aug_prob:
                Transform = T.RandomRotation((90,90))
                image1 = Transform(image1)
                image2 = Transform(image2)
                if read_sgt:
                    sgt = Transform(sgt)
                
            if random.random() < self.aug_prob:
                image1 = F.hflip(image1)
                image2 = F.hflip(image2)
                if read_sgt:
                    sgt = F.hflip(sgt)
                
            if random.random() < self.aug_prob:
                image1 = F.vflip(image1)
                image2 = F.vflip(image2)
                if read_sgt:
                    sgt = F.vflip(sgt)

            Transform = T.Resize((self.image_width,self.image_height))
            image1 = Transform(image1)
            image2 = Transform(image2)
            if read_sgt:
                sgt = Transform(sgt)
            
        if sgt.shape[0] != 1 and self.learning_mode != 'Dist' and self.teacher_type != '7B':
            sgt = T.Grayscale(num_output_channels=1)(sgt)
            sgt[sgt<.3] = 0.
            sgt[sgt>=.3] = 1.
        elif sgt.shape[0] == 1 and self.learning_mode != 'Dist' and self.teacher_type != '7B':
            sgt[sgt<.5] = 0.
            sgt[sgt>=.5] = 1.

        if image1.shape[0] != 3 and image2.shape[0] != 3:
            image1 = image1.repeat(3, 1, 1)
            image2 = image2.repeat(3, 1, 1)

        return [image1, image2], [sgt], [image_path, sgt_path]


def get_loader(cfg, mode, aug_prob):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(cfg, mode=mode, aug_prob=aug_prob)
    
    dataloader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=True if mode=='train' else False,
                             num_workers=cfg.num_workers,
                             pin_memory=True,
                             drop_last=False)
    
    return dataloader
