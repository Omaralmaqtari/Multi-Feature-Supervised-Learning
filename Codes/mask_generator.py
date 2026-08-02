# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 00:15:12 2026

@author: Omar Al-maqtari
"""

import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ncut_pytorch import Ncut
import kornia.morphology as morph
from kornia.filters import laplacian
import matplotlib.pyplot as plt

eps = 1e-8

def _minmax_norm(x):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min).clamp_min(eps)

def _resize_like(x, size, mode="bilinear"):
    if mode in ["nearest", "area"]:
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)

@torch.no_grad()
def _estimate_k_clusters(image, min_k=2, max_k=8,bins=8,
                        peak_mass_thr=0.015, cover_mass=0.90):
    """
    Estimate adaptive K for each image from raw pixel distribution.

    Args:
        image: [B,C,H,W]
        min_k, max_k: cluster range.
        bins: histogram bins per channel.
        peak_mass_thr: minimum probability mass for a bin to count as a dominant.
        cover_mass: number of top bins needed to explain this much image mass.

    Returns:
        k_list: list[int] of length B.
    """
    image = torch.nan_to_num(image).detach().float()
    B, C, H, W = image.shape

    # Normalize per image to [0,1].
    x_min = image.amin(dim=(1, 2, 3), keepdim=True)
    x_max = image.amax(dim=(1, 2, 3), keepdim=True)
    x = (image - x_min) / (x_max - x_min).clamp_min(eps)

    k_list = []

    for b in range(B):
        xb = x[b]  # [C,H,W]

        # -------- grayscale / single-channel case --------
        if C == 1:
            vals = xb[0].reshape(-1)
            q = torch.clamp((vals * bins).long(), 0, bins - 1)

            hist = torch.bincount(q, minlength=bins).float()
            p = hist / hist.sum().clamp_min(eps)

            # Local maxima.
            left = torch.cat([p[:1], p[:-1]])
            right = torch.cat([p[1:], p[-1:]])
            peaks = (p >= left) & (p >= right) & (p >= peak_mass_thr)
            peak_count = int(peaks.sum().item())

            # Number of dominant bins covering most mass.
            sorted_p = torch.sort(p, descending=True).values
            top_count = int((torch.cumsum(sorted_p, dim=0) < cover_mass).sum().item()) + 1

        # -------- RGB / multi-channel case --------
        else:
            # Use first 3 channels if image has more than 3.
            rgb = xb[:3]

            r = torch.clamp((rgb[0].reshape(-1) * bins).long(), 0, bins - 1)
            g = torch.clamp((rgb[1].reshape(-1) * bins).long(), 0, bins - 1)
            bb = torch.clamp((rgb[2].reshape(-1) * bins).long(), 0, bins - 1)

            # Flatten 3D RGB bin index.
            idx = r * bins * bins + g * bins + bb
            hist = torch.bincount(idx, minlength=bins ** 3).float()
            p = hist / hist.sum().clamp_min(eps)

            # 3D local-peak detection.
            p3 = p.view(1, 1, bins, bins, bins)
            local_max = F.max_pool3d(p3, kernel_size=3, stride=1, padding=1)
            peaks = (p3 == local_max) & (p3 >= peak_mass_thr)
            peak_count = int(peaks.sum().item())

            # Number of top color bins needed to cover most pixels.
            sorted_p = torch.sort(p, descending=True).values
            top_count = int((torch.cumsum(sorted_p, dim=0) < cover_mass).sum().item()) + 1

        # Combine both:
        # peak_count = dominant color modes
        # top_count = distribution spread
        k_raw = int(round(0.7 * peak_count + 0.3 * math.sqrt(top_count)))

        # Add foreground/background minimum behavior.
        k = max(min_k, min(max_k, k_raw))

        k_list.append(int(k))

    return k_list

def _pairwise_distance(x, centers, distance="euclidean"):
    """
    x:       [B,N,C]
    centers: [B,K,C]
    returns: [B,N,K]
    """
    if distance == "euclidean":
        return ((x[:, :, None, :] - centers[:, None, :, :]) ** 2).sum(dim=-1)

    if distance == "cosine":
        x_n = F.normalize(x, dim=-1, eps=eps)
        c_n = F.normalize(centers, dim=-1, eps=eps)
        return 1.0 - torch.einsum("bnc,bkc->bnk", x_n, c_n)

    raise ValueError(f"Unknown distance: {distance}")

def _relabel_clusters(clustered, mask_score, n_clusters,
                      binary=False, score_thr_ratio=0.1):
    """
    Relabel clusters based on intersection/shared area with mask_score.

    clustered:
        [B,H,W] long cluster IDs in [0, n_clusters-1]

    mask_score:
        [B,1,H,W] or [B,H,W]

    Output:
        [B,1,H,W] float in [0,1]

    Meaning:
        1.0 -> cluster with largest shared area with mask_score
        0.0 -> cluster with smallest/no shared area with mask_score

    Ranking:
        highest shared area  -> highest value
        second shared area   -> second highest value
        ...
    """
    if mask_score.dim() == 4:
        mask_score = mask_score[:, 0]

    B, H, W = clustered.shape
    device = clustered.device

    flat_clustered = clustered.reshape(B, -1).long()       # [B,N]
    flat_score = mask_score.reshape(B, -1).float()         # [B,N]

    score_max = flat_score.amax(dim=1, keepdim=True)
    score_thr = score_max * float(score_thr_ratio)
    flat_support = (flat_score >= score_thr).float()       # [B,N]

    shared_area = torch.zeros(B, n_clusters, device=device, dtype=flat_score.dtype)
    counts = torch.zeros(B, n_clusters, device=device, dtype=flat_score.dtype)

    shared_area.scatter_add_(1, flat_clustered, flat_support)
    counts.scatter_add_(1, flat_clustered, torch.ones_like(flat_support))

    valid = counts > 0
    shared_area = torch.where(valid, shared_area, torch.full_like(shared_area, -1e8))

    order = torch.argsort(shared_area, dim=1, descending=False)  # [B,K]

    cluster_values = torch.zeros(B, n_clusters, device=device, dtype=flat_score.dtype)

    if binary:
        best = order[:, -1:]
        cluster_values.scatter_(1, best, 1.0)
    else:
        denom = max(n_clusters - 1, 1)
        rank_values = (torch.arange(n_clusters, device=device, dtype=flat_score.dtype).view(1, n_clusters) / float(denom))
        cluster_values.scatter_(1, order, rank_values.expand(B, -1))

    relabeled = torch.gather(cluster_values, 1, flat_clustered)
    return relabeled.view(B, 1, H, W)

def _kmeans(x, num_clusters, distance="euclidean",
            max_iter=16, tol=1e-4):
    """
    x: [B,N,C]
    returns:
        labels:  [B,N]
        centers: [B,K,C]
    """
    B, N, C = x.shape
    K = int(num_clusters)
    device = x.device
    dtype = x.dtype
    labels = 0.

    x = x.float()

    # deterministic initialization: evenly spaced pixels per image
    init_idx = torch.linspace(0, N - 1, K, device=device).long()
    centers = x[:, init_idx, :].contiguous()  # [B,K,C]

    for _ in range(max_iter):
        dist = _pairwise_distance(x, centers, distance=distance)  # [B,N,K]
        labels = dist.argmin(dim=-1)                              # [B,N]

        old_centers = centers.clone()

        one_hot = F.one_hot(labels, num_classes=K).float()        # [B,N,K]
        counts = one_hot.sum(dim=1).clamp_min(1.0)                # [B,K]
        new_centers = torch.einsum("bnk,bnc->bkc", one_hot, x) / counts[:, :, None]

        # keep old center for empty clusters
        empty = one_hot.sum(dim=1) == 0
        centers = torch.where(empty[:, :, None], old_centers, new_centers)

        shift = ((centers - old_centers) ** 2).sum(dim=-1).sqrt().sum(dim=1)
        if torch.all(shift < tol):
            break

    return labels, centers.to(dtype=dtype)

@torch.no_grad()
def _voting(candidates, guide=None, area_min=0.018,
            area_max=0.80, use_soft_vote=True, temperature=0.10):
    """
    Candidate selection/voting.

    Args:
        candidates:
            list of [B,1,H,W] masks

        guide:
            optional [B,1,H,W], e.g. CAM / Hebb / seg support

        use_soft_vote:
            False -> select best candidate per image
            True  -> weighted average of candidates

    Returns:
        voted: [B,1,H,W]
    """
    cand = torch.stack(candidates, dim=1).float()   # [B,M,1,H,W]
    B, M, _, H, W = cand.shape

    cand_bin = (cand >= 0.5).float()
    flat = cand_bin.view(B, M, -1)                  # [B,M,P]

    # Pairwise IoU: [B,M,M]
    inter = torch.bmm(flat, flat.transpose(1, 2))
    area = flat.sum(dim=-1)                         # [B,M]
    union = area[:, :, None] + area[:, None, :] - inter
    iou = inter / union.clamp_min(eps)

    agreement = iou.mean(dim=-1)                    # [B,M]

    area_ratio = area / float(H * W)
    valid = (area_ratio >= area_min) & (area_ratio <= area_max)

    # Guide score
    if guide is not None:
        g = F.interpolate(guide.float(), size=(H, W), mode="bilinear", align_corners=False).view(B, 1, -1)  # [B,1,P]

        guide_score = (flat * g).sum(dim=-1) / area.clamp_min(1.0)
        score = 0.5*agreement + 0.5*guide_score
    else:
        score = agreement

    score = torch.where(valid, score, torch.full_like(score, -1e8))

    if use_soft_vote:
        weights = F.softmax(score / temperature, dim=1)  # [B,M]
        voted = (cand * weights[:, :, None, None, None]).sum(dim=1)
        voted = _minmax_norm(voted)
    else:
        best = score.argmax(dim=1)                       # [B]
        idx = best.view(B, 1, 1, 1, 1).expand(B, 1, 1, H, W)
        voted = torch.gather(cand, dim=1, index=idx).squeeze(1)

    return voted

class Clustering(nn.Module):
    def __init__(self):
        super().__init__()

    def LoG(self, x):
        """
        x: [B,C,H,W]
        returns: [B,C,H,W] in [0,1]
        """
        x = torch.nan_to_num(x)
        x1 = laplacian(x, (3, 3), border_type="constant")
        x2 = laplacian(x, (5, 5), border_type="constant")
        x3 = laplacian(x, (7, 7), border_type="constant")
        y = (x1 + x2 + x3) / 3.0

        thr = y.amax(dim=(-2, -1), keepdim=True) * 0.1
        y = torch.where(y >= thr, y, torch.zeros_like(y))

        return _minmax_norm(y)

    def adaptive_kmeans(self, x, mask_score, k_list,
                        neighborhood=1, min_k=2, max_k=8):
        """
        Adaptive K-Means based on per-image histogram-estimated K.

        Args:
            x:          [B,C,H,W]
            mask_score: [B,1,H,W]
            k_list:     list[int], one estimated K per image
            neighborhood:
                0 -> return [K]
                1 -> return [K-1, K]
            min_k, max_k:
                clamp range for K

        Returns:
            maps: list of [B,1,H,W]
        """
        B, C, H, W = x.shape
        device = x.device

        assert len(k_list) == B, f"len(k_list)={len(k_list)} but B={B}"

        x_tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        mask_score = mask_score.reshape(B, 1, H, W)

        N = H * W
        maps = []

        offsets = list(range(-neighborhood, neighborhood))

        for delta in offsets:
            full_map = torch.zeros(B, 1, H, W, device=device, dtype=x.dtype)

            k_for_images = []
            for k in k_list:
                kk = int(k) + int(delta)
                kk = max(min_k, min(max_k, kk))
                kk = min(kk, N)
                k_for_images.append(kk)

            unique_ks = sorted(set(k_for_images))

            for k in unique_ks:
                idx = [i for i, kk in enumerate(k_for_images) if kk == k]

                if len(idx) == 0:
                    continue

                idx_t = torch.tensor(idx, device=device, dtype=torch.long)

                x_sub = x_tokens.index_select(0, idx_t)
                mask_sub = mask_score.index_select(0, idx_t)

                labels_euc, _ = _kmeans(x_sub, num_clusters=k, distance="euclidean", max_iter=18, tol=1e-4)
                labels_cos, _ = _kmeans(x_sub, num_clusters=k, distance="cosine", max_iter=18, tol=1e-4)
                labels_euc = labels_euc.view(len(idx), H, W)
                labels_cos = labels_cos.view(len(idx), H, W)
                map_euc = _relabel_clusters(labels_euc, mask_sub, k, binary=False, score_thr_ratio=0.1)
                map_cos = _relabel_clusters(labels_cos, mask_sub, k, binary=False, score_thr_ratio=0.1)
                cluster_map = torch.cat([map_euc, map_cos], dim=1).mean(dim=1, keepdim=True)
                cluster_map = _minmax_norm(cluster_map)

                # Now every image in this offset slot receives a valid map.
                full_map.index_copy_(0, idx_t, cluster_map.to(full_map.dtype))

            maps.append(full_map)

        return maps

    def kmeans(self, x, mask_score, n_clusters):
        """
        x:         [B,C,H,W]
        Mask_score: [B,1,H,W]
        returns:   list of [B,1,H,W], each map normalized to [0,1]
        """
        B, C, H, W = x.shape

        x_tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
        mask_score = mask_score.reshape(B, 1, H, W)

        maps = []

        for k in n_clusters:
            labels_euc, _ = _kmeans(x_tokens, num_clusters=k, distance="euclidean", max_iter=18, tol=1e-4)
            labels_cos, _ = _kmeans(x_tokens, num_clusters=k, distance="cosine", max_iter=18, tol=1e-4)
            labels_euc = labels_euc.view(B, H, W)
            labels_cos = labels_cos.view(B, H, W)
            map_euc = _relabel_clusters(labels_euc, mask_score, k, binary=False, score_thr_ratio=0.1)
            map_cos = _relabel_clusters(labels_cos, mask_score, k, binary=False, score_thr_ratio=0.1)

            # Keep all clusters and average the two clustering distances.
            cluster_map = torch.cat([map_euc, map_cos], dim=1).mean(dim=1, keepdim=True)
            cluster_map = _minmax_norm(cluster_map)

            maps.append(cluster_map)

        return maps

    def ncut_single(self, x, n_clusters):
        """
        x:         [C,H,W]
        cam_score: [1,H,W]
        returns:   [1,H,W]

        ncut_pytorch.Ncut.fit_transform() a single image operation.
        """
        C, H, W = x.shape
        x_tokens = x.permute(1, 2, 0).reshape(H * W, C).contiguous()

        maps = []
        for k in n_clusters:
            ncut = Ncut(n_eig=k, device=x.device)
            labels = ncut.fit_transform(x_tokens).to(x.device)
            labels = labels[:, 1:5].permute(1, 0).contiguous()
            y = labels.view(4, H, W).float()
            y = _minmax_norm(y)

            maps.append(y)

        return torch.cat(maps, dim=0)

    def ncut(self, x, n_clusters):
        """
        x:         [B,C,H,W]
        cam_score: [B,1,H,W]
        returns:   [B,1,H,W]
        """
        outs = []
        for b in range(x.shape[0]):
            outs.append(self.ncut_single(x[b], n_clusters))
        return torch.stack(outs, dim=0)

class Mask_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.threshold = 0.1
        self.clusters = Clustering().to(cfg.device)
        self.register_buffer("morph_kernel", torch.ones(3, 3))

    @torch.no_grad()
    def Generate(self, image, sr1, sr2, feats, mgt):
        image_full_hw = image.shape[-2:]

        image = F.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=False)
        image = torch.nan_to_num(image)
        B, C, H, W = image.shape

        sr_t1 = torch.where(sr1 >= self.threshold, sr1, torch.zeros_like(sr1))
        sr_t2 = torch.where(sr2 >= self.threshold, sr2, torch.zeros_like(sr2))
        mgt_t = torch.where(mgt >= self.threshold, mgt, torch.zeros_like(mgt))
        sr_t1 = _resize_like(sr_t1, (H, W), mode="nearest")
        sr_t2 = _resize_like(sr_t2, (H, W), mode="nearest")
        mgt_t = _resize_like(mgt_t, (H, W), mode="nearest")
        sr_mask1 = morph.closing(sr_t1, self.morph_kernel)
        sr_mask2 = morph.closing(sr_t2, self.morph_kernel)
        mgt_mask = morph.closing(mgt_t, self.morph_kernel)

        # Combined support mask
        mask_t = 0.45*sr_mask1 + 0.2*sr_mask2 + 0.35*mgt_mask
        mask_t = _minmax_norm(morph.closing(mask_t, self.morph_kernel))

        k_list = _estimate_k_clusters(image, min_k=2, max_k=8, bins=8, peak_mass_thr=0.018, cover_mass=0.90)
        image_kmaps = self.clusters.adaptive_kmeans(image, mask_t, k_list=k_list, neighborhood=1, min_k=2, max_k=8)

        image_ncut = self.clusters.ncut(image, [20])
        ncut_kmaps = self.clusters.adaptive_kmeans(image_ncut, mask_t, k_list=k_list, neighborhood=0, min_k=2, max_k=8)

        f1, f2, f3, f4 = feats
        f_n1 = self.clusters.kmeans(_resize_like(self.clusters.ncut(f1, [20]), (H, W), mode="nearest"), mask_t, [3])[0]
        f_n2 = self.clusters.kmeans(_resize_like(self.clusters.ncut(f2, [20]), (H, W), mode="nearest"), mask_t, [3])[0]
        f_n3 = self.clusters.kmeans(_resize_like(self.clusters.ncut(f3, [20]), (H, W), mode="nearest"), mask_t, [2])[0]
        f_n4 = self.clusters.kmeans(_resize_like(self.clusters.ncut(f4, [20]), (H, W), mode="nearest"), mask_t, [2])[0]

        edges = self.clusters.LoG(image).mean(dim=1, keepdim=True)

        candidates = image_kmaps + ncut_kmaps + [f_n1, f_n2, f_n3, f_n4, edges]
        mask = _voting(candidates, mask_t)
        mask = F.interpolate(mask, size=image_full_hw, mode="bilinear", align_corners=False)
        mask = _minmax_norm(mask)

        return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    cfg = parser.parse_args()

    image = torch.rand(8, 1, 256, 256).to('cuda:0')
    sr1 = torch.rand(8, 1, 256, 256).to('cuda:0')
    sr2 = torch.rand(8, 1, 256, 256).to('cuda:0')
    feats = [torch.rand(8, 1024, 16, 16).to('cuda:0') for _ in range(4)]
    mgt = torch.rand(8, 1, 256, 256).to('cuda:0')
    mask_generator = Mask_Generator(cfg).to('cuda:0')
    mask = mask_generator.Generate(image, sr1, sr2, feats, mgt)

    print(mask.shape)
