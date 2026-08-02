# hebbian_fastseg.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8

class HebbConv2d(nn.Module):
    """
    Hebbian convolution.

    Input:
        x: [B, Cin, H, W]

    Output:
        y: [B, Cout, H, W]
    """

    def __init__(self, ch_in, ch_out = 32, kernel_size = 5, padding = None, momentum = 0.97, temperature = 0.20, top_k = 2,
        use_reconst_rule = True):
        super().__init__()

        self.ch_in = int(ch_in)
        self.ch_out = int(ch_out)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)

        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.use_reconst_rule = bool(use_reconst_rule)

        fan = self.ch_in * self.kernel_size * self.kernel_size
        w = torch.randn(self.ch_out, fan) * (1.0 / fan ** 0.5)
        w = F.normalize(w, dim=1)

        norm_group = ch_out//4 if ch_out % 4 == 0 else 1
        self.norm = nn.GroupNorm(norm_group, ch_out)

        self.register_buffer("weight_flat", w)
        self.register_buffer("initialized", torch.tensor(True))

    def _weight_4d(self, dtype, device):
        w = F.normalize(self.weight_flat.float(), dim=1)
        w = w.to(device=device, dtype=dtype)
        return w.view(self.ch_out, self.ch_in, self.kernel_size, self.kernel_size)

    def _extract_patches(self, x):
        """
        Returns:
            patches: [B*H*W, Cin*K*K]
        """

        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)  # [B, Cin*K*K, H*W]

        patches = patches.transpose(1, 2).reshape(-1, patches.shape[1])
        patches = F.normalize(patches.float(), dim=1)

        return patches

    def _competition(self, response):
        """
        response: [N, Cout]

        returns:
            z: [N, Cout]
        """
        # Soft-WTA
        z = F.softmax(response / self.temperature, dim=1)

        # Optional k-WTA sparsification
        if self.top_k > 0 and self.top_k < z.shape[1]:
            vals, idx = torch.topk(z, k=self.top_k, dim=1)
            z_sparse = torch.zeros_like(z)
            z_sparse.scatter_(1, idx, vals)
            z = z_sparse / z_sparse.sum(dim=1, keepdim=True).clamp_min(eps)

        return z

    @torch.no_grad()
    def update(self, x, seed=None, lr=None):
        """
        x:
            [B,C,H,W]

        seed:
            optional [B,1,H,W]
        """
        patches = self._extract_patches(x)  # [N, D]
        w = F.normalize(self.weight_flat.float(), dim=1)  # [O, D]

        response = patches @ w.t()  # [N, O]
        z = self._competition(response)  # [N, O]

        if seed is not None:
            s = seed.flatten().float().to(z.device)
            s = s.view(-1, 1)
            z = z * s

        denom = z.sum(dim=0, keepdim=True).t().clamp_min(1.0)  # [O,1]

        if self.use_reconst_rule:
            # Repo-like reconstruction view:
            # Δw_i ≈ z_i * (x - reconstructed_x_i)
            recon = z @ w  # [N,D]
            error = patches - recon
            delta = z.t() @ error / denom
        else:
            # Plain Hebbian / HPCA-like correlation update.
            delta = z.t() @ patches / denom
            delta = delta - (delta * w).sum(dim=1, keepdim=True) * w

        new_w = w + lr * delta
        new_w = F.normalize(new_w, dim=1)

        self.weight_flat.mul_(self.momentum).add_(new_w.to(self.weight_flat.dtype) * (1.0 - self.momentum))
        self.weight_flat.copy_(F.normalize(self.weight_flat, dim=1))

    def forward(self, x, seed=None, lr=None, update_hebbian=False):
        x = torch.nan_to_num(x)

        if update_hebbian:
            self.update(x, seed=seed, lr=lr)

        w = self._weight_4d(dtype=x.dtype, device=x.device)
        x = F.conv2d(x, w, bias=None, padding=self.padding)
        x = F.relu(self.norm(x))

        return x

class HebbSeed(nn.Module):
    def __init__(self, ch, momentum = 0.97, bg_weight = 0.4, init_scale = 8.0):
        super().__init__()

        self.ch = int(ch)
        self.momentum = float(momentum)
        self.bg_weight = float(bg_weight)

        fg = F.normalize(torch.randn(1, ch), dim=1)
        bg = F.normalize(torch.randn(1, ch), dim=1)

        self.register_buffer("fg_proto", fg)
        self.register_buffer("bg_proto", bg)
        self.register_buffer("initialized", torch.tensor(False))

        self.logit_scale = nn.Parameter(torch.tensor(float(init_scale)).log())

    @torch.no_grad()
    def update(self, feat, seed=None, lr=None):
        """
        feat:
            [B,C,H,W]

        seed:
            [B,1,H,W]
        """
        if seed is None:
            return

        tokens = feat.detach().flatten(2).transpose(1, 2)  # [B,N,C]
        tokens = F.normalize(tokens.float(), dim=-1)

        seed_flat = seed.flatten(1)

        fg_tokens = tokens[seed_flat >= 0.5]
        bg_tokens = tokens[seed_flat < 0.5]

        if fg_tokens.numel() == 0:
            return

        fg_new = F.normalize(fg_tokens.mean(dim=0, keepdim=True), dim=-1) * lr

        # Background is huge, so subsample it to prevent dominance.
        if bg_tokens.numel() > 0:
            max_bg = min(bg_tokens.shape[0], max(256, fg_tokens.shape[0] * 8))
            if bg_tokens.shape[0] > max_bg:
                idx = torch.randperm(bg_tokens.shape[0], device=bg_tokens.device)[:max_bg]
                bg_tokens = bg_tokens[idx]

        if bg_tokens.numel() > 0:
            bg_new = F.normalize(bg_tokens.mean(dim=0, keepdim=True), dim=-1) * lr
        else:
            bg_new = self.bg_proto.float() * lr

        if not bool(self.initialized):
            self.fg_proto.copy_(fg_new.to(self.fg_proto.dtype))
            self.bg_proto.copy_(bg_new.to(self.bg_proto.dtype))
            self.initialized.fill_(True)
        else:
            self.fg_proto.mul_(self.momentum).add_(fg_new.to(self.fg_proto.dtype) * (1.0 - self.momentum))
            self.bg_proto.mul_(self.momentum).add_(bg_new.to(self.bg_proto.dtype) * (1.0 - self.momentum))

            self.fg_proto.copy_(F.normalize(self.fg_proto, dim=-1))
            self.bg_proto.copy_(F.normalize(self.bg_proto, dim=-1))

    def run(self, feat):
        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)
        tokens = F.normalize(tokens.float(), dim=-1)

        fg = self.fg_proto.float()
        bg = self.bg_proto.float()

        fg_score = tokens @ fg.t()
        bg_score = tokens @ bg.t()

        score = fg_score - self.bg_weight*bg_score
        score = score.view(B, 1, H, W)

        scale = self.logit_scale.exp().clamp(1.0, 30.0)
        prior = torch.sigmoid(score * scale)

        return prior

    def forward(self, x, seed=None, lr=None, update_hebbian=False):
        x = torch.nan_to_num(x)

        if update_hebbian:
            self.update(x, seed=seed, lr=lr)

        x = self.run(x)
        return x


class Hebbian(nn.Module):
    """
    Input:
        image: [B,3,H,W]
        pred:  [B,1,H,W], raw decoder sigmoid output

    Output:
        refined: [B,1,H,W]
    """
    def __init__(self, ch_in = 4, momentum = 0.95):
        super().__init__()
        self.hebb_conv1 = HebbConv2d(ch_in=ch_in, ch_out=32, kernel_size=3, momentum=momentum,
                                     temperature=0.2, top_k=1, use_reconst_rule=True)
        self.hebb_conv2 = HebbConv2d(ch_in=32, ch_out=1, kernel_size=3, momentum=momentum,
                                     temperature=0.2, top_k=1, use_reconst_rule=False)
        self.hebb_seed = HebbSeed(ch=ch_in, momentum=momentum, bg_weight=0.2, init_scale=8.0)
        self.gate = nn.Sequential(
            nn.Conv2d(ch_in, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32, init=0.01),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
            )

    def forward(self, dist_feat, pred, seed=None, hebb_lr=None, update_hebbian=False):
        b, c, h, w = pred.shape
        dist_feat = F.interpolate(dist_feat, size=[h//4,w//4], mode="bilinear", align_corners=False)
        pred = F.interpolate(pred, size=[h//4,w//4], mode="bilinear", align_corners=False)
        dist_feat = dist_feat.clamp(0.0, 1.0)
        pred = pred.clamp(0.0, 1.0)
        x = torch.cat([dist_feat, pred], dim=1)

        seed1 = None
        if seed is not None:
            if seed.shape[-2:] != pred.shape[-2:]:
                seed = F.interpolate(seed.float(), size=[h//4,w//4], mode="nearest")

            seed_bin = (seed > 0.5).float()
            seed1 = 1.0 + 2.0 * seed_bin

        feat = self.hebb_conv1(x, seed=seed1, lr=hebb_lr, update_hebbian=update_hebbian)
        feat = self.hebb_conv2(feat, seed=seed1, lr=hebb_lr, update_hebbian=update_hebbian)
        feat = torch.sigmoid(feat)
        prior = self.hebb_seed(x, seed=seed, lr=hebb_lr, update_hebbian=update_hebbian)
        gate = self.gate(x)

        hebb = feat * prior
        out = gate*pred + (1.-gate)*hebb

        out = F.interpolate(out, size=[h, w], mode="bilinear", align_corners=False)

        return out