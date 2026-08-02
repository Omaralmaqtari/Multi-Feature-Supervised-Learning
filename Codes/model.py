# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 22:48:59 2026

@author: Omar Al-maqtari
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath

from modelscope import AutoModel

from hebbian_layer import Hebbian
from memory_module import MaskMemory
from utils2 import EEA, FRCM, RoPE, FourierSeriesMapping, FrequencyFilter

def _apply_rope(x, sin, cos):
    x1, x2 = x.chunk(2, dim=-1)
    rotated_x = torch.cat([-x2, x1], dim=-1)
    return (x * cos) + (rotated_x * sin)

# Squeeze and Excitation Attention
class SEM(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(ch, reduction, kernel_size=1, bias=False),
                                  nn.PReLU(num_parameters=reduction, init=0.05),
                                  nn.Conv2d(reduction, ch, kernel_size=1, bias=False),
                                  nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y.expand_as(x)

# Convolution layer
class iConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel, stride=1, dilation=1, groups=1, bias=False, norm='identity',
                 act='identity'):
        super().__init__()
        assert stride in [1, 2]
        padding = "same" if stride == 1 else 1
        self.iconv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel, padding=padding, stride=stride, dilation=dilation,
                               groups=groups, bias=bias)

        if norm == 'identity':
            self.norm = nn.Identity()
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(ch_out)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, ch_out)

        if act == 'identity':
            self.act = nn.Identity()
        elif act == 'silu':
            self.act = nn.SiLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=ch_out, init=0.05)
        elif act == 'relu':
            self.act = nn.ReLU(True)

    def forward(self, x):
        return self.act(self.norm(self.iconv(x)))

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(iConv(ch_in, int(ch_in * 2), 1, norm='bn', act='silu'),
                                  iConv(int(ch_in * 2), int(ch_in * 2), 3, stride=stride, groups=int(ch_in * 2),
                                        norm='bn', act='silu'),
                                  iConv(int(ch_in * 2), ch_out, 1, norm='bn'))
        self.shortcut = stride == 1 and ch_in == ch_out

    def forward(self, x):
        if self.shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MLP
class MLP(nn.Module):
    def __init__(self, ch):
        super(MLP, self).__init__()
        self.c1 = iConv(ch, int(ch * 3), 1, bias=True, act='gelu')
        self.c2 = iConv(int(ch * 3), ch, 1, bias=True)

    def forward(self, x):
        x = self.c2(self.c1(x))

        return x

class LLRA(nn.Module):
    """Local Learnable Residual Attention.
    history item: [B, C, H, W]
    output:       [B, C, H, W]
    """

    def __init__(self, ch, eps=1e-6):
        super(LLRA, self).__init__()
        self.ch = int(ch)
        self.eps = float(eps)

        self.query = nn.Parameter(torch.zeros(self.ch, dtype=torch.float32))
        self.rmsnorm = nn.RMSNorm(self.ch, eps=1e-6, elementwise_affine=True)

    def forward(self, history, return_weights=False):
        if len(history) == 0:
            raise ValueError("LLRA history cannot be empty.")

        shape = history[0].shape
        for i, value in enumerate(history):
            if value.shape != shape:
                raise ValueError(f"LLRA history[{i}] has shape {tuple(value.shape)}, "
                                 f"expected {tuple(shape)}.")

        # [L, B, C, H, W]
        values = torch.stack(history, dim=0)

        # RMS normalization across channels for the keys.
        keys = values.float()
        keys = values.permute(0, 1, 3, 4, 2)  # [L,B,H,W,C]
        keys = self.rmsnorm(keys)
        keys = keys.permute(0, 1, 4, 2, 3)  # [L,B,C,H,W]

        # [L, B, H, W]: one depth score at each spatial position.
        logits = torch.einsum("c,lbchw->lbhw", self.query, keys)

        weights = torch.softmax(logits, dim=0).to(values.dtype)
        output = (weights.unsqueeze(2) * values).sum(dim=0)

        if return_weights:
            return output, weights
        return output

# Vision Mixture of Experts
class VMoE(nn.Module):
    """
    Token-level MoE.

    Router:
        x      : [B,C,H,W]
        tokens : [B,N,C], where N = H*W
        logits : [B,N,E]

    Expert:
        selected tokens are reshaped to [M,C,1,1]

    Output:
        [B,C,H,W]
    """

    def __init__(self, ch):
        super(VMoE, self).__init__()
        self.ch = ch

        self.num_experts = 4
        self.top_k = 2
        self.capacity_factor = 1.25
        self.bias_clip = 0.25

        self.router = nn.Linear(self.ch, self.num_experts)
        self.experts = nn.ModuleList([MLP(self.ch) for _ in range(self.num_experts)])
        self.shared_expert = MLP(self.ch)

        self.register_buffer("expert_bias", torch.zeros(self.num_experts))

    @torch.no_grad()
    def update_expert_bias(self, loads):
        target = loads.sum() / self.num_experts

        overloaded = loads > target
        underloaded = loads < target

        gamma = F.softmax(loads, dim=-1) * 0.2

        self.expert_bias[overloaded] = self.expert_bias[overloaded] - gamma[overloaded]
        self.expert_bias[underloaded] = self.expert_bias[underloaded] + gamma[underloaded]

        if self.bias_clip is not None:
            self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        E = self.num_experts
        K = self.top_k

        # Convert feature map to tokens
        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B,N,C]
        tokens = F.layer_norm(tokens, (C,))

        # Token-level routing
        logits = self.router(tokens)  # [B,N,E]

        # Bias is used only for selection.
        biased_logits = logits + self.expert_bias.view(1, 1, E)
        _, topk_idx = torch.topk(biased_logits, k=K, dim=-1)  # [B,N,K]

        # Use original logits for weights.
        topk_logits = logits.gather(dim=-1, index=topk_idx)  # [B,N,K]
        topk_w = F.softmax(topk_logits, dim=-1)  # [B,N,K]

        # Flatten token assignments
        # Global token index: 0 ... B*N-1
        token_ids = torch.arange(B * N, device=x.device).view(B, N, 1)
        token_ids = token_ids.expand(B, N, K)  # [B,N,K]

        flat_token = token_ids.reshape(-1)  # [B*N*K]
        flat_expert = topk_idx.reshape(-1)  # [B*N*K]
        flat_weight = topk_w.reshape(-1)  # [B*N*K]

        loads = torch.bincount(flat_expert, minlength=E).to(logits.dtype)

        # Capacity per expert, based on token assignments.
        # Expected assignments per expert = B*N*K / E
        cap = int(math.ceil(self.capacity_factor * (B * N * K / E)))
        cap = max(1, cap)

        # Sort assignments by expert.
        perm = torch.argsort(flat_expert)
        flat_expert = flat_expert[perm]
        flat_token = flat_token[perm]
        flat_weight = flat_weight[perm]

        # Original unnormalized tokens for expert processing.
        # Shape: [B*N,C]
        x_tokens = x.flatten(2).transpose(1, 2).contiguous().view(B * N, C)

        # Output tokens.
        y_tokens = torch.zeros_like(x_tokens)  # [B*N,C]
        counts = torch.bincount(flat_expert, minlength=E)
        offsets = torch.cumsum(torch.cat([counts.new_zeros(1), counts]), dim=0)
        active_experts = torch.nonzero(counts > 0, as_tuple=False).flatten().tolist()

        # Expert processing token-by-token
        for e in active_experts:
            start = int(offsets[e].item())
            end = int(offsets[e + 1].item())

            tok = flat_token[start:end]  # [M]
            w = flat_weight[start:end]  # [M]

            # Capacity: keep highest routing weights for this expert.
            if cap is not None and tok.numel() > cap:
                order = torch.argsort(w, descending=True)
                tok = tok[order][:cap]
                w = w[order][:cap]

            # Selected tokens: [M,C]
            x_e = x_tokens.index_select(0, tok)

            # Convert tokens to 1x1 feature maps for Conv expert.
            x_e = x_e.view(-1, C, 1, 1)  # [M,C,1,1]

            # Conv expert on token-as-pixel.
            y_e = self.experts[e](x_e)  # [M,C,1,1]
            y_e = y_e.view(-1, C)  # [M,C]

            # Weighted combine.
            y_e = y_e * w.view(-1, 1)

            # Scatter-add back to token grid.
            y_tokens.index_add_(0, tok, y_e.to(y_tokens.dtype))

        # Restore grid
        y = y_tokens.view(B, N, C).transpose(1, 2).contiguous().view(B, C, H, W)

        # Shared expert remains dense convolutional path.
        return self.shared_expert(x) + y.contiguous(), loads

class Attention(nn.Module):
    def __init__(self, cfg, ch, heads, ws, st, drop_path, blk):
        super(Attention, self).__init__()
        self.heads = heads
        self.ws = int(ws)
        self.st = int(st)
        self.blk = blk
        self.win = 0.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        rope = RoPE(cfg, ws, ch, heads)
        rope._init_weights()
        self.rope = rope._forward()
        self.ff = FrequencyFilter(ws, 60)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))), requires_grad=True)
        self.qk_proj = nn.Conv2d(ch, ch * 2, 1, bias=True)
        self.eea = EEA(ch)
        self.fsm = FourierSeriesMapping(ch)
        self.softmax = nn.Softmax(dim=-1)
        self.out_proj = nn.Conv2d(ch, ch, 1, bias=True)

        if self.blk == 'vmoe':
            self.mlp = VMoE(ch)
        elif self.blk == 'conv':
            self.mlp = MLP(ch)

        self.n1 = nn.GroupNorm(2, ch)
        self.n2 = nn.GroupNorm(2, ch)

    @torch.no_grad()
    def update_expert_bias(self, loads):
        if self.blk == 'vmoe':
            self.mlp.update_expert_bias(loads)

    def apply_rope(self, q, k, rope):
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = _apply_rope(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = _apply_rope(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def unfold(self, x, win, ws, st):
        b, c, h, w = x.shape
        x = x.unfold(2, ws, st).unfold(3, ws, st).contiguous()
        x = x.view(b, c, win * win, ws, ws)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * win * win, c, ws, ws)
        return x

    def fold(self, x, b, c, h, w, win, ws, st):
        x = x.view(b, win * win, c * ws * ws).transpose(1, 2).contiguous()
        x = F.fold(x, output_size=(h, w), kernel_size=ws, stride=st)

        # Normalize by overlap counts
        ones = torch.ones((b, 1, h, w), device=x.device, dtype=x.dtype)
        ones_win = F.unfold(ones, kernel_size=self.ws, stride=self.st)
        overlap_mask = F.fold(ones_win, output_size=(h, w), kernel_size=self.ws, stride=self.st)
        x = x / overlap_mask.clamp_min(1e-6)
        return x

    # Patch Self-attention
    def PSA(self, q, k, v):
        b, c, h, w = q.shape

        q = self.ff(q)
        q = q.view(b, -1, self.heads, h * w).permute(0, 2, 3, 1).contiguous()
        k = k.view(b, -1, self.heads, h * w).permute(0, 2, 3, 1).contiguous()
        v = v.view(b, -1, self.heads, h * w).permute(0, 2, 3, 1).contiguous()
        q, k = self.apply_rope(q, k, self.rope)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale,
                                  max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device))).exp()
        attn = attn * logit_scale
        attn = self.softmax(attn)
        v = (attn @ v)

        v = v.permute(0, 3, 1, 2).contiguous().view(b, c, h, w)

        return v

    def attention_delta(self, x):
        x_ = self.n1(x)
        q, k = torch.chunk(self.qk_proj(x_), 2, dim=1)
        v = (q + k) / 2
        v = self.fsm(v) + self.eea(v)
        return self.drop_path(self.out_proj(self.PSA(q, k, v)))

    def mlp_delta(self, x):
        x_ = self.n2(x)
        loads = 0.
        if self.blk == 'vmoe':
            x_, loads = self.mlp(x_)
        elif self.blk == 'conv':
            x_ = self.mlp(x_)
        return self.drop_path(x_), loads

    def forward(self, x):
        b, c, h, w = x.shape
        self.win = int((h - self.ws) / self.st + 1)

        x1 = self.unfold(x, self.win, self.ws, self.st)
        x1 = x1 + self.attention_delta(x1)
        mlp_delta, loads = self.mlp_delta(x1)
        x1 = x1 + mlp_delta

        x1 = self.fold(x1, b, c, h, w, self.win, self.ws, self.st)
        return x + x1, loads

class BasicBlock(nn.Module):
    def __init__(self, cfg, ch_in, ch_out, heads, ws, st, depth, drop_path, blk):
        super(BasicBlock, self).__init__()
        ch_mid = ch_out // 2
        self.depth = int(depth)
        self.use_llra = True
        self.d = ConvBlock(ch_in, ch_out, 2)

        self.c1 = nn.Sequential(iConv(ch_out, ch_out, 3, groups=ch_out, norm='bn', act='silu'),
                                iConv(ch_out, ch_mid, 1))
        self.attn = nn.ModuleList([Attention(cfg, ch_mid, heads, ws, st,
                                             drop_path[i] if isinstance(drop_path, list) else drop_path,
                                             blk[i])
                                   for i in range(depth)])

        if self.use_llra:
            self.llra_reads = nn.ModuleList([LLRA(ch_mid) for _ in range(2 * depth + 1)])
        else:
            self.llra_reads = nn.ModuleList()

        self.c2 = iConv(2 * ch_mid, ch_out, 1, norm='bn')
        self.sem = SEM(ch_out)

    @torch.no_grad()
    def update_expert_bias(self, loads_list):
        for attn, loads in zip(self.attn, loads_list):
            attn.update_expert_bias(loads)

    def forward(self, x):
        x = self.d(x)
        loads_list = []

        x1 = self.c1(x)

        if self.use_llra:
            b, c, h, w = x1.shape
            ref_attn = self.attn[0]
            win = int((h - ref_attn.ws) / ref_attn.st + 1)
            xw = ref_attn.unfold(x1, win, ref_attn.ws, ref_attn.st)

            history = [xw]
            read_index = 0
            for attn in self.attn:
                h_attn = self.llra_reads[read_index](history)
                read_index += 1
                delta_attn = attn.attention_delta(h_attn)
                history.append(delta_attn)

                h_mlp = self.llra_reads[read_index](history)
                read_index += 1
                delta_mlp, loads = attn.mlp_delta(h_mlp)
                history.append(delta_mlp)
                loads_list.append(loads)

            xw = self.llra_reads[read_index](history)
            x2 = ref_attn.fold(xw, b, c, h, w, win, ref_attn.ws, ref_attn.st)
        else:
            x2 = x1
            for attn in self.attn:
                x2, loads = attn(x2)
                loads_list.append(loads)

        x1 = self.c2(torch.cat([x1, x2], dim=1))

        x1 = x1 + self.sem(x1)
        return x + x1, loads_list

class Encoder(nn.Module):
    def __init__(self, cfg, img_in, ch_in):
        super(Encoder, self).__init__()
        self.cfg = cfg
        depths = [1, 2, 4, 3]
        heads = [2, 4, 6, 8]
        drop_path_rates = [i.item() for i in torch.linspace(0, 0.2, sum(depths))]
        dpr = [drop_path_rates[sum(depths[:i]):sum(depths[:i + 1])] for i in range(len(depths))]
        blk = []
        for n in range(len(depths)):
            blk.append([])
            for i in range(depths[n]):
                if n == 0 or n == 1 or n == 3:
                    blk[n].append('conv')
                    continue
                if i % 2 == 0:
                    blk[n].append('vmoe')
                else:
                    blk[n].append('conv')

        self.Conv = nn.Sequential(nn.Conv2d(img_in, 64, 3, 2, padding=1),
                                  ConvBlock(64, 64, 1))

        self.E1 = BasicBlock(self.cfg, 64, ch_in[0], heads[0], 16, 16, depths[0], dpr[0], blk[0])
        self.E2 = BasicBlock(self.cfg, ch_in[0], ch_in[1], heads[1], 16, 16, depths[1], dpr[1], blk[1])
        self.E3 = BasicBlock(self.cfg, ch_in[1], ch_in[2], heads[2], 8, 8, depths[2], dpr[2], blk[2])
        self.E4 = BasicBlock(self.cfg, ch_in[2], ch_in[3], heads[3], 8, 8, depths[3], dpr[3], blk[3])

        self.patch_proj1 = nn.Conv2d(ch_in[0], ch_in[4], 1, groups=64, bias=False)
        self.patch_proj2 = nn.Conv2d(ch_in[1], ch_in[4], 1, groups=128, bias=False)
        self.patch_proj3 = nn.Conv2d(ch_in[2], ch_in[4], 1, groups=128, bias=False)
        self.patch_proj4 = nn.Conv2d(ch_in[3], ch_in[4], 1, groups=256, bias=False)

        self.d = nn.Dropout(0.05)

    @torch.no_grad()
    def update_expert_bias(self, loads_list):
        for e, loads in zip([self.E3], loads_list):
            e.update_expert_bias(loads)

    def forward(self, x):
        x0 = self.Conv(x)

        loads_list = []
        x1 = self.d(x0)
        x1, _ = self.E1(x1)

        x2 = self.d(x1)
        x2, _ = self.E2(x2)

        x3 = self.d(x2)
        x3, loads3 = self.E3(x3)
        loads_list.append(loads3)

        x4 = self.d(x3)
        x4, _ = self.E4(x4)

        d1 = F.normalize(self.patch_proj1(F.adaptive_avg_pool2d(x1, 16)), dim=1)
        d2 = F.normalize(self.patch_proj2(F.adaptive_avg_pool2d(x2, 16)), dim=1)
        d3 = F.normalize(self.patch_proj3(F.adaptive_avg_pool2d(x3, 16)), dim=1)
        d4 = F.normalize(self.patch_proj4(F.adaptive_avg_pool2d(x4, 16)), dim=1)

        return [x1, x2, x3, x4], [d1, d2, d3, d4], loads_list

class Decoder(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Decoder, self).__init__()
        self.D1 = nn.ModuleList([iConv(ch_in[0], ch_in[0] // 2, 1, dilation=1, groups=1, norm='bn', act='relu'),
                                 iConv(ch_in[0], ch_in[0] // 2, 1, dilation=2, groups=4, norm='bn', act='silu'),
                                 iConv(ch_in[0], 64, 1, dilation=1, groups=2, act='prelu')])

        self.D2 = nn.ModuleList([iConv(ch_in[1], ch_in[1] // 2, 1, dilation=1, groups=1, norm='bn', act='relu'),
                                 iConv(ch_in[1], ch_in[1] // 2, 1, dilation=2, groups=4, norm='bn', act='silu'),
                                 nn.MaxPool2d(3, stride=1, padding=1),
                                 iConv(ch_in[1], ch_in[0], 3, dilation=1, groups=2, act='prelu')])

        self.D3 = nn.ModuleList([iConv(ch_in[2], ch_in[2] // 2, 1, dilation=1, groups=1, norm='bn', act='relu'),
                                 iConv(ch_in[2], ch_in[2] // 2, 1, dilation=2, groups=4, norm='bn', act='silu'),
                                 nn.MaxPool2d(3, stride=1, padding=1),
                                 iConv(ch_in[2], ch_in[1], 1, dilation=1, groups=4, act='prelu')])

        self.D4 = nn.ModuleList([iConv(ch_in[3], ch_in[2], 1, dilation=1, groups=2, act='prelu')])

        self.frcm1 = FRCM(ch_ins=[64, ch_in[0], ch_in[1], ch_in[2], ch_in[3]], ch_out=4)
        self.frcm2 = FRCM(ch_ins=[ch_in[4], ch_in[4], ch_in[4], ch_in[4]], ch_out=8)

        ch_in = 64 + 24 + 40
        self.sem = SEM(ch_in, reduction=4)
        self.conv = iConv(ch_in, 64, 3, groups=64, bias=True, norm='bn', act='prelu')
        self.out = nn.Sequential(iConv(64, 64, 1, bias=False, act='prelu'),
                                 nn.Conv2d(64, ch_out, kernel_size=1, bias=False),
                                 nn.Sigmoid())
        self.d = nn.Dropout(0.1)

    def forward(self, x, d, img_shape):
        x1, x2, x3, x4 = x
        d1, d2, d3, d4 = d

        d4c = self.D4[0]
        x = d4c(x4)
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)

        d3c1a, d3c1b, mp, d3c2 = self.D3
        x_ = torch.cat([d3c1a(x3), d3c1b(x3)], 1)
        x = d3c2(mp(x_ + x))
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)

        d2c1a, d2c1b, mp, d2c2 = self.D2
        x_ = torch.cat([d2c1a(x2), d2c1b(x2)], 1)
        x = d2c2(mp(self.d(x_) + x))
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)

        d1c1a, d1c1b, d1c2 = self.D1
        x_ = torch.cat([d1c1a(x1), d1c1b(x1)], 1)
        x = d1c2(self.d(x_) + x)
        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)

        x = F.interpolate(x, scale_factor=(2), mode='bilinear', align_corners=False)
        sides1 = self.frcm1([x, x1, x2, x3, x4], img_shape)
        sides2 = self.frcm2([d1, d2, d3, d4], img_shape)
        x = torch.cat([x, sides1, sides2], 1)

        return self.out(self.conv(x + self.sem(x)))

class DINOv3_distill(nn.Module):
    def __init__(self, cfg, teacher_type, teacher_dtype):
        super().__init__()
        self.device = cfg.device
        teacher_name = ""

        if teacher_type == 'large':
            teacher_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        elif teacher_type == '7B':
            teacher_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

        self.teacher = AutoModel.from_pretrained(teacher_name, dtype=teacher_dtype)
        self.teacher = self.teacher.to(self.device)

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        x_t_patch = []
        R = int(self.teacher.config.num_register_tokens)

        x = x.to(self.device)
        self.teacher.train(False)
        with torch.no_grad():
            x = self.teacher(x, output_hidden_states=True)
            x = [x.hidden_states[6], x.hidden_states[12], x.hidden_states[18], x.hidden_states[24]]

        for x_i in x:
            # patch tokens
            p_i = F.normalize(x_i[:, 1 + R:, :], dim=-1)  # (B, 1 + R + N, D)
            b, n, c = p_i.shape
            hw = int(n ** 0.5)
            p_i = p_i.permute(0, 2, 1).view(b, c, hw, hw).contiguous()
            p_i = torch.nan_to_num(p_i)
            x_t_patch.append(p_i)
        return x_t_patch

class M(nn.Module):
    def __init__(self, cfg, img_in, segout):
        super(M, self).__init__()
        ch_in = [128, 256, 384, 512]
        self.cfg = cfg
        self.learning_mode = cfg.learning_mode
        self.teacher_type = cfg.teacher_type

        if self.teacher_type == 'large':
            ch_in.append(1024)
        elif self.teacher_type == '7B':
            ch_in.append(4096)
        self.E = Encoder(self.cfg, img_in, ch_in)

        if self.learning_mode in ['Seed', 'Seg']:
            self.D = Decoder(ch_in, segout)
            self.H = Hebbian(ch_in[-1] + 1, 0.95)
            if self.learning_mode == 'Seg':
                self.mask_memory = MaskMemory(capacity=cfg.memory_capacity,
                                              grid_size=cfg.memory_grid_size,
                                              sim_threshold=cfg.memory_sim_threshold,
                                              metric=cfg.memory_metric)

    @torch.no_grad()
    def update_expert_bias(self, loads_list):
        self.E.update_expert_bias(loads_list)

    def forward(self, x, seed=None, hebb_lr=None, update_hebbian=False):
        x = torch.nan_to_num(x)
        img_shape = x.shape[2:]

        x_s, x_s_patch, loads_list = self.E(x)

        if self.learning_mode in ['Seed', 'Seg']:
            r = self.D(x_s, x_s_patch, img_shape)
            r = self.H(x_s_patch[-3], r, seed, hebb_lr, update_hebbian)

            return r, x_s_patch, loads_list

        return x_s_patch, loads_list


if __name__ == '__main__':
    import argparse

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
    x = torch.rand(2, 3, 256, 256).to(cfg.device)
    y, _, _ = model(x)
    print(y.shape)