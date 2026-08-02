# memory_module.py
# -*- coding: utf-8 -*-

import os
import re
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8

def _minmax_norm(x):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min).clamp_min(eps)

def _binary(x, threshold=0.5):
    return (x >= threshold).float()

def _stable_hash_id(text: str):
    """
    Stable 63-bit integer hash for saving sample IDs as tensors.
    """
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()
    return int(h[:15], 16)

class MaskMemory(nn.Module):
    """
    Memory bank for adaptive Seg-stage mask correction.
    """

    def __init__(self, capacity=262, grid_size=8, sim_threshold=0.85, metric="iou"):
        super().__init__()
        self.capacity = int(capacity)
        self.grid_size = int(grid_size)
        self.sim_threshold = float(sim_threshold)
        self.metric = str(metric)

        self.register_buffer("memory_masks", torch.zeros(self.capacity, 1, self.grid_size, self.grid_size), persistent=True)
        self.register_buffer("valid", torch.zeros(self.capacity, dtype=torch.bool), persistent=True)
        self.register_buffer("sample_hashes", torch.zeros(self.capacity, dtype=torch.long), persistent=True)
        self.register_buffer("num_inserted", torch.zeros((), dtype=torch.long), persistent=True)

        # String IDs are saved via get_extra_state / set_extra_state.
        self.memory_ids = [""] * self.capacity

    @property
    def count(self):
        return int(self.valid.sum().item())

    def to_grid_mask(self, mask):
        """
        mask: [1,H,W]
        returns: [1,G,G]
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = F.interpolate(mask.unsqueeze(0).float(), size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)[0]

        return _minmax_norm(mask)

    def resize_from_grid(self, mask8, out_hw):
        """
        mask8: [1,G,G]
        """
        return F.interpolate(mask8.unsqueeze(0).float(), size=out_hw, mode="bilinear", align_corners=False)[0]

    def normalize_file_id(self, file_id):
        """
        Normalize image/mask filename for matching.

        Example:
            2007_000033.jpg -> 2007_000033
            2007_000033.png -> 2007_000033
        """
        file_id = str(file_id)
        base = os.path.basename(file_id)
        stem, _ = os.path.splitext(base)
        return stem.lower().strip()

    @torch.no_grad()
    def find_by_file_id(self, file_id):
        """
        Find memory mask by GT-mask filename similarity.

        Returns:
            memory_mask: [1,G,G] or None
            memory_idx: int or None
        """
        if self.count == 0:
            return None, None

        query = self.normalize_file_id(file_id)

        for idx in torch.nonzero(self.valid, as_tuple=False).flatten().tolist():
            mem_id = self.memory_ids[int(idx)]
            mem_key = self.normalize_file_id(mem_id)

            # Stem-level match handles .jpg/.png extension differences.
            if mem_key == query:
                return self.memory_masks[int(idx)].detach().clone(), int(idx)

        return None, None

    @torch.no_grad()
    def is_unique_mask(self, mask_grid, unique_threshold=0.9):
        """
        Check whether a candidate memory-grid mask is unique.

        mask_grid:
            [1,G,G] or [G,G]

        Returns:
            True  -> mask is sufficiently different from existing memory masks
            False -> mask is duplicate / near-duplicate
        """
        if mask_grid.dim() == 2:
            mask_grid = mask_grid.unsqueeze(0)

        # Empty memory means the first valid mask is unique.
        if self.count == 0:
            return True

        sims, _, _ = self.similarity(mask_grid)

        if sims is None:
            return True

        max_sim = float(sims.max().item())

        # If similarity is too high, this mask already exists in memory.
        return max_sim < float(unique_threshold)

    @torch.no_grad()
    def add(self, mask8, sample_id):
        """
        If memory is not full:
            append at first free slot.
        If memory is full:
            remove oldest slot 0, shift left, insert at last slot.
        """
        if mask8.dim() == 2:
            mask8 = mask8.unsqueeze(0)

        sample_id = str(sample_id)
        sample_hash = _stable_hash_id(sample_id)

        if self.count < self.capacity:
            idx = self.count
        else:
            # FIFO discard oldest.
            self.memory_masks[:-1].copy_(self.memory_masks[1:].clone())
            self.valid[:-1].copy_(self.valid[1:].clone())
            self.sample_hashes[:-1].copy_(self.sample_hashes[1:].clone())
            self.memory_ids = self.memory_ids[1:] + [""]
            idx = self.capacity - 1

        self.memory_masks[idx].copy_(mask8.to(self.memory_masks.device))
        self.valid[idx] = True
        self.sample_hashes[idx] = int(sample_hash)
        self.memory_ids[idx] = sample_id
        self.num_inserted.add_(1)

    @torch.no_grad()
    def add_unique_gt_batch(self, gt_masks, sample_ids=None, max_add=None, unique_threshold=0.9, min_fg_pixels=2):
        """
        Insert only unique GT masks into memory.

        gt_masks:
            [B,1,H,W] or [B,H,W]

        Returns:
            added:   number of inserted masks
            skipped: number of duplicate/invalid masks
        """
        if gt_masks is None:
            return 0, 0

        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        gt_masks = gt_masks.detach().float().to(self.memory_masks.device)
        B = gt_masks.shape[0]

        if sample_ids is None:
            sample_ids = [f"init_gt_{int(self.num_inserted.item())}_{i}" for i in range(B)]
        else:
            sample_ids = [str(x) for x in sample_ids]

        if max_add is None:
            max_add = B

        added = 0
        skipped = 0
        for b in range(B):
            if added >= int(max_add):
                break

            gt = gt_masks[b]

            if gt.dim() == 2:
                gt = gt.unsqueeze(0)

            gt = _binary(gt)

            # Convert GT to memory grid.
            gt_grid = self.to_grid_mask(gt)
            gt_grid = _binary(gt_grid)

            # Skip empty masks or nearly empty masks after grid conversion.
            if gt_grid.sum().item() < int(min_fg_pixels):
                skipped += 1
                continue

            # Skip repeated / near-repeated masks.
            if not self.is_unique_mask(gt_grid, unique_threshold=unique_threshold):
                skipped += 1
                continue

            self.add(gt_grid, sample_id=sample_ids[b])
            added += 1

        return added, skipped

    @torch.no_grad()
    def similarity(self, mask8):
        """
        mask8: [1,G,G]

        returns:
            sims: [M]
            mem:  [M,1,G,G]
            ids:  list[str]
        """
        if self.count == 0:
            return None, None, []

        valid_idx = torch.nonzero(self.valid, as_tuple=False).flatten()
        mem = self.memory_masks[valid_idx].to(mask8.device).float()
        ids = [self.memory_ids[int(i)] for i in valid_idx.cpu().tolist()]

        q = _binary(mask8).unsqueeze(0)  # [1,1,G,G]

        if self.metric == "dice":
            inter = (mem * q).sum(dim=(1, 2, 3))
            den = mem.sum(dim=(1, 2, 3)) + q.sum(dim=(1, 2, 3))
            sims = (2.0 * inter) / den.clamp_min(eps)
        elif self.metric == "iou":
            inter = (mem * q).sum(dim=(1, 2, 3))
            union = ((mem + q) > 0).float().sum(dim=(1, 2, 3))
            sims = inter / union.clamp_min(eps)
        else:
            raise ValueError(f"Unknown memory metric: {self.metric}")

        return sims, mem, ids

    @torch.no_grad()
    def match_best_and_refine_one(self, gen_mask):
        """
        gen_mask: [1,H,W]

        Compare the selected generated mask with ALL valid masks in memory.
        If the best similarity >= threshold:
            refined = mean(generated_mask, best_memory_mask)
        else:
            return original generated mask.

        returns:
            refined_mask: [1,H,W]
            matched: bool
            best_sim: float
            matched_id: str | None
        """
        out_hw = gen_mask.shape[-2:]
        gen8 = self.to_grid_mask(gen_mask)

        sims, mem, mem_ids = self.similarity(gen8)

        if sims is None:
            return gen_mask, False, 0.0, None

        best_sim, best_idx = sims.max(dim=0)
        best_sim_value = float(best_sim.item())

        if best_sim_value < self.sim_threshold:
            return gen_mask, False, best_sim_value, None

        best_mem_mask = mem[best_idx]  # [1,G,G]
        matched_id = mem_ids[int(best_idx.item())]

        refined8 = 0.5*_minmax_norm(gen8) + 0.5*best_mem_mask
        refined8 = _minmax_norm(refined8)

        refined = self.resize_from_grid(refined8, out_hw)
        refined = _minmax_norm(refined)

        return refined, True, best_sim_value, matched_id

    @torch.no_grad()
    def refine_selected_from_batch(self, gen_masks, enable_memory_match=False, selected_index=None, sample_ids=None):
        """
        Memory behavior:

        1. For every sample in the batch:
            if its maskcut filename exists in memory:
                use the memory maskcut directly.

        2. For samples not found in memory:
            generated mask remains unchanged.

        3. If enable_memory_match=True:
            choose one selected image.
            If that selected image was already replaced by filename-memory:
                do nothing else.
            Otherwise:
                compare generated mask with memory.
                if best similarity >= threshold:
                    average generated mask with the best memory mask.
                else:
                    keep generated mask unchanged.
        """
        B = gen_masks.shape[0]
        device = gen_masks.device
        out = gen_masks.detach().clone()

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(B)]
        else:
            sample_ids = [str(x) for x in sample_ids]

        info = {"memory_count": self.count,
                "selected_index": None,
                "selected_sample_id": None,
                "filename_hits": 0,
                "filename_hit_indices": [],
                "used_filename_memory": False,
                "matched": False,
                "best_sim": 0.0,
                "matched_id": None}

        filename_replaced = torch.zeros(B, device=device, dtype=torch.bool)

        for b in range(B):
            mem_grid, mem_idx = self.find_by_file_id(sample_ids[b])

            if mem_grid is None:
                continue

            mem_mask = self.resize_from_grid(mem_grid.to(device), out_hw=out.shape[-2:])

            out[b] = mem_mask
            filename_replaced[b] = True

            info["filename_hits"] += 1
            info["filename_hit_indices"].append(int(b))

        if not enable_memory_match:
            return out, info

        if selected_index is None:
            selected_index = int(torch.randint(0, B, (1,), device=device).item())

        selected_index = int(selected_index)
        selected_index = max(0, min(selected_index, B - 1))

        selected_id = sample_ids[selected_index]

        info["selected_index"] = selected_index
        info["selected_sample_id"] = selected_id

        if bool(filename_replaced[selected_index]):
            info["used_filename_memory"] = True
            return out, info

        # Otherwise, apply normal generated-mask matching.
        refined, matched, best_sim, matched_id = self.match_best_and_refine_one(out[selected_index])

        info["matched"] = bool(matched)
        info["best_sim"] = float(best_sim)
        info["matched_id"] = matched_id

        if matched:
            out[selected_index] = refined

        return out, info