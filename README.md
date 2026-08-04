# MFSL-Lightweight Distillation and Adaptive Mask Learning Framework for Unsupervised Object Segmentation

Official PyTorch implementation of **MFSL**, a three-stage framework for compact dense-feature distillation and unsupervised binary object segmentation.

MFSL combines:

1. **DINOv3 feature distillation** into a lightweight student encoder.
2. **MaskCut-based seed learning** for decoder initialization.
3. **Adaptive multi-feature mask learning** using clustering, voting, memory matching, and cyclic pseudo-mask weighting.

> **Task:** class-agnostic foreground/background segmentation  
> **Training stages:** `Dist` → `Seed` → `Seg`  
> **Datasets used in the paper:** PASCAL VOC 2012 and COCO20K derived from COCO 2017

---

### Main components

- Lightweight hierarchical encoder with four feature levels.
- Integrated attention block with:
  - local window processing;
  - cosine attention;
  - rotary position encoding;
  - frequency filtering;
  - Fourier-series feature mapping;
  - edge-enhanced value features.
- Token-level vision mixture of experts:
- Local learnable residual attention over preceding residual features.
- Multi-level decoder with feature reduction and concatenation.
- Three-stage unsupervised learning procedure.
- Adaptive K estimation from image color distributions.
- Multi-feature K-means and normalized-cut mask generation.
- Soft candidate voting, mask memory, and cyclic pseudo-mask fusion.

---

<p align="center">
  <img src="https://github.com/Omaralmaqtari/Multi-Feature-Supervised-Learning/blob/main/Figures/learning%20algorithm.png" alt="MFSL learning algorithm" width="33%">
</p>

**Figure 1.** Distillation–Seed–Segmentation training process.

<p align="center">
  <img src="https://github.com/Omaralmaqtari/Multi-Feature-Supervised-Learning/blob/main/Figures/model%20architecture.png" alt="MFSL model architecture" width="33%">
</p>

**Figure 2.** MFSL architecture and the proposed basic blocks.

<p align="center">
  <img src="https://github.com/Omaralmaqtari/Multi-Feature-Supervised-Learning/blob/main/Figures/K%20estimation.png?raw=true" alt="Adaptive K estimation" width="33%">
</p>

**Figure 3.** Adaptive K estimation from the normalized RGB histogram.

---

## Requirements

A CUDA-capable GPU.

### Core packages

- Python 3.12+
- PyTorch
- torchvision
- NumPy
- Pillow
- SciPy
- Kornia
- timm
- ModelScope
- Matplotlib
- tqdm
- ptflops
- ncut_pytorch
- Muon.SingleDeviceMuonWithAuxAdam

---

## Dataset preparation

Set `dataset_path` in `main.py` to the parent directory of the selected dataset.

The loader expects the following structure.

### PASCAL VOC 2012

```text
datasets/
└── VOC 2012/
    ├── train/
    ├── train_mask/      # Maskcut masks
    ├── train_t/
    ├── train_t_seed/
    ├── val/
    ├── val_mask/       # Original masks for evaluation
    ├── val_t/
    └── val_t_seed/
```

### COCO20K

```text
datasets/
└── COCO 2017/
    └── coco20k/
        ├── train/
        ├── train_mask/      # Maskcut masks
        ├── train_t/
        ├── train_t_seed/
        ├── val/
        ├── val_mask/        # Original masks for evaluation
        ├── val_t/
        └── val_t_seed/
```

---

## Pretrained weights

Place the uploaded checkpoints in `weights/`.

The training code builds checkpoint names using:

```text
<model_type>_<dataset>_<teacher_type>_<learning_mode>_<experiment_num>.pth
```

Examples:

```text
weights/
├── M_VOC 2012_large_Dist_1.pth
├── M_VOC 2012_large_Seed_1.pth
├── M_VOC 2012_large_Seg_1.pth
├── M_COCO 2017_large_Dist_1.pth
├── M_COCO 2017_large_Seed_1.pth
└── M_COCO 2017_large_Seg_1.pth
```

---

## Acknowledgments

This implementation builds on components from DINOv3, MaskCut, and normalized cuts. Please cite the corresponding original works when using this repository.
