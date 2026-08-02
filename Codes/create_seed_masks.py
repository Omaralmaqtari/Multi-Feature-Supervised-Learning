# create_seed_masks_from_binary.py
# -*- coding: utf-8 -*-

"""
Create sparse center seed masks from binary segmentation masks.

Input binary masks:
    foreground: any value > 0
    background: value == 0

Output seed masks:
    1   = foreground center seed
    0   = optional background seed
    255 = ignored / unlabeled pixel

Recommended for Seed mode:
    use sparse loss where only pixels with value 0 or 1 are supervised,
    and pixels with 255 are ignored.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import (label, distance_transform_edt, binary_erosion, binary_dilation)


FG_SEED_VALUE = 1
BG_SEED_VALUE = 0


def draw_disk(out, cy, cx, radius, value):
    h, w = out.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    out[disk] = value


def get_component_center(component, mode="edt"):
    """
    component: bool [H,W]

    mode:
        "edt":
            center = pixel farthest from boundary.
            Most reliable for seed learning.

        "centroid_snap":
            center = foreground pixel closest to geometric centroid.
            More geometric, but may be closer to boundary.

        "hybrid":
            balances centroid closeness and boundary distance.
    """
    ys, xs = np.where(component)
    if len(ys) == 0:
        return None

    if mode == "edt":
        dist = distance_transform_edt(component)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        return int(cy), int(cx)

    cy_float = ys.mean()
    cx_float = xs.mean()

    if mode == "centroid_snap":
        d2 = (ys - cy_float) ** 2 + (xs - cx_float) ** 2
        idx = int(np.argmin(d2))
        return int(ys[idx]), int(xs[idx])

    if mode == "hybrid":
        dist = distance_transform_edt(component)
        d_center = np.sqrt((ys - cy_float) ** 2 + (xs - cx_float) ** 2)
        d_center = d_center / (d_center.max() + 1e-6)

        boundary_score = dist[ys, xs]
        boundary_score = boundary_score / (boundary_score.max() + 1e-6)

        # high score = close to centroid and far from boundary
        score = 0.55 * boundary_score + 0.45 * (1.0 - d_center)
        idx = int(np.argmax(score))
        return int(ys[idx]), int(xs[idx])

    raise ValueError(f"Unknown center mode: {mode}")


def make_foreground_seeds(
    binary_mask,
    seed_radius=3,
    min_area=20,
    center_mode="edt",
    erode_before_center=False,
    erosion_iters=1,
)   :
    """
    binary_mask: bool [H,W]

    returns:
        seed: uint8 [H,W], foreground seeds only; ignored elsewhere
        centers: list of (cy, cx, area)
    """
    h, w = binary_mask.shape
    seed = np.full((h, w), BG_SEED_VALUE, dtype=np.uint8)

    comp_map, n_comp = label(binary_mask.astype(np.uint8))
    centers = []

    for comp_id in range(1, n_comp + 1):
        component = comp_map == comp_id
        area = int(component.sum())

        if area < min_area:
            continue

        center_component = component

        # Optional: makes the selected center more conservative.
        # If erosion destroys a small object, fall back to original component.
        if erode_before_center:
            eroded = binary_erosion(component, iterations=erosion_iters)
            if eroded.sum() > 0:
                center_component = eroded

        center = get_component_center(center_component, mode=center_mode)
        if center is None:
            continue

        cy, cx = center

        # Ensure center is inside original component.
        if not component[cy, cx]:
            dist = distance_transform_edt(component)
            cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
            cy, cx = int(cy), int(cx)

        draw_disk(seed, cy, cx, seed_radius, FG_SEED_VALUE)
        centers.append((cy, cx, area))

    return seed, centers


def add_background_seeds(
    seed,
    binary_mask,
    radius=3,
    seeds_per_image=5,
    min_distance_from_fg=8,
    ):
    """
    Add sparse background seeds far from foreground objects.
    This is optional but useful for decoder warm-up.
    """
    if seeds_per_image <= 0:
        return seed

    bg = ~binary_mask

    # Exclude a margin around foreground so background seeds are not ambiguous.
    fg_dilated = binary_dilation(binary_mask, iterations=min_distance_from_fg)
    safe_bg = bg & (~fg_dilated)

    if safe_bg.sum() == 0:
        safe_bg = bg

    bg_score = distance_transform_edt(safe_bg)

    for _ in range(seeds_per_image):
        if bg_score.max() <= 0:
            break

        by, bx = np.unravel_index(np.argmax(bg_score), bg_score.shape)
        by, bx = int(by), int(bx)

        draw_disk(seed, by, bx, radius, BG_SEED_VALUE)

        # Suppress nearby area to spread background seeds.
        yy, xx = np.ogrid[:seed.shape[0], :seed.shape[1]]
        suppress_radius = max(radius * 5, 12)
        suppress = (yy - by) ** 2 + (xx - bx) ** 2 <= suppress_radius ** 2
        bg_score[suppress] = 0

    return seed


def create_seed_mask_from_binary(
    mask_array,
    seed_radius=3,
    min_area=20,
    center_mode="edt",
    add_bg=False,
    bg_radius=3,
    bg_seeds_per_image=5,
    min_distance_from_fg=8,
    erode_before_center=False,
    erosion_iters=1,
    ):
    """
    mask_array:
        binary mask where foreground is > 0.
    """
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    binary = mask_array > 0

    seed, centers = make_foreground_seeds(
        binary_mask=binary,
        seed_radius=seed_radius,
        min_area=min_area,
        center_mode=center_mode,
        erode_before_center=erode_before_center,
        erosion_iters=erosion_iters,
        )

    if add_bg:
        seed = add_background_seeds(
            seed=seed,
            binary_mask=binary,
            radius=bg_radius,
            seeds_per_image=bg_seeds_per_image,
            min_distance_from_fg=min_distance_from_fg,
            )

    return seed, centers


def save_debug_overlay(mask_array, seed_array, save_path):
    """
    Debug RGB overlay:
        foreground mask = gray
        foreground seed = red
        background seed = blue
    """
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]

    binary = mask_array > 0
    h, w = binary.shape

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[binary] = np.array([120, 120, 120], dtype=np.uint8)

    fg_seed = seed_array == FG_SEED_VALUE
    bg_seed = seed_array == BG_SEED_VALUE

    rgb[fg_seed] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[bg_seed] = np.array([0, 80, 255], dtype=np.uint8)

    Image.fromarray(rgb).save(save_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src", type=str, default="D:/Postdoc-SWJTU/Research/Datasets/COCO 2017/coco20k/val_t_mask/",
                        help="Folder containing binary masks.")
    parser.add_argument("--dst", type=str, default="D:/Postdoc-SWJTU/Research/Datasets/COCO 2017/coco20k/val_t_seed/",
                        help="Folder to save seed masks.")

    parser.add_argument("--seed-radius", type=int, default=6)
    parser.add_argument("--min-area", type=int, default=16)

    parser.add_argument("--center-mode", type=str, default="centroid_snap",
                        choices=["edt", "centroid_snap", "hybrid"])

    parser.add_argument("--erode-before-center", action="store_true")
    parser.add_argument("--erosion-iters", type=int, default=1)

    parser.add_argument("--add-bg", action="store_true")
    parser.add_argument("--bg-radius", type=int, default=3)
    parser.add_argument("--bg-seeds-per-image", type=int, default=5)
    parser.add_argument("--min-distance-from-fg", type=int, default=5)

    parser.add_argument("--debug", action="store_true", help="Save RGB overlays for checking seed positions.")
    parser.add_argument("--debug-dir", type=str, default=None)

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if args.debug:
        debug_dir = Path(args.debug_dir) if args.debug_dir is not None else dst / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted([
        p for p in src.iterdir()
        if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

    total_objects = 0
    empty_images = 0

    for p in mask_paths:
        mask = np.array(Image.open(p))

        seed, centers = create_seed_mask_from_binary(
            mask_array=mask,
            seed_radius=args.seed_radius,
            min_area=args.min_area,
            center_mode=args.center_mode,
            add_bg=args.add_bg,
            bg_radius=args.bg_radius,
            bg_seeds_per_image=args.bg_seeds_per_image,
            min_distance_from_fg=args.min_distance_from_fg,
            erode_before_center=args.erode_before_center,
            erosion_iters=args.erosion_iters,
            )

        if len(centers) == 0:
            empty_images += 1

        total_objects += len(centers)

        Image.fromarray(seed*255).save(dst / p.name)

        if args.debug:
            save_debug_overlay(mask, seed, debug_dir / p.name)

    print(f"Processed masks: {len(mask_paths)}")
    print(f"Total foreground components seeded: {total_objects}")
    print(f"Images with no valid components: {empty_images}")
    print(f"Seed masks saved to: {dst}")

    if args.debug:
        print(f"Debug overlays saved to: {debug_dir}")


if __name__ == "__main__":
    main()