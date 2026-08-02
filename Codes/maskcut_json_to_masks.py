import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from pycocotools import mask as mask_utils

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, dict):
        # RLE
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)

    elif isinstance(segmentation, list):
        # polygon(s)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rles)

    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    return mask

def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def build_image_maps(data):
    image_info_by_id = {}
    annotations_by_image = defaultdict(list)

    # Case 1: full COCO-style dict
    if isinstance(data, dict) and "annotations" in data:
        for img in data.get("images", []):
            image_info_by_id[img["id"]] = {"file_name": img["file_name"],
                                           "height": img["height"],
                                           "width": img["width"]}

        for ann in data["annotations"]:
            annotations_by_image[ann["image_id"]].append(ann)

    # Case 2: flat result list
    elif isinstance(data, list):
        for ann in data:
            annotations_by_image[ann["image_id"]].append(ann)

    else:
        raise ValueError("Unsupported JSON format.")

    return image_info_by_id, annotations_by_image


def build_filename_lookup_from_folder(image_dir):
    lookup = {}
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        lookup[stem] = name
    return lookup

def resolve_filename(image_id, image_info_by_id, stem_lookup):
    if image_id in image_info_by_id:
        return image_info_by_id[image_id]["file_name"]

    image_id_str = str(image_id)
    if image_id_str in stem_lookup:
        return stem_lookup[image_id_str]

    # Sometimes file names may already be full names
    if image_id_str in stem_lookup.values():
        return image_id_str

    return None

def resolve_size(image_id, anns, image_info_by_id, image_dir, file_name):
    if image_id in image_info_by_id:
        return image_info_by_id[image_id]["height"], image_info_by_id[image_id]["width"]

    # try annotation bbox/mask size from RLE
    for ann in anns:
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict) and "size" in seg:
            h, w = seg["size"]
            return int(h), int(w)

    # fallback: read image
    img = Image.open(os.path.join(image_dir, file_name))
    w, h = img.size
    return h, w

def convert_json_to_masks(json_path, image_dir, out_dir, score_thr=0.0, keep_topk=None, merge_mode="or"):
    ensure_dir(out_dir)

    data = load_json(json_path)
    image_info_by_id, annotations_by_image = build_image_maps(data)
    stem_lookup = build_filename_lookup_from_folder(image_dir)

    saved = 0
    skipped = 0

    for image_id, anns in annotations_by_image.items():
        file_name = resolve_filename(image_id, image_info_by_id, stem_lookup)
        if file_name is None:
            print(f"[WARN] Could not resolve filename for image_id={image_id}")
            skipped += 1
            continue

        height, width = resolve_size(image_id, anns, image_info_by_id, image_dir, file_name)

        # Filter by score
        filtered = []
        for ann in anns:
            score = ann.get("score", 1.0)
            if score >= score_thr:
                filtered.append(ann)

        if len(filtered) == 0:
            merged = np.zeros((height, width), dtype=np.uint8)
        else:
            # Optional top-k
            filtered = sorted(filtered, key=lambda x: x.get("score", 1.0), reverse=True)
            if keep_topk is not None:
                filtered = filtered[:keep_topk]

            if merge_mode == "or":
                merged = np.zeros((width, height), dtype=np.uint8)
                for ann in filtered:
                    seg = ann["segmentation"]
                    mask = decode_segmentation(seg, height, width)
                    merged = np.logical_or(merged, mask).astype(np.uint8)

            elif merge_mode == "best":
                best_ann = filtered[0]
                merged = decode_segmentation(best_ann["segmentation"], height, width)

            else:
                raise ValueError(f"Unknown merge_mode: {merge_mode}")

        out_name = os.path.splitext(file_name)[0] + ".png"
        out_path = os.path.join(out_dir, out_name)

        Image.fromarray(merged * 255).save(out_path)
        saved += 1

    print(f"[DONE] Saved {saved} masks to: {out_dir}")
    print(f"[INFO] Skipped {skipped} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--keep_topk", type=int, default=None)
    parser.add_argument("--merge_mode", type=str, default="or", choices=["or", "best"])
    args = parser.parse_args()

    convert_json_to_masks(json_path=args.json_path,
                          image_dir=args.image_dir,
                          out_dir=args.out_dir,
                          score_thr=args.score_thr,
                          keep_topk=args.keep_topk,
                          merge_mode=args.merge_mode)

