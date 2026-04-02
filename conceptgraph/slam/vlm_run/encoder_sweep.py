"""
Stage 3: Encoder Sweep.

Loads merge groupings produced by Stage 2, re-crops from the original images
using (color_path, xyxy) pairs, encodes with a specified vision encoder, and
saves per-view + running-average embeddings per object.

Usage:
    python encoder_sweep.py \
        --groupings /path/to/merge_groupings.pkl.gz \
        --encoder "openclip:ViT-bigG-14:laion2b_s39b_b160k" \
        --output_dir /path/to/embeddings/ \
        --device cuda \
        [--use_scaled_crop] [--crop_scale_factor 1.5]
"""

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import sys, os
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conceptgraph.utils.vlms.vlm_encoder import create_encoder


def generate_crop(
    image: Image.Image,
    xyxy,
    use_scaled_crop: bool = True,
    crop_scale_factor: float = 1.5,
    fixed_padding: int = 20,
) -> Image.Image:
    """Generate a single crop from an image given xyxy bbox."""
    img_w, img_h = image.size
    x_min, y_min, x_max, y_max = xyxy
    if use_scaled_crop:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        s = crop_scale_factor
        x_min = max(0, cx - w * s / 2)
        y_min = max(0, cy - h * s / 2)
        x_max = min(img_w, cx + w * s / 2)
        y_max = min(img_h, cy + h * s / 2)
    else:
        x_min = max(0, x_min - fixed_padding)
        y_min = max(0, y_min - fixed_padding)
        x_max = min(img_w, x_max + fixed_padding)
        y_max = min(img_h, y_max + fixed_padding)
    return image.crop((x_min, y_min, x_max, y_max))


def entropy_select_best_feature(
    embeddings_path: Path,
    prompt_list_path: Path,
    encoder_id: str,
    device: str = "cuda",
):
    """
    Post-processing pass: select the minimum-entropy per-view feature as best_feature.

    1. Load a compact prompt list (~50 Replica GT labels).
    2. Encode the prompt list with the text encoder.
    3. For each object's per-view features, compute cosine sim -> softmax -> entropy.
    4. Select the min-entropy view as best_feature.

    Updates the embeddings file in-place.
    """
    print(f"[Entropy Selection] Loading prompt list from {prompt_list_path}...")
    with open(prompt_list_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"[Entropy Selection] {len(prompts)} prompts loaded")

    enc = create_encoder(encoder_id, device)
    text_feats = enc.encode_text(prompts)  # (P, D)

    print(f"[Entropy Selection] Loading embeddings from {embeddings_path}...")
    with gzip.open(embeddings_path, "rb") as f:
        results = pickle.load(f)

    updated = 0
    for obj_id, obj_data in results.items():
        per_view = obj_data.get('per_view_features')
        if per_view is None or len(per_view) == 0:
            continue

        sims = per_view @ text_feats.T  # (V, P)
        probs = np.exp(sims - sims.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        best_idx = int(np.argmin(entropies))

        obj_data['best_feature'] = per_view[best_idx]
        obj_data['best_entropy'] = float(entropies[best_idx])
        obj_data['best_view_idx'] = best_idx
        updated += 1

    with gzip.open(embeddings_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[Entropy Selection] Updated {updated}/{len(results)} objects")

    enc.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Encoder Sweep")
    parser.add_argument("--groupings", type=str, required=True,
                        help="Path to merge_groupings.pkl.gz from Stage 2")
    parser.add_argument("--encoder", type=str, required=True,
                        help="Encoder ID (openclip:<arch>:<pretrained> or HF model ID)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save encoder embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_scaled_crop", action="store_true", default=True)
    parser.add_argument("--no_scaled_crop", dest="use_scaled_crop", action="store_false")
    parser.add_argument("--crop_scale_factor", type=float, default=1.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--entropy_prompt_list", type=str, default=None,
                        help="Path to prompt list for entropy-based feature selection (Stage 5c)")
    args = parser.parse_args()

    groupings_path = Path(args.groupings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Encoder Sweep] Loading groupings from {groupings_path}...")
    with gzip.open(groupings_path, "rb") as f:
        groupings: Dict = pickle.load(f)
    print(f"[Encoder Sweep] {len(groupings)} objects loaded")

    encoder_name = args.encoder.replace("/", "_").replace(":", "_")
    print(f"[Encoder Sweep] Loading encoder: {args.encoder}")
    enc = create_encoder(args.encoder, args.device)

    results: Dict[str, Dict] = {}
    image_cache: Dict[str, Image.Image] = {}

    for obj_id, obj_data in tqdm(groupings.items(), desc="Encoding objects"):
        color_paths = obj_data['color_path']
        xyxy_list = obj_data['xyxy']
        n_points_list = obj_data.get('n_points_per_view', [])

        crops: List[Image.Image] = []
        view_metadata: List[Dict] = []

        for view_idx, (cp, xyxy) in enumerate(zip(color_paths, xyxy_list)):
            if cp not in image_cache:
                image_cache[cp] = Image.open(cp).convert("RGB")
            img = image_cache[cp]

            crop = generate_crop(
                img, xyxy,
                use_scaled_crop=args.use_scaled_crop,
                crop_scale_factor=args.crop_scale_factor,
            )
            crops.append(crop)
            n_pts = n_points_list[view_idx] if view_idx < len(n_points_list) else 0
            view_metadata.append({
                'view_idx': view_idx,
                'image_idx': obj_data['image_idx'][view_idx] if view_idx < len(obj_data['image_idx']) else -1,
                'n_points': n_pts,
            })

            if len(image_cache) > 200:
                image_cache.clear()

        if not crops:
            continue

        all_feats = []
        for batch_start in range(0, len(crops), args.batch_size):
            batch = crops[batch_start:batch_start + args.batch_size]
            feats, _ = enc.encode_crops(batch)
            all_feats.append(feats)
        per_view_features = np.concatenate(all_feats, axis=0)  # (V, D)

        n_dets = len(per_view_features)
        weights = np.ones(n_dets, dtype=np.float32)
        running_avg = np.average(per_view_features, axis=0, weights=weights)
        running_avg = running_avg / (np.linalg.norm(running_avg) + 1e-8)

        results[obj_id] = {
            'running_avg': running_avg,
            'per_view_features': per_view_features,
            'view_metadata': view_metadata,
            'best_feature': None,
        }

    out_path = output_dir / f"{encoder_name}.pkl.gz"
    with gzip.open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"[Encoder Sweep] Saved {len(results)} object embeddings to {out_path}")

    enc.cleanup()

    if args.entropy_prompt_list:
        entropy_select_best_feature(
            out_path,
            Path(args.entropy_prompt_list),
            args.encoder,
            args.device,
        )


if __name__ == "__main__":
    main()
