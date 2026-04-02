"""
Stage 4: VLM Caption Sweep.

Loads merge groupings and object PKL from Stage 2, selects the top-K views
per object by n_points, sends crops to a VLM API for captioning (caption,
color, material), then summarizes per-view captions into a canonical tag.

Usage:
    python vlm_caption_sweep.py \
        --groupings /path/to/merge_groupings.pkl.gz \
        --vlm_api_url http://localhost:8000/v1 \
        --vlm_model_name Qwen/Qwen3-VL-2B-Instruct \
        --output_dir /path/to/captions/ \
        --top_k_views 10 \
        [--use_scaled_crop] [--crop_scale_factor 1.5]
"""

import argparse
import base64
import gzip
import io
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from conceptgraph.utils.vlms.vlm_api import VLMAPIClient, wait_for_server


CAPTION_SYSTEM_PROMPT = (
    "You are an assistant that describes objects in indoor scenes. "
    "Be concise and specific."
)
CAPTION_TEMPLATE = (
    "Describe this object in one sentence. Focus on what the object is, "
    "not its surroundings."
)
COLOR_TEMPLATE = "What is the primary color of this object? Reply with just the color name."
MATERIAL_TEMPLATE = "What material is this object made of? Reply with just the material name."
SUMMARIZE_TEMPLATE = (
    "Given these per-view descriptions of the same object, produce:\n"
    "1. A single canonical object tag (e.g. 'wooden chair', 'white wall')\n"
    "2. A list of candidate tags\n"
    "3. A brief summary caption\n\n"
    "Per-view descriptions:\n{descriptions}\n\n"
    "Respond in JSON format:\n"
    '{{ "canonical_tag": "...", "candidate_tags": ["..."], "summary": "..." }}'
)


def generate_crop(
    image: Image.Image, xyxy,
    use_scaled_crop: bool = True, crop_scale_factor: float = 1.5,
) -> Image.Image:
    img_w, img_h = image.size
    x_min, y_min, x_max, y_max = xyxy
    if use_scaled_crop:
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        s = crop_scale_factor
        x_min = max(0, cx - w * s / 2)
        y_min = max(0, cy - h * s / 2)
        x_max = min(img_w, cx + w * s / 2)
        y_max = min(img_h, cy + h * s / 2)
    return image.crop((x_min, y_min, x_max, y_max))


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def select_top_views(obj_data: Dict, top_k: int) -> List[int]:
    """Select up to top_k view indices, ranked by n_points_per_view descending."""
    n_points = obj_data.get('n_points_per_view', [])
    if not n_points:
        return list(range(min(top_k, len(obj_data['color_path']))))
    indexed = list(enumerate(n_points))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:top_k]]


def query_vlm(client: VLMAPIClient, crop: Image.Image, prompt: str) -> str:
    """Send a single image+prompt query to the VLM API."""
    b64 = pil_to_base64(crop)
    messages = [
        {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]},
    ]
    try:
        resp = client.client.chat.completions.create(
            model=client.model_name,
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[error: {e}]"


def summarize_captions(client: VLMAPIClient, descriptions: List[str]) -> Dict:
    """Ask the VLM to summarize per-view captions into canonical_tag + candidates."""
    desc_text = "\n".join(f"- {d}" for d in descriptions)
    prompt = SUMMARIZE_TEMPLATE.format(descriptions=desc_text)
    messages = [
        {"role": "system", "content": "You are a JSON-only assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.client.chat.completions.create(
            model=client.model_name,
            messages=messages,
            max_tokens=300,
            temperature=0.0,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return {
            "canonical_tag": descriptions[0] if descriptions else "unknown",
            "candidate_tags": descriptions[:3],
            "summary": " | ".join(descriptions[:3]),
        }


def main():
    parser = argparse.ArgumentParser(description="Stage 4: VLM Caption Sweep")
    parser.add_argument("--groupings", type=str, required=True)
    parser.add_argument("--vlm_api_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--vlm_model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k_views", type=int, default=10)
    parser.add_argument("--use_scaled_crop", action="store_true", default=True)
    parser.add_argument("--no_scaled_crop", dest="use_scaled_crop", action="store_false")
    parser.add_argument("--crop_scale_factor", type=float, default=1.5)
    args = parser.parse_args()

    groupings_path = Path(args.groupings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Caption Sweep] Loading groupings from {groupings_path}...")
    with gzip.open(groupings_path, "rb") as f:
        groupings: Dict = pickle.load(f)
    print(f"[Caption Sweep] {len(groupings)} objects")

    vlm_name = args.vlm_model_name.replace("/", "_")

    print(f"[Caption Sweep] Connecting to VLM API at {args.vlm_api_url}...")
    wait_for_server(args.vlm_api_url)
    client = VLMAPIClient(
        base_url=args.vlm_api_url,
        model_name=args.vlm_model_name,
    )

    obj_json: Dict[str, Dict] = {}
    image_cache: Dict[str, Image.Image] = {}

    for obj_id, obj_data in tqdm(groupings.items(), desc="Captioning"):
        view_indices = select_top_views(obj_data, args.top_k_views)
        color_paths = obj_data['color_path']
        xyxy_list = obj_data['xyxy']

        per_view_captions = []
        colors = []
        materials = []

        for vi in view_indices:
            cp = color_paths[vi]
            if cp not in image_cache:
                image_cache[cp] = Image.open(cp).convert("RGB")
            img = image_cache[cp]
            crop = generate_crop(img, xyxy_list[vi],
                                 use_scaled_crop=args.use_scaled_crop,
                                 crop_scale_factor=args.crop_scale_factor)

            caption = query_vlm(client, crop, CAPTION_TEMPLATE)
            color = query_vlm(client, crop, COLOR_TEMPLATE)
            material = query_vlm(client, crop, MATERIAL_TEMPLATE)

            per_view_captions.append(caption)
            colors.append(color)
            materials.append(material)

            if len(image_cache) > 200:
                image_cache.clear()

        summary = summarize_captions(client, per_view_captions)

        from collections import Counter
        color_counts = Counter(colors)
        material_counts = Counter(materials)

        obj_json[obj_id] = {
            'obj_idx': obj_data.get('obj_idx', -1),
            'class_name': obj_data.get('class_name', 'unknown'),
            'canonical_tag': summary.get('canonical_tag', ''),
            'candidate_tags': summary.get('candidate_tags', []),
            'summary': summary.get('summary', ''),
            'color': color_counts.most_common(1)[0][0] if color_counts else '',
            'material': material_counts.most_common(1)[0][0] if material_counts else '',
            'per_view_captions': per_view_captions,
            'n_views': len(view_indices),
            'bbox_center': obj_data.get('bbox_center', []),
            'bbox_extent': obj_data.get('bbox_extent', []),
        }

    out_path = output_dir / f"obj_json_{vlm_name}.json"
    with open(out_path, "w") as f:
        json.dump(obj_json, f, indent=2)
    print(f"[Caption Sweep] Saved {len(obj_json)} objects to {out_path}")

    client.cleanup()


if __name__ == "__main__":
    main()
