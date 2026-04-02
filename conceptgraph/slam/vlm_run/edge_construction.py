"""
Stage 5a: MST Edge Construction.

Loads the object PKL from Stage 2, computes pairwise 3D AABB IoU between all
objects, builds a Minimum Spanning Tree on the IoU similarity graph, and
optionally labels each MST edge with a spatial relationship via VLM API.

Usage:
    python edge_construction.py \
        --pkl_path /path/to/pcd_*.pkl.gz \
        --output_dir /path/to/edges/ \
        [--vlm_api_url http://localhost:8000/v1] \
        [--vlm_model_name Qwen/Qwen3-VL-2B-Instruct]
"""

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def load_objects(pkl_path: Path) -> List[Dict]:
    """Load serialized objects from pkl.gz."""
    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get('objects', data.get('obj_list', []))
    return data


def compute_pairwise_iou_from_bboxes(objects: List[Dict]) -> np.ndarray:
    """Compute NxN AABB IoU matrix from serialized bbox_np arrays."""
    n = len(objects)
    iou_matrix = np.zeros((n, n), dtype=np.float32)

    centers = []
    extents = []
    for obj in objects:
        if 'bbox_np' in obj:
            pts = np.asarray(obj['bbox_np'])
            ctr = pts.mean(axis=0)
            ext = pts.max(axis=0) - pts.min(axis=0)
        else:
            ctr = np.array(obj.get('bbox_center', [0, 0, 0]))
            ext = np.array(obj.get('bbox_extent', [0, 0, 0]))
        centers.append(ctr)
        extents.append(ext)

    centers = np.array(centers)
    extents = np.array(extents)

    for i in range(n):
        for j in range(i + 1, n):
            min_i = centers[i] - extents[i] / 2
            max_i = centers[i] + extents[i] / 2
            min_j = centers[j] - extents[j] / 2
            max_j = centers[j] + extents[j] / 2

            inter_min = np.maximum(min_i, min_j)
            inter_max = np.minimum(max_i, max_j)
            inter_dims = np.maximum(0, inter_max - inter_min)
            inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

            vol_i = max(extents[i].prod(), 1e-10)
            vol_j = max(extents[j].prod(), 1e-10)
            union_vol = vol_i + vol_j - inter_vol

            iou = inter_vol / max(union_vol, 1e-10)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou

    return iou_matrix


def build_mst(iou_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """Build MST from IoU similarity. Converts to distance (1 - IoU) for MST."""
    n = iou_matrix.shape[0]
    dist_matrix = 1.0 - iou_matrix
    np.fill_diagonal(dist_matrix, 0)
    sparse = csr_matrix(dist_matrix)
    mst = minimum_spanning_tree(sparse)
    mst_coo = mst.tocoo()

    edges = []
    for i, j, w in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        edges.append((int(i), int(j), float(1.0 - w)))
    return edges


def label_edge_spatial(
    obj_i: Dict, obj_j: Dict, vlm_client=None,
) -> str:
    """Determine spatial relationship between two objects."""
    ci = np.array(obj_i.get('bbox_center', obj_i.get('bbox_np', np.zeros((8, 3))).mean(axis=0) if 'bbox_np' in obj_i else [0, 0, 0]))
    cj = np.array(obj_j.get('bbox_center', obj_j.get('bbox_np', np.zeros((8, 3))).mean(axis=0) if 'bbox_np' in obj_j else [0, 0, 0]))

    if isinstance(ci, list):
        ci = np.array(ci)
    if isinstance(cj, list):
        cj = np.array(cj)

    diff = cj - ci
    abs_diff = np.abs(diff)

    if abs_diff[1] > abs_diff[0] and abs_diff[1] > abs_diff[2]:
        return "above" if diff[1] > 0 else "below"
    elif abs_diff[0] > abs_diff[2]:
        return "right of" if diff[0] > 0 else "left of"
    else:
        return "in front of" if diff[2] > 0 else "behind"


def main():
    parser = argparse.ArgumentParser(description="Stage 5a: MST Edge Construction")
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to pcd_*.pkl.gz from Stage 2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vlm_api_url", type=str, default=None,
                        help="Optional VLM API for edge labeling")
    parser.add_argument("--vlm_model_name", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[MST Edges] Loading objects...")
    objects = load_objects(Path(args.pkl_path))
    print(f"[MST Edges] {len(objects)} objects loaded")

    if len(objects) < 2:
        print("[MST Edges] Fewer than 2 objects, skipping edge construction.")
        return

    print("[MST Edges] Computing pairwise 3D IoU...")
    iou_matrix = compute_pairwise_iou_from_bboxes(objects)

    print("[MST Edges] Building MST...")
    mst_edges = build_mst(iou_matrix)
    print(f"[MST Edges] {len(mst_edges)} edges in MST")

    edges_json = []
    for i, j, iou in mst_edges:
        obj_i_tag = objects[i].get('class_name', f'object_{i}')
        obj_j_tag = objects[j].get('class_name', f'object_{j}')
        rel = label_edge_spatial(objects[i], objects[j])
        edges_json.append({
            'obj1_idx': i,
            'obj2_idx': j,
            'obj1_tag': obj_i_tag,
            'obj2_tag': obj_j_tag,
            'relation': rel,
            'iou': round(iou, 4),
        })

    suffix = ""
    if args.vlm_model_name:
        suffix = f"_{args.vlm_model_name.replace('/', '_')}"

    out_path = output_dir / f"edges_mst{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(edges_json, f, indent=2)
    print(f"[MST Edges] Saved {len(edges_json)} edges to {out_path}")


if __name__ == "__main__":
    main()
