"""
Stage 5b: HPSG (Hierarchical Plane-Enhanced Scene Graph) Construction.

Loads the object PKL from Stage 2, detects structural planes via RANSAC on the
combined point cloud, clusters them via DBSCAN, labels planes (floor/ceiling/wall),
anchors objects to their nearest plane, and infers a scene type via LLM.

Produces a 3-level hierarchy: scene -> planes -> objects.

Usage:
    python hpsg_construction.py \
        --pkl_path /path/to/pcd_*.pkl.gz \
        --output_dir /path/to/hpsg/ \
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# =============================================================================
# Plane Detection
# =============================================================================

def ransac_plane_fit(points: np.ndarray, n_iterations: int = 1000,
                     distance_threshold: float = 0.02, min_inliers: int = 100
                     ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """
    RANSAC plane fitting on a (N, 3) point cloud.

    Returns (normal, offset, inlier_mask) or None if no plane found.
    """
    if len(points) < min_inliers:
        return None

    best_inliers = None
    best_normal = None
    best_offset = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(n_iterations):
        idx = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal = normal / norm_len
        offset = -np.dot(normal, p0)

        distances = np.abs(np.dot(points, normal) + offset)
        inlier_mask = distances < distance_threshold
        count = inlier_mask.sum()

        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
            best_inliers = inlier_mask

    if best_count < min_inliers:
        return None
    return best_normal, best_offset, best_inliers


def detect_planes(all_points: np.ndarray, max_planes: int = 10,
                  distance_threshold: float = 0.02,
                  min_inlier_ratio: float = 0.01) -> List[Dict]:
    """
    Iteratively detect planes via RANSAC.

    Returns list of plane dicts with 'normal', 'offset', 'inlier_indices', 'n_points'.
    """
    planes = []
    remaining_mask = np.ones(len(all_points), dtype=bool)
    min_inliers = max(int(len(all_points) * min_inlier_ratio), 50)

    for _ in range(max_planes):
        pts = all_points[remaining_mask]
        result = ransac_plane_fit(pts, distance_threshold=distance_threshold,
                                  min_inliers=min_inliers)
        if result is None:
            break

        normal, offset, local_inlier_mask = result
        global_indices = np.where(remaining_mask)[0]
        inlier_indices = global_indices[local_inlier_mask]

        planes.append({
            'normal': normal.tolist(),
            'offset': float(offset),
            'inlier_indices': inlier_indices.tolist(),
            'n_points': int(local_inlier_mask.sum()),
        })

        remaining_mask[inlier_indices] = False

    return planes


# =============================================================================
# Plane Clustering & Labeling
# =============================================================================

def cluster_planes(planes: List[Dict], normal_threshold: float = 0.1,
                   offset_threshold: float = 0.5) -> List[List[int]]:
    """
    Simple greedy clustering of planes in (normal, offset) parameter space.
    Merges planes with similar normals and offsets.
    """
    n = len(planes)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = [i]
        visited[i] = True
        ni = np.array(planes[i]['normal'])
        oi = planes[i]['offset']

        for j in range(i + 1, n):
            if visited[j]:
                continue
            nj = np.array(planes[j]['normal'])
            oj = planes[j]['offset']

            normal_sim = abs(np.dot(ni, nj))
            offset_diff = abs(oi - oj)

            if normal_sim > (1.0 - normal_threshold) and offset_diff < offset_threshold:
                cluster.append(j)
                visited[j] = True

        clusters.append(cluster)

    return clusters


def label_plane(normal: np.ndarray, gravity: np.ndarray = np.array([0, -1, 0])) -> str:
    """Classify a plane as floor, ceiling, or wall based on normal alignment."""
    normal = np.array(normal)
    alignment = abs(np.dot(normal, gravity))

    if alignment > 0.8:
        if np.dot(normal, gravity) > 0:
            return "floor"
        else:
            return "ceiling"
    return "wall"


# =============================================================================
# Object Anchoring
# =============================================================================

def anchor_objects_to_planes(
    objects: List[Dict], plane_groups: List[Dict],
) -> Dict[int, int]:
    """
    For each object, find the nearest structural plane by centroid distance.

    Returns {obj_idx: plane_group_idx}.
    """
    anchoring = {}
    if not plane_groups:
        return anchoring

    plane_normals = [np.array(pg['normal']) for pg in plane_groups]
    plane_offsets = [pg['offset'] for pg in plane_groups]

    for obj_idx, obj in enumerate(objects):
        if 'bbox_np' in obj:
            centroid = np.asarray(obj['bbox_np']).mean(axis=0)
        elif 'pcd_np' in obj:
            centroid = np.asarray(obj['pcd_np']).mean(axis=0)
        else:
            anchoring[obj_idx] = 0
            continue

        min_dist = float('inf')
        best_plane = 0
        for pg_idx, (normal, offset) in enumerate(zip(plane_normals, plane_offsets)):
            dist = abs(np.dot(centroid, normal) + offset)
            if dist < min_dist:
                min_dist = dist
                best_plane = pg_idx

        anchoring[obj_idx] = best_plane

    return anchoring


# =============================================================================
# Scene Type Inference
# =============================================================================

def infer_scene_type(object_tags: List[str], vlm_client=None) -> str:
    """
    Infer scene type from object tags.
    Falls back to heuristic if no VLM is available.
    """
    tags_lower = [t.lower() for t in object_tags]
    tag_set = set(tags_lower)

    if vlm_client is not None:
        try:
            tag_list = ", ".join(object_tags[:50])
            prompt = (
                f"Given these objects detected in a scene: [{tag_list}]\n"
                f"What type of room or space is this? Reply with just the room type "
                f"(e.g. 'bedroom', 'kitchen', 'living room', 'office', 'bathroom')."
            )
            messages = [
                {"role": "user", "content": prompt},
            ]
            resp = vlm_client.client.chat.completions.create(
                model=vlm_client.model_name,
                messages=messages,
                max_tokens=20,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip().lower()
        except Exception:
            pass

    kitchen_words = {'oven', 'stove', 'refrigerator', 'microwave', 'sink', 'dishwasher', 'kitchen'}
    bedroom_words = {'bed', 'pillow', 'mattress', 'nightstand', 'dresser', 'bedroom'}
    bathroom_words = {'toilet', 'bathtub', 'shower', 'towel', 'bathroom'}
    office_words = {'desk', 'computer', 'monitor', 'keyboard', 'office'}
    living_words = {'sofa', 'couch', 'tv', 'television', 'coffee table', 'living'}

    for word_set, label in [
        (kitchen_words, "kitchen"), (bedroom_words, "bedroom"),
        (bathroom_words, "bathroom"), (office_words, "office"),
        (living_words, "living room"),
    ]:
        if tag_set & word_set:
            return label

    return "room"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 5b: HPSG Construction")
    parser.add_argument("--pkl_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_planes", type=int, default=10)
    parser.add_argument("--distance_threshold", type=float, default=0.02)
    parser.add_argument("--vlm_api_url", type=str, default=None)
    parser.add_argument("--vlm_model_name", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[HPSG] Loading objects...")
    with gzip.open(args.pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        objects = data.get('objects', data.get('obj_list', []))
    else:
        objects = data
    print(f"[HPSG] {len(objects)} objects loaded")

    all_points_list = []
    for obj in objects:
        if 'pcd_np' in obj:
            all_points_list.append(np.asarray(obj['pcd_np']))
    if all_points_list:
        all_points = np.concatenate(all_points_list, axis=0)
    else:
        print("[HPSG] No point cloud data found. Exiting.")
        return

    print(f"[HPSG] Total points: {len(all_points)}")
    print("[HPSG] Detecting planes via RANSAC...")
    planes = detect_planes(all_points, max_planes=args.max_planes,
                           distance_threshold=args.distance_threshold)
    print(f"[HPSG] {len(planes)} raw planes detected")

    print("[HPSG] Clustering planes...")
    clusters = cluster_planes(planes)
    print(f"[HPSG] {len(clusters)} plane groups after clustering")

    plane_groups = []
    for cluster_idx, cluster in enumerate(clusters):
        normals = [np.array(planes[i]['normal']) for i in cluster]
        avg_normal = np.mean(normals, axis=0)
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
        avg_offset = np.mean([planes[i]['offset'] for i in cluster])
        total_points = sum(planes[i]['n_points'] for i in cluster)

        plane_label = label_plane(avg_normal)
        plane_groups.append({
            'plane_id': cluster_idx,
            'normal': avg_normal.tolist(),
            'offset': float(avg_offset),
            'label': plane_label,
            'n_points': total_points,
            'n_raw_planes': len(cluster),
        })

    print("[HPSG] Anchoring objects to planes...")
    anchoring = anchor_objects_to_planes(objects, plane_groups)

    object_tags = [obj.get('class_name', 'unknown') for obj in objects]
    vlm_client = None
    if args.vlm_api_url and args.vlm_model_name:
        try:
            from conceptgraph.utils.vlms.vlm_api import VLMAPIClient, wait_for_server
            wait_for_server(args.vlm_api_url)
            vlm_client = VLMAPIClient(
                base_url=args.vlm_api_url,
                model_name=args.vlm_model_name,
            )
        except Exception:
            pass

    scene_type = infer_scene_type(object_tags, vlm_client)
    print(f"[HPSG] Inferred scene type: {scene_type}")

    hpsg = {
        'scene_type': scene_type,
        'planes': plane_groups,
        'objects': [],
    }

    for obj_idx, obj in enumerate(objects):
        plane_id = anchoring.get(obj_idx)
        obj_entry = {
            'obj_idx': obj_idx,
            'class_name': obj.get('class_name', 'unknown'),
            'parent_plane_id': plane_id,
            'parent_plane_label': plane_groups[plane_id]['label'] if plane_id is not None and plane_id < len(plane_groups) else None,
        }
        if 'bbox_np' in obj:
            pts = np.asarray(obj['bbox_np'])
            obj_entry['bbox_center'] = pts.mean(axis=0).tolist()
            obj_entry['bbox_extent'] = (pts.max(axis=0) - pts.min(axis=0)).tolist()
        hpsg['objects'].append(obj_entry)

    out_path = output_dir / "hpsg.json"
    with open(out_path, "w") as f:
        json.dump(hpsg, f, indent=2)
    print(f"[HPSG] Saved hierarchy to {out_path}")

    if vlm_client is not None:
        vlm_client.cleanup()


if __name__ == "__main__":
    main()
