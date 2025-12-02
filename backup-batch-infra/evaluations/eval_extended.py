import argparse
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    HAS_COCO_EVAL = True
except ImportError:
    print("Warning: pycocoevalcap not found.")
    HAS_COCO_EVAL = False


def load_captions_from_json(json_path):
    """Loads object captions from obj_json file."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    captions = []
    obj_id_to_caption = {}  # For triplet construction

    if isinstance(data, dict):
        for key, obj_data in data.items():
            cap = obj_data.get("object_caption") or obj_data.get("consolidated_caption")
            class_name = obj_data.get("class_name", "object")

            if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                captions.append(cap)
                obj_id_to_caption[key] = {"caption": cap, "class_name": class_name}
            elif cap and isinstance(cap, list) and len(cap) > 0:
                captions.append(str(cap[0]))
                obj_id_to_caption[key] = {"caption": str(cap[0]), "class_name": class_name}

    return captions, obj_id_to_caption


def load_edges_from_json(json_path):
    """Loads edges from edge_json file."""
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Warning: Edge file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        data = json.load(f)

    edges = []
    # Adjust based on your actual edge_json structure
    if isinstance(data, dict):
        for edge_key, edge_data in data.items():
            edges.append({
                "obj1_idx": edge_data.get("obj1_idx"),
                "obj2_idx": edge_data.get("obj2_idx"),
                "obj1_class": edge_data.get("obj1_class", ""),
                "obj2_class": edge_data.get("obj2_class", ""),
                "relation": edge_data.get("relation", "related_to"),
                "caption": edge_data.get("caption", ""),  # If edges have captions
            })
    elif isinstance(data, list):
        for edge_data in data:
            edges.append({
                "obj1_idx": edge_data.get("obj1_idx"),
                "obj2_idx": edge_data.get("obj2_idx"),
                "obj1_class": edge_data.get("obj1_class", ""),
                "obj2_class": edge_data.get("obj2_class", ""),
                "relation": edge_data.get("relation", "related_to"),
                "caption": edge_data.get("caption", ""),
            })

    return edges


def build_triplet_captions(obj_id_to_caption, edges):
    """
    Builds triplet strings: "[class1] caption1 | relation | [class2] caption2"
    """
    triplets = []

    for edge in edges:
        obj1_key = str(edge.get("obj1_idx"))
        obj2_key = str(edge.get("obj2_idx"))
        relation = edge.get("relation", "related_to")
        edge_caption = edge.get("caption", "")

        obj1_info = obj_id_to_caption.get(obj1_key, {})
        obj2_info = obj_id_to_caption.get(obj2_key, {})

        obj1_class = obj1_info.get("class_name", edge.get("obj1_class", "object"))
        obj2_class = obj2_info.get("class_name", edge.get("obj2_class", "object"))
        obj1_cap = obj1_info.get("caption", obj1_class)
        obj2_cap = obj2_info.get("caption", obj2_class)

        # Format 1: Full triplet with captions
        triplet_full = f"[{obj1_class}] {obj1_cap} | {relation} | [{obj2_class}] {obj2_cap}"

        # Format 2: Compact triplet (class + relation only)
        triplet_compact = f"{obj1_class} {relation} {obj2_class}"

        # Format 3: Include edge caption if available
        if edge_caption:
            triplet_full += f" ({edge_caption})"

        triplets.append(triplet_full)

    return triplets


def compute_pairwise_matrix(scorer, row_caps, col_caps, metric_name="metric"):
    """Computes NxM pairwise score matrix."""
    gts = {}
    res = {}
    idx_map = []

    count = 0
    for i, r_cap in enumerate(row_caps):
        for j, c_cap in enumerate(col_caps):
            key = str(count)
            gts[key] = [r_cap]
            res[key] = [c_cap]
            idx_map.append((i, j))
            count += 1

    if count == 0:
        return np.zeros((len(row_caps), len(col_caps)))

    avg_score, scores = scorer.compute_score(gts, res)

    matrix = np.zeros((len(row_caps), len(col_caps)))

    if isinstance(scores, list) and len(scores) == count:
        scores_list = scores
    elif isinstance(scores, np.ndarray):
        scores_list = scores.tolist()
    else:
        return matrix

    for k, score_val in enumerate(scores_list):
        i, j = idx_map[k]
        if isinstance(score_val, dict):
            matrix[i, j] = score_val.get('All', {}).get('f', 0.0)
        else:
            matrix[i, j] = score_val

    return matrix


def compute_set_metrics(row_caps, col_caps, scorers, label=""):
    """Computes Average-Max Recall and Precision."""
    results = {}

    if len(row_caps) == 0 or len(col_caps) == 0:
        print(f"  Warning: Empty caption set for {label}")
        return {name: {"recall_avg_max": 0.0, "precision_avg_max": 0.0} for name in scorers}

    for name, scorer in scorers.items():
        print(f"  Computing {label} Pairwise Matrix for {name}...")
        sim_matrix = compute_pairwise_matrix(scorer, row_caps, col_caps, name)

        max_row_to_col = np.max(sim_matrix, axis=1)
        recall = np.mean(max_row_to_col)

        max_col_to_row = np.max(sim_matrix, axis=0)
        precision = np.mean(max_col_to_row)

        results[name] = {
            "matrix_mean": float(np.mean(sim_matrix)),
            "recall_avg_max": float(recall),
            "precision_avg_max": float(precision),
            "count_oracle": len(row_caps),
            "count_candidate": len(col_caps),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Scene Graph using CIDEr/SPICE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--oracle_suffix", type=str, required=True)
    parser.add_argument("--candidate_suffix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_spice", action="store_true")
    parser.add_argument("--skip_edges", action="store_true", help="Skip edge/triplet evaluation")

    args = parser.parse_args()

    if not HAS_COCO_EVAL:
        print("Cannot run evaluation without pycocoevalcap.")
        sys.exit(1)

    dataset_root = Path(args.dataset_root)

    # --- Load Object Captions ---
    oracle_obj_path = dataset_root / args.scene_id / "exps" / args.oracle_suffix / f"obj_json_{args.oracle_suffix}.json"
    cand_obj_path = dataset_root / args.scene_id / "exps" / args.candidate_suffix / f"obj_json_{args.candidate_suffix}.json"

    print(f"Loading Oracle Objects: {oracle_obj_path}")
    oracle_caps, oracle_obj_map = load_captions_from_json(oracle_obj_path)
    print(f"  Found {len(oracle_caps)} object captions.")

    print(f"Loading Candidate Objects: {cand_obj_path}")
    cand_caps, cand_obj_map = load_captions_from_json(cand_obj_path)
    print(f"  Found {len(cand_caps)} object captions.")

    # --- Load Edges ---
    oracle_triplets = []
    cand_triplets = []

    if not args.skip_edges:
        oracle_edge_path = dataset_root / args.scene_id / "exps" / args.oracle_suffix / f"edge_json_{args.oracle_suffix}.json"
        cand_edge_path = dataset_root / args.scene_id / "exps" / args.candidate_suffix / f"edge_json_{args.candidate_suffix}.json"

        print(f"Loading Oracle Edges: {oracle_edge_path}")
        oracle_edges = load_edges_from_json(oracle_edge_path)
        print(f"  Found {len(oracle_edges)} edges.")

        print(f"Loading Candidate Edges: {cand_edge_path}")
        cand_edges = load_edges_from_json(cand_edge_path)
        print(f"  Found {len(cand_edges)} edges.")

        # Build triplets
        oracle_triplets = build_triplet_captions(oracle_obj_map, oracle_edges)
        cand_triplets = build_triplet_captions(cand_obj_map, cand_edges)
        print(f"  Built {len(oracle_triplets)} oracle triplets, {len(cand_triplets)} candidate triplets.")

    # --- Initialize Scorers ---
    scorers = {"CIDEr": Cider()}
    if not args.skip_spice:
        scorers["SPICE"] = Spice()

    metrics_results = {}

    # --- 1. Object Caption Metrics ---
    print("\n=== Object Caption Metrics (Nodes) ===")
    obj_scores = compute_set_metrics(oracle_caps, cand_caps, scorers, label="Object")
    metrics_results["object_captions"] = obj_scores
    for k, v in obj_scores.items():
        print(f"  {k} Recall: {v['recall_avg_max']:.4f}, Precision: {v['precision_avg_max']:.4f}")

    # --- 2. Triplet Metrics (Nodes + Edges) ---
    if not args.skip_edges and len(oracle_triplets) > 0 and len(cand_triplets) > 0:
        print("\n=== Triplet Metrics (Node-Edge-Node) ===")
        triplet_scores = compute_set_metrics(oracle_triplets, cand_triplets, scorers, label="Triplet")
        metrics_results["triplets"] = triplet_scores
        for k, v in triplet_scores.items():
            print(f"  {k} Recall: {v['recall_avg_max']:.4f}, Precision: {v['precision_avg_max']:.4f}")

    # --- 3. Combined (Objects + Triplets) ---
    if not args.skip_edges and len(oracle_triplets) > 0:
        print("\n=== Combined Metrics (Objects + Triplets) ===")
        oracle_combined = oracle_caps + oracle_triplets
        cand_combined = cand_caps + cand_triplets
        combined_scores = compute_set_metrics(oracle_combined, cand_combined, scorers, label="Combined")
        metrics_results["combined"] = combined_scores
        for k, v in combined_scores.items():
            print(f"  {k} Recall: {v['recall_avg_max']:.4f}, Precision: {v['precision_avg_max']:.4f}")

    # --- Save ---
    output_dir = Path(args.output_dir) if args.output_dir else cand_obj_path.parent / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = {
        "scene_id": args.scene_id,
        "oracle_suffix": args.oracle_suffix,
        "candidate_suffix": args.candidate_suffix,
        "metrics": metrics_results,
    }

    out_file = output_dir / "vlm_graph_eval.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
