#!/usr/bin/env python3
"""
VLM Text Evaluation Script

Evaluates information quality between scene configurations using text captions.
Implements two complementary approaches:

1. Document-Level: Treats entire scene as sorted concatenated document
2. Pairwise Set Matching: Computes N×M similarity matrix for per-fact recall

Metrics: CIDEr and SPICE (when available)
Symmetric evaluation: Recall (Oracle→Config) and Precision (Config→Oracle)

Usage:
python eval_text_vlm.py --dataset_root /path/to/data --scene_id office0 \
    --oracle_suffix oracle_exp --candidate_suffix vlm_config
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Attempt to import evaluation metrics
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.bleu.bleu import Bleu
    HAS_COCO_EVAL = True
except ImportError:
    HAS_COCO_EVAL = False


def load_scene_captions(json_path):
    """
    Load object captions from obj_json file.

    Args:
        json_path (Path): Path to obj_json_*.json file

    Returns:
        list: List of caption strings
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"Warning: File not found: {json_path}")
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

    captions = []

    # Handle different JSON structures
    if isinstance(data, dict):
        for key, obj_data in data.items():
            if isinstance(obj_data, dict):
                # Try different caption field names
                caption = None
                for field in ['object_caption', 'consolidated_caption', 'caption']:
                    if field in obj_data and obj_data[field]:
                        caption = obj_data[field]
                        break

                if caption and isinstance(caption, str) and len(caption.strip()) > 0:
                    captions.append(caption.strip())
                elif caption and isinstance(caption, list) and len(caption) > 0:
                    # Handle list of captions - take first or join
                    captions.append(str(caption[0]).strip())

    elif isinstance(data, list):
        for obj_data in data:
            if isinstance(obj_data, dict):
                caption = obj_data.get('object_caption') or obj_data.get('consolidated_caption')
                if caption and isinstance(caption, str) and len(caption.strip()) > 0:
                    captions.append(caption.strip())

    return captions


def compute_document_level_metrics(oracle_captions, candidate_captions, scorers):
    """
    Compute document-level metrics by treating scenes as concatenated documents.

    This approach sorts captions alphabetically to make comparison order-invariant,
    then concatenates them into single "scene documents" for holistic evaluation.

    Args:
        oracle_captions (list): List of oracle caption strings
        candidate_captions (list): List of candidate caption strings
        scorers (dict): Dictionary of scorer objects (CIDEr, SPICE, etc.)

    Returns:
        dict: Document-level scores for each metric
    """
    # Sort captions alphabetically for order invariance
    oracle_doc = ". ".join(sorted(oracle_captions))
    candidate_doc = ". ".join(sorted(candidate_captions))

    # Prepare inputs for pycocoevalcap format
    gts = {"scene": [oracle_doc]}
    res = {"scene": [candidate_doc]}

    results = {}
    for name, scorer in scorers.items():
        try:
            avg_score, _ = scorer.compute_score(gts, res)

            # Handle different return types
            if isinstance(avg_score, list):
                # BLEU returns list of scores for different n-grams
                score = float(avg_score[-1])  # Use highest n-gram (BLEU-4)
            else:
                score = float(avg_score)

            results[name.lower()] = score

        except Exception as e:
            print(f"Warning: Failed to compute {name} document score: {e}")
            results[name.lower()] = 0.0

    return results


def compute_pairwise_matrix(scorer, ref_captions, cand_captions, metric_name="metric"):
    """
    Compute N×M similarity matrix between reference and candidate captions.

    Args:
        scorer: pycocoevalcap scorer object
        ref_captions (list): Reference captions (oracle)
        cand_captions (list): Candidate captions (config)
        metric_name (str): Name of metric for logging

    Returns:
        np.ndarray: Similarity matrix of shape (len(ref_captions), len(cand_captions))
    """
    N, M = len(ref_captions), len(cand_captions)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    # Prepare batch evaluation
    gts = {}
    res = {}
    idx_map = []  # Maps batch index back to (ref_idx, cand_idx)

    for i, ref_cap in enumerate(ref_captions):
        for j, cand_cap in enumerate(cand_captions):
            batch_key = f"{i}_{j}"
            gts[batch_key] = [ref_cap]
            res[batch_key] = [cand_cap]
            idx_map.append((i, j))

    # Compute scores
    try:
        avg_score, scores = scorer.compute_score(gts, res)

        # Initialize similarity matrix
        sim_matrix = np.zeros((N, M))

        # Handle different score formats
        if isinstance(scores, (list, np.ndarray)):
            if len(scores) == len(idx_map):
                for k, score_val in enumerate(scores):
                    i, j = idx_map[k]
                    sim_matrix[i, j] = float(score_val)
        else:
            print(f"Warning: Unexpected score format for {metric_name}")
            return np.zeros((N, M))

        return sim_matrix

    except Exception as e:
        print(f"Warning: Failed to compute {metric_name} pairwise matrix: {e}")
        return np.zeros((N, M))


def compute_set_matching_metrics(oracle_captions, candidate_captions, scorers):
    """
    Compute pairwise set matching metrics using average-max similarity.

    This implements the "CIDEr for sets" approach:
    - For each oracle caption, find the best matching candidate caption
    - For each candidate caption, find the best matching oracle caption
    - Average these max scores to get recall and precision

    Args:
        oracle_captions (list): Oracle reference captions
        candidate_captions (list): Candidate configuration captions
        scorers (dict): Dictionary of scorer objects

    Returns:
        dict: Set-level metrics for each scorer
    """
    results = {}

    for name, scorer in scorers.items():
        print(f"  Computing {name} pairwise similarity matrix...")

        # Compute N×M similarity matrix (Oracle × Candidate)
        sim_matrix = compute_pairwise_matrix(scorer, oracle_captions, candidate_captions, name)

        if sim_matrix.size == 0:
            results[name.lower()] = {
                'recall_avg_max': 0.0,
                'precision_avg_max': 0.0,
                'f1': 0.0,
                'matrix_stats': {'mean': 0.0, 'max': 0.0, 'min': 0.0}
            }
            continue

        # Recall: Oracle → Candidate (Coverage)
        # For each oracle caption, what was the best candidate match?
        max_oracle_to_cand = np.max(sim_matrix, axis=1)  # Max over candidates for each oracle
        recall = np.mean(max_oracle_to_cand)

        # Precision: Candidate → Oracle (Grounding)
        # For each candidate caption, what was the best oracle match?
        max_cand_to_oracle = np.max(sim_matrix, axis=0)  # Max over oracles for each candidate
        precision = np.mean(max_cand_to_oracle)

        # F1 Score
        f1 = 0.0
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)

        results[name.lower()] = {
            'recall_avg_max': float(recall),
            'precision_avg_max': float(precision),
            'f1': float(f1),
            'matrix_stats': {
                'mean': float(np.mean(sim_matrix)),
                'max': float(np.max(sim_matrix)),
                'min': float(np.min(sim_matrix)),
                'std': float(np.std(sim_matrix))
            }
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLM Text Captions using CIDEr/SPICE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Approaches:
  1. Document-Level: Sort and concatenate all captions into scene documents
  2. Pairwise Set Matching: N×M similarity matrix with average-max scoring

Metrics:
  - Recall: Oracle→Config coverage (does config contain oracle facts?)
  - Precision: Config→Oracle grounding (are config facts supported?)

Examples:
  python eval_text_vlm.py --dataset_root /data --scene_id office0 \\
      --oracle_suffix oracle_exp --candidate_suffix vlm_config

  # Skip expensive SPICE computation
  python eval_text_vlm.py --dataset_root /data --scene_id office0 \\
      --oracle_suffix oracle_exp --candidate_suffix vlm_config --skip_spice
        """
    )

    parser.add_argument("--dataset_root", type=str, required=True,
                       help="Path to dataset root directory")
    parser.add_argument("--scene_id", type=str, required=True,
                       help="Scene ID to evaluate (e.g., office0)")
    parser.add_argument("--oracle_suffix", type=str, required=True,
                       help="Experiment suffix for oracle/reference scene")
    parser.add_argument("--candidate_suffix", type=str, required=True,
                       help="Experiment suffix for candidate/configuration to evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save results (default: candidate exp dir/evaluations)")
    parser.add_argument("--skip_spice", action="store_true",
                       help="Skip SPICE computation for faster evaluation")
    parser.add_argument("--use_bleu", action="store_true",
                       help="Include BLEU metrics (faster alternative to CIDEr)")

    args = parser.parse_args()

    if not HAS_COCO_EVAL:
        print("ERROR: pycocoevalcap not found. Please install it:")
        print("  pip install pycocoevalcap")
        sys.exit(1)

    dataset_root = Path(args.dataset_root)
    scene_id = args.scene_id

    # Construct file paths
    oracle_json_path = dataset_root / scene_id / "exps" / args.oracle_suffix / f"obj_json_{args.oracle_suffix}.json"
    candidate_json_path = dataset_root / scene_id / "exps" / args.candidate_suffix / f"obj_json_{args.candidate_suffix}.json"

    print(f"Evaluating scene: {scene_id}")
    print(f"Oracle (reference): {args.oracle_suffix}")
    print(f"Candidate (config): {args.candidate_suffix}")
    print()

    # Load captions
    print(f"Loading oracle captions from: {oracle_json_path}")
    oracle_captions = load_scene_captions(oracle_json_path)
    print(f"  Oracle: {len(oracle_captions)} captions")

    print(f"Loading candidate captions from: {candidate_json_path}")
    candidate_captions = load_scene_captions(candidate_json_path)
    print(f"  Candidate: {len(candidate_captions)} captions")

    if len(oracle_captions) == 0 or len(candidate_captions) == 0:
        print("ERROR: One or both caption sets are empty. Cannot evaluate.")
        sys.exit(1)

    # Initialize scorers
    scorers = {}
    scorers['CIDEr'] = Cider()

    if not args.skip_spice:
        scorers['SPICE'] = Spice()
        print("Including SPICE (this may be slow for large caption sets)")
    else:
        print("Skipping SPICE for faster evaluation")

    if args.use_bleu:
        scorers['BLEU'] = Bleu(4)  # BLEU-4

    print(f"Using metrics: {list(scorers.keys())}")
    print()

    # Compute metrics
    results = {}

    # 1. Document-Level Metrics (Fast, holistic)
    print("Computing document-level metrics (sorted concatenation)...")
    doc_metrics = compute_document_level_metrics(oracle_captions, candidate_captions, scorers)
    results['document_level'] = doc_metrics

    print("Document-level results:")
    for metric, score in doc_metrics.items():
        print(".4f")
    print()

    # 2. Pairwise Set Matching Metrics (Rigorous, per-fact)
    print("Computing pairwise set matching metrics (N×M matrix)...")
    set_metrics = compute_set_matching_metrics(oracle_captions, candidate_captions, scorers)
    results['set_matching'] = set_metrics

    print("Set matching results:")
    for metric_name, metric_data in set_metrics.items():
        print(f"  {metric_name.upper()}:")
        print(".4f")
        print(".4f")
        print(".4f")
    print()

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else candidate_json_path.parent / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {
        "evaluation_type": "vlm_text_evaluation",
        "scene_id": scene_id,
        "oracle_suffix": args.oracle_suffix,
        "candidate_suffix": args.candidate_suffix,
        "oracle_json_path": str(oracle_json_path),
        "candidate_json_path": str(candidate_json_path),
        "oracle_captions_count": len(oracle_captions),
        "candidate_captions_count": len(candidate_captions),
        "metrics_used": list(scorers.keys()),
        "results": results
    }

    output_file = output_dir / "vlm_text_eval.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
