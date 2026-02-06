#!/usr/bin/env python3
"""
CLIP Embeddings Evaluation Script

Evaluates information quality between scene configurations using CLIP embeddings.
Implements Chamfer-style semantic recall as the "CIDEr equivalent" for vector spaces.

Metrics:
- Recall (Oracle→Config): For each Oracle object, how close is the best match in Config?
- Precision (Config→Oracle): For each Config object, is there supporting evidence in Oracle?
- F1: Harmonic mean of recall and precision

Usage:
python eval_embeddings_clip.py --dataset_root /path/to/data --scene_id office0 \
    --oracle_suffix oracle_exp --candidate_suffix test_config
"""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def load_scene_embeddings(pkl_path):
    """
    Load CLIP embeddings from a point cloud PKL file.

    Args:
        pkl_path (Path): Path to pcd_*.pkl.gz file

    Returns:
        tuple: (objects_list, embeddings_tensor) where embeddings_tensor is (N, D)
               Returns (None, None) if loading fails or no valid embeddings found
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        print(f"Warning: File not found: {pkl_path}")
        return None, None

    try:
        with gzip.open(pkl_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None, None

    # Handle different saving formats
    if isinstance(data, dict) and 'objects' in data:
        objects_data = data['objects']
    else:
        print(f"Warning: Unexpected data format in {pkl_path}")
        return None, None

    # Extract CLIP embeddings
    embeddings = []
    valid_objects = []

    for obj in objects_data:
        if 'clip_ft' in obj and obj['clip_ft'] is not None:
            emb = obj['clip_ft']

            # Convert to tensor if needed
            if isinstance(emb, list):
                emb = np.array(emb)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()

            # Ensure proper shape and normalize
            emb = emb.view(-1)  # Flatten to 1D
            if emb.numel() > 0:
                emb = F.normalize(emb, dim=0)  # L2 normalize
                embeddings.append(emb)
                valid_objects.append(obj)

    if len(embeddings) == 0:
        print(f"Warning: No valid CLIP embeddings found in {pkl_path}")
        return valid_objects, None

    # Stack into (N, D) tensor
    embeddings_tensor = torch.stack(embeddings, dim=0)
    return valid_objects, embeddings_tensor


def compute_semantic_recall_metrics(oracle_emb, candidate_emb):
    """
    Compute Chamfer-style semantic recall metrics between two embedding sets.

    This implements the "CIDEr for embeddings" - measuring how much semantic
    information from the oracle scene is preserved in the candidate scene.

    Args:
        oracle_emb (torch.Tensor): Oracle embeddings (N, D)
        candidate_emb (torch.Tensor): Candidate embeddings (M, D)

    Returns:
        dict: Metrics including recall, precision, f1, and similarity statistics
    """
    if oracle_emb is None or candidate_emb is None:
        return {
            'recall': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'oracle_objects': 0,
            'candidate_objects': 0,
            'sim_matrix_stats': {'mean': 0.0, 'max': 0.0, 'min': 0.0}
        }

    N, D = oracle_emb.shape
    M, _ = candidate_emb.shape

    # Compute cosine similarity matrix (N x M)
    # Each row i: similarities between oracle object i and all candidate objects
    # Each col j: similarities between candidate object j and all oracle objects
    sim_matrix = torch.mm(oracle_emb, candidate_emb.t())

    # Recall: Oracle → Candidate (Coverage)
    # For each oracle object, find the best matching candidate object
    # This measures: "How much of the oracle's semantic content is present?"
    max_sim_oracle_to_cand, _ = torch.max(sim_matrix, dim=1)
    recall = torch.mean(max_sim_oracle_to_cand).item()

    # Precision: Candidate → Oracle (Spuriousness)
    # For each candidate object, find the best matching oracle object
    # This measures: "How many candidate objects are actually grounded in reality?"
    max_sim_cand_to_oracle, _ = torch.max(sim_matrix, dim=0)
    precision = torch.mean(max_sim_cand_to_oracle).item()

    # F1 Score
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'recall': recall,  # Oracle→Candidate coverage
        'precision': precision,  # Candidate→Oracle grounding
        'f1': f1,  # Balanced score
        'oracle_objects': N,
        'candidate_objects': M,
        'sim_matrix_stats': {
            'mean': torch.mean(sim_matrix).item(),
            'max': torch.max(sim_matrix).item(),
            'min': torch.min(sim_matrix).item(),
            'std': torch.std(sim_matrix).item()
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP Embeddings using Chamfer-style Semantic Recall",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_embeddings_clip.py --dataset_root /data --scene_id office0 \\
      --oracle_suffix oracle_exp --candidate_suffix vlm_config

Metrics:
  - Recall: For each oracle object, how close is the best candidate match?
  - Precision: For each candidate object, is there oracle support?
  - F1: Harmonic mean of recall and precision
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

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    scene_id = args.scene_id

    # Construct file paths
    oracle_exp_path = dataset_root / scene_id / "exps" / args.oracle_suffix
    oracle_pkl_path = oracle_exp_path / f"pcd_{args.oracle_suffix}.pkl.gz"

    candidate_exp_path = dataset_root / scene_id / "exps" / args.candidate_suffix
    candidate_pkl_path = candidate_exp_path / f"pcd_{args.candidate_suffix}.pkl.gz"

    print(f"Evaluating scene: {scene_id}")
    print(f"Oracle (reference): {args.oracle_suffix}")
    print(f"Candidate (config): {args.candidate_suffix}")
    print()

    # Load oracle embeddings
    print(f"Loading oracle embeddings from: {oracle_pkl_path}")
    oracle_objects, oracle_emb = load_scene_embeddings(oracle_pkl_path)
    if oracle_emb is not None:
        print(f"  Oracle: {oracle_emb.shape[0]} objects with {oracle_emb.shape[1]}D embeddings")
    else:
        print("  Failed to load oracle embeddings")
        sys.exit(1)

    # Load candidate embeddings
    print(f"Loading candidate embeddings from: {candidate_pkl_path}")
    candidate_objects, candidate_emb = load_scene_embeddings(candidate_pkl_path)
    if candidate_emb is not None:
        print(f"  Candidate: {candidate_emb.shape[0]} objects with {candidate_emb.shape[1]}D embeddings")
    else:
        print("  Failed to load candidate embeddings")
        sys.exit(1)

    # Compute metrics
    print("\nComputing semantic recall metrics...")
    metrics = compute_semantic_recall_metrics(oracle_emb, candidate_emb)

    # Display results
    print("\n" + "="*60)
    print("SEMANTIC RECALL EVALUATION RESULTS")
    print("="*60)
    print(".4f")
    print(".4f")
    print(".4f")
    print(f"Objects - Oracle: {metrics['oracle_objects']}, Candidate: {metrics['candidate_objects']}")
    print("\nSimilarity Matrix Statistics:")
    stats = metrics['sim_matrix_stats']
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print("="*60)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else candidate_exp_path / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "evaluation_type": "clip_embeddings_semantic_recall",
        "scene_id": scene_id,
        "oracle_suffix": args.oracle_suffix,
        "candidate_suffix": args.candidate_suffix,
        "oracle_pkl_path": str(oracle_pkl_path),
        "candidate_pkl_path": str(candidate_pkl_path),
        "timestamp": str(torch.randint(0, 1000000, (1,)).item()),  # Simple timestamp
        "metrics": metrics
    }

    output_file = output_dir / "clip_embedding_eval.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
