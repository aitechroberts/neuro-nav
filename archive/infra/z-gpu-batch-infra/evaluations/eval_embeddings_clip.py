import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure conceptgraph is in the path if running from a different directory
# This assumes the script is run from the repository root or appropriate python path is set
try:
    from conceptgraph.slam.slam_classes import MapObjectList
except ImportError:
    print("Could not import conceptgraph modules. Please ensure PYTHONPATH includes the 'neuro-nav' directory.")
    sys.exit(1)

def load_objects_from_pkl(pkl_path):
    """
    Loads the objects list from a .pkl.gz file.
    Returns a list of objects (dicts) and their embeddings.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    # Handle different saving formats
    if isinstance(data, dict):
        if 'objects' in data:
            # Standard format from save_pointcloud
            objects_data = data['objects']
            # If it was serialized as a list of dicts, we can use it directly
            # If it was serialized as MapObjectList, it might need conversion, 
            # but save_pointcloud usually calls .to_serializable() which returns a list of dicts.
        else:
            # Fallback or check other keys
            print(f"Warning: 'objects' key not found in {pkl_path}. Keys: {data.keys()}")
            return None, None
    else:
        # Assuming data itself is the list
        objects_data = data

    # Extract embeddings
    embeddings = []
    valid_objects = []
    
    for obj in objects_data:
        # Check for clip_ft
        if 'clip_ft' in obj and obj['clip_ft'] is not None:
            emb = obj['clip_ft']
            # Ensure it's a numpy array or torch tensor
            if isinstance(emb, list):
                emb = np.array(emb)
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            
            # Normalize if not already (CLIP embeddings should be, but good to ensure)
            emb = F.normalize(emb.float().view(1, -1), dim=1)
            embeddings.append(emb)
            valid_objects.append(obj)
    
    if len(embeddings) == 0:
        return [], None
        
    # Stack into (N, D) tensor
    embeddings_tensor = torch.cat(embeddings, dim=0)
    return valid_objects, embeddings_tensor

def compute_chamfer_metrics(ref_emb, cand_emb):
    """
    Computes the Chamfer-style recall and precision metrics between two sets of embeddings.
    
    Args:
        ref_emb (torch.Tensor): Reference embeddings (Oracle) of shape (N, D)
        cand_emb (torch.Tensor): Candidate embeddings (Config) of shape (M, D)
        
    Returns:
        dict: containing 'recall', 'precision', 'f1'
    """
    if ref_emb is None or cand_emb is None:
        return {'recall': 0.0, 'precision': 0.0, 'f1': 0.0}
    
    # Compute Cosine Similarity Matrix (N x M)
    # ref_emb: (N, D), cand_emb: (M, D) -> sim_matrix: (N, M)
    sim_matrix = torch.mm(ref_emb, cand_emb.t())
    
    # 1. Recall: Oracle -> Config
    # For each Oracle object, what is the max similarity in Config?
    # "How much of the Oracle is covered by Config?"
    max_sim_ref_to_cand, _ = torch.max(sim_matrix, dim=1)
    recall = torch.mean(max_sim_ref_to_cand).item()
    
    # 2. Precision (Spuriousness check): Config -> Oracle
    # For each Config object, what is the max similarity in Oracle?
    # "How many Config objects are actually grounded in the Oracle?"
    max_sim_cand_to_ref, _ = torch.max(sim_matrix, dim=0)
    precision = torch.mean(max_sim_cand_to_ref).item()
    
    # 3. F1 Score
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {
        'recall': recall,
        'precision': precision, 
        'f1': f1,
        'sim_matrix_stats': {
            'mean': torch.mean(sim_matrix).item(),
            'max': torch.max(sim_matrix).item(),
            'min': torch.min(sim_matrix).item()
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP Embeddings using Chamfer-style Semantic Recall")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--scene_id", type=str, required=True, help="Scene ID to evaluate")
    parser.add_argument("--oracle_suffix", type=str, required=True, help="Experiment suffix for Oracle (Reference)")
    parser.add_argument("--candidate_suffix", type=str, required=True, help="Experiment suffix for Candidate (Configuration)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results. Defaults to candidate exp dir.")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    scene_id = args.scene_id
    
    # Construct Paths
    # Oracle Path
    oracle_exp_path = dataset_root / scene_id / "exps" / args.oracle_suffix
    oracle_pkl_path = oracle_exp_path / f"pcd_{args.oracle_suffix}.pkl.gz"
    
    # Candidate Path
    candidate_exp_path = dataset_root / scene_id / "exps" / args.candidate_suffix
    candidate_pkl_path = candidate_exp_path / f"pcd_{args.candidate_suffix}.pkl.gz"
    
    print(f"Loading Oracle from: {oracle_pkl_path}")
    _, oracle_emb = load_objects_from_pkl(oracle_pkl_path)
    print(f"Loaded Oracle: {oracle_emb.shape[0] if oracle_emb is not None else 0} objects")

    print(f"Loading Candidate from: {candidate_pkl_path}")
    _, candidate_emb = load_objects_from_pkl(candidate_pkl_path)
    print(f"Loaded Candidate: {candidate_emb.shape[0] if candidate_emb is not None else 0} objects")
    
    # Evaluation
    metrics = compute_chamfer_metrics(oracle_emb, candidate_emb)
    
    print("\n--- Evaluation Results ---")
    print(f"Semantic Recall (Oracle -> Config): {metrics['recall']:.4f}")
    print(f"Semantic Precision (Config -> Oracle): {metrics['precision']:.4f}")
    print(f"Semantic F1 Score: {metrics['f1']:.4f}")
    
    # Save Results
    output_dir = Path(args.output_dir) if args.output_dir else candidate_exp_path / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "scene_id": scene_id,
        "oracle_suffix": args.oracle_suffix,
        "candidate_suffix": args.candidate_suffix,
        "metrics": metrics
    }
    
    output_file = output_dir / "clip_embedding_eval.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()

