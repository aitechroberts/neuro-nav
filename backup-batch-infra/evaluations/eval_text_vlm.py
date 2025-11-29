import argparse
import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict

# Attempt to import pycocoevalcap metrics
try:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.bleu.bleu import Bleu
    HAS_COCO_EVAL = True
except ImportError:
    print("Warning: pycocoevalcap not found. Please install it to run CIDEr/SPICE metrics.")
    print("pip install pycocoevalcap")
    HAS_COCO_EVAL = False

def load_captions_from_json(json_path):
    """
    Loads object captions from the obj_json file.
    Returns a list of valid caption strings.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    captions = []
    # The structure is { "object_1": { "object_caption": "..." }, ... }
    # or list of dicts? Check batch_vlm_mapping.py: save_obj_json saves a Dict {key: dict}
    
    if isinstance(data, dict):
        for key, obj_data in data.items():
            # Try 'object_caption' first, then fallback to 'consolidated_caption'
            cap = obj_data.get("object_caption")
            if not cap:
                cap = obj_data.get("consolidated_caption")
            
            # Sometimes caption might be None or empty
            if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                captions.append(cap)
            elif cap and isinstance(cap, list): 
                 # In case it's a list of captions, take the first or join?
                 # consolidated_caption is usually a string
                 if len(cap) > 0:
                     captions.append(str(cap[0]))
    elif isinstance(data, list):
        for obj_data in data:
             cap = obj_data.get("object_caption")
             if not cap:
                 cap = obj_data.get("consolidated_caption")
                 
             if cap and isinstance(cap, str) and len(cap.strip()) > 0:
                captions.append(cap)

    return captions

def compute_pairwise_matrix(scorer, row_caps, col_caps, metric_name="metric"):
    """
    Computes an NxM matrix of scores where S[i,j] = Score(ref=row_caps[i], cand=col_caps[j]).
    Uses a flattened batch approach to maximize efficiency.
    """
    # Prepare flattened batch
    gts = {}
    res = {}
    
    idx_map = [] # List of (i, j) tuples corresponding to keys
    
    count = 0
    for i, r_cap in enumerate(row_caps):
        for j, c_cap in enumerate(col_caps):
            key = str(count)
            gts[key] = [r_cap] # Ground Truth (List of 1 ref)
            res[key] = [c_cap] # Candidate (List of 1 cand)
            idx_map.append((i, j))
            count += 1
            
    if count == 0:
        return np.zeros((len(row_caps), len(col_caps)))

    # Run Scorer
    # scorer.compute_score returns (avg_score, scores_array)
    # For CIDEr, scores_array is (N, 1)? Or (N,)?
    avg_score, scores = scorer.compute_score(gts, res)
    
    # Remap to Matrix
    matrix = np.zeros((len(row_caps), len(col_caps)))
    
    # Handle different return types from different scorers
    # BLEU returns list of scores (one for each n-gram), we usually take Bleu_4 (index 3) or average?
    # Typically compute_score returns a float and a list of floats/arrays
    
    if isinstance(scores, list) and len(scores) == count:
        # Simple list of scores
        scores_list = scores
    elif isinstance(scores, np.ndarray):
        scores_list = scores.tolist()
    else:
        # Some metrics might return something else, handle if necessary
        print(f"Warning: Unexpected score format for {metric_name}")
        return matrix

    for k, score_val in enumerate(scores_list):
        i, j = idx_map[k]
        
        # HANDLE SPICE DICT RETURN FORMAT
        if isinstance(score_val, dict):
            if 'All' in score_val and 'f' in score_val['All']:
                matrix[i, j] = score_val['All']['f']
            else:
                # Fallback or warning
                matrix[i, j] = 0.0
        else:
            matrix[i, j] = score_val
        
    return matrix

def compute_set_metrics(row_caps, col_caps, scorers):
    """
    Computes Average-Max Recall (Row->Col) and Precision (Col->Row).
    Row = Oracle, Col = Candidate
    """
    results = {}
    
    for name, scorer in scorers.items():
        print(f"  Computing Pairwise Matrix for {name}...")
        # Matrix S[i,j] = Score(Ref=Row[i], Cand=Col[j])
        sim_matrix = compute_pairwise_matrix(scorer, row_caps, col_caps, name)
        
        # Recall: Oracle(Row) -> Config(Col)
        # For each Row i, max over cols j
        max_row_to_col = np.max(sim_matrix, axis=1)
        recall = np.mean(max_row_to_col)
        
        # Precision: Config(Col) -> Oracle(Row)
        # For each Col j, max over rows i
        # Note: We use the SAME matrix. S[i,j] is similarity. 
        # Is similarity symmetric? 
        # CIDEr(Ref=A, Cand=B) is NOT necessarily equal to CIDEr(Ref=B, Cand=A) because of TF-IDF weighting on Reference Set.
        # However, for 1-to-1 comparison, it's often treated as roughly symmetric or we should strictly recompute 
        # with swapped roles if we want strict adherence to "Ref=Target".
        # User asked for: "Mirror it config->oracle".
        # To be rigorous: Matrix_Fwd = Score(Ref=Oracle, Cand=Config)
        # To check "Is Config Spurious?", we treat Config as Query (Cand) and Oracle as Database (Ref)? 
        # Or Config as Database (Ref) and Oracle as Query (Cand)?
        # 
        # Interpretation A: "Recall" = Does Oracle Fact exist in Config? Ref=Oracle, Cand=Config.
        # Interpretation B: "Precision" = Is Config Fact supported by Oracle? Ref=Oracle, Cand=Config. 
        # If we use the SAME matrix (Ref=Oracle, Cand=Config):
        #   Max over rows (for each col) = "Does this Config caption match ANY Oracle caption?"
        #   This seems correct for Precision using the standard definition where Oracle defines the "ground truth language".
        #   We check if the Candidate string is valid under the Reference distribution.
        
        max_col_to_row = np.max(sim_matrix, axis=0)
        precision = np.mean(max_col_to_row)
        
        results[name] = {
            "matrix_mean": float(np.mean(sim_matrix)),
            "recall_avg_max": float(recall),
            "precision_avg_max": float(precision)
        }
        
    return results

def compute_doc_metrics(ref_caps, cand_caps, scorers):
    """
    Computes Whole-Scene Document metrics by sorting and concatenating.
    """
    # Sort
    ref_doc = ". ".join(sorted(ref_caps))
    cand_doc = ". ".join(sorted(cand_caps))
    
    gts = {"scene": [ref_doc]}
    res = {"scene": [cand_doc]}
    
    results = {}
    for name, scorer in scorers.items():
        # print(f"  Computing Doc Score for {name}...")
        avg_score, _ = scorer.compute_score(gts, res)
        
        # If BLEU, it returns a list of 4 scores
        if isinstance(avg_score, list):
            avg_score = avg_score[-1] # Use Bleu_4
            
        results[name] = float(avg_score)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Captions using CIDEr/SPICE")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--oracle_suffix", type=str, required=True)
    parser.add_argument("--candidate_suffix", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_spice", action="store_true", help="Skip SPICE metric (faster)")
    
    args = parser.parse_args()
    
    if not HAS_COCO_EVAL:
        print("Cannot run evaluation without pycocoevalcap.")
        sys.exit(1)

    # Paths
    dataset_root = Path(args.dataset_root)
    oracle_path = dataset_root / args.scene_id / "exps" / args.oracle_suffix / f"obj_json_{args.oracle_suffix}.json"
    cand_path = dataset_root / args.scene_id / "exps" / args.candidate_suffix / f"obj_json_{args.candidate_suffix}.json"
    
    print(f"Loading Oracle: {oracle_path}")
    oracle_caps = load_captions_from_json(oracle_path)
    print(f"  Found {len(oracle_caps)} captions.")
    
    print(f"Loading Candidate: {cand_path}")
    cand_caps = load_captions_from_json(cand_path)
    print(f"  Found {len(cand_caps)} captions.")
    
    if len(oracle_caps) == 0 or len(cand_caps) == 0:
        print("Error: One or both caption sets are empty.")
        sys.exit(1)
        
    # Initialize Scorers
    scorers = {}
    scorers["CIDEr"] = Cider()
    # scorers["Bleu"] = Bleu(4) 
    
    if not args.skip_spice:
        scorers["SPICE"] = Spice()
        
    metrics_results = {}
    
    # 1. Sorted Concat Document Metrics
    print("\n--- Computing Sorted Concat Scene Document Metrics ---")
    doc_scores = compute_doc_metrics(oracle_caps, cand_caps, scorers)
    metrics_results["doc_level"] = doc_scores
    for k, v in doc_scores.items():
        print(f"{k}: {v:.4f}")
        
    # 2. Pairwise Set Matching
    print("\n--- Computing Pairwise Set Matching (Average Max) ---")
    set_scores = compute_set_metrics(oracle_caps, cand_caps, scorers)
    metrics_results["set_level"] = set_scores
    for k, v in set_scores.items():
        print(f"{k} Recall (Oracle->Config): {v['recall_avg_max']:.4f}")
        print(f"{k} Precision (Config->Oracle): {v['precision_avg_max']:.4f}")

    # Save
    output_dir = Path(args.output_dir) if args.output_dir else cand_path.parent / "evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    final_output = {
        "scene_id": args.scene_id,
        "oracle_suffix": args.oracle_suffix,
        "candidate_suffix": args.candidate_suffix,
        "metrics": metrics_results
    }
    
    out_file = output_dir / "vlm_text_eval.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
        
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    main()

