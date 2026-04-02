"""
Evaluation Metrics for Scene Graph Quality.

Provides:
- mIoU: mean Intersection-over-Union (per-class)
- F-mIoU: frequency-weighted mIoU
- mAcc: mean per-class accuracy
- QA wrappers: EM@1, BLEU, ROUGE-L, METEOR, CIDEr

Usage:
    from z_evaluations.metrics import compute_miou, compute_f_miou, compute_macc
"""

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# Semantic Segmentation Metrics
# =============================================================================

def compute_miou(
    predictions: List[int],
    ground_truth: List[int],
    num_classes: Optional[int] = None,
) -> float:
    """
    Mean Intersection-over-Union across all classes present in ground_truth.

    Args:
        predictions: list of predicted class labels (integer).
        ground_truth: list of ground-truth class labels (integer).
        num_classes: if None, inferred from the union of prediction/gt labels.

    Returns:
        mIoU as a float in [0, 1].
    """
    assert len(predictions) == len(ground_truth)
    all_classes = set(ground_truth)
    if num_classes is not None:
        all_classes = set(range(num_classes))

    per_class_iou = []
    for c in sorted(all_classes):
        pred_c = set(i for i, p in enumerate(predictions) if p == c)
        gt_c = set(i for i, g in enumerate(ground_truth) if g == c)
        inter = len(pred_c & gt_c)
        union = len(pred_c | gt_c)
        if union == 0:
            continue
        per_class_iou.append(inter / union)

    return float(np.mean(per_class_iou)) if per_class_iou else 0.0


def compute_f_miou(
    predictions: List[int],
    ground_truth: List[int],
    num_classes: Optional[int] = None,
) -> float:
    """
    Frequency-weighted mIoU.

    Each class IoU is weighted by its frequency in ground_truth.
    """
    assert len(predictions) == len(ground_truth)
    all_classes = set(ground_truth)
    if num_classes is not None:
        all_classes = set(range(num_classes))

    gt_counts = Counter(ground_truth)
    total = len(ground_truth)

    weighted_sum = 0.0
    weight_sum = 0.0

    for c in sorted(all_classes):
        pred_c = set(i for i, p in enumerate(predictions) if p == c)
        gt_c = set(i for i, g in enumerate(ground_truth) if g == c)
        inter = len(pred_c & gt_c)
        union = len(pred_c | gt_c)
        if union == 0:
            continue
        iou = inter / union
        freq = gt_counts[c] / total
        weighted_sum += freq * iou
        weight_sum += freq

    return float(weighted_sum / max(weight_sum, 1e-10))


def compute_macc(
    predictions: List[int],
    ground_truth: List[int],
    num_classes: Optional[int] = None,
) -> float:
    """
    Mean per-class accuracy.

    For each class, accuracy = (correctly predicted) / (total in GT for that class).
    """
    assert len(predictions) == len(ground_truth)
    all_classes = set(ground_truth)
    if num_classes is not None:
        all_classes = set(range(num_classes))

    per_class_acc = []
    for c in sorted(all_classes):
        gt_indices = [i for i, g in enumerate(ground_truth) if g == c]
        if not gt_indices:
            continue
        correct = sum(1 for i in gt_indices if predictions[i] == c)
        per_class_acc.append(correct / len(gt_indices))

    return float(np.mean(per_class_acc)) if per_class_acc else 0.0


# =============================================================================
# QA Metrics
# =============================================================================

def exact_match_at_1(predictions: List[str], ground_truth: List[str]) -> float:
    """EM@1: fraction of predictions that exactly match ground truth."""
    assert len(predictions) == len(ground_truth)
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth)
                  if p.strip().lower() == g.strip().lower())
    return correct / len(predictions)


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-max_n.

    Simple token-level BLEU without brevity penalty for quick evaluation.
    For official scores, use sacrebleu or pycocoevalcap.
    """
    results = {}
    for n in range(1, max_n + 1):
        scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.strip().lower().split()
            ref_tokens = ref.strip().lower().split()

            if len(pred_tokens) < n or len(ref_tokens) < n:
                scores.append(0.0)
                continue

            pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(len(pred_tokens) - n + 1))
            ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))

            clipped = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
            total = sum(pred_ngrams.values())
            scores.append(clipped / max(total, 1))

        results[f"BLEU-{n}"] = float(np.mean(scores)) if scores else 0.0

    return results


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    ROUGE-L F1 score based on longest common subsequence.
    """
    def _lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.strip().lower().split()
        ref_tokens = ref.strip().lower().split()
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs_length(pred_tokens, ref_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return float(np.mean(scores)) if scores else 0.0
