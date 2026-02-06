# Scene Graph Information Quality Evaluation

This document outlines the methodology for evaluating the "Information Quality" of generated 3D Scene Graphs. Our goal is to quantify how well a generated scene graph (the **Candidate**) captures the semantic information present in a ground-truth or high-fidelity baseline (the **Oracle**), without relying on strict geometric alignment or object ID matching.

We employ a dual-pipeline strategy to handle the two primary modalities of our scene graphs: **Text** (VLM Captions) and **Embeddings** (CLIP Vectors).

---

## 1. The Challenge: Implicit Alignment
Traditional 3D metrics (like IoU) fail for information quality because:
1.  **Fragmentation:** One Oracle object (e.g., "Kitchen Counter") might be split into three Config objects ("Counter left", "Counter right", "Sink area").
2.  **Semantic Equivalence:** A generated graph might perfectly describe the scene semantics ("Red armchair") even if the bounding box is slightly shifted or the ID is different.
3.  **Permutation Invariance:** The order of objects in the scene list is arbitrary.

To address this, we treat the scene as a **Bag of Semantics**. We evaluate whether the *set of facts* in the Oracle exists in the *set of facts* in the Candidate, regardless of order or fragmentation.

---

## 2. Embedding Evaluation (CLIP Vectors)
**Script:** `evaluations/eval_embeddings_clip.py`

When using CLIP-based backends, objects are represented by high-dimensional vectors. We cannot use text metrics (like CIDEr) on vectors. Instead, we use a vector-space equivalent based on **Chamfer Similarity**.

### Methodology: Semantic Recall (Chamfer-Style)
**Chamfer Distance** is a metric typically used in 3D Point Cloud processing to compare two clouds of points. It measures the average distance from every point in Cloud A to its *nearest neighbor* in Cloud B, plus the reverse.

We adapt this for the **Semantic Vector Space**:

1.  **Forward Recall (Oracle $\to$ Config):**
    *   Question: *"For every object in the Oracle, is there a semantically similar object in the Candidate?"*
    *   Metric: Average Max Cosine Similarity.
    $$ \text{Recall} = \frac{1}{N} \sum_{i=1}^{N} \max_{j} \text{CosineSim}(O_i, C_j) $$

2.  **Backward Precision (Config $\to$ Oracle):**
    *   Question: *"Are the objects in the Candidate grounded in reality (the Oracle), or are they hallucinations?"*
    *   Metric: Average Max Cosine Similarity (reversed).
    $$ \text{Precision} = \frac{1}{M} \sum_{j=1}^{M} \max_{i} \text{CosineSim}(C_j, O_i) $$

### Robustness
*   **Multi-View Robustness:** By using the aggregated object embeddings (averaged across frames), we compare the *stable* semantic representation of the object, filtering out per-frame noise.
*   **Soft Matching:** Unlike a hard threshold ("Is it a match? Yes/No"), Cosine Similarity provides a continuous score. A candidate that is "semantically close" (e.g., "Dark red chair" vs "Maroon chair") yields a high score, while a hallucination yields a low score.

---

## 3. Text Evaluation (VLM Captions)
**Script:** `evaluations/eval_text_vlm.py`

For backends that generate textual descriptions (VLMs), we use standard Captioning metrics (**CIDEr**, **SPICE**). However, standard implementations expect a 1-to-1 mapping. We implement two strategies to handle the "Bag of Captions" structure.

### Method A: Pairwise Set Matching (Fine-Grained)
This is the direct text analogue of the Embedding Chamfer metric.
1.  Compute a full pairwise score matrix ($N \times M$) between all Oracle captions and all Candidate captions using CIDEr or SPICE.
2.  **Avg-Max Recall:** For each Oracle caption, find the Candidate caption that yields the highest score. Average these max scores.
    *   *Why:* This gives credit if the information exists *anywhere* in the generated scene, even if it's attached to a fragmented object.

### Method B: Sorted Scene Document (Holistic)
1.  Sort all Oracle captions alphabetically and concatenate them into a single "Oracle Scene Document".
2.  Sort all Candidate captions alphabetically and concatenate them into a single "Candidate Scene Document".
3.  Run CIDEr/SPICE on this single document pair.

*   **Why:** This creates a global "fingerprint" of the scene's information content. Sorting ensures permutation invariance (the order of detection doesn't matter).
*   **Interpretation:** High scores indicate the Candidate's "Word Cloud" (distribution of n-grams and semantic propositions) closely matches the Oracle.

---

## Summary of Metrics

| Metric | Modality | What it Measures |
| :--- | :--- | :--- |
| **Semantic Recall** | Embeddings | How much of the Oracle's semantic content (in vector space) was retrieved? |
| **Semantic Precision** | Embeddings | How "clean" is the generation? (Low precision = many hallucinations). |
| **Set-Level CIDEr/SPICE** | Text | **Per-Fact Fidelity.** Does every specific caption in the Oracle have a match in the Candidate? |
| **Doc-Level CIDEr/SPICE** | Text | **Global Fidelity.** Do the two scenes describe the same overall "stuff"? |

## How to Run

**1. CLIP Embedding Evaluation**
```bash
python evaluations/eval_embeddings_clip.py \
    --dataset_root /path/to/data \
    --scene_id scene_01 \
    --oracle_suffix oracle_run \
    --candidate_suffix vlm_run_v1
```

**2. VLM Text Evaluation**
*Requires `pycocoevalcap` installed.*
```bash
python evaluations/eval_text_vlm.py \
    --dataset_root /path/to/data \
    --scene_id scene_01 \
    --oracle_suffix oracle_run \
    --candidate_suffix vlm_run_v1
```

