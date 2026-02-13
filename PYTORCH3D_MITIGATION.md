# PyTorch3D Mitigation Guide

## Why this exists

This project currently relies on `pytorch3d.ops.box3d_overlap()` for accurate 3D oriented-box overlap in key ConceptGraphs spatial logic.

If `pytorch3d` is unavailable, the mapping pipeline can fail during spatial similarity and object merging, which directly affects object identity consistency and therefore 3D scene understanding quality.

---

## Where the dependency matters in this repo

- `conceptgraph/utils/ious.py`
  - `compute_3d_iou_accurate_batch()`
  - `compute_3d_giou_accurate_batch()`
  - `compute_3d_contain_ratio_accurate_batch()`
  - each imports `pytorch3d.ops as ops` and calls `ops.box3d_overlap(...)`
- `conceptgraph/slam/mapping.py`
  - uses the accurate IoU/GIoU functions depending on `spatial_sim_type`
- `conceptgraph/slam/utils.py`
  - `compute_overlap_matrix_general()` uses `compute_3d_iou_accurate_batch()` as a geometric prefilter
  - `merge_objects()` depends on `compute_overlap_matrix_general()`

Operationally, this means merge quality and reliability are coupled to the PyTorch3D overlap primitive.

---

## What breaks if PyTorch3D is missing

1. Runtime import error when code path reaches `import pytorch3d.ops as ops`.
2. Merge stage may fail or be disabled, depending on your mitigation.
3. Without merge consolidation, map quality typically degrades:
   - duplicate 3D objects for the same entity,
   - fragmented per-object evidence across frames,
   - noisier edge graph relations,
   - weaker downstream semantic scene consistency.

---

## Mitigation strategy (recommended order)

Use a layered approach:

1. **Immediate safe mode** (keep pipeline running now).
2. **Fallback implementation** (remove hard runtime dependency).
3. **Replacement method** (restore higher geometric fidelity).

---

## 1) Immediate safe mode (no code change)

Use Hydra/runtime config to avoid paths requiring accurate OBB IoU:

- set `spatial_sim_type=iou` (or `giou`)
- avoid overlap-based merge path:
  - `merge_overlap_thresh=-1` (early return in `merge_objects()`), or
  - disable merge intervals/final merge (`merge_interval=-1`, `run_merge_final_frame=false`)

Example override pattern:

```bash
python conceptgraph/slam/vlm_run/batch_vlm_mapping_qwen.py \
  spatial_sim_type=iou \
  merge_overlap_thresh=-1 \
  merge_interval=-1 \
  run_merge_final_frame=false
```

This avoids crashes but reduces map consolidation quality.

---

## 2) Repo-level fallback (recommended baseline mitigation)

Implement a runtime fallback in `conceptgraph/utils/ious.py`:

- try `pytorch3d.ops.box3d_overlap`
- if unavailable, fall back to axis-aligned IoU (`compute_iou_batch`) using AABB corners extracted from OBB points

Then use that fallback from `compute_overlap_matrix_general()` and mapping similarity code paths.

### Why this is a good baseline

- Very low engineering risk.
- Keeps all pipeline stages running.
- Maintains deterministic behavior.
- Accuracy is lower for rotated objects but usually acceptable for coarse merge gating.

### Suggested fallback behavior

- Log once: `PyTorch3D unavailable; using AABB IoU fallback`.
- Keep return tensor shapes identical to the current API.
- Keep threshold semantics unchanged (`<1e-6` skip logic still works).

---

## Two most capable and widely used replacement methods

Below are the two strongest practical replacements when teams cannot rely on PyTorch3D.

## Method A: BEV 3D IoU kernels (OpenPCDet/MMDetection3D style)

### Summary

Use bird's-eye-view polygon intersection + height overlap (common in LiDAR/3D detection stacks), typically via CUDA/C++ kernels such as `iou3d_nms` in OpenPCDet/MMCV ecosystems.

### Why it is widely used

- Standard in production-grade 3D detection frameworks.
- Strong performance and mature implementations.
- Good geometric fidelity for yaw-rotated 3D boxes common in robotics/autonomy.

### Pros

- High throughput on GPU.
- Better geometric realism than AABB fallback.
- Well validated in the 3D vision community.

### Cons

- Integration complexity (build toolchain, CUDA compatibility).
- Usually expects parameterized boxes `(x, y, z, dx, dy, dz, yaw)` rather than arbitrary 8-corner OBBs.
- Less straightforward when boxes are not upright/yaw-parameterized.

### Integration notes for ConceptGraphs

1. Convert Open3D box points to canonical box parameters where possible.
2. Add adapter function to produce pairwise IoU tensor shaped `(M, N)`.
3. Use this adapter in:
   - `compute_3d_iou_accurate_batch()` replacement path
   - `compute_overlap_matrix_general()` prefilter
4. Keep `compute_iou_batch` fallback as a final safety net.

---

## Method B: Exact geometric OBB IoU via convex polyhedron intersection (Trimesh/Open3D pipeline)

### Summary

Compute intersection volume between two oriented box polyhedra with geometry tooling (for example `trimesh` plus robust boolean/intersection backend), then IoU from intersection and union volumes.

### Why it is widely used

- Uses general computational geometry primitives.
- Works with arbitrary oriented boxes from corner points.
- Good CPU-side compatibility without requiring CUDA kernels.

### Pros

- High geometric fidelity (closest conceptual replacement to `box3d_overlap`).
- Flexible for non-yaw-only orientations.
- Framework-agnostic (can run in CPU-only environments).

### Cons

- Slower than GPU kernels for large pairwise matrices.
- Requires careful robustness handling for near-degenerate geometry.
- Intersection backends can vary by platform.

### Integration notes for ConceptGraphs

1. Keep current IoU prefilter pattern to reduce pair count before expensive exact intersection.
2. Cache box mesh/polyhedron construction per object to avoid repeated conversions.
3. Use batched/block processing for large scenes.
4. Add robust fallback chain:
   - exact OBB intersection method
   - if fails: AABB IoU
   - if invalid geometry: treat as zero overlap and continue

---

## Recommended decision matrix

- Need fastest rollout with minimal risk: **Fallback to AABB IoU** first.
- Need high throughput and have CUDA/kernel expertise: **Method A (BEV kernel)**.
- Need high geometric fidelity with arbitrary orientations and CPU compatibility: **Method B (convex polyhedron exact IoU)**.

In practice, many teams deploy:

1. immediate AABB fallback for reliability,
2. then add Method A or Method B as an optional high-fidelity backend.

---

## Validation checklist after mitigation

1. Run a representative multi-scene batch.
2. Compare against PyTorch3D baseline on:
   - number of final objects,
   - duplicate rate,
   - merge count per frame,
   - edge count stability over time.
3. Inspect difficult scenes (clutter, rotations, thin structures).
4. Confirm no hard failures when PyTorch3D is absent.
5. Ensure logs clearly state which IoU backend was active.

---

## Practical recommendation for this repository

For this codebase, the most robust path is:

1. add a built-in fallback in `conceptgraph/utils/ious.py`,
2. keep merge enabled,
3. start with AABB fallback for stability,
4. optionally add Method A or Method B as a selectable backend via config (for example `iou_backend: pytorch3d|bev_kernel|exact_poly|aabb`).

This preserves operational reliability while keeping a path to high-quality 3D merge behavior without hard-coupling to PyTorch3D.

