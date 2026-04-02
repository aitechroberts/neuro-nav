# Fix: Remove unbounded merge_list memory leak in MappingTracker

## Problem

`MappingTracker.track_merge()` appends `(obj1, obj2)` references to
`self.merge_list` on every merge event. These are live references to full
object dicts containing Open3D point clouds, SAM masks, feature vectors, and
all detection metadata.

Because `obj2` (the absorbed detection) would otherwise be garbage-collected
after the merge, `merge_list` keeps it alive indefinitely. `obj1` references
continue mutating in place as subsequent merges modify the same map object.

**The list is write-only.** No code path ever reads, iterates, serializes, or
logs `merge_list`. It serves no function.

### Impact

On ScanNet scene0046_00 (stride 10, ~550 frames):

- Each merge stores ~2–5 MB of object data (PCDs, masks, numpy arrays).
- By frame ~180 (~1/3 of the scene), hundreds of merges have accumulated,
  consuming **1.5–3 GB** of CPU RAM that is never freed.
- On a 30 GB workstation this causes swap thrashing. On an 8 GB Jetson Orin
  it would OOM the process.

### Root cause

```python
# conceptgraph/utils/logging_metrics.py, line 64-66
def track_merge(self, obj1, obj2):
    self.total_merges += 1
    self.merge_list.append((obj1, obj2))  # <-- unbounded leak
```

### Additional issue: design flaw

Even if `merge_list` were consumed, it would not provide useful merge
provenance. `obj1` is a mutable dict modified in-place by every subsequent
merge, so all `obj1` entries in the list point to the **final** state of the
object, not the state at the time of the recorded merge.

## Fix

Remove the `merge_list.append()` call. Retain `self.total_merges += 1` (the
scalar counter) which is used elsewhere.

### Files changed

| File | Change |
|---|---|
| `conceptgraph/utils/logging_metrics.py` | Remove `self.merge_list.append((obj1, obj2))` from `track_merge()` |

### What is NOT affected

- `total_merges` counter — preserved, still incremented.
- `merge_obj2_into_obj1()` — no change to merge logic, PCD reconstruction,
  feature averaging, or any extend/add attributes.
- `to_serializable()` / pkl output — `merge_list` was never serialized.
- All evaluation scripts in `z_evaluations/` — none reference `merge_list`.
- Point cloud reconstruction, bounding boxes, CLIP/VLM features, captions,
  edge mapping — completely untouched.

## Companion fix (same PR)

`merge_obj2_into_obj1()` in `conceptgraph/slam/utils.py` now trims
`obj['mask']` to the last 2 entries after each merge, and
`to_serializable()` in `conceptgraph/slam/slam_classes.py` explicitly strips
masks before pickling. Historical masks were never used downstream; capping
them prevents a secondary source of unbounded memory growth (~1.25 MB per
mask at ScanNet resolution).

## Testing

Run any ScanNet or Replica scene end-to-end and confirm:

1. Output pkl/json/pcd files are byte-identical (masks were already absent
   from serialized output).
2. RSS of the mapping process stays flat after initial model loading rather
   than growing linearly with frame count.
3. `tracker.total_merges` still reports the correct count in logs.
