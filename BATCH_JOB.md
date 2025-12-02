### Local GPU Batch Job – Quick Reference

This file documents how the `vlm-batch-gpu` container is wired and how to run a small local test batch on Replica `office0`.

---

### 1. Runtime stack

- **Image**: built from `gpu-batch-infra/Dockerfile.vlm-batch-gpu`
- **Entrypoint script**: `gpu-batch-infra/entrypoint_batch_vlm.sh`
- **Python module invoked**:

  ```bash
  python3 -u -m conceptgraph.slam.batch_vlm_mapping
  ```

- **Hydra config root**: `conceptgraph/hydra_configs/`
  - Main config: `batch_vlm_mapping.yaml`
  - Pulls in: `base.yaml`, `base_mapping.yaml`, `replica.yaml`, `sam.yaml`, `classes.yaml`, `logging_level.yaml`, etc.

---

### 2. Env vars → Hydra overrides

`entrypoint_batch_vlm.sh` turns environment variables into Hydra overrides:

- Paths:
  - `REPO_ROOT` → `repo_root` (default `/app/neuro-nav`)
  - `DATA_ROOT` → `data_root` (default `/mnt/data`)
  - `DATASET_ROOT` → `dataset_root` (if set)
  - `SCENE_ID` → `scene_id` (if set)
- Experiment / behavior:
  - `EXP_SUFFIX` → `exp_suffix` (default `batch_vlm`)
  - `DET_EXP_SUFFIX` → `detections_exp_suffix` (default `s_detections`)
  - `DEVICE` → `device` (default `cuda`)
  - `START` → `start` (default `0`)
  - `END` → `end` (default `-1` → all frames)
  - `STRIDE` → `stride` (default `1`)
  - `MAKE_EDGES` → `make_edges` (default `true`)
  - `FORCE_DET` → `force_detection` (default `true`)
  - `SAVE_JSON` → `save_json` (default `true`)
  - `SAVE_PCD` → `save_pcd` (default `true`)
  - `VIS_RENDER` → `vis_render` (default `false`)
  - `PERIODIC_PCD`, `PERIODIC_PCD_INTERVAL` → `periodically_save_pcd`, `periodically_save_pcd_interval`

These override the values in `rerun_realtime_mapping.yaml` / `base_mapping.yaml` / `replica.yaml`.

Inside `batch_vlm_mapping.py`, Hydra gives you `cfg`, and the dataset is created as:

```python
dataset = get_dataset(
    dataconfig=cfg.dataset_config,
    start=cfg.start,
    end=cfg.end,
    stride=cfg.stride,
    basedir=cfg.dataset_root,
    sequence=cfg.scene_id,
    ...
)
```

---

### 3. Local test – first 100 frames of Replica `office0`

Assumptions:

- Repo root on host:

  ```bash
  /home/jroberts/cmu-grad/neuro-nav/neuro-nav
  ```

- Replica data on host:

  ```bash
  /home/jroberts/cmu-grad/neuro-nav/data/Replica/office0/...
  ```

#### 3.1 Build the image

From repo root:

```bash
cd /home/jroberts/cmu-grad/neuro-nav/neuro-nav

docker build \
  -t vlm-batch-gpu:local \
  -f gpu-batch-infra/Dockerfile.vlm-batch-gpu .
```

#### 3.2 Run frames 0–99 of `office0`

```bash
HOST_DATA_ROOT=/home/jroberts/cmu-grad/neuro-nav/data

docker run --rm \
  --gpus all \
  -v "${HOST_DATA_ROOT}":/mnt/data:ro \
  -e REPO_ROOT=/app/neuro-nav \
  -e DATA_ROOT=/mnt/data \
  -e DATASET_ROOT=/mnt/data/Replica \
  -e SCENE_ID=office0 \
  -e START=0 \
  -e END=100 \
  -e STRIDE=1 \
  -e MAKE_EDGES=false \  # optional: skip OpenAI edges for a fast smoke test
  -e SAVE_PCD=true \
  -e SAVE_JSON=true \
  vlm-batch-gpu:local
```

What this yields inside the container:

- `cfg.repo_root = /app/neuro-nav`
- `cfg.data_root = /mnt/data`
- `cfg.dataset_root = /mnt/data/Replica`
- `cfg.dataset_config = /app/neuro-nav/conceptgraph/dataset/dataconfigs/replica/replica.yaml`
- `cfg.scene_id = "office0"`
- `cfg.start = 0`, `cfg.end = 100`, `cfg.stride = 1`

and `batch_vlm_mapping` runs the full mapping pipeline on the first 100 frames only.

You should see logs like:

```text
[entrypoint] Using overrides:
  repo_root=/app/neuro-nav
  data_root=/mnt/data
  dataset_root=/mnt/data/Replica
  scene_id=office0
  exp_suffix=batch_vlm
  detections_exp_suffix=s_detections
  device=cuda
  start=0
  end=100
  stride=1
  ...
```

Once this smoke test succeeds, you can:

- Increase `END` or change `STRIDE` for larger local runs.
- Turn `MAKE_EDGES=true` and set `OPENAI_API_KEY` for full VLM edges.
- Mirror these env vars into your AWS Batch job definition for cloud runs.

---

### 4. Local FSx-style smoke test (10 frames, separate mounts)

To mirror the future FSx layout without needing an actual FSx file system, use the
`gpu-batch-infra/Dockerfile.gpu-test-local` image. It runs `conceptgraph.slam.batch_test_local`
and defaults to `END=10` for quick smoke testing.

Build once:

```bash
cd /home/jroberts/cmu-grad/neuro-nav/neuro-nav
docker build \
  -t vlm-batch-gpu:test-local \
  -f gpu-batch-infra/Dockerfile.gpu-test-local .
```

Run with three bind mounts (inputs, checkpoints, outputs):

```bash
HOST_DATA_ROOT=/home/jroberts/cmu-grad/neuro-nav/data
HOST_CKPT_ROOT=/home/jroberts/cmu-grad/neuro-nav/checkpoints
HOST_OUTPUT_ROOT=/home/jroberts/cmu-grad/neuro-nav/local-outputs

docker run --rm --gpus all \
  -v "${HOST_DATA_ROOT}":/mnt/local-data:ro \
  -v "${HOST_CKPT_ROOT}":/mnt/checkpoints \
  -v "${HOST_OUTPUT_ROOT}":/mnt/local-output \
  -e SCENE_ID=office0 \
  -e START=0 \
  -e END=10 \
  -e MAKE_EDGES=false \  # enable + set OPENAI_API_KEY when you need VLM edges
  vlm-batch-gpu:test-local
```

What this gives you:
- Dataset files are read from `/mnt/local-data/Replica/...` (read-only mount).
- Outputs land under `/mnt/local-output/<scene>/exps/<exp_suffix>` and are immediately visible
  on the host inside `${HOST_OUTPUT_ROOT}/office0/exps/batch_vlm/...`.
- Checkpoints are accessible at `/mnt/checkpoints` so you can mimic the Batch job’s CKPT mount.

Adjust `END`, `STRIDE`, or `MAKE_EDGES` as needed; once the pipeline behaves locally, reuse the
same directory structure when pointing an AWS Batch job at a real FSx file system.


### Push to AWS

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com"
```