## Local GPU Smoke Tests

These steps mirror the Batch/FSx layout locally so you can iterate on the VLM
pipeline before deploying to AWS.

### 1. Host directory layout

Create (or reuse) three directories:

| Purpose            | Host path (example)                              | Container mount          |
|--------------------|--------------------------------------------------|--------------------------|
| Replica datasets   | `/home/jroberts/cmu-grad/neuro-nav/data`         | `/mnt/local-data` (ro)   |
| Checkpoints/cache  | `/home/jroberts/cmu-grad/neuro-nav/checkpoints`  | `/mnt/checkpoints`       |
| Output artifacts   | `/home/jroberts/cmu-grad/neuro-nav/local-outputs`| `/mnt/local-output`      |

The mapping code writes to `<scene>/exps/<suffix>` under the output mount, so you
can inspect results on the host at `local-outputs/office0/exps/...`.

### 2. Build the local image

```bash
cd /home/jroberts/cmu-grad/neuro-nav/neuro-nav
docker build \
  -t vlm-batch-gpu:test-local \
  -f gpu-batch-infra/Dockerfile.gpu-test-local .
```

This Dockerfile installs the project, sets `BATCH_MAIN=conceptgraph.slam.batch_test_local`,
and defaults to the `batch_test_local.yaml` Hydra config (10 frames, edges disabled).

### 3. Run a 10-frame smoke test

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

Key points:

- Inputs come from `/mnt/local-data/Replica/...`.
- Outputs land in `/mnt/local-output/office0/exps/batch_vlm_local/...`.
- Checkpoints are accessible at `/mnt/checkpoints`.

### 4. Turning on edges or longer runs

- Set `MAKE_EDGES=true` and export `OPENAI_API_KEY=...` to exercise the full VLM flow.
- Adjust `START/END/STRIDE` env vars for larger slices.
- Override `EXP_SUFFIX`, `DET_EXP_SUFFIX`, or `OUTPUT_ROOT` to keep runs organized.

Once things look good, reuse the same environment overrides in AWS Batch (with a
real FSx mount) and switch back to the standard `Dockerfile.vlm-batch-gpu`.

