#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURE THESE PATHS / NAMES FOR YOUR MACHINE
# ---------------------------------------------------------------------------

# Docker image name built from Dockerfile.multi-batch
IMAGE_NAME="conceptgraphs-multi-batch:local"

# Repo root on your laptop (where conceptgraph + Dockerfile live)
REPO_HOST_ROOT="$HOME/cmu-grad/neuro-nav"

# Your local Replica scenes live here:
#   ~/cmu-grad/neuro-data/replica/{room0,room1,office2,office3}
DATASET_HOST_ROOT="$HOME/cmu-grad/neuro-data/replica"

# Checkpoints (YOLO/SAM/etc) on host
CHECKPOINTS_HOST_ROOT="$HOME/cmu-grad/neuro-data/checkpoints"

# Where you want CG outputs to be staged on the host
OUTPUT_HOST_ROOT="$HOME/cmu-grad/neuro-data/cg-output"

# Your finished-data S3 bucket prefix (EDIT THIS!)
# e.g. s3://neuro-nav-data-finished/replica/
S3_OUTPUT_URI="s3://YOUR-DATA-FINISHED-BUCKET/replica/"

# Only these scenes will be processed by the entrypoint
SCENES_TO_RUN="room0 room1 office2 office3"

# AWS region for S3 + Secrets Manager
AWS_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

# ---------------------------------------------------------------------------
# BUILD IMAGE (if needed)
# ---------------------------------------------------------------------------
echo "[run-local] Building Docker image ${IMAGE_NAME} from Dockerfile.multi-batch..."
docker build -f "${REPO_HOST_ROOT}/Dockerfile.multi-batch" -t "${IMAGE_NAME}" "${REPO_HOST_ROOT}"

# Ensure output directory exists on host
mkdir -p "${OUTPUT_HOST_ROOT}"
mkdir -p "${CHECKPOINTS_HOST_ROOT}"

# ---------------------------------------------------------------------------
# RUN CONTAINER (single multi-scene batch)
# ---------------------------------------------------------------------------
echo "[run-local] Starting multi-scene batch on laptop..."

# Build optional checkpoints mount
CKPT_ARGS=()
if [[ -n "${CHECKPOINTS_HOST_ROOT}" ]]; then
  mkdir -p "${CHECKPOINTS_HOST_ROOT}"
  CKPT_ARGS+=(-v "${CHECKPOINTS_HOST_ROOT}:/app/checkpoints")
  CKPT_ARGS+=(-e CKPT_DIR="/app/checkpoints")
else
  # No explicit checkpoints directory; allow HF/ultralytics to manage their own
  CKPT_ARGS+=(-e CKPT_DIR="")
fi


docker run --gpus all --rm \
  # Mount code
  -v "${REPO_HOST_ROOT}":/app/neuro-nav \
  # Mount Replica scenes as /app/data/replica
  -v "${DATASET_HOST_ROOT}":/app/data/replica \
  # Mount checkpoints
  # Mount output directory
  -v "${OUTPUT_HOST_ROOT}":/app/output \
  # Mount AWS credentials
  -v "$HOME/.aws:/root/.aws:ro" \
  \
  "${CKPT_ARGS[@]}" \
  -e REPO_ROOT="/app/neuro-nav" \
  -e DATA_ROOT="/app/data/replica" \
  -e OUTPUT_ROOT="/app/output" \
  \
  # Use the local batch runner (no ReRun, no viz)
  -e BATCH_MAIN="conceptgraph.slam.batch_test_local" \
  \
  # Scene whitelist for entrypoint
  -e SCENES_TO_RUN="${SCENES_TO_RUN}" \
  \
  # We are *not* downloading inputs from S3 in local mode
  -e S3_INPUT_URI="" \
  # But we *do* want to upload outputs to finished bucket
  -e S3_OUTPUT_URI="${S3_OUTPUT_URI}" \
  -e AWS_DEFAULT_REGION="${AWS_REGION}" \
  \
  # Mapping / batch defaults (can tweak)
  -e EXP_SUFFIX="batch_vlm_local" \
  -e DET_EXP_SUFFIX="s_detections_batch" \
  -e DEVICE="cuda" \
  -e STRIDE="10" \
  -e START="0" \
  -e END="-1" \
  -e MAKE_EDGES="false" \
  -e SAVE_JSON="true" \
  -e SAVE_PCD="true" \
  -e SAVE_SEMANTIC_SNAPSHOT="true" \
  -e VIS_RENDER="false" \
  -e USE_WANDB="false" \
  \
  # DO NOT delete input scene directories on laptop
  -e DELETE_INPUT_AFTER_RUN="false" \
  \
  "${IMAGE_NAME}"
