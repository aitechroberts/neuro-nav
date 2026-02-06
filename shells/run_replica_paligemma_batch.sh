#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# PaliGemma Batch Processing Script
# Runs fully local inference (no OpenAI API calls)
# ---------------------------------------------------------------------------

export AWS_PROFILE=acct2

# Path to the ConceptGraphs repo
REPO_ROOT="$HOME/cmu-grad/neuro-nav"

# Data paths
DATA_ROOT="$HOME/cmu-grad/neuro-data"
REPLICA_ROOT="${DATA_ROOT}/Replica"

# Scenes to process
SCENES=("room0" "room1" "office2" "office3")

# 

# Experiment label - differentiates from GPT-4 runs
EXP_SUFFIX="batch_paligemma"
DET_EXP_SUFFIX="s_detections_paligemma"

# PaliGemma mapping script
PY_SCRIPT="conceptgraph/slam/batch_vlm_mapping_paligemma.py"

# Checkpoints directory
CKPT_DIR="$HOME/cmu-grad/neuro-data/checkpoints"

# Output root
OUTPUT_ROOT="${REPLICA_ROOT}"

# S3 location for results
S3_OUTPUT_URI="s3://data-finished-585780419748-us-east-1/paligemma448/"

# Mapping controls
DEVICE="cuda"
START=0
END=-1
STRIDE=10
MAKE_EDGES="true"
FORCE_DET="true"
SAVE_JSON="true"
SAVE_PCD="true"
SAVE_SEMANTIC_SNAPSHOT="true"
VIS_RENDER="false"
USE_WANDB="true"  # Disabled by default for local runs

# PaliGemma model (can be overridden)
# Options: "google/paligemma-3b-mix-224" or "google/paligemma2-3b-mix-448"
PALIGEMMA_MODEL="google/paligemma2-3b-mix-448"

# Python interpreter
PYTHON_BIN="python3"

# Cache cleanup
CLEAN_HF_CACHE="true"   # Keep HF cache since PaliGemma needs it
CLEAN_TORCH_CACHE="false"
CLEAN_ULTRA_CACHE="false"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if ! command -v aws >/dev/null 2>&1; then
  echo "[run-paligemma] ERROR: aws CLI not found in PATH."
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[run-paligemma] ERROR: ${PYTHON_BIN} not found in PATH."
  exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[run-paligemma] ERROR: REPO_ROOT does not exist: ${REPO_ROOT}"
  exit 1
fi

if [[ ! -d "${REPLICA_ROOT}" ]]; then
  echo "[run-paligemma] ERROR: REPLICA_ROOT does not exist: ${REPLICA_ROOT}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${REPLICA_ROOT}/${SCENE}"
  if [[ ! -d "${SCENE_DIR}" ]]; then
    echo "[run-paligemma][skip] Scene directory not found: ${SCENE_DIR}"
    continue
  fi

  echo "=================================================================="
  echo "[run-paligemma] Processing scene: ${SCENE} with PaliGemma"
  echo "=================================================================="

  # Hydra overrides
  OVERRIDES=()
  OVERRIDES+=("repo_root=${REPO_ROOT}")
  OVERRIDES+=("data_root=${DATA_ROOT}")
  OVERRIDES+=("dataset_root=${REPLICA_ROOT}")
  OVERRIDES+=("scene_id=${SCENE}")

  OVERRIDES+=("exp_suffix=${EXP_SUFFIX}")
  OVERRIDES+=("detections_exp_suffix=${DET_EXP_SUFFIX}")
  OVERRIDES+=("device=${DEVICE}")
  OVERRIDES+=("start=${START}")
  OVERRIDES+=("end=${END}")
  OVERRIDES+=("stride=${STRIDE}")
  OVERRIDES+=("make_edges=${MAKE_EDGES}")
  OVERRIDES+=("force_detection=${FORCE_DET}")
  OVERRIDES+=("save_json=${SAVE_JSON}")
  OVERRIDES+=("save_pcd=${SAVE_PCD}")
  OVERRIDES+=("save_semantic_snapshot=${SAVE_SEMANTIC_SNAPSHOT}")
  OVERRIDES+=("vis_render=${VIS_RENDER}")
  OVERRIDES+=("use_wandb=${USE_WANDB}")
  OVERRIDES+=("paligemma_model=${PALIGEMMA_MODEL}")

  # Environment variables
  export REPO_ROOT
  export DATA_ROOT
  export CKPT_DIR
  export OUTPUT_ROOT

  echo "[run-paligemma] Launching ${PYTHON_BIN} ${PY_SCRIPT} for scene ${SCENE}..."
  set +e
  "${PYTHON_BIN}" "${PY_SCRIPT}" "${OVERRIDES[@]}"
  RET=$?
  set -e

  if [[ ${RET} -ne 0 ]]; then
    echo "[run-paligemma][error] Mapping failed for ${SCENE} (exit code ${RET}). Skipping upload."
    continue
  fi

  echo "[run-paligemma] Mapping succeeded for ${SCENE}. Uploading results to S3..."

  SCENE_OUTPUT_DIR="${REPLICA_ROOT}/${SCENE}/exps/${EXP_SUFFIX}"

  if [[ ! -d "${SCENE_OUTPUT_DIR}" ]]; then
    echo "[run-paligemma][warn] Output directory not found for scene ${SCENE}: ${SCENE_OUTPUT_DIR}"
    continue
  fi

  S3_SCENE_URI="${S3_OUTPUT_URI%/}/${SCENE}"

  echo "[run-paligemma] aws s3 sync \"${SCENE_OUTPUT_DIR}\" \"${S3_SCENE_URI}\""
  aws s3 sync "${SCENE_OUTPUT_DIR}" "${S3_SCENE_URI}"

  echo "[run-paligemma] Upload complete. Deleting local outputs for scene ${SCENE}..."
  rm -rf "${SCENE_OUTPUT_DIR}"

  echo "[run-paligemma] Done with scene ${SCENE}."
  echo
done

echo "[run-paligemma] All requested scenes processed with PaliGemma."

# ---------------------------------------------------------------------------
# Optional cache cleanup
# ---------------------------------------------------------------------------

if [[ "${CLEAN_HF_CACHE}" == "true" ]]; then
  echo "[run-paligemma] Cleaning HuggingFace caches..."
  rm -rf "${HOME}/.cache/huggingface" 2>/dev/null || true
  echo "[run-paligemma] HuggingFace cache cleanup done."
fi

if [[ "${CLEAN_TORCH_CACHE}" == "true" ]]; then
  echo "[run-paligemma] Cleaning Torch cache..."
  rm -rf "${HOME}/.cache/torch" 2>/dev/null || true
  echo "[run-paligemma] Torch cache cleanup done."
fi

if [[ "${CLEAN_ULTRA_CACHE}" == "true" ]]; then
  echo "[run-paligemma] Cleaning Ultralytics cache..."
  rm -rf "${HOME}/.cache/ultralytics" 2>/dev/null || true
  echo "[run-paligemma] Ultralytics cache cleanup done."
fi
