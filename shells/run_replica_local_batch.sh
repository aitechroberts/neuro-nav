#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration - EDIT these paths / bucket name for your laptop setup
# ---------------------------------------------------------------------------

export AWS_PROFILE=acct2

# Path to the ConceptGraphs repo (where `conceptgraph/` package lives)
REPO_ROOT="$HOME/cmu-grad/neuro-nav"

# Parent folder that contains `replica/` (and maybe other datasets)
DATA_ROOT="$HOME/cmu-grad/neuro-data"

# Replica scenes are assumed to be here: ${DATA_ROOT}/replica/{room0,room1,...}
REPLICA_ROOT="${DATA_ROOT}/Replica"

# Scenes to process in order
SCENES=("room0" "room1" "office2" "office3")

# Experiment label (used in output folder and filenames)
EXP_SUFFIX="batch_vlm_local"
DET_EXP_SUFFIX="s_detections_batch"

# Choose which mapping script to run.
# Default: original batch VLM mapping script (no ReRun UI)
PY_SCRIPT="conceptgraph/slam/batch_vlm_mapping.py"
# If you prefer the `batch_test_local` variant (with OUTPUT_ROOT / CKPT_DIR hooks),
# change to:
# PY_SCRIPT="conceptgraph/slam/batch_test_local.py"

# Optional: local checkpoints directory if your scripts use it (e.g. batch_test_local.py)
CKPT_DIR="$HOME/cmu-grad/neuro-data/checkpoints"

# Where CG outputs will be logically considered to live for upload/cleanup.
# For batch_vlm_mapping, outputs live under: ${REPLICA_ROOT}/${SCENE}/exps/${EXP_SUFFIX}
OUTPUT_ROOT="${REPLICA_ROOT}"

# S3 location for "data-finished" results (EDIT THIS!)
# Example: s3://neuro-nav-data-finished/replica/
S3_OUTPUT_URI="s3://data-finished-585780419748-us-east-1/oracle/"

# Mapping controls
DEVICE="cuda"
START=0
END=-1
STRIDE=10
MAKE_EDGES="true"                # Set to "true" to enable VLM edges / OpenAI calls
FORCE_DET="true"
SAVE_JSON="true"
SAVE_PCD="true"
SAVE_SEMANTIC_SNAPSHOT="true"
VIS_RENDER="false"
USE_WANDB="true"

# Python interpreter (use your venv's python here if desired)
PYTHON_BIN="python3"

# Whether to clean HuggingFace (and optionally Torch/Ultralytics) caches after all scenes
CLEAN_HF_CACHE="true"     # set to "false" if you want to keep the cache
CLEAN_TORCH_CACHE="false" # optional; set "true" if you want Torch cache gone too
CLEAN_ULTRA_CACHE="false" # optional; for ultralytics cache


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

if ! command -v aws >/dev/null 2>&1; then
  echo "[run-replica-local] ERROR: aws CLI not found in PATH."
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[run-replica-local] ERROR: ${PYTHON_BIN} not found in PATH."
  exit 1
fi

if [[ -z "${S3_OUTPUT_URI}" || "${S3_OUTPUT_URI}" == "s3://YOUR-DATA-FINISHED-BUCKET/replica/" ]]; then
  echo "[run-replica-local] ERROR: Please set S3_OUTPUT_URI to your data-finished bucket path."
  exit 1
fi

# Ensure repo exists
if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[run-replica-local] ERROR: REPO_ROOT does not exist: ${REPO_ROOT}"
  exit 1
fi

# Ensure replica root exists
if [[ ! -d "${REPLICA_ROOT}" ]]; then
  echo "[run-replica-local] ERROR: REPLICA_ROOT does not exist: ${REPLICA_ROOT}"
  exit 1
fi

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

cd "${REPO_ROOT}"

# Ensure Python can import the `conceptgraph` package from this repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

for SCENE in "${SCENES[@]}"; do
  SCENE_DIR="${REPLICA_ROOT}/${SCENE}"
  if [[ ! -d "${SCENE_DIR}" ]]; then
    echo "[run-replica-local][skip] Scene directory not found: ${SCENE_DIR}"
    continue
  fi

  echo "=================================================================="
  echo "[run-replica-local] Processing scene: ${SCENE}"
  echo "=================================================================="

  # Hydra overrides for this scene
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

  # Environment variables (for scripts that look at them)
  export REPO_ROOT
  export DATA_ROOT
  export CKPT_DIR
  export OUTPUT_ROOT

  echo "[run-replica-local] Launching ${PYTHON_BIN} ${PY_SCRIPT} for scene ${SCENE}..."
  set +e
  "${PYTHON_BIN}" "${PY_SCRIPT}" "${OVERRIDES[@]}"
  RET=$?
  set -e

  if [[ ${RET} -ne 0 ]]; then
    echo "[run-replica-local][error] Mapping failed for ${SCENE} (exit code ${RET}). Skipping upload."
    continue
  fi

  echo "[run-replica-local] Mapping succeeded for ${SCENE}. Uploading results to S3..."

  # -----------------------------------------------------------------------
  # Determine local output folder for this scene and sync to S3
  # -----------------------------------------------------------------------
  # For batch_vlm_mapping, results live at:
  #   ${REPLICA_ROOT}/${SCENE}/exps/${EXP_SUFFIX}
  # If you switch to batch_test_local and its OUTPUT_ROOT-based paths,
  # update SCENE_OUTPUT_DIR accordingly.
  SCENE_OUTPUT_DIR="${REPLICA_ROOT}/${SCENE}/exps/${EXP_SUFFIX}"

  if [[ ! -d "${SCENE_OUTPUT_DIR}" ]]; then
    echo "[run-replica-local][warn] Output directory not found for scene ${SCENE}: ${SCENE_OUTPUT_DIR}"
    echo "  Nothing to upload or delete."
    continue
  fi

  # Append scene to S3_OUTPUT_URI if it looks like just a prefix
  # e.g. s3://bucket/data-finished/replica/ -> s3://bucket/data-finished/replica/room0
  S3_SCENE_URI="${S3_OUTPUT_URI%/}/${SCENE}"

  echo "[run-replica-local] aws s3 sync \"${SCENE_OUTPUT_DIR}\" \"${S3_SCENE_URI}\""
  aws s3 sync "${SCENE_OUTPUT_DIR}" "${S3_SCENE_URI}"

  echo "[run-replica-local] Upload complete. Deleting local CG outputs for scene ${SCENE}..."
  rm -rf "${SCENE_OUTPUT_DIR}"

  echo "[run-replica-local] Done with scene ${SCENE}."
  echo
done

echo "[run-replica-local] All requested scenes processed."

# ---------------------------------------------------------------------------
# Optional cache cleanup (local machine, not Docker)
# ---------------------------------------------------------------------------

if [[ "${CLEAN_HF_CACHE}" == "true" ]]; then
  echo "[run-replica-local] Cleaning HuggingFace caches..."

  # Default HF cache
  rm -rf "${HOME}/.cache/huggingface" 2>/dev/null || true

  # Respect env overrides if you ever set them
  if [[ -n "${HF_HOME:-}" ]]; then
    rm -rf "${HF_HOME}" 2>/dev/null || true
  fi

  if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    rm -rf "${HF_HUB_CACHE}" 2>/dev/null || true
  fi

  echo "[run-replica-local] HuggingFace cache cleanup done."
fi

if [[ "${CLEAN_TORCH_CACHE}" == "true" ]]; then
  echo "[run-replica-local] Cleaning Torch cache..."
  rm -rf "${HOME}/.cache/torch" 2>/dev/null || true

  if [[ -n "${TORCH_HOME:-}" ]]; then
    rm -rf "${TORCH_HOME}" 2>/dev/null || true
  fi

  echo "[run-replica-local] Torch cache cleanup done."
fi

if [[ "${CLEAN_ULTRA_CACHE}" == "true" ]]; then
  echo "[run-replica-local] Cleaning Ultralytics cache..."
  rm -rf "${HOME}/.cache/ultralytics" 2>/dev/null || true
  echo "[run-replica-local] Ultralytics cache cleanup done."
fi

