#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Batch VLM Mapping start"

# Build Hydra overrides from env (with sensible defaults provided in Dockerfile)
OVERRIDES=()
OVERRIDES+=("repo_root=${REPO_ROOT:-/app/neuro-nav}")
OVERRIDES+=("data_root=${DATA_ROOT:-/mnt/data}")

if [[ -n "${DATASET_ROOT:-}" ]]; then
  OVERRIDES+=("dataset_root=${DATASET_ROOT}")
fi
if [[ -n "${SCENE_ID:-}" ]]; then
  OVERRIDES+=("scene_id=${SCENE_ID}")
fi

OVERRIDES+=("exp_suffix=${EXP_SUFFIX:-batch_vlm}")
OVERRIDES+=("detections_exp_suffix=${DET_EXP_SUFFIX:-s_detections}")
OVERRIDES+=("device=${DEVICE:-cuda}")
OVERRIDES+=("start=${START:-0}")
OVERRIDES+=("end=${END:--1}")
OVERRIDES+=("stride=${STRIDE:-1}")
OVERRIDES+=("make_edges=${MAKE_EDGES:-true}")
OVERRIDES+=("force_detection=${FORCE_DET:-true}")
OVERRIDES+=("save_json=${SAVE_JSON:-true}")
OVERRIDES+=("save_pcd=${SAVE_PCD:-true}")
OVERRIDES+=("vis_render=${VIS_RENDER:-false}")
OVERRIDES+=("periodically_save_pcd=${PERIODIC_PCD:-false}")
OVERRIDES+=("periodically_save_pcd_interval=${PERIODIC_PCD_INTERVAL:-10}")

# Disable ReRun (hybrid script doesn't use it, but ensure overrides if config has it)
OVERRIDES+=("use_rerun=false")
OVERRIDES+=("save_rerun=false")

echo "[entrypoint] Using overrides:"
for o in "${OVERRIDES[@]}"; do
  echo "  ${o}"
done

# If OPENAI API is not set and edges requested, warn (but don't fail)
if [[ "${MAKE_EDGES:-true}" == "true" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[entrypoint][warn] MAKE_EDGES=true but OPENAI_API_KEY is not set. VLM edges will be skipped."
fi

exec python3 -u -m conceptgraph.slam.batch_vlm_mapping "${OVERRIDES[@]}"


