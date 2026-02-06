#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] Batch VLM Mapping start"

# =============================================================================
# SECRETS BOOTSTRAP: Fetch secrets from AWS Secrets Manager
# =============================================================================
# Fetch OpenAI API key if not already set and secret name is provided
OPENAI_SECRET_NAME="${OPENAI_SECRET_NAME:-OpenAI}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[entrypoint] Fetching OPENAI_API_KEY from Secrets Manager (${OPENAI_SECRET_NAME})..."
  # Try to fetch the secret; don't fail if it doesn't exist
  if OPENAI_SECRET=$(aws secretsmanager get-secret-value --secret-id "${OPENAI_SECRET_NAME}" --query SecretString --output text 2>/dev/null); then
    # Check if it's JSON (has quotes/braces) or plain string
    if [[ "${OPENAI_SECRET}" == \{* ]]; then
      # JSON format - extract the key (try common field names)
      export OPENAI_API_KEY=$(echo "${OPENAI_SECRET}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('OPENAI_API_KEY') or d.get('api_key') or d.get('key') or '')" 2>/dev/null || echo "")
    else
      # Plain string
      export OPENAI_API_KEY="${OPENAI_SECRET}"
    fi
    if [[ -n "${OPENAI_API_KEY}" ]]; then
      echo "[entrypoint] Loaded OPENAI_API_KEY from Secrets Manager"
    else
      echo "[entrypoint][warn] Secret found but could not extract API key"
    fi
  else
    echo "[entrypoint][warn] Could not fetch OpenAI secret '${OPENAI_SECRET_NAME}' from Secrets Manager"
  fi
fi

# Fetch WandB API key if not set
# We assume it might be in the same secret or a different one.
# For simplicity, let's look for a specific WandB secret or try to parse it from User-Keys if typical.
# Or simpler: user provides WANDB_SECRET_NAME env var.
WANDB_SECRET_NAME="${WANDB_SECRET_NAME:-WandB}"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
   echo "[entrypoint] Attempting to fetch WANDB_API_KEY from Secrets Manager (${WANDB_SECRET_NAME})..."
   if WANDB_SECRET=$(aws secretsmanager get-secret-value --secret-id "${WANDB_SECRET_NAME}" --query SecretString --output text 2>/dev/null); then
       if [[ "${WANDB_SECRET}" == \{* ]]; then
           export WANDB_API_KEY=$(echo "${WANDB_SECRET}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('WANDB_API_KEY') or d.get('api_key') or d.get('key') or '')" 2>/dev/null || echo "")
       else
           export WANDB_API_KEY="${WANDB_SECRET}"
       fi
       if [[ -n "${WANDB_API_KEY}" ]]; then
           echo "[entrypoint] Loaded WANDB_API_KEY from Secrets Manager"
       fi
   fi
fi

ENTRYPOINT_MODULE="${BATCH_MAIN:-conceptgraph.slam.batch_vlm_mapping}"

# S3 Ingestion (Run BEFORE Python so data is ready)
if [[ -n "${S3_INPUT_URI:-}" ]]; then
    # Ensure DATA_ROOT exists
    mkdir -p "${DATA_ROOT}"
    
    # Check if input is a single file (ends in .zip) or a directory
    if [[ "${S3_INPUT_URI}" == *.zip ]]; then
        echo "[entrypoint] Downloading SINGLE file from ${S3_INPUT_URI} to ${DATA_ROOT}..."
        aws s3 cp "${S3_INPUT_URI}" "${DATA_ROOT}/"
    else
        echo "[entrypoint] Downloading DIRECTORY from ${S3_INPUT_URI} to ${DATA_ROOT}..."
        aws s3 sync "${S3_INPUT_URI}" "${DATA_ROOT}"
    fi

    # AUTO-UNZIP LOGIC
    # If zipped scenes are present, unzip them.
    if compgen -G "${DATA_ROOT}/*.zip" > /dev/null; then
        echo "[entrypoint] Found zip files in ${DATA_ROOT}. Unzipping..."
        for zipfile in "${DATA_ROOT}"/*.zip; do
            echo "  Unzipping ${zipfile}..."
            unzip -o -q "${zipfile}" -d "${DATA_ROOT}"
        done
    fi
    
    # PATH CORRECTION LOGIC
    # We need to find where the SCENE_ID folder ended up.
    # It could be:
    #   /app/data/room0           (if zip matched root)
    #   /app/data/replica/room0   (if zip contained 'replica/room0')
    #   /app/data/scans/scene...  (if ScanNet)
    
    FOUND_PATH=$(find "${DATA_ROOT}" -maxdepth 3 -type d -name "${SCENE_ID}" | head -n 1)
    
    if [[ -n "${FOUND_PATH}" ]]; then
        # The dataset_root is the PARENT of the scene folder
        PARENT_DIR=$(dirname "${FOUND_PATH}")
        echo "[entrypoint] Found scene '${SCENE_ID}' at '${FOUND_PATH}'"
        echo "[entrypoint] Setting dataset_root to '${PARENT_DIR}'"
        export DATASET_ROOT="${PARENT_DIR}"
    else
        echo "[entrypoint][warn] Could not find folder named '${SCENE_ID}' inside '${DATA_ROOT}'."
        echo "[entrypoint][warn] Proceeding with defaults, but this may fail."
    fi
fi

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
OVERRIDES+=("save_semantic_snapshot=${SAVE_SEMANTIC_SNAPSHOT:-true}")
OVERRIDES+=("vis_render=${VIS_RENDER:-false}")
OVERRIDES+=("periodically_save_pcd=${PERIODIC_PCD:-false}")
OVERRIDES+=("periodically_save_pcd_interval=${PERIODIC_PCD_INTERVAL:-10}")

# Enable WandB if key is present
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    OVERRIDES+=("use_wandb=true")
else
    OVERRIDES+=("use_wandb=false")
fi

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

echo "[entrypoint] Launching module: ${ENTRYPOINT_MODULE}"

# Checkpoints are downloaded automatically by libraries (YOLO, HF) if not present locally.
# We skip explicit S3 sync for checkpoints to simplify the pipeline.

python3 -u -m "${ENTRYPOINT_MODULE}" "${OVERRIDES[@]}" "$@"
PYTHON_EXIT_CODE=$?

# S3 Egress
if [[ -n "${S3_OUTPUT_URI:-}" ]]; then
    echo "[entrypoint] Uploading results from ${OUTPUT_ROOT} to ${S3_OUTPUT_URI}..."
    aws s3 sync "${OUTPUT_ROOT}" "${S3_OUTPUT_URI}"
fi

exit $PYTHON_EXIT_CODE


