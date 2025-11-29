#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint-multi] Multi-Batch VLM Mapping start"

# =============================================================================
# SECRETS BOOTSTRAP: Fetch secrets from AWS Secrets Manager
# =============================================================================
OPENAI_SECRET_NAME="${OPENAI_SECRET_NAME:-OpenAI}"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[entrypoint] Fetching OPENAI_API_KEY from Secrets Manager (${OPENAI_SECRET_NAME})..."
  if OPENAI_SECRET=$(aws secretsmanager get-secret-value --secret-id "${OPENAI_SECRET_NAME}" --query SecretString --output text 2>/dev/null); then
    if [[ "${OPENAI_SECRET}" == \{* ]]; then
      export OPENAI_API_KEY=$(echo "${OPENAI_SECRET}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('OPENAI_API_KEY') or d.get('api_key') or d.get('key') or '')" 2>/dev/null || echo "")
    else
      export OPENAI_API_KEY="${OPENAI_SECRET}"
    fi
    if [[ -n "${OPENAI_API_KEY}" ]]; then
      echo "[entrypoint] Loaded OPENAI_API_KEY from Secrets Manager"
    fi
  else
    echo "[entrypoint][warn] Could not fetch OpenAI secret from Secrets Manager"
  fi
fi

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
DATA_ROOT="${DATA_ROOT:-/app/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/app/output}"

# =============================================================================
# STEP 1: BULK DOWNLOAD
# =============================================================================
if [[ -n "${S3_INPUT_URI:-}" ]]; then
    echo "[entrypoint] Bulk downloading from ${S3_INPUT_URI} to ${DATA_ROOT}..."
    mkdir -p "${DATA_ROOT}"
    aws s3 sync "${S3_INPUT_URI}" "${DATA_ROOT}"
    
    echo "[entrypoint] Unzipping all .zip files..."
    # Handle case where no zip files exist gracefully
    if compgen -G "${DATA_ROOT}/*.zip" > /dev/null; then
        for zipfile in "${DATA_ROOT}"/*.zip; do
            echo "  Unzipping ${zipfile}..."
            unzip -o -q "${zipfile}" -d "${DATA_ROOT}"
            # Optional: remove zip to save space immediately
            rm "${zipfile}"
        done
    fi
else
    echo "[entrypoint][error] S3_INPUT_URI is not set. Cannot proceed with batch."
    exit 1
fi

# =============================================================================
# STEP 2: ITERATE SCENES
# =============================================================================
# Find all directories in DATA_ROOT that look like scene folders (contain 'traj.txt' or similar, or just iterate all dirs)
# Assuming structure: /app/data/room0, /app/data/office1 ...
# Or /app/data/replica/room0 if zip structure was nested.

# Find directories at depth 1 or 2. We verify if it's a scene by checking for some common file if possible,
# or just assume every subfolder is a scene.
# Let's assume every directory inside DATA_ROOT that is NOT '__MACOSX' is a scene.

echo "[entrypoint] Searching for scenes in ${DATA_ROOT}..."
find "${DATA_ROOT}" -mindepth 1 -maxdepth 2 -type d -not -name "__MACOSX" | while read -r SCENE_DIR; do
    
    CURRENT_SCENE_ID=$(basename "${SCENE_DIR}")
    # Simple check to skip likely non-scene dirs (e.g. 'Replica' container folder if not flat)
    # If it contains other directories, it might be a container. If it contains files, it's a scene.
    if [[ -z "$(ls -A "${SCENE_DIR}")" ]]; then
        echo "[entrypoint][skip] Empty directory: ${SCENE_DIR}"
        continue
    fi

    echo "=================================================================="
    echo "[entrypoint] Processing Scene: ${CURRENT_SCENE_ID}"
    echo "=================================================================="

    # Determine DATASET_ROOT (parent of scene dir)
    CURRENT_DATASET_ROOT=$(dirname "${SCENE_DIR}")

    # Build Hydra Overrides for this specific scene
    OVERRIDES=()
    OVERRIDES+=("repo_root=${REPO_ROOT:-/app/neuro-nav}")
    OVERRIDES+=("data_root=${DATA_ROOT}")
    OVERRIDES+=("dataset_root=${CURRENT_DATASET_ROOT}")
    OVERRIDES+=("scene_id=${CURRENT_SCENE_ID}")
    
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
    OVERRIDES+=("use_wandb=${use_wandb:-false}")
    
    # Ensure WandB override matches detected key
    if [[ -n "${WANDB_API_KEY:-}" ]]; then
        OVERRIDES+=("use_wandb=true")
    else
        OVERRIDES+=("use_wandb=false")
    fi

    echo "[entrypoint] Launching python module..."
    python3 -u -m "${ENTRYPOINT_MODULE}" "${OVERRIDES[@]}"
    
    # Check exit code? 
    # If python fails, we log it but maybe continue to next scene?
    # Let's capture it.
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "[entrypoint][error] Processing failed for ${CURRENT_SCENE_ID} (exit code $RET). Skipping upload."
        # Continue loop?
    else
        echo "[entrypoint] Processing success for ${CURRENT_SCENE_ID}"
        
        # =============================================================================
        # STEP 3: INCREMENTAL UPLOAD & CLEANUP
        # =============================================================================
        if [[ -n "${S3_OUTPUT_URI:-}" ]]; then
            echo "[entrypoint] Uploading results for ${CURRENT_SCENE_ID}..."
            # Sync ONLY the output folder for this scene to the S3 destination
            # Structure in OUTPUT_ROOT is usually: OUTPUT_ROOT/SCENE_ID/exps/...
            
            LOCAL_SCENE_OUTPUT="${OUTPUT_ROOT}/${CURRENT_SCENE_ID}"
            
            if [ -d "${LOCAL_SCENE_OUTPUT}" ]; then
                # We append SCENE_ID to S3 URI to keep buckets organized if S3_OUTPUT_URI is a root
                # S3_OUTPUT_URI should end with / ideally.
                # e.g. s3://bucket/finished/ -> s3://bucket/finished/room0
                aws s3 sync "${LOCAL_SCENE_OUTPUT}" "${S3_OUTPUT_URI}${CURRENT_SCENE_ID}"
                
                echo "[entrypoint] Upload complete. Cleaning up local output..."
                rm -rf "${LOCAL_SCENE_OUTPUT}"
            else
                echo "[entrypoint][warn] No output directory found at ${LOCAL_SCENE_OUTPUT}"
            fi
        fi
    fi

    echo "[entrypoint] Cleaning up input data for ${CURRENT_SCENE_ID}..."
    rm -rf "${SCENE_DIR}"
    
done

echo "[entrypoint] All scenes processed."

