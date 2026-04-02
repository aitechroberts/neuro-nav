#!/usr/bin/env bash
# =============================================================================
# Staged Pipeline Orchestration
#
# Runs the multi-stage scene graph pipeline with configurable stage selection,
# encoder lists, and VLM lists.
#
# Usage:
#   STAGE=all SCENE_IDS="room0 room1" ./run_staged_pipeline.sh
#   STAGE=detect SCENE_IDS="room0" ./run_staged_pipeline.sh
#   STAGE=encode ENCODER_LIST="openclip:ViT-bigG-14:laion2b_s39b_b160k" ./run_staged_pipeline.sh
#   STAGE=caption VLM_LIST="Qwen/Qwen3-VL-2B-Instruct" ./run_staged_pipeline.sh
# =============================================================================

set -euo pipefail

STAGE="${STAGE:-all}"
SCENE_IDS="${SCENE_IDS:-room0}"
DATASET_ROOT="${DATASET_ROOT:-/home/jrob/cmu-grad/neuro-data/Replica}"
DATASET_CONFIG="${DATASET_CONFIG:-replica}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DEVICE="${DEVICE:-cuda}"

# Stage 2 matching backbone
MATCHING_MODEL="${MATCHING_MODEL:-ViT-bigG-14}"
MATCHING_PRETRAINED="${MATCHING_PRETRAINED:-laion2b_s39b_b160k}"

# Stage 3 encoder sweep
ENCODER_LIST="${ENCODER_LIST:-openclip:ViT-bigG-14:laion2b_s39b_b160k}"
ENTROPY_PROMPT_LIST="${ENTROPY_PROMPT_LIST:-config/replica_50_labels.txt}"

# Stage 4 VLM caption sweep
VLM_LIST="${VLM_LIST:-}"
VLM_API_URL="${VLM_API_URL:-http://localhost:8000/v1}"

# Config overrides
STRIDE="${STRIDE:-50}"
EXP_SUFFIX="${EXP_SUFFIX:-batch_api}"
MAKE_EDGES="${MAKE_EDGES:-false}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAPPING_SCRIPT="$PROJECT_ROOT/conceptgraph/slam/vlm_run/batch_vlm_mapping_api.py"
ENCODER_SWEEP="$PROJECT_ROOT/conceptgraph/slam/vlm_run/encoder_sweep.py"
CAPTION_SWEEP="$PROJECT_ROOT/conceptgraph/slam/vlm_run/vlm_caption_sweep.py"
EDGE_SCRIPT="$PROJECT_ROOT/conceptgraph/slam/vlm_run/edge_construction.py"
HPSG_SCRIPT="$PROJECT_ROOT/conceptgraph/slam/vlm_run/hpsg_construction.py"

# =============================================================================
# Helper functions
# =============================================================================

resolve_exp_path() {
    local scene_id="$1"
    if [ -n "$OUTPUT_ROOT" ]; then
        echo "$OUTPUT_ROOT/$scene_id/exps/$EXP_SUFFIX"
    else
        echo "$DATASET_ROOT/$scene_id/exps/$EXP_SUFFIX"
    fi
}

# =============================================================================
# Stage 1: Detection (YOLO + SAM only)
# =============================================================================

run_detect() {
    echo "========================================="
    echo "  Stage 1: Detection"
    echo "========================================="
    for scene in $SCENE_IDS; do
        echo "[Stage 1] Running detections for scene: $scene"
        DETECTION_ONLY=true \
        MAKE_EDGES=false \
        python "$MAPPING_SCRIPT" \
            dataset_config="$DATASET_CONFIG" \
            dataset_root="$DATASET_ROOT" \
            scene_id="$scene" \
            stride="$STRIDE" \
            exp_suffix="$EXP_SUFFIX" \
            detection_only=true \
            save_detections=true \
            force_detection=true \
            make_edges=false \
            ${OUTPUT_ROOT:+output_root=$OUTPUT_ROOT}
    done
    echo "[Stage 1] Detection complete for all scenes."
}

# =============================================================================
# Stage 2: Definitive Mapping (ViT-bigG-14)
# =============================================================================

run_map() {
    echo "========================================="
    echo "  Stage 2: Definitive Mapping"
    echo "========================================="
    for scene in $SCENE_IDS; do
        echo "[Stage 2] Mapping scene: $scene with $MATCHING_MODEL"
        MATCHING_MODEL="$MATCHING_MODEL" \
        MATCHING_MODEL_PRETRAINED="$MATCHING_PRETRAINED" \
        MAKE_EDGES="$MAKE_EDGES" \
        python "$MAPPING_SCRIPT" \
            dataset_config="$DATASET_CONFIG" \
            dataset_root="$DATASET_ROOT" \
            scene_id="$scene" \
            stride="$STRIDE" \
            exp_suffix="$EXP_SUFFIX" \
            detection_only=false \
            force_detection=false \
            save_detections=false \
            matching_model="$MATCHING_MODEL" \
            matching_model_pretrained="$MATCHING_PRETRAINED" \
            make_edges="$MAKE_EDGES" \
            ${OUTPUT_ROOT:+output_root=$OUTPUT_ROOT}
    done
    echo "[Stage 2] Mapping complete for all scenes."
}

# =============================================================================
# Stage 3: Encoder Sweep
# =============================================================================

run_encode() {
    echo "========================================="
    echo "  Stage 3: Encoder Sweep"
    echo "========================================="
    for scene in $SCENE_IDS; do
        exp_path=$(resolve_exp_path "$scene")
        groupings_path="$exp_path/merge_groupings.pkl.gz"

        if [ ! -f "$groupings_path" ]; then
            echo "[Stage 3] WARNING: No groupings found at $groupings_path -- skipping $scene"
            continue
        fi

        for encoder in $ENCODER_LIST; do
            echo "[Stage 3] Encoding scene=$scene encoder=$encoder"
            python "$ENCODER_SWEEP" \
                --groupings "$groupings_path" \
                --encoder "$encoder" \
                --output_dir "$exp_path/embeddings" \
                --device "$DEVICE" \
                --use_scaled_crop \
                ${ENTROPY_PROMPT_LIST:+--entropy_prompt_list "$PROJECT_ROOT/$ENTROPY_PROMPT_LIST"}
        done
    done
    echo "[Stage 3] Encoder sweep complete."
}

# =============================================================================
# Stage 4: VLM Caption Sweep
# =============================================================================

run_caption() {
    echo "========================================="
    echo "  Stage 4: VLM Caption Sweep"
    echo "========================================="
    if [ -z "$VLM_LIST" ]; then
        echo "[Stage 4] No VLM_LIST specified, skipping."
        return
    fi

    for vlm_model in $VLM_LIST; do
        echo "[Stage 4] Starting vLLM for $vlm_model ..."

        for scene in $SCENE_IDS; do
            exp_path=$(resolve_exp_path "$scene")
            groupings_path="$exp_path/merge_groupings.pkl.gz"

            if [ ! -f "$groupings_path" ]; then
                echo "[Stage 4] WARNING: No groupings at $groupings_path -- skipping"
                continue
            fi

            echo "[Stage 4] Captioning scene=$scene vlm=$vlm_model"
            python "$CAPTION_SWEEP" \
                --groupings "$groupings_path" \
                --vlm_api_url "$VLM_API_URL" \
                --vlm_model_name "$vlm_model" \
                --output_dir "$exp_path/captions" \
                --use_scaled_crop
        done
    done
    echo "[Stage 4] Caption sweep complete."
}

# =============================================================================
# Stage 5: Post-Processing (MST Edges + HPSG)
# =============================================================================

run_post() {
    echo "========================================="
    echo "  Stage 5: Post-Processing"
    echo "========================================="
    for scene in $SCENE_IDS; do
        exp_path=$(resolve_exp_path "$scene")

        pkl_path=$(find "$exp_path" -name "pcd_*.pkl.gz" | head -1)
        if [ -z "$pkl_path" ]; then
            echo "[Stage 5] WARNING: No PKL found in $exp_path -- skipping"
            continue
        fi

        echo "[Stage 5a] MST edges for scene=$scene"
        python "$EDGE_SCRIPT" \
            --pkl_path "$pkl_path" \
            --output_dir "$exp_path/edges"

        echo "[Stage 5b] HPSG hierarchy for scene=$scene"
        python "$HPSG_SCRIPT" \
            --pkl_path "$pkl_path" \
            --output_dir "$exp_path/hpsg"
    done
    echo "[Stage 5] Post-processing complete."
}

# =============================================================================
# Stage Dispatch
# =============================================================================

case "$STAGE" in
    detect)
        run_detect
        ;;
    map)
        run_map
        ;;
    encode)
        run_encode
        ;;
    caption)
        run_caption
        ;;
    post)
        run_post
        ;;
    all)
        run_detect
        run_map
        run_encode
        run_caption
        run_post
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: detect, map, encode, caption, post, all"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "  Pipeline complete (stage=$STAGE)"
echo "========================================="
