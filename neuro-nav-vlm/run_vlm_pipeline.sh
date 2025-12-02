#!/bin/bash

# VLM Pipeline Runner for neuro-nav-vlm
# This script runs the complete VLM-based scene graph pipeline

set -e  # Exit on error

echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║          VLM-Based Scene Graph Pipeline Runner                     ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Activate environment
echo "→ Activating Python environment..."
if [ -f "../neuro-nav/.venv/bin/activate" ]; then
    source ../neuro-nav/.venv/bin/activate
    echo "  ✓ Using neuro-nav environment"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "  ✓ Using local environment"
else
    echo "  ✗ Error: No virtual environment found!"
    echo "    Run: python -m venv .venv && source .venv/bin/activate"
    exit 1
fi

# Load CUDA if available
if [ -f "../neuro-nav/use-cuda-126.sh" ]; then
    source ../neuro-nav/use-cuda-126.sh
    echo "  ✓ CUDA 12.6 loaded"
fi

# Find the scene map file
echo ""
echo "→ Looking for scene map..."

# Check if user provided paths
if [ ! -z "$1" ]; then
    CACHEDIR="$1"
    MAPFILE="${CACHEDIR}/map/scene_map_cfslam.pkl.gz"
    echo "  Using provided path: ${CACHEDIR}"
else
    # Try to find latest output automatically
    if [ -d "data/outputs" ]; then
        LATEST_ROOM=$(find data/outputs -type d -name "room0" 2>/dev/null | sort -r | head -1)
        if [ ! -z "$LATEST_ROOM" ]; then
            # Extract parent directory (timestamp directory)
            TIMESTAMP_DIR=$(dirname "$LATEST_ROOM")
            CACHEDIR=$(dirname "$TIMESTAMP_DIR")
            MAPFILE="${LATEST_ROOM}/map/scene_map_cfslam.pkl.gz"
            echo "  Found latest scene: ${LATEST_ROOM}"
        else
            echo "  ✗ Error: No scene map found in data/outputs"
            echo ""
            echo "Usage: $0 [CACHEDIR]"
            echo "Example: $0 data/outputs/2025-11-12/10-30-00"
            echo ""
            echo "Or run SLAM first:"
            echo "  cd ../neuro-nav"
            echo "  python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test"
            exit 1
        fi
    else
        echo "  ✗ Error: data/outputs directory not found"
        echo "  Did you create a symlink? Run: ln -s ../neuro-nav/data data"
        exit 1
    fi
fi

# Verify map file exists
if [ ! -f "$MAPFILE" ]; then
    echo "  ✗ Error: Scene map not found at: ${MAPFILE}"
    exit 1
fi

echo "  ✓ Scene map found"
echo ""
echo "Configuration:"
echo "  Cache dir: ${CACHEDIR}"
echo "  Map file:  ${MAPFILE}"
echo ""

# Model selection
FLORENCE_MODEL="${FLORENCE_MODEL:-microsoft/Florence-2-large}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
DEVICE="${DEVICE:-cuda:0}"

echo "Models:"
echo "  Florence-2: ${FLORENCE_MODEL}"
echo "  Qwen2-VL:   ${QWEN_MODEL}"
echo "  Device:     ${DEVICE}"
echo ""

# Ask for confirmation
read -p "Continue with VLM pipeline? (y/n) [y]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Extract captions
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  Step 1/4: Extract Node Captions with Florence-2                   ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir "${CACHEDIR}" \
    --mapfile "${MAPFILE}" \
    --florence-model "${FLORENCE_MODEL}" \
    --qwen-model "${QWEN_MODEL}" \
    --device "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "✗ Error in caption extraction"
    exit 1
fi

echo ""
echo "✓ Caption extraction complete"

# Step 2: Refine captions
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  Step 2/4: Refine Node Captions with Qwen2-VL                      ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir "${CACHEDIR}" \
    --mapfile "${MAPFILE}" \
    --qwen-model "${QWEN_MODEL}" \
    --device "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "✗ Error in caption refinement"
    exit 1
fi

echo ""
echo "✓ Caption refinement complete"

# Step 3: Build scene graph
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  Step 3/4: Build Scene Graph with Relationships                    ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir "${CACHEDIR}" \
    --mapfile "${MAPFILE}" \
    --qwen-model "${QWEN_MODEL}" \
    --device "${DEVICE}"

if [ $? -ne 0 ]; then
    echo "✗ Error in scene graph building"
    exit 1
fi

echo ""
echo "✓ Scene graph building complete"

# Step 4: Generate JSON
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║  Step 4/4: Generate Scene Graph JSON                               ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir "${CACHEDIR}" \
    --mapfile "${MAPFILE}"

if [ $? -ne 0 ]; then
    echo "✗ Error in JSON generation"
    exit 1
fi

echo ""
echo "✓ JSON generation complete"

# Summary
echo ""
echo "╔═════════════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Complete! 🎉                            ║"
echo "╚═════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files:"
echo "  Scene graph:       ${CACHEDIR}/scene_graph.json"
echo "  Raw captions:      ${CACHEDIR}/cfslam_florence_captions.json"
echo "  Refined captions:  ${CACHEDIR}/cfslam_qwen_responses/"
echo "  Relationships:     ${CACHEDIR}/cfslam_object_relations.json"
echo "  Debug images:      ${CACHEDIR}/cfslam_captions_florence_debug/"
echo ""
echo "Next steps:"
echo "  1. View scene graph:   cat ${CACHEDIR}/scene_graph.json | jq '.'"
echo "  2. Query the scene:    python query_vlm_scene.py"
echo "  3. Visualize:          Use Rerun viewer"
echo ""

