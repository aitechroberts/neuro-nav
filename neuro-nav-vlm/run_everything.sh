#!/bin/bash
# Complete VLM Pipeline Runner
# Run this after following steps 1-5 in START_HERE.md

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║              Qwen2-VL Scene Graph Pipeline                        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to correct directory
cd "$(dirname "$0")"

# Set Python path
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

# Configuration - CHANGE THIS IF USING A DIFFERENT SCENE MAP
SCENE_DIR="${1:-r_mapping_with_llm}"
CACHEDIR="data/Replica/room0/exps/${SCENE_DIR}"
MAPFILE="${CACHEDIR}/map/scene_map_cfslam.pkl.gz"

echo "Configuration:"
echo "  Scene: ${SCENE_DIR}"
echo "  Cache: ${CACHEDIR}"
echo "  Map:   ${MAPFILE}"
echo ""

# Verify map file exists
if [ ! -f "${MAPFILE}" ]; then
    echo "✗ Error: Scene map not found at: ${MAPFILE}"
    echo ""
    echo "Please run these commands first:"
    echo "  mkdir -p ${CACHEDIR}/map"
    echo "  ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/${SCENE_DIR}/pcd_${SCENE_DIR}.pkl.gz \\"
    echo "         ${MAPFILE}"
    exit 1
fi

echo "✓ Scene map found"
echo ""

# Confirm before starting
read -p "Start VLM pipeline? This will take ~25-30 minutes. (y/n) [y]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Step 1/4: Extract Node Captions with Qwen2-VL                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "✗ Error in caption extraction"
    exit 1
fi

echo ""
echo "✓ Caption extraction complete"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Step 2/4: Refine Node Captions with Qwen2-VL                    ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "✗ Error in caption refinement"
    exit 1
fi

echo ""
echo "✓ Caption refinement complete"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Step 3/4: Build Scene Graph with Relationships                  ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

if [ $? -ne 0 ]; then
    echo "✗ Error in scene graph building"
    exit 1
fi

echo ""
echo "✓ Scene graph building complete"
echo ""

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  Step 4/4: Generate Scene Graph JSON                             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

if [ $? -ne 0 ]; then
    echo "✗ Error in JSON generation"
    exit 1
fi

echo ""
echo "✓ JSON generation complete"
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Complete! 🎉                          ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files:"
echo "  Scene graph:       ${CACHEDIR}/scene_graph.json"
echo "  Raw captions:      ${CACHEDIR}/cfslam_qwen_captions.json"
echo "  Refined captions:  ${CACHEDIR}/cfslam_qwen_responses/"
echo "  Relationships:     ${CACHEDIR}/cfslam_object_relations.json"
echo "  Debug images:      ${CACHEDIR}/cfslam_captions_qwen_debug/"
echo ""
echo "View results:"
echo "  cat ${CACHEDIR}/scene_graph.json | jq '.'"
echo ""
echo "  Or: cat ${CACHEDIR}/scene_graph.json | head -100"
echo ""


