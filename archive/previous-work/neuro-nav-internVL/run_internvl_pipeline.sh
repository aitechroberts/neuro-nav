#!/bin/bash
# Run the complete InternVL2-based scene graph pipeline

# Configuration
MAPFILE="data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz"
CACHEDIR="data/Replica/room0/exps/r_mapping_with_llm"
DEVICE="cuda:0"

echo "========================================="
echo "InternVL2 Scene Graph Pipeline"
echo "========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "Activate your venv first:"
    echo "  source /path/to/.venv/bin/activate"
    echo ""
fi

# Check if scene map exists
if [ ! -f "$MAPFILE" ]; then
    echo "❌ Error: Scene map not found at $MAPFILE"
    echo ""
    echo "Run SLAM pipeline first:"
    echo "  cd ../neuro-nav"
    echo "  python conceptgraph/slam/rerun_realtime_mapping.py \\"
    echo "    --config-name=rerun_simple_test end=30"
    exit 1
fi

echo "✅ Scene map found: $MAPFILE"
echo ""

# Step 1: Extract node captions
echo "Step 1/4: Extracting node captions with InternVL2..."
python conceptgraph/scenegraph/build_scenegraph_internvl.py \
    --mode extract-node-captions \
    --mapfile "$MAPFILE" \
    --cachedir "$CACHEDIR" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "❌ Error in caption extraction"
    exit 1
fi

# Step 2: Refine node captions
echo ""
echo "Step 2/4: Refining node captions..."
python conceptgraph/scenegraph/build_scenegraph_internvl.py \
    --mode refine-node-captions \
    --mapfile "$MAPFILE" \
    --cachedir "$CACHEDIR" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "❌ Error in caption refinement"
    exit 1
fi

# Step 3: Build scene graph
echo ""
echo "Step 3/4: Building scene graph..."
python conceptgraph/scenegraph/build_scenegraph_internvl.py \
    --mode build-scenegraph \
    --mapfile "$MAPFILE" \
    --cachedir "$CACHEDIR" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "❌ Error in scene graph construction"
    exit 1
fi

# Step 4: Generate scene graph JSON
echo ""
echo "Step 4/4: Generating scene graph JSON..."
python conceptgraph/scenegraph/build_scenegraph_internvl.py \
    --mode generate-scenegraph-json \
    --mapfile "$MAPFILE" \
    --cachedir "$CACHEDIR"

if [ $? -ne 0 ]; then
    echo "❌ Error in JSON generation"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ Pipeline complete!"
echo "========================================="
echo ""
echo "Scene graph saved to:"
echo "  $CACHEDIR/scene_graph.json"
echo ""
echo "To query the scene:"
echo "  python query_internvl_scene.py \\"
echo "    --scene-graph $CACHEDIR/scene_graph.json \\"
echo "    --query 'Where can I sit?'"
echo ""

