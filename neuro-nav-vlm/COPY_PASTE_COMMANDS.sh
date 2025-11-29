#!/bin/bash
# Copy and paste these commands one section at a time
# Read START_HERE.md for detailed explanations

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: FIRST TIME SETUP (Do this once)
# ═══════════════════════════════════════════════════════════════════════════

cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh

# Install the VLM package
pip install -e .

# Link to neuro-nav data
rm -f data
ln -s /home/nick/Project_dir/neuro-nav/data data

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: PREPARE SCENE MAP (Do this once per scene)
# ═══════════════════════════════════════════════════════════════════════════

# Find available scene maps
find /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_* -name "*.pkl.gz"

# Choose one and set it here (default: r_mapping_with_llm)
SCENE_NAME="r_mapping_with_llm"

# Create map directory and link
mkdir -p data/Replica/room0/exps/${SCENE_NAME}/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/${SCENE_NAME}/pcd_${SCENE_NAME}.pkl.gz \
       data/Replica/room0/exps/${SCENE_NAME}/map/scene_map_cfslam.pkl.gz

# Verify it worked
ls -la data/Replica/room0/exps/${SCENE_NAME}/map/scene_map_cfslam.pkl.gz

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: RUN PIPELINE (Easy way - automated)
# ═══════════════════════════════════════════════════════════════════════════

cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

# Run everything (takes ~25-30 minutes)
./run_everything.sh

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: RUN PIPELINE (Manual way - step by step)
# ═══════════════════════════════════════════════════════════════════════════

cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

# Set these variables
SCENE_NAME="r_mapping_with_llm"
CACHEDIR="data/Replica/room0/exps/${SCENE_NAME}"
MAPFILE="${CACHEDIR}/map/scene_map_cfslam.pkl.gz"

# Step 1: Extract captions (~10-15 min)
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 2: Refine captions (~5 min)
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 3: Build scene graph (~10 min)
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 4: Generate JSON (~1 min)
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: VIEW RESULTS
# ═══════════════════════════════════════════════════════════════════════════

SCENE_NAME="r_mapping_with_llm"

# View scene graph (with jq)
cat data/Replica/room0/exps/${SCENE_NAME}/scene_graph.json | jq '.[:3]'

# View scene graph (without jq)
cat data/Replica/room0/exps/${SCENE_NAME}/scene_graph.json | head -100

# View raw captions
cat data/Replica/room0/exps/${SCENE_NAME}/cfslam_qwen_captions.json | jq '.'

# View relationships
cat data/Replica/room0/exps/${SCENE_NAME}/cfslam_object_relations.json | jq '.'

# Count objects
cat data/Replica/room0/exps/${SCENE_NAME}/scene_graph.json | jq 'length'

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: QUICK RESTART (After reboot/closing terminal)
# ═══════════════════════════════════════════════════════════════════════════

cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
./run_everything.sh

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: TROUBLESHOOTING
# ═══════════════════════════════════════════════════════════════════════════

# Fix "ModuleNotFoundError: No module named 'conceptgraph.vlm'"
cd /home/nick/Project_dir/neuro-nav-vlm
pip install -e .
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

# Fix "No module named 'transformers'"
pip install transformers timm qwen-vl-utils accelerate einops torchvision bitsandbytes

# Fix "Scene map not found"
SCENE_NAME="r_mapping_with_llm"  # Change this
mkdir -p data/Replica/room0/exps/${SCENE_NAME}/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/${SCENE_NAME}/pcd_${SCENE_NAME}.pkl.gz \
       data/Replica/room0/exps/${SCENE_NAME}/map/scene_map_cfslam.pkl.gz

# Clear GPU memory if stuck
pkill -9 python

# Check GPU usage
nvidia-smi

# ═══════════════════════════════════════════════════════════════════════════
# DONE! Read START_HERE.md for detailed explanations
# ═══════════════════════════════════════════════════════════════════════════


