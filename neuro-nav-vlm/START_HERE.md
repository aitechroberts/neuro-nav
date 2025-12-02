# START HERE - Complete Setup Guide

**Simple, explicit instructions to get the VLM pipeline running.**

---

## Prerequisites

- ✅ You have: Qwen2-VL-2B model (already downloaded)
- ✅ You have: Existing scene maps in neuro-nav
- ✅ You need: 8GB GPU, Python 3.10, CUDA 12.6

---

## Step 1: Activate Environment (Every Time)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
```

---

## Step 2: Install neuro-nav-vlm (One Time Only)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
pip install -e .
```

**This makes Python able to find the VLM modules.**

---

## Step 3: Link to Data (One Time Only)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm

# Remove old link if it exists
rm -f data

# Create proper link
ln -s /home/nick/Project_dir/neuro-nav/data data
```

---

## Step 4: Find Your Scene Map

```bash
# List all available scene maps
find /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_* -name "*.pkl.gz"
```

**You should see something like:**
```
/home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/pcd_r_mapping_with_llm.pkl.gz
```

**Pick one** to use. We'll use `r_mapping_with_llm` as an example.

---

## Step 5: Setup Scene Map Structure

```bash
cd /home/nick/Project_dir/neuro-nav-vlm

# Create the expected directory structure
mkdir -p data/Replica/room0/exps/r_mapping_with_llm/map

# Link the scene map with the expected name
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/pcd_r_mapping_with_llm.pkl.gz \
       data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz
```

**Replace `r_mapping_with_llm` with whatever scene map you chose in Step 4.**

---

## Step 6: Set Python Path (Every Time You Run)

```bash
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
```

---

## Step 7: Run VLM Pipeline

### Step 7a: Extract Captions (~10-15 minutes)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir data/Replica/room0/exps/r_mapping_with_llm \
    --mapfile data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0
```

**This will:**
- Load Qwen2-VL
- Process each object
- Generate captions
- Save to: `data/Replica/room0/exps/r_mapping_with_llm/cfslam_qwen_captions.json`

---

### Step 7b: Refine Captions (~5 minutes)

```bash
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir data/Replica/room0/exps/r_mapping_with_llm \
    --mapfile data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0
```

**This will:**
- Refine the captions
- Extract object tags
- Save to: `data/Replica/room0/exps/r_mapping_with_llm/cfslam_qwen_responses/`

---

### Step 7c: Build Scene Graph (~10 minutes)

```bash
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir data/Replica/room0/exps/r_mapping_with_llm \
    --mapfile data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0
```

**This will:**
- Determine spatial relationships
- Build scene graph
- Save to: `data/Replica/room0/exps/r_mapping_with_llm/cfslam_object_relations.json`

---

### Step 7d: Generate Final JSON (~1 minute)

```bash
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir data/Replica/room0/exps/r_mapping_with_llm \
    --mapfile data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz
```

**This will:**
- Create final scene graph
- Save to: `data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json`

---

## Step 8: View Results

```bash
# View the scene graph (first 3 objects)
cat data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json | jq '.[0:3]'

# Or without jq
cat data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json | head -50
```

---

## Quick Reference: Complete Command Sequence

**Copy-paste this entire block:**

```bash
# 1. Navigate and activate
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh

# 2. Set Python path
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

# 3. Set variables (CHANGE THE SCENE MAP NAME IF NEEDED)
SCENE_DIR="r_mapping_with_llm"
CACHEDIR="data/Replica/room0/exps/${SCENE_DIR}"
MAPFILE="${CACHEDIR}/map/scene_map_cfslam.pkl.gz"

# 4. Run all steps
echo "Step 1/4: Extracting captions..."
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

echo "Step 2/4: Refining captions..."
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

echo "Step 3/4: Building scene graph..."
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE} \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

echo "Step 4/4: Generating JSON..."
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

echo "✅ Done! Scene graph saved to: ${CACHEDIR}/scene_graph.json"
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'conceptgraph.vlm'"

**Fix:**
```bash
cd /home/nick/Project_dir/neuro-nav-vlm
pip install -e .
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
```

---

### "No such file or directory: scene_map_cfslam.pkl.gz"

**Fix:** Make sure you created the symlink in Step 5.

```bash
# Check if it exists
ls -la data/Replica/room0/exps/r_mapping_with_llm/map/

# If not, create it
mkdir -p data/Replica/room0/exps/r_mapping_with_llm/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/pcd_r_mapping_with_llm.pkl.gz \
       data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz
```

---

### "CUDA out of memory"

**Fix:** Close other programs or use fewer detections:

```bash
python ... --max-detections-per-object 5  # Default is 10
```

---

### "No module named 'transformers'"

**Fix:** The venv is broken. Reinstall dependencies:

```bash
cd /home/nick/Project_dir/neuro-nav
pip install transformers timm qwen-vl-utils accelerate einops torchvision bitsandbytes
```

---

## Different Scene Maps

To use a different scene map, just change the `SCENE_DIR` variable:

```bash
# For r_mapping_with_gpt4v
SCENE_DIR="r_mapping_with_gpt4v"

# For r_direction1_semantic
SCENE_DIR="r_direction1_semantic"

# Then create the map link:
mkdir -p data/Replica/room0/exps/${SCENE_DIR}/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/${SCENE_DIR}/pcd_${SCENE_DIR}.pkl.gz \
       data/Replica/room0/exps/${SCENE_DIR}/map/scene_map_cfslam.pkl.gz
```

---

## What Each Step Does

| Step | What It Does | Time | Output |
|------|--------------|------|--------|
| **extract-node-captions** | Captions each object with Qwen2-VL | 10-15 min | Raw captions JSON |
| **refine-node-captions** | Refines and extracts object tags | 5 min | Refined captions |
| **build-scenegraph** | Determines spatial relationships | 10 min | Relationships JSON |
| **generate-scenegraph-json** | Creates final human-readable output | 1 min | scene_graph.json |

**Total time: ~25-30 minutes**

---

## That's It!

You now have:
- ✅ Qwen2-VL pipeline running
- ✅ Scene graph generated
- ✅ No API costs (all local)
- ✅ Ready to query!

**Next:** Query the scene with `python query_vlm_scene.py` (needs to be implemented)


