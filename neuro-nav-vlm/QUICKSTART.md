# VLM Neuro-Nav Quick Start

Get started with the VLM-powered scene graph pipeline in 5 minutes!

## Prerequisites

- Linux system with NVIDIA GPU (8GB+ VRAM)
- CUDA 12.6+ installed
- Python 3.8+

## Quick Setup (5 Steps)

### 1. Activate Environment
```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
```

### 2. Install VLM Dependencies
```bash
pip install -r requirements_vlm.txt
```

### 3. Create Data Symlink
```bash
ln -s /home/nick/Project_dir/neuro-nav/data data
```

### 4. Download Models (~5GB, one-time)
```bash
python download_models.py
# Choose option 5 (recommended models)
```

### 5. Test Setup
```bash
python test_vlm_setup.py
```

## Run the Pipeline

### Option 1: Automatic (Recommended)
```bash
./run_vlm_pipeline.sh
```

This automatically finds your latest scene map and runs all 4 steps.

### Option 2: Manual (If you need control)
```bash
# Set your scene directory
export CACHEDIR="data/outputs/2025-11-12/10-30-00"  # Change this!
export MAPFILE="${CACHEDIR}/room0/map/scene_map_cfslam.pkl.gz"

# Run pipeline
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}

python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir ${CACHEDIR} \
    --mapfile ${MAPFILE}
```

## View Results

```bash
# View scene graph (first 3 objects)
cat data/outputs/[latest]/room0/scene_graph.json | jq '.[0:3]'

# View debug visualizations
ls data/outputs/[latest]/room0/cfslam_captions_florence_debug/

# Query the scene interactively
python query_vlm_scene.py
```

## Need a Scene Map First?

If you don't have a scene map yet, run the SLAM pipeline first:

```bash
cd /home/nick/Project_dir/neuro-nav
source .venv/bin/activate
source ./use-cuda-126.sh
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

This creates the scene map that the VLM pipeline needs.

## What You Get

After running the pipeline, you'll have:

- ✅ **Scene Graph JSON** - Human-readable scene description
- ✅ **Object Captions** - Detailed descriptions from Florence-2
- ✅ **Refined Descriptions** - Enhanced by Qwen2-VL
- ✅ **Spatial Relationships** - Object relationships (on, in, etc.)
- ✅ **Debug Visualizations** - Images with captions overlaid
- ✅ **Feature Embeddings** - For similarity comparisons

## Common Issues

### "No scene map found"
→ Run SLAM first (see above) or provide the correct path

### "CUDA out of memory"
→ Use smaller models:
```bash
export FLORENCE_MODEL="microsoft/Florence-2-base"
export QWEN_MODEL="Qwen/Qwen2-VL-2B-Instruct"
./run_vlm_pipeline.sh
```

### "qwen-vl-utils not found"
→ Install it:
```bash
pip install qwen-vl-utils
```

### "Model download fails"
→ Check internet connection and disk space. Models are ~5GB total.

## Configuration

Customize by setting environment variables:

```bash
# Use smaller/faster models
export FLORENCE_MODEL="microsoft/Florence-2-base"
export QWEN_MODEL="Qwen/Qwen2-VL-2B-Instruct"

# Use larger/better models (needs more VRAM)
export FLORENCE_MODEL="microsoft/Florence-2-large"
export QWEN_MODEL="Qwen/Qwen2-VL-7B-Instruct"

# Change GPU
export DEVICE="cuda:1"

# Then run
./run_vlm_pipeline.sh
```

## Documentation

- **SETUP_GUIDE.md** - Detailed setup instructions
- **README_VLM.md** - Architecture and technical details
- **requirements_vlm.txt** - Python dependencies

## Model Sizes

| Model | Size | Speed | Quality | VRAM |
|-------|------|-------|---------|------|
| Florence-2-base | 230MB | Fast | Good | 2GB |
| Florence-2-large | 770MB | Medium | Better | 3GB |
| Qwen2-VL-2B | 4GB | Fast | Good | 4GB |
| Qwen2-VL-7B | 14GB | Slow | Best | 12GB |

**Recommended combo:** Florence-2-large + Qwen2-VL-2B (Total: 8GB VRAM)

## Next Steps

1. ✅ Run the pipeline
2. ✅ View the scene graph
3. ✅ Query the scene with `query_vlm_scene.py`
4. ✅ Compare with original neuro-nav
5. ✅ Integrate with your robot!

## Support

For issues or questions:
1. Check SETUP_GUIDE.md for detailed troubleshooting
2. Review error messages carefully
3. Verify GPU memory with `nvidia-smi`
4. Test imports with `python test_vlm_setup.py`

Happy mapping! 🚀🤖

