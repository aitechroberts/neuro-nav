# Qwen2-VL Only Setup

**Simplified setup using only Qwen2-VL for everything!**

## What Changed?

Instead of using Florence-2 + Qwen2-VL, we now use **just Qwen2-VL** for:
- ✅ Object captioning
- ✅ Caption refinement
- ✅ Spatial relationships
- ✅ Scene querying

**Why?** Qwen2-VL is more capable and you already have it downloaded!

---

## Quick Setup

### 1. You Already Have Qwen2-VL! ✓

The model is downloaded at: `~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct`

### 2. Setup Data Link

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
./setup_data_link.sh
```

### 3. Run the Pipeline

```bash
./run_vlm_pipeline.sh
```

That's it! The pipeline will use Qwen2-VL for everything.

---

## What the Pipeline Does

### Step 1: Extract Captions (Qwen2-VL)
- Processes each object from multiple viewpoints
- Generates detailed captions
- Saves to `cfslam_qwen_captions.json`

### Step 2: Refine Captions (Qwen2-VL)
- Consolidates captions from multiple views
- Extracts object tags and summaries
- Saves to `cfslam_qwen_responses/`

### Step 3: Build Scene Graph (Qwen2-VL)
- Determines spatial relationships
- Creates scene graph structure
- Saves to `cfslam_object_relations.json`

### Step 4: Generate JSON
- Creates human-readable scene graph
- Saves to `scene_graph.json`

---

## Output Files

```
your_cache_dir/
├── scene_graph.json                    # Main output
├── cfslam_qwen_captions.json          # Raw captions
├── cfslam_qwen_responses/             # Refined captions per object
├── cfslam_object_relations.json       # Spatial relationships
├── cfslam_captions_qwen_debug/        # Debug visualizations
└── map/scene_map_cfslam_pruned.pkl.gz # Pruned 3D map
```

---

## Performance

Using only Qwen2-VL:
- **GPU Memory**: ~6-8GB
- **Processing Time**: ~20-30 minutes for 30 frames
- **Quality**: Excellent (Qwen2-VL is very capable!)
- **Cost**: $0 (completely local)

---

## Run Commands

### Full Pipeline
```bash
./run_vlm_pipeline.sh
```

### Individual Steps
```bash
# Step 1: Extract captions
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode extract-node-captions \
    --cachedir data/outputs/[timestamp] \
    --mapfile data/outputs/[timestamp]/room0/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 2: Refine captions
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode refine-node-captions \
    --cachedir data/outputs/[timestamp] \
    --mapfile data/outputs/[timestamp]/room0/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 3: Build scene graph
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode build-scenegraph \
    --cachedir data/outputs/[timestamp] \
    --mapfile data/outputs/[timestamp]/room0/map/scene_map_cfslam.pkl.gz \
    --qwen-model Qwen/Qwen2-VL-2B-Instruct \
    --device cuda:0

# Step 4: Generate JSON
python conceptgraph/scenegraph/build_scenegraph_vlm.py \
    --mode generate-scenegraph-json \
    --cachedir data/outputs/[timestamp] \
    --mapfile data/outputs/[timestamp]/room0/map/scene_map_cfslam.pkl.gz
```

---

## Query the Scene

After building the scene graph:

```bash
python query_vlm_scene.py
```

Example queries:
- "Where is the chair?"
- "What objects are on the table?"
- "Describe the room"
- "Find the laptop"

---

## Troubleshooting

### "No scene map found"
Run the SLAM pipeline first:
```bash
cd /home/nick/Project_dir/neuro-nav
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

### "CUDA out of memory"
Reduce the number of detections per object:
```bash
python ... --max-detections-per-object 5  # Default is 10
```

### Model not loading
Check that Qwen2-VL is downloaded:
```bash
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct
```

---

## Advantages of Qwen-Only

✅ **Simpler**: One model instead of two  
✅ **More capable**: Qwen2-VL handles vision + language better  
✅ **Consistent**: Same model for all steps  
✅ **Working**: Florence-2 had compatibility issues  
✅ **Efficient**: Less model switching = faster  

---

## Next Steps

1. ✅ Setup data link
2. ✅ Run pipeline
3. ✅ View scene graph
4. ✅ Query the scene
5. ✅ Compare with original neuro-nav

Happy mapping with Qwen2-VL! 🚀



