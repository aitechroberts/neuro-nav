# 🚀 Run Me First - Qwen2-VL Setup

## You're Ready to Go!

Qwen2-VL is already downloaded. Let's run the pipeline!

---

## Step 1: Setup Data (30 seconds)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
./setup_data_link.sh
```

This creates a link to your neuro-nav data.

---

## Step 2: Run the Pipeline (~20-30 minutes)

```bash
./run_vlm_pipeline.sh
```

**What happens:**
1. Extracts captions with Qwen2-VL (~10 min)
2. Refines captions (~5 min)  
3. Builds scene graph (~10 min)
4. Generates JSON (~1 min)

---

## Step 3: View Results

```bash
# View scene graph (first 3 objects)
cat data/outputs/*/room0/scene_graph.json | jq '.[0:3]'

# View debug images
ls data/outputs/*/room0/cfslam_captions_qwen_debug/
```

---

## Step 4: Query the Scene

```bash
python query_vlm_scene.py
```

Try asking:
- "Where is the chair?"
- "What's on the desk?"
- "Describe the room"

---

## Need a Scene First?

If you don't have a scene map yet:

```bash
cd /home/nick/Project_dir/neuro-nav
source .venv/bin/activate
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

This takes ~5-10 minutes and creates the scene map.

---

## What You Have

✅ **Qwen2-VL-2B** - Downloaded and ready  
✅ **Modified pipeline** - Uses only Qwen2-VL  
✅ **All scripts** - Ready to run  

---

## Commands Summary

```bash
# 1. Setup (once)
./setup_data_link.sh

# 2. Run pipeline
./run_vlm_pipeline.sh

# 3. Query scene
python query_vlm_scene.py
```

---

## GPU Requirements

- **Memory**: 6-8GB VRAM
- **Model**: Qwen2-VL-2B-Instruct (~4GB)
- **Processing**: ~20-30 min for 30 frames

---

## Troubleshooting

### "Command not found"
```bash
chmod +x run_vlm_pipeline.sh setup_data_link.sh query_vlm_scene.py
```

### "No scene map"
Run SLAM first (see above)

### "CUDA out of memory"
Close other GPU applications or reduce `--max-detections-per-object 5`

---

**Ready? Run this:**

```bash
./setup_data_link.sh && ./run_vlm_pipeline.sh
```

That's it! 🎉



