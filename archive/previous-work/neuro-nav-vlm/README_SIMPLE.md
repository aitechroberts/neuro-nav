# Neuro-Nav VLM - Simple Guide

**Use Qwen2-VL for scene understanding - no API costs!**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Setup (one time)
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
pip install -e .

# 2. Prepare scene map (one time per scene)
mkdir -p data/Replica/room0/exps/r_mapping_with_llm/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/pcd_r_mapping_with_llm.pkl.gz \
       data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz

# 3. Run pipeline (~25-30 min)
./run_everything.sh
```

**That's it!** Scene graph will be saved to:
`data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json`

---

## 📖 Need More Details?

**Read: [`START_HERE.md`](START_HERE.md)** - Complete step-by-step guide

---

## 🔄 Run Again (After Restart)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
./run_everything.sh
```

---

## 📁 Files You Need

| File | Purpose |
|------|---------|
| **START_HERE.md** | Complete setup guide |
| **README_SIMPLE.md** | This file - quick reference |
| **run_everything.sh** | Automated pipeline runner |
| **query_vlm_scene.py** | Query the scene (interactive) |

---

## ⚙️ What It Does

1. **Uses Qwen2-VL** (already downloaded, ~4GB)
2. **Processes scene map** from neuro-nav
3. **Generates captions** for each object
4. **Builds scene graph** with spatial relationships
5. **Creates JSON** output for queries

**No API costs. All local. ~25-30 minutes.**

---

## 🆘 Problems?

### "ModuleNotFoundError"
```bash
pip install -e .
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
```

### "Scene map not found"
```bash
# Find your scene maps
find /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_* -name "*.pkl.gz"

# Then link it (see Step 2 above)
```

### "CUDA out of memory"
```bash
# Close other GPU programs or reduce detections
python ... --max-detections-per-object 5
```

**See START_HERE.md for detailed troubleshooting.**

---

## 📊 Output

After running, you get:

```
data/Replica/room0/exps/r_mapping_with_llm/
├── scene_graph.json                    ← Main output!
├── cfslam_qwen_captions.json          ← Raw captions
├── cfslam_qwen_responses/             ← Refined captions
├── cfslam_object_relations.json       ← Spatial relationships
└── cfslam_captions_qwen_debug/        ← Debug images
```

**View it:**
```bash
cat data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json | jq '.'
```

---

## 🎯 Different Scene Map?

```bash
# Change this in run_everything.sh or pass as argument:
./run_everything.sh r_mapping_with_gpt4v

# Or r_direction1_semantic, etc.
```

---

## 💡 What's Different from Original?

| Original | VLM Version |
|----------|-------------|
| YOLO + CLIP + LLaVA + GPT-4 | Just Qwen2-VL |
| $2-5 per scene (API) | $0 (local) |
| 45 minutes | 25 minutes |
| 4 separate models | 1 model |

**Same output format. Same quality or better!**

---

**Ready?** → `./run_everything.sh` 🚀


