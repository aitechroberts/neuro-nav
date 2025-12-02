# 📖 READ ME FIRST

**Welcome to Neuro-Nav VLM!** This folder contains everything you need.

---

## 🚀 I Want to Run It RIGHT NOW

**Copy-paste these 3 commands:**

```bash
cd /home/nick/Project_dir/neuro-nav-vlm && \
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate && \
pip install -e . && \
mkdir -p data/Replica/room0/exps/r_mapping_with_llm/map && \
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/pcd_r_mapping_with_llm.pkl.gz \
       data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz && \
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh && \
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH && \
./run_everything.sh
```

**That's it!** Wait ~25-30 minutes and you're done.

---

## 📚 Pick Your Documentation Style

### Option 1: Visual Learner
👉 **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** - Flowcharts and diagrams

### Option 2: Just Give Me Commands
👉 **[COPY_PASTE_COMMANDS.sh](COPY_PASTE_COMMANDS.sh)** - Commands only

### Option 3: Simple Explanation
👉 **[README_SIMPLE.md](README_SIMPLE.md)** - Quick start (3 commands)

### Option 4: Complete Guide
👉 **[START_HERE.md](START_HERE.md)** - Detailed walkthrough

### Option 5: Not Sure Which to Read?
👉 **[WHICH_FILE_TO_READ.md](WHICH_FILE_TO_READ.md)** - Guide to guides

---

## 📁 All Documentation Files

```
📖_READ_ME_FIRST.md              ← You are here!
│
├── 🚀 Quick Start
│   ├── VISUAL_GUIDE.md          ← Flowcharts & visuals
│   ├── README_SIMPLE.md         ← 3-command start
│   └── COPY_PASTE_COMMANDS.sh   ← Just commands
│
├── 📖 Complete Guides
│   ├── START_HERE.md            ← Best starting point
│   ├── WHICH_FILE_TO_READ.md    ← Guide to documentation
│   └── INSTALLATION_CHECKLIST.md ← Checklist format
│
├── 🔧 Technical Docs
│   ├── README.md                ← Full documentation
│   ├── README_VLM.md            ← VLM architecture
│   ├── QWEN_ONLY_SETUP.md       ← Qwen2-VL details
│   └── SETUP_GUIDE.md           ← Detailed setup
│
└── 🚀 Scripts
    ├── run_everything.sh        ← Run complete pipeline
    ├── setup_data_link.sh       ← Link data directory
    ├── download_models.py       ← Download VLM models
    └── query_vlm_scene.py       ← Query the scene
```

---

## ⏱️ Time Investment

| If you have... | Read this... | Time |
|----------------|--------------|------|
| 30 seconds | **This file** | 30s |
| 2 minutes | **[README_SIMPLE.md](README_SIMPLE.md)** | 2m |
| 5 minutes | **[VISUAL_GUIDE.md](VISUAL_GUIDE.md)** | 5m |
| 10 minutes | **[START_HERE.md](START_HERE.md)** | 10m |
| 30 minutes | **[README.md](README.md)** | 30m |

---

## 🎯 What This Does

```
Input:  Scene map from neuro-nav (3D objects + images)
   ↓
Process: Qwen2-VL labels, captions, and understands relationships
   ↓
Output: scene_graph.json (queryable scene understanding)
```

**Replaces:** YOLO + CLIP + LLaVA + GPT-4  
**With:** Just Qwen2-VL  
**Cost:** $0 (was $2-5)  
**Time:** 25 min (was 45 min)  

---

## ✅ Prerequisites (You Should Have)

- [x] GPU with 8GB+ VRAM
- [x] Python 3.10
- [x] CUDA 12.6
- [x] neuro-nav working
- [x] Scene maps generated (`*.pkl.gz` files)
- [x] Qwen2-VL model downloaded (~4GB)

**Missing something?** → See [START_HERE.md](START_HERE.md)

---

## 🆘 Quick Troubleshooting

### Error: "ModuleNotFoundError"
```bash
pip install -e .
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
```

### Error: "Scene map not found"
```bash
find /home/nick/Project_dir/neuro-nav/data -name "*.pkl.gz"
# Then see Step 2 in VISUAL_GUIDE.md
```

### Error: "CUDA out of memory"
```bash
pkill -9 python
nvidia-smi  # Check what's using GPU
```

**More problems?** → See "Troubleshooting" in [START_HERE.md](START_HERE.md)

---

## 🎓 Recommended Learning Path

```
1. Read THIS FILE (you're here!)              [30 seconds]
   ↓
2. Read VISUAL_GUIDE.md                       [5 minutes]
   ↓
3. Copy-paste the 3 commands above            [1 minute]
   ↓
4. Wait for pipeline to complete              [25-30 minutes]
   ↓
5. View your scene graph!                     [DONE! 🎉]
```

**Then optionally:** Read [START_HERE.md](START_HERE.md) to understand details.

---

## 💡 Pro Tips

1. **First time?** Just run the 3 commands at the top. Understand later.
2. **Stuck?** Read [WHICH_FILE_TO_READ.md](WHICH_FILE_TO_READ.md) to find help.
3. **Want to understand?** Read [START_HERE.md](START_HERE.md) step-by-step.
4. **Just commands?** Use [COPY_PASTE_COMMANDS.sh](COPY_PASTE_COMMANDS.sh).
5. **Visual learner?** Start with [VISUAL_GUIDE.md](VISUAL_GUIDE.md).

---

## 🎁 What You Get

After running, you'll have:

```
✓ Qwen2-VL scene understanding
✓ Object labels and captions
✓ Spatial relationships
✓ Queryable scene graph (JSON)
✓ Debug visualizations
✓ $0 API costs (all local)
```

---

## 📞 Next Steps

1. **Run the pipeline** (commands at top)
2. **View results:** `cat data/.../scene_graph.json | jq '.'`
3. **Query the scene:** `python query_vlm_scene.py` (coming soon)
4. **Iterate:** Try different scene maps!

---

## 🚀 Ready?

**Fastest path:** Copy the 3 commands at the top → paste → wait → done!

**Careful path:** Read [START_HERE.md](START_HERE.md) → understand → run → done!

**Visual path:** Read [VISUAL_GUIDE.md](VISUAL_GUIDE.md) → see flowchart → run → done!

---

**Your choice! All paths lead to success.** 🎯

---

<div align="center">

**Made with ❤️ to replace YOLO+CLIP+LLaVA+GPT-4 with just Qwen2-VL**

[START_HERE.md](START_HERE.md) • [VISUAL_GUIDE.md](VISUAL_GUIDE.md) • [README_SIMPLE.md](README_SIMPLE.md)

</div>


