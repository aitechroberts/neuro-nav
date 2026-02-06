# Visual Quick Start Guide

```
╔════════════════════════════════════════════════════════════════════════╗
║                   Neuro-Nav VLM Pipeline                                ║
║                    (Qwen2-VL Version)                                   ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 The 3-Step Process

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   1. SETUP      │  →   │   2. PREPARE     │  →   │   3. RUN        │
│   (One Time)    │      │   (Per Scene)    │      │   (25-30 min)   │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

---

## Step 1: Setup (One Time)

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
pip install -e .
```

**✓ Done? You're 33% there!**

---

## Step 2: Prepare Scene Map (Per Scene)

```bash
# Find your scene maps
find /home/nick/Project_dir/neuro-nav/data -name "*.pkl.gz"

# Pick one, then:
SCENE="r_mapping_with_llm"
mkdir -p data/Replica/room0/exps/${SCENE}/map
ln -sf /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/${SCENE}/pcd_${SCENE}.pkl.gz \
       data/Replica/room0/exps/${SCENE}/map/scene_map_cfslam.pkl.gz
```

**✓ Done? You're 66% there!**

---

## Step 3: Run Pipeline

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

./run_everything.sh
```

**✓ Done? You're 100% there! 🎉**

---

## 📊 What Happens During Pipeline

```
Input: Scene Map (from neuro-nav)
   ↓
┌──────────────────────────────────────────────┐
│ Step 1: Extract Captions (10-15 min)        │
│ - Load Qwen2-VL model                        │
│ - Process each object crop                   │
│ - Generate raw captions                      │
└──────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────┐
│ Step 2: Refine Captions (5 min)             │
│ - Refine captions with context               │
│ - Extract object tags                        │
│ - Generate structured descriptions           │
└──────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────┐
│ Step 3: Build Scene Graph (10 min)          │
│ - Determine spatial relationships            │
│ - Create object-object connections           │
│ - Build scene understanding                  │
└──────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────┐
│ Step 4: Generate JSON (1 min)               │
│ - Create human-readable output               │
│ - Save scene graph                           │
└──────────────────────────────────────────────┘
   ↓
Output: scene_graph.json ✓
```

---

## 📁 Output Files

```
data/Replica/room0/exps/r_mapping_with_llm/
│
├── 🎯 scene_graph.json              ← YOUR MAIN OUTPUT!
│
├── cfslam_qwen_captions.json        ← Raw captions
├── cfslam_qwen_responses/           ← Refined captions
├── cfslam_object_relations.json     ← Relationships
└── cfslam_captions_qwen_debug/      ← Debug images
```

---

## 🔄 To Run Again (After Restart)

```
┌────────────┐
│  Terminal  │
│  Restart   │
└─────┬──────┘
      │
      ↓
┌─────────────────────────────────────┐
│ cd /home/nick/Project_dir/neuro-nav-vlm
│ source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
│ source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
│ export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
└─────────────────────────────────────┘
      │
      ↓
┌─────────────────────┐
│ ./run_everything.sh │
└─────────────────────┘
      │
      ↓
    ✓ Done
```

---

## ⚡ Quick Commands Reference

| Task | Command |
|------|---------|
| **Setup** | `pip install -e .` |
| **Run** | `./run_everything.sh` |
| **View output** | `cat data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \| jq '.'` |
| **Count objects** | `cat ... \| jq 'length'` |
| **Kill GPU** | `pkill -9 python` |
| **Check GPU** | `nvidia-smi` |

---

## 🆘 Common Errors

```
┌────────────────────────────────────────┐
│ Error: "ModuleNotFoundError"           │
├────────────────────────────────────────┤
│ Fix:                                   │
│   pip install -e .                     │
│   export PYTHONPATH=...:$PYTHONPATH    │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Error: "Scene map not found"           │
├────────────────────────────────────────┤
│ Fix:                                   │
│   See Step 2 above                     │
│   Check with: ls -la data/.../map/     │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Error: "CUDA out of memory"            │
├────────────────────────────────────────┤
│ Fix:                                   │
│   pkill -9 python                      │
│   Close other GPU programs             │
└────────────────────────────────────────┘
```

---

## 💰 Cost Comparison

```
┌─────────────────────────────────────────────────────┐
│              Original   →   VLM Version             │
├─────────────────────────────────────────────────────┤
│ Models:      4 models   →   1 model                │
│ Cost:        $2-5       →   $0                      │
│ Time:        45 min     →   25 min                  │
│ GPU RAM:     ~12 GB     →   ~8 GB                   │
│ Setup:       Complex    →   Simple                  │
└─────────────────────────────────────────────────────┘
```

---

## 📚 Need More Help?

```
┌────────────────────┬─────────────────────────────────┐
│ What You Need      │ File to Read                    │
├────────────────────┼─────────────────────────────────┤
│ Just run it        │ COPY_PASTE_COMMANDS.sh          │
│ Quick start        │ README_SIMPLE.md                │
│ Full guide         │ START_HERE.md                   │
│ Troubleshooting    │ START_HERE.md (bottom)          │
│ Architecture       │ README.md                       │
└────────────────────┴─────────────────────────────────┘
```

---

## ✨ Success Looks Like This

```bash
$ ./run_everything.sh

╔═══════════════════════════════════════════╗
║   Qwen2-VL Scene Graph Pipeline           ║
╚═══════════════════════════════════════════╝

✓ Scene map found

Step 1/4: Extract Node Captions with Qwen2-VL
[████████████████████] 100%

Step 2/4: Refine Node Captions with Qwen2-VL
[████████████████████] 100%

Step 3/4: Build Scene Graph with Relationships
[████████████████████] 100%

Step 4/4: Generate Scene Graph JSON
[████████████████████] 100%

╔═══════════════════════════════════════════╗
║        Pipeline Complete! 🎉              ║
╚═══════════════════════════════════════════╝

Output: data/.../scene_graph.json
```

---

**Ready to start?** → Open **[START_HERE.md](START_HERE.md)** 🚀


