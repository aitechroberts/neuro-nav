# MiniCPM-V Pipeline Quickstart

## 3 Commands to Get Started

```bash
# 1. Setup (one-time, 10 minutes)
cd /home/nick/Project_dir/neuro-nav-miniCPM
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
bash install_minicpm.sh && pip install -e . && bash setup_data_link.sh
python download_models.py

# 2. Run Pipeline (15-30 minutes)
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-miniCPM:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash run_minicpm_pipeline.sh

# 3. Query Scene (instant)
python query_minicpm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I sit?"
```

---

## What is This?

**neuro-nav-miniCPM** uses **MiniCPM-V-2.6** (2.4B parameter VLM) to:
- Caption objects in 3D scenes
- Build spatial scene graphs
- Answer questions about the environment

**Advantages:**
- 2.4B parameters (efficient!)
- Fast inference
- Low memory usage
- No API costs

---

## Complete Setup Steps

### Step 1: Environment (2 min)
```bash
cd /home/nick/Project_dir/neuro-nav-miniCPM
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
bash install_minicpm.sh
pip install -e .
bash setup_data_link.sh
```

### Step 2: Download Model (5-10 min)
```bash
python download_models.py
# Downloads ~4-5GB to ~/.cache/huggingface/
```

### Step 3: Run Pipeline (15-30 min)
```bash
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-miniCPM:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash run_minicpm_pipeline.sh
```

### Step 4: Query (instant)
```bash
python query_minicpm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I work?"
```

---

## Example Queries

```bash
# Find seating
python query_minicpm_scene.py --scene-graph data/.../scene_graph.json \
  --query "Where can I sit?"

# Find objects
python query_minicpm_scene.py --scene-graph data/.../scene_graph.json \
  --query "Where is the table?"

# Room description
python query_minicpm_scene.py --scene-graph data/.../scene_graph.json \
  --query "Describe the room"

# Task-oriented
python query_minicpm_scene.py --scene-graph data/.../scene_graph.json \
  --query "Where can I place a cup?"
```

---

## Troubleshooting

**Out of disk space?**
```bash
df -h /home/nick
pip cache purge
```

**GPU out of memory?**
```bash
pkill -9 python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Model not downloading?**
```bash
rm -rf ~/.cache/huggingface/hub/models--openbmb--MiniCPM-V-2_6
python download_models.py
```

---

**Ready to go!** 🚀

For full documentation, see `README.md`

