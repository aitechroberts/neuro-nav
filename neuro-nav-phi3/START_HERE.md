# 🚀 START HERE - MiniCPM-V-2.6 Pipeline

## Copy-Paste This:

```bash
# Setup (one-time, 10 minutes)
cd /home/nick/Project_dir/neuro-nav-miniCPM
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
bash install_minicpm.sh && pip install -e . && bash setup_data_link.sh
python download_models.py

# Run Pipeline (15-30 minutes)
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-miniCPM:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash run_minicpm_pipeline.sh

# Query Scene (instant)
python query_minicpm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I sit?"
```

---

## What You Get

- ✅ **MiniCPM-V-2.6** (2.4B parameters)
- ✅ Efficient, fast inference
- ✅ Low memory usage (~4-5GB VRAM)
- ✅ Detailed captions (100-300 words)
- ✅ No API costs
- ✅ Works offline

---

## Compare with Qwen2-VL

You now have **2 VLMs** to compare:
1. **Qwen2-VL-2B** (neuro-nav-vlm)
2. **MiniCPM-V-2.6** (neuro-nav-miniCPM)

Run both and compare outputs!

---

Created: 2025-11-19

