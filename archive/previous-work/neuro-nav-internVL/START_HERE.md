# 🚀 START HERE - InternVL2 Pipeline

## Copy-Paste This:

```bash
# Setup (one-time, 10 minutes)
cd /home/nick/Project_dir/neuro-nav-internVL
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
bash install_internvl.sh && pip install -e . && bash setup_data_link.sh
python download_models.py

# Run Pipeline (15-30 minutes)
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash run_internvl_pipeline.sh

# Query Scene (instant)
python query_internvl_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I sit?"
```

---

## What is This?

**neuro-nav-internVL** uses **InternVL2-2B** (2 billion parameter VLM) to:
- Caption objects in 3D scenes (200-400 word descriptions)
- Build spatial scene graphs
- Answer questions about the environment

**vs old pipeline:**
- 3x smaller (2.7B vs 8B params)
- 2-3x faster
- $0 cost (vs $1.20/scene for GPT-4)
- Fully offline

**vs neuro-nav-vlm (Qwen2-VL):**
- Same size (~2B params)
- Different architecture (InternVL2 vs Qwen2)
- Strong OCR + multi-lingual support

---

## Requirements

✅ CUDA GPU with 8GB+ VRAM (e.g., RTX 3060)  
✅ Python 3.8+  
✅ SLAM scene map from `neuro-nav`

---

## Documentation

- **Quickstart**: See `QUICKSTART.md` (step-by-step guide)
- **Full README**: See `README.md` (complete documentation)
- **Comparison**: See `PIPELINE_COMPARISON.md` in neuro-nav-vlm folder

---

## Need Help?

1. **Model not downloading?**
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-2B
   python download_models.py
   ```

2. **Scene map not found?**
   ```bash
   cd ../neuro-nav
   python conceptgraph/slam/rerun_realtime_mapping.py \
     --config-name=rerun_simple_test end=30
   ```

3. **GPU out of memory?**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   pkill -9 python  # Kill old processes
   ```

---

## Quick Comparison

| Feature | InternVL2 | Qwen2-VL | Old (GPT-4) |
|---------|-----------|----------|-------------|
| Params | 2B | 2B | 8B + API |
| Memory | 7GB | 7GB | 18GB |
| Cost | $0 | $0 | $1.20/scene |
| Speed | 1-2s/obj | 1-2s/obj | 5-8s/obj |
| OCR | **Strong** | Good | N/A |

---

**Choose InternVL2 for**: Strong OCR, multi-lingual, alternative architecture  
**Choose Qwen2-VL for**: Chinese focus, proven in your existing setup  
**Both are excellent!**

---

Created: 2025-11-19

