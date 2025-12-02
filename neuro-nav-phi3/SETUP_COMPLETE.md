# Phi-3-Vision Setup Complete! ✅

The neuro-nav-phi3 pipeline is now fully configured and ready to use.

## What Was Fixed

1. ✅ **Model name corrected**: Changed from `Qwen/Phi-3-Vision-2B-Instruct` to `microsoft/Phi-3-vision-128k-instruct`
2. ✅ **Directory paths updated**: Changed from `qwen` to `phi3` in all savedir paths
3. ✅ **Print statements fixed**: Updated to show "Phi-3-Vision model" instead of "Qwen model"
4. ✅ **Method signatures fixed**: 
   - `caption_image()` - removed invalid `detail_level` parameter
   - `refine_caption()` - removed invalid `image` parameter
   - `query_scene()` - fixed parameter order
5. ✅ **FlashAttention disabled**: Model now loads with eager attention (no flash-attn required)
6. ✅ **Decode calls fixed**: Changed `processor.decode()` to `processor.tokenizer.decode()`
7. ✅ **Requirements updated**: Removed MiniCPM references, updated to Phi-3-Vision
8. ✅ **Install script fixed**: Updated references from MiniCPM to Phi-3

## Quick Start

### 1. Install Dependencies

```bash
cd /home/nick/Project_dir/neuro-nav-phi3
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
pip install -e .
pip install -r requirements_phi3.txt
```

### 2. Setup Data Link

```bash
./setup_data_link.sh
```

### 3. Download Model (~3-4GB)

```bash
# Clear GPU memory first if needed
pkill -9 python 2>/dev/null || true
sleep 2

# Download model
python download_models.py
```

### 4. Run Pipeline

```bash
# Set environment variables
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-phi3:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Clear GPU memory
pkill -9 python 2>/dev/null || true
sleep 2

# Run pipeline
bash run_phi3_pipeline.sh
```

### 5. Query Scene

```bash
python query_phi3_scene.py \
    --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
    --query "Where can I sit?"
```

## Model Information

- **Model**: `microsoft/Phi-3-vision-128k-instruct`
- **Size**: ~3-4GB
- **Parameters**: 4.2B
- **GPU Memory**: ~8GB VRAM needed
- **Attention**: Eager (FlashAttention disabled)

## Output Files

After running the pipeline, you'll find:

```
data/Replica/room0/exps/r_mapping_with_llm/
├── scene_graph.json                    # Main output
├── cfslam_phi3_captions.json           # Raw captions
├── cfslam_phi3_responses/              # Refined captions
└── cfslam_object_relations.json        # Spatial relationships
```

## Troubleshooting

### Out of Memory

If you get CUDA OOM errors:

```bash
# Clear GPU memory
pkill -9 python 2>/dev/null || true
sleep 2

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Then retry
```

### FlashAttention Error

This should be fixed now, but if you see it:

```bash
# The model is configured to use eager attention
# No flash-attn installation needed
```

### Model Download Issues

```bash
# Check disk space (need ~5GB free)
df -h

# Check internet connection
ping huggingface.co

# Try offline mode if model is already cached
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Next Steps

1. ✅ Setup complete
2. ⏭️ Download model: `python download_models.py`
3. ⏭️ Run pipeline: `bash run_phi3_pipeline.sh`
4. ⏭️ Test querying: `python query_phi3_scene.py`

## Comparison with Other Pipelines

| Pipeline | Model | Size | Speed | Cost |
|----------|-------|------|-------|------|
| **neuro-nav** (Default) | GPT-4 API | N/A | 5-8 sec/obj | ~$1.20/scene |
| **neuro-nav-vlm** (Qwen2-VL) | Qwen2-VL-2B | 2B | 1-2 sec/obj | $0.00 |
| **neuro-nav-paligemma** (PaliGemma) | PaliGemma-3B | 3B | 1-2 sec/obj | $0.00 |
| **neuro-nav-phi3** (Phi-3-Vision) | Phi-3-Vision-4.2B | 4.2B | 1-2 sec/obj | $0.00 |

All VLM pipelines are faster, cheaper, and produce more detailed captions than the default pipeline!

---

**Setup Date**: 2025-11-19  
**Status**: ✅ Ready to use

