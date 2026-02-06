# neuro-nav-internVL

3D Scene Graph Construction using **InternVL2-2B** Vision-Language Model

This repository replaces the old YOLO+CLIP+LLaVA+GPT-4 pipeline with a modern, unified VLM approach using InternVL2-2B from OpenGVLab.

---

## 🎯 What's Different?

### Old Pipeline (neuro-nav)
- **Models**: YOLO + SAM + CLIP + LLaVA (7B) + GPT-4 (API)
- **Parameters**: ~8B (local) + GPT-4 (cloud)
- **GPU Memory**: ~18GB
- **Cost**: ~$1.20 per scene (GPT-4 API)
- **Captions**: Short (1 sentence)

### New Pipeline (neuro-nav-internVL)
- **Models**: YOLO + SAM + InternVL2-2B
- **Parameters**: ~2.7B (fully local)
- **GPU Memory**: ~7GB
- **Cost**: $0.00 (no API calls)
- **Captions**: Detailed (200-400 words)

### Comparison with neuro-nav-vlm (Qwen2-VL)
- **InternVL2-2B**: Strong OCR, captioning, multi-lingual support
- **Qwen2-VL-2B**: Similar size, different architecture
- **Both**: ~2B parameters, ~4GB memory, excellent performance

InternVL2 offers a slightly different approach with strong OCR capabilities and multilingual support.

---

## 📋 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (use neuro-nav's venv)
cd /home/nick/Project_dir/neuro-nav
source .venv/bin/activate
source use-cuda-126.sh

# Navigate to neuro-nav-internVL
cd /home/nick/Project_dir/neuro-nav-internVL

# Install dependencies
bash install_internvl.sh

# Install package
pip install -e .

# Setup data link (to share SLAM outputs)
bash setup_data_link.sh
```

### 2. Download Models

```bash
# Download InternVL2-2B (~4-5GB)
python download_models.py
```

### 3. Generate SLAM Scene Map

```bash
# If you haven't run SLAM yet, do it in neuro-nav:
cd ../neuro-nav
python conceptgraph/slam/rerun_realtime_mapping.py \
  --config-name=rerun_simple_test end=30

# This creates: data/Replica/room0/exps/r_mapping_with_llm/map/scene_map_cfslam.pkl.gz
```

### 4. Run InternVL2 Pipeline

```bash
cd /home/nick/Project_dir/neuro-nav-internVL
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run complete pipeline (4 stages)
bash run_internvl_pipeline.sh
```

### 5. Query the Scene

```bash
# Single query
python query_internvl_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I sit?"

# Interactive mode
python query_internvl_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json
```

---

## 🔧 Pipeline Stages

The InternVL2 pipeline consists of 4 stages:

### Stage 1: Extract Node Captions
```bash
python conceptgraph/scenegraph/build_scenegraph_internvl.py extract-node-captions \
  --mapfile data/.../scene_map_cfslam.pkl.gz \
  --cachedir data/.../
```
- Uses InternVL2 to caption each detected object
- Generates detailed 200-400 word descriptions
- Saves to `cfslam_internvl_captions.json`

### Stage 2: Refine Node Captions
```bash
python conceptgraph/scenegraph/build_scenegraph_internvl.py refine-node-captions \
  --mapfile data/.../scene_map_cfslam.pkl.gz \
  --cachedir data/.../
```
- Uses InternVL2 to consolidate multiple views
- Extracts object tags and summaries
- Saves to `cfslam_internvl_responses.json`

### Stage 3: Build Scene Graph
```bash
python conceptgraph/scenegraph/build_scenegraph_internvl.py build-scenegraph \
  --mapfile data/.../scene_map_cfslam.pkl.gz \
  --cachedir data/.../
```
- Merges 3D objects using geometric reasoning
- Extracts spatial relationships with InternVL2
- Builds connected scene graph

### Stage 4: Generate JSON
```bash
python conceptgraph/scenegraph/build_scenegraph_internvl.py generate-scenegraph-json \
  --mapfile data/.../scene_map_cfslam.pkl.gz \
  --cachedir data/.../
```
- Exports scene graph to JSON format
- Ready for querying and visualization

---

## 📊 Model Information

### InternVL2-2B Specs
- **Parameters**: 2 billion
- **Architecture**: Vision-Language Transformer
- **Input**: Images + Text
- **Output**: Text (captions, answers, descriptions)
- **GPU Memory**: ~4GB (fp16/bf16)
- **Source**: OpenGVLab/Shanghai AI Lab

### InternVL2 Capabilities
- ✅ Image captioning (detailed, 200+ words)
- ✅ Visual question answering (VQA)
- ✅ Spatial relationship extraction
- ✅ Scene understanding
- ✅ OCR and text recognition
- ✅ Multi-lingual support (80+ languages)
- ✅ Open-domain reasoning

---

## 🗂️ Directory Structure

```
neuro-nav-internVL/
├── conceptgraph/
│   ├── vlm/
│   │   ├── __init__.py
│   │   └── internvl2_model.py          # InternVL2 wrapper
│   ├── scenegraph/
│   │   └── build_scenegraph_internvl.py # Main pipeline script
│   ├── slam/                            # Shared from neuro-nav
│   └── utils/                           # Shared from neuro-nav
├── data/                                # Symlink to neuro-nav/data
├── requirements_internvl.txt            # InternVL2 dependencies
├── setup.py                             # Package setup
├── download_models.py                   # Model downloader
├── install_internvl.sh                  # Install script
├── setup_data_link.sh                   # Data symlink script
├── run_internvl_pipeline.sh             # Pipeline runner
├── query_internvl_scene.py              # Query interface
└── README.md                            # This file
```

---

## 🔍 Usage Examples

### Example 1: Find Seating
```bash
python query_internvl_scene.py \
  --scene-graph data/.../scene_graph.json \
  --query "Where can I sit?"

# Output:
# "You can sit on the white sofa with a smooth, curved backrest 
# positioned at (2.8, -0.7, -0.7). The sofa features a minimalist
# design with decorative pillows..."
```

### Example 2: Locate Objects
```bash
python query_internvl_scene.py \
  --scene-graph data/.../scene_graph.json \
  --query "Where is the laptop?"

# Output:
# "The laptop can be found at Object 5, positioned at (1.2, 0.3, 0.8).
# It is a silver laptop with a standard keyboard layout..."
```

### Example 3: Scene Description
```bash
python query_internvl_scene.py \
  --scene-graph data/.../scene_graph.json \
  --query "Describe the room"

# Output:
# "The room is a modern living space featuring a white minimalist
# sofa as the central piece of furniture, positioned at (2.8, -0.7, -0.7)..."
```

---

## 🐛 Troubleshooting

### Model Loading Issues
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-2B
python download_models.py
```

### GPU Out of Memory
```bash
# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Or reduce batch size / use 8-bit quantization
# Edit internvl2_model.py: load_in_8bit=True
```

### Import Errors
```bash
# Ensure package is installed
pip install -e .

# Set PYTHONPATH
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
```

### SLAM Scene Map Not Found
```bash
# Generate scene map in neuro-nav first
cd ../neuro-nav
python conceptgraph/slam/rerun_realtime_mapping.py \
  --config-name=rerun_simple_test end=30

# Then link it
cd ../neuro-nav-internVL
bash setup_data_link.sh
```

---

## 📈 Performance

### Speed
- **Caption extraction**: ~1-2 seconds per object
- **Caption refinement**: ~0.5-1 second per object
- **Relationship extraction**: ~0.3 seconds per pair
- **Query response**: ~0.5-1 second per query

### Memory Usage
- **Model loading**: ~4GB GPU VRAM (bf16)
- **Peak inference**: ~6-7GB GPU VRAM
- **Minimum GPU**: 8GB VRAM (e.g., RTX 3060)

### Accuracy
- InternVL2-2B achieves competitive results on vision-language benchmarks
- Strong performance on OCR, captioning, and VQA tasks
- Comparable to Qwen2-VL-2B in most tasks

---

## 🔄 Comparison: InternVL2 vs Qwen2-VL

| Feature | InternVL2-2B | Qwen2-VL-2B |
|---------|-------------|-------------|
| **Parameters** | 2B | 2B |
| **GPU Memory** | ~4GB | ~4GB |
| **Captioning** | Excellent | Excellent |
| **OCR** | **Strong** | Good |
| **Multi-lingual** | **80+ langs** | Chinese focus |
| **API** | HF Transformers | HF Transformers |
| **Speed** | Fast | Fast |
| **Open Source** | ✅ Yes | ✅ Yes |

**When to use InternVL2:**
- Need strong OCR capabilities
- Multi-lingual support required
- Alternative architecture preference

**When to use Qwen2-VL:**
- Chinese language focus
- Specific Qwen ecosystem preference

---

## 🚀 Next Steps

1. **Run on your own data**:
   - Replace Replica dataset with your robot's RGB-D captures
   - Run SLAM on your data
   - Run InternVL2 pipeline

2. **Integrate with navigation**:
   - Use `query_internvl_scene.py` API in your robot code
   - Parse scene graph JSON for path planning
   - Query in real-time for dynamic navigation

3. **Optimize for your GPU**:
   - Try 8-bit quantization for lower memory
   - Adjust batch sizes for your hardware
   - Profile inference speed

4. **Compare with Qwen2-VL**:
   - Run both pipelines on same data
   - Compare caption quality
   - Evaluate query response quality

---

## 📚 Additional Resources

- **InternVL2 Paper**: [arXiv:2404.16821](https://arxiv.org/abs/2404.16821)
- **HuggingFace Model**: [OpenGVLab/InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B)
- **Documentation**: See `QUICKSTART.md` for step-by-step guide
- **Comparison**: See `PIPELINE_COMPARISON.md` for detailed comparison with old pipeline

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{internvl2,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation in this repo
- Compare with neuro-nav and neuro-nav-vlm implementations

**Created**: 2025-11-19  
**Last Updated**: 2025-11-19  
**Version**: 1.0.0

