# Run InternVL2 and Compare with Qwen2-VL

## 🚀 Quick Start: Run Both and Compare

### Step 1: Run InternVL2 Pipeline (15-30 minutes)

```bash
# Navigate and setup
cd /home/nick/Project_dir/neuro-nav-internVL
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh

# Set environment
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run pipeline
bash run_internvl_pipeline.sh
```

**Output Location:**
```
data/Replica/room0/exps/r_mapping_with_llm/
├── cfslam_internvl_captions.json      # InternVL2 captions
├── cfslam_internvl_responses.json     # Refined captions
└── scene_graph.json                   # Final scene graph
```

---

### Step 2: Run Qwen2-VL Pipeline (if not done yet)

```bash
# Navigate to Qwen folder
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh

# Set environment
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run pipeline
bash run_vlm_pipeline.sh
```

**Output Location:**
```
data/Replica/room0/exps/r_mapping_with_llm/
├── cfslam_qwen_captions.json          # Qwen2-VL captions
├── cfslam_qwen_responses.json         # Refined captions
└── scene_graph.json                   # Final scene graph
```

---

## 📊 Compare Outputs

### Method 1: Automated Comparison Script

Create a comparison script:

```bash
cat > /home/nick/Project_dir/compare_vlms.py << 'EOF'
#!/usr/bin/env python3
"""Compare InternVL2 and Qwen2-VL outputs"""

import json
from pathlib import Path

DATA_DIR = Path("/home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def compare_captions():
    """Compare initial captions"""
    print("\n" + "="*70)
    print("CAPTION COMPARISON")
    print("="*70 + "\n")
    
    internvl_caps = load_json(DATA_DIR / "cfslam_internvl_captions.json")
    qwen_caps = load_json(DATA_DIR / "cfslam_qwen_captions.json")
    
    print(f"InternVL2: {len(internvl_caps)} objects")
    print(f"Qwen2-VL: {len(qwen_caps)} objects")
    
    # Compare first object
    if internvl_caps and qwen_caps:
        print(f"\n{'─'*70}")
        print("OBJECT 0 COMPARISON")
        print(f"{'─'*70}\n")
        
        print("InternVL2 Caption (first view):")
        print("─" * 70)
        internvl_cap = internvl_caps[0]['captions'][0]
        print(internvl_cap[:300] + "..." if len(internvl_cap) > 300 else internvl_cap)
        
        print("\nQwen2-VL Caption (first view):")
        print("─" * 70)
        qwen_cap = qwen_caps[0]['captions'][0]
        print(qwen_cap[:300] + "..." if len(qwen_cap) > 300 else qwen_cap)
        
        # Stats
        internvl_words = sum(len(c.split()) for c in internvl_caps[0]['captions'])
        qwen_words = sum(len(c.split()) for c in qwen_caps[0]['captions'])
        
        print(f"\n{'─'*70}")
        print("STATISTICS (Object 0):")
        print(f"{'─'*70}")
        print(f"InternVL2: {internvl_words} total words across {len(internvl_caps[0]['captions'])} views")
        print(f"Qwen2-VL:  {qwen_words} total words across {len(qwen_caps[0]['captions'])} views")
        print(f"Word ratio: InternVL2 is {(internvl_words/qwen_words):.2f}x compared to Qwen2-VL")

def compare_refinements():
    """Compare refined captions"""
    print("\n" + "="*70)
    print("REFINED CAPTION COMPARISON")
    print("="*70 + "\n")
    
    internvl_ref = load_json(DATA_DIR / "cfslam_internvl_responses.json")
    qwen_ref = load_json(DATA_DIR / "cfslam_qwen_responses.json")
    
    print(f"InternVL2 responses: {len(internvl_ref)}")
    print(f"Qwen2-VL responses: {len(qwen_ref)}")
    
    # Compare first object
    if internvl_ref and qwen_ref:
        print(f"\n{'─'*70}")
        print("OBJECT 0 REFINEMENT")
        print(f"{'─'*70}\n")
        
        print("InternVL2 Object Tag:")
        print(f"  → {internvl_ref[0].get('object_tag', 'N/A')}")
        
        print("\nQwen2-VL Object Tag:")
        print(f"  → {qwen_ref[0].get('object_tag', 'N/A')}")
        
        print("\nInternVL2 Summary:")
        print(f"  → {internvl_ref[0].get('summary', 'N/A')[:100]}...")
        
        print("\nQwen2-VL Summary:")
        print(f"  → {qwen_ref[0].get('summary', 'N/A')[:100]}...")

def compare_scene_graphs():
    """Compare final scene graphs"""
    print("\n" + "="*70)
    print("SCENE GRAPH COMPARISON")
    print("="*70 + "\n")
    
    # Note: Both pipelines overwrite scene_graph.json
    # We need to save them separately or compare during runtime
    print("⚠️  NOTE: Both pipelines write to the same scene_graph.json")
    print("To compare, you need to:")
    print("1. Copy scene_graph.json after running InternVL2:")
    print("   cp data/.../scene_graph.json data/.../scene_graph_internvl.json")
    print("2. Run Qwen2-VL pipeline")
    print("3. Compare the two files")
    print("")
    
    # Check which one exists
    sg_path = DATA_DIR / "scene_graph.json"
    if sg_path.exists():
        sg = load_json(sg_path)
        print(f"Current scene_graph.json has {len(sg)} objects")
        
        # Show first object
        if sg:
            obj = sg[0]
            print(f"\nObject 0:")
            print(f"  ID: {obj.get('id')}")
            print(f"  Tag: {obj.get('object_tag', 'N/A')}")
            print(f"  Position: {obj.get('bbox_center', 'N/A')}")
            print(f"  Caption length: {len(obj.get('caption', ''))} chars")

def main():
    print("\n" + "="*70)
    print("InternVL2 vs Qwen2-VL COMPARISON")
    print("="*70)
    
    try:
        compare_captions()
        compare_refinements()
        compare_scene_graphs()
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure both pipelines have been run:")
        print("  1. cd neuro-nav-internVL && bash run_internvl_pipeline.sh")
        print("  2. cd neuro-nav-vlm && bash run_vlm_pipeline.sh")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
EOF

chmod +x /home/nick/Project_dir/compare_vlms.py
```

**Run the comparison:**

```bash
python /home/nick/Project_dir/compare_vlms.py
```

---

### Method 2: Manual File Comparison

#### Compare Captions

```bash
# InternVL2 captions
cat /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/cfslam_internvl_captions.json | jq '.[0]'

# Qwen2-VL captions
cat /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/cfslam_qwen_captions.json | jq '.[0]'
```

#### Compare Refined Responses

```bash
# InternVL2 responses
cat /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/cfslam_internvl_responses.json | jq '.[0]'

# Qwen2-VL responses
cat /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/cfslam_qwen_responses.json | jq '.[0]'
```

---

### Method 3: Save Scene Graphs Separately

To compare final scene graphs, save them with different names:

```bash
# After running InternVL2, backup scene graph
cd /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm
cp scene_graph.json scene_graph_internvl.json

# Run Qwen2-VL pipeline
cd /home/nick/Project_dir/neuro-nav-vlm
bash run_vlm_pipeline.sh

# Backup Qwen scene graph
cd /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm
cp scene_graph.json scene_graph_qwen.json

# Compare
diff -u scene_graph_internvl.json scene_graph_qwen.json | head -50
```

---

## 🔍 Query Comparison

Test the same queries on both models:

### InternVL2 Query

```bash
cd /home/nick/Project_dir/neuro-nav-internVL
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python query_internvl_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph_internvl.json \
  --query "Where can I sit?"
```

### Qwen2-VL Query

```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph_qwen.json \
  --query "Where can I sit?"
```

---

## 📈 Performance Comparison

### Measure Processing Time

**InternVL2:**
```bash
cd /home/nick/Project_dir/neuro-nav-internVL
time bash run_internvl_pipeline.sh 2>&1 | tee internvl_timing.log
```

**Qwen2-VL:**
```bash
cd /home/nick/Project_dir/neuro-nav-vlm
time bash run_vlm_pipeline.sh 2>&1 | tee qwen_timing.log
```

### Measure GPU Memory

```bash
# In another terminal, monitor GPU usage
watch -n 1 nvidia-smi

# Or log it
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > gpu_usage.log
```

---

## 📊 Comparison Criteria

### 1. Caption Quality
- **Length**: Word count per object
- **Detail**: Specific details mentioned (colors, materials, positions)
- **Consistency**: Similar captions across multiple views of same object

### 2. Object Tags
- **Accuracy**: Does tag match the object?
- **Specificity**: "white sofa" vs "sofa"
- **Consistency**: Same object gets same tag

### 3. Speed
- **Caption extraction**: Seconds per object
- **Refinement**: Seconds per object
- **Total time**: Minutes for entire scene

### 4. Memory Usage
- **Peak VRAM**: Maximum GPU memory used
- **Average VRAM**: Typical memory footprint

### 5. Query Quality
- **Relevance**: Does answer address the question?
- **Detail**: How descriptive is the answer?
- **Accuracy**: Are spatial references correct?

---

## 🎯 Expected Differences

### InternVL2 Strengths
- ✅ Better OCR (if scene has text/signs)
- ✅ Multi-lingual descriptions
- ✅ Document/chart understanding
- ✅ Different caption style/phrasing

### Qwen2-VL Strengths
- ✅ Proven results (already tested)
- ✅ Excellent Chinese support
- ✅ Strong instruction-following
- ✅ Very detailed captions

### Should Be Similar
- 🟰 Caption length (~200-400 words)
- 🟰 Processing speed (~1-2 sec/object)
- 🟰 Memory usage (~6-7GB VRAM)
- 🟰 Scene graph structure (same algorithm)
- 🟰 Number of objects detected (same YOLO+SAM)

---

## 🔧 Troubleshooting Comparison

### Issue: Can't find both caption files

**Solution:**
```bash
# Check which files exist
ls -lh /home/nick/Project_dir/neuro-nav/data/Replica/room0/exps/r_mapping_with_llm/cfslam_*_captions.json

# Make sure both pipelines have run
cd /home/nick/Project_dir/neuro-nav-internVL && bash run_internvl_pipeline.sh
cd /home/nick/Project_dir/neuro-nav-vlm && bash run_vlm_pipeline.sh
```

### Issue: Scene graphs overwrite each other

**Solution:** Save them separately after each run:
```bash
# After InternVL2
cp data/.../scene_graph.json data/.../scene_graph_internvl.json

# After Qwen2-VL
cp data/.../scene_graph.json data/.../scene_graph_qwen.json
```

### Issue: GPU memory error when switching models

**Solution:** Clear GPU between runs:
```bash
# Kill all Python processes
pkill -9 python

# Check GPU is clear
nvidia-smi

# Then run next pipeline
```

---

## 📝 Summary Commands

**Complete comparison workflow:**

```bash
# 1. Run InternVL2
cd /home/nick/Project_dir/neuro-nav-internVL
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1 && export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash run_internvl_pipeline.sh

# 2. Backup InternVL2 scene graph
cp data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
   data/Replica/room0/exps/r_mapping_with_llm/scene_graph_internvl.json

# 3. Run Qwen2-VL
cd /home/nick/Project_dir/neuro-nav-vlm
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
bash run_vlm_pipeline.sh

# 4. Backup Qwen2-VL scene graph
cp data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
   data/Replica/room0/exps/r_mapping_with_llm/scene_graph_qwen.json

# 5. Run comparison
python /home/nick/Project_dir/compare_vlms.py
```

---

**Created**: 2025-11-19  
**For**: Comparing InternVL2-2B vs Qwen2-VL-2B scene graph outputs

