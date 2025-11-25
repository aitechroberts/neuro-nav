# Query System for ConceptGraphs

**Research Question**: *What computational mechanisms are necessary for semantic understanding in robotics?*

This module implements 5 different query engines that test whether vision, language, or reasoning contribute most to semantic understanding.

---

## 🚀 Quick Start

### **1. Activate Environment**
```bash
cd ~/projects/neuro-nav
source .venv/bin/activate && source ./use-cuda-126.sh
```

### **2. Generate Scene Data** (first time only)
```bash
./run_exploration1.sh
```

### **3. Run Queries**
```bash
# Interactive mode (recommended)
python query_system/run_query_comparison.py --setups 3,4,5 --interactive

# Single query
python query_system/run_query_comparison.py --setups 5 --query "What color is the pillow?"
```

---

## 🔬 The Five Setups

### **Setup 1: GPT-4 Text Reasoning**
**What**: CLIP retrieval → GPT-4 text-based reasoning
**Why**: Tests if advanced reasoning helps without seeing images
**How**: `--setups 1 --query "Find a chair"`

### **Setup 2: Qwen2.5-3B Local Reasoning**
**What**: CLIP retrieval → Local LLM text reasoning
**Why**: Can small local models match GPT-4?
**How**: `--setups 2 --query "Find a chair"`

### **Setup 3: CLIP Visual Grounding**
**What**: Pure visual-semantic matching (no reasoning)
**Why**: Does retrieval alone work?
**How**: `--setups 3 --query "Find something red"`

### **Setup 4: YOLO Text-Only**
**What**: Text label matching (no vision)
**Why**: Baseline - how much does vision matter?
**How**: `--setups 4 --query "Find a chair"`

### **Setup 5: GPT-4V Vision** ⭐ **NEW**
**What**: CLIP retrieval → GPT-4V with actual RGB images
**Why**: Does seeing actual images improve reasoning?
**How**: `--setups 5 --query "What color is the pillow?"`

**Note**: Setup 5 can see colors, patterns, textures - things text can't describe!

---

## 📊 Key Comparisons

| Comparison | Tests |
|------------|-------|
| **Setup 1 vs 5** | Text reasoning vs Vision reasoning |
| **Setup 2 vs 5** | Local vs Cloud with vision |
| **Setup 3 vs 5** | Pure retrieval vs Reasoning |
| **Setup 4 vs 3** | Symbolic vs Visual grounding |

---

## 💡 Usage Examples

```bash
# Test visual reasoning (Setup 5 can see colors!)
python query_system/run_query_comparison.py --setups 5 \
  --query "What color is the pillow?"

# Compare text vs vision reasoning
python query_system/run_query_comparison.py --setups 1 \
  --query "Find something to sit on"
# Then separately (to avoid VRAM issues):
python query_system/run_query_comparison.py --setups 5 \
  --query "Find something to sit on"

# Fast comparison (no VRAM needed)
python query_system/run_query_comparison.py --setups 3,4 --interactive

# All setups (run separately due to VRAM):
python query_system/run_query_comparison.py --setups 3,4
python query_system/run_query_comparison.py --setups 5
```

---

## ⚠️ Important Notes

**VRAM Constraints**:
- Setups 3, 4, 5 each load CLIP separately
- Run Setup 5 alone: `--setups 5`
- Or run 3,4 together: `--setups 3,4`

**API Key** (for Setups 1 & 5):
```bash
export OPENAI_API_KEY=your-key-here
```

---

## 📈 Expected Performance

| Setup | Strength | Limitation |
|-------|----------|------------|
| **1: GPT-4 Text** | Best text reasoning | Can't see colors/patterns |
| **2: Qwen2.5** | Free, local | Weaker reasoning than GPT-4 |
| **3: CLIP** | Fast visual matching | No reasoning |
| **4: YOLO** | Fastest, minimal compute | No visual info |
| **5: GPT-4V** | Sees actual images! | Needs API key, costs ~$0.01/query |

---

## 🎯 Good Test Queries

**Visual queries** (Setup 5 excels):
- "What color is the pillow?"
- "Find something with a pattern"
- "Describe the stool"

**Reasoning queries** (Setups 1, 2, 5):
- "Find something to sit on"
- "What can I use to hold water?"

**Simple matching** (All setups):
- "Find a chair"
- "Find a table"

---

**Built for**: CMU 11851 - Talking to Robots
**Author**: Jesse (jesse-dev branch)
**Date**: November 2025
