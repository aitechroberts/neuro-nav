# Jesse's Guide: ConceptGraphs 3D Scene Understanding

**Course**: Talking to Robots (CMU 11851)  
**Focus**: Exploring what "meaning" means for AI and robotics  
**Hardware**: Lenovo Legion Laptop (RTX 4060, 8GB VRAM)

---

## 🎯 What This Does

This system performs **3D semantic scene understanding** - it takes RGB-D images and creates a meaningful 3D map with:
- **Object detection** (YOLO) - "What things are in the scene?"
- **3D segmentation** (MobileSAM) - "Where exactly are they in 3D space?"
- **Semantic features** (CLIP) - "What do these objects mean semantically?"
- **Scene reconstruction** - Building a coherent 3D understanding

Perfect for exploring: **What does a robot actually need to understand about a scene to function?**

---

## 🔄 Quick Restart (After Closing)

**Everything is saved!** Just run these 3 commands:

```bash
cd ~/projects/neuro-nav
git checkout jesse-dev
source .venv/bin/activate && source ./use-cuda-126.sh
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

**What's Persistent:**
- ✅ Code committed to `jesse-dev` branch
- ✅ Virtual environment (`.venv`) stays on disk
- ✅ Replica dataset (12GB) stays in `data/`
- ✅ Downloaded models cached in `~/.cache/`
- ✅ All your configurations saved

**Nothing needs to be reinstalled!** Everything just picks up where you left off.

---

## 🚀 Full Setup Guide (First Time or Troubleshooting)

### 1. Environment Setup

```bash
# Navigate to project
cd ~/projects/neuro-nav

# Make sure you're on the correct git branch
git checkout jesse-dev

# Activate the Python 3.10.12 virtual environment
source .venv/bin/activate

# Set CUDA 12.6 environment
source ./use-cuda-126.sh

# Verify environment
python --version  # Should show Python 3.10.12
echo $CUDA_HOME   # Should show /usr/local/cuda-12.6
```

### 2. Run the Rerun Visualization (Recommended)

```bash
cd ~/projects/neuro-nav
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test
```

**What you'll see:**
- Beautiful 3D visualization in browser/window
- Real-time object detection and tracking
- Color-coded point clouds for each object
- Timeline controls to scrub through frames
- Dual view: 3D scene + camera RGB feed

### 3. Alternative: Open3D GUI (Simpler)

```bash
cd ~/projects/neuro-nav
python conceptgraph/slam/gui_realtime_mapping.py
```

Opens a native 3D viewer with interactive controls.

---

## 🌿 Git Branch Info

**Branch:** `jesse-dev`  
This is your working branch with all the fixes and optimizations for your system.

**Key differences from main:**
- Optimized for RTX 4060 Laptop (8GB VRAM)
- All paths configured for your system
- PyTorch3D-dependent features disabled
- Bug fixes for Open3D API compatibility

**To see what changed:**
```bash
git diff main jesse-dev
```

---

## 📝 Key Configuration Files Modified

All paths are now set for your system (`/home/jesse/projects/neuro-nav`):

1. **`conceptgraph/hydra_configs/base_paths.yaml`**
   - `repo_root` and `data_root` updated

2. **`conceptgraph/hydra_configs/rerun_simple_test.yaml`**
   - Simple test config: 30 frames, every 3rd frame
   - Uses MobileSAM (low memory)
   - GIOU similarity (no PyTorch3D needed)
   - Merge/denoise disabled (avoid PyTorch3D issues)

3. **Code Fixes Applied:**
   - `conceptgraph/slam/gui_realtime_mapping.py` - Uses MobileSAM
   - `conceptgraph/slam/rerun_realtime_mapping.py` - Uses MobileSAM
   - `conceptgraph/slam/utils.py` - Made 'captions' optional (2 places)
   - `conceptgraph/utils/optional_rerun_wrapper.py` - Fixed Open3D bbox API

---

## 💾 Data & Models

**Dataset Location:**
```
~/projects/neuro-nav/data/Replica/
├── room0/      # Indoor office scene
├── room1/      # Another room
├── office0/    # Office scenes
└── ...
```

**Models (auto-downloaded on first run):**
- YOLO: `yolov8l-world.pt` (~80MB)
- SAM: `mobile_sam.pt` (~40MB) - *Chosen for 8GB VRAM*
- CLIP: `ViT-H-14` (~2.5GB) - Cached in `~/.cache/huggingface/`

---

## 🔧 Hardware Optimizations for Your Laptop

We optimized for **RTX 4060 Laptop GPU (8GB VRAM)**:

1. **MobileSAM** instead of SAM-Large (saves ~5GB VRAM)
2. **GIOU similarity** instead of PyTorch3D IOU (simpler, faster)
3. **Disabled operations** that require PyTorch3D:
   - Object merging
   - Advanced denoising
   - Overlap-based filtering

These are **post-processing refinements** - the core detection and reconstruction still works perfectly!

---

## 🤔 Exploring "Meaning" for Robots: Future Directions

### **Category 1: What Information Actually Matters?**

#### 1.1 **Minimal Semantic Requirements**
*Question: What's the minimum semantic information a robot needs?*

**Experiments:**
- Run with **no CLIP features** - just geometric boxes
- Try **generic labels** ("object1", "object2") vs semantic names
- Test: Can a robot navigate/manipulate with just geometry?

**Implementation:**
```bash
# Disable CLIP embedding in config
force_detection: false  # Use pre-computed detections without features
```

**Philosophical Question:** Is "chair" fundamentally different from "object_type_7" for a robot?

#### 1.2 **Geometric vs Semantic Understanding**
*Do robots need to know "what" things are, or just "where" they are?*

**Experiments:**
- Navigation tasks with/without semantic labels
- Manipulation: "Pick up the red thing" vs "Pick up the mug"
- Compare performance: pure geometry vs full semantics

**Code Exploration:**
- Modify `class_agnostic: True` to ignore class information
- Test robot behaviors with only bounding boxes

---

### **Category 2: Abstraction Levels**

#### 2.1 **Granularity of Meaning**
*How detailed should a robot's understanding be?*

**Experiments:**
- **Coarse**: "furniture" (merge chairs, tables, sofas)
- **Medium**: "chair", "table", "sofa"  
- **Fine**: "office_chair", "dining_chair", "armchair"

**Implementation:**
```python
# Modify conceptgraph/scannet200_classes.txt
# Group classes into hierarchies
```

**Question:** Does finer granularity help or hurt robot performance?

#### 2.2 **Affordance-Based Understanding**
*What if robots understood "what you can do with things" vs "what things are"?*

**Experiments:**
- Relabel objects by affordance:
  - "sittable" (chairs, sofas, stools)
  - "supportable" (tables, shelves)
  - "openable" (doors, drawers, cabinets)
  - "graspable" (mugs, bottles, books)

**Deep Question:** Is affordance more "meaningful" than category for robots?

---

### **Category 3: Context and Relationships**

#### 3.1 **Spatial Relationships**
*Does a robot need to understand "the cup ON the table"?*

**Current System:**
- Has spatial information (IoU, overlap)
- Tracks objects independently

**Enhancement:**
```python
# Add to system: spatial relationship detection
"cup_1 ON table_2"
"chair_3 NEAR table_2"  
"lamp_4 ABOVE table_2"
```

**Experiment:** Give robot tasks requiring spatial reasoning:
- "Move the cup from the table to the shelf"
- Compare: explicit relationships vs pure geometry

#### 3.2 **Functional Scenes**
*Should robots understand "this is a kitchen" vs just "these are objects"?*

**Experiments:**
- Scene-level classification (kitchen, office, bedroom)
- Test: Does scene context improve object recognition?
- Question: Is "kitchen" a meaningful concept for a robot?

---

### **Category 4: Temporal Understanding**

#### 4.1 **Object Persistence**
*What does it mean for a robot to "remember" an object?*

**Current System:**
- Tracks objects across frames
- Maintains object identity (`uuid`)

**Experiments:**
- Object disappears and reappears - same or new?
- Moving objects vs static scene
- "When did I last see the coffee mug?"

**Explore:**
```python
# In conceptgraph/slam/slam_classes.py
# MapObject has 'num_detections', 'image_idx' history
# Explore: temporal reasoning about objects
```

#### 4.2 **Change Detection**
*Should robots notice when things change?*

**Ideas:**
- Detect moved objects
- Notice new/missing objects
- "The room is different than before"

---

### **Category 5: Uncertainty and Confidence**

#### 5.1 **Epistemic Meaning**
*What does a robot "know" vs "guess"?*

**Experiments:**
- Threshold confidence scores: `mask_conf_threshold`
- Show robot's uncertainty in visualization
- Task: "Only manipulate objects you're confident about"

**Implementation:**
```yaml
# In config
mask_conf_threshold: 0.25  # Try 0.5, 0.75
obj_min_detections: 1      # Require multiple sightings
```

**Question:** Should robots act differently on uncertain information?

#### 5.2 **Partial Observability**
*How do robots handle "I can't see everything"?*

**Experiments:**
- Process only partial scans
- Occluded objects
- Compare: full scene vs partial understanding

---

### **Category 6: Grounding Language to Vision**

#### 6.1 **Language-Driven Scene Understanding**
*Can you ask the robot about the scene in natural language?*

**Enhancement Ideas:**
```python
# Add language queries
"Where is the red mug?"
"How many chairs are there?"
"What's on the table?"
```

**Explore:**
- CLIP already provides text-image matching!
- Use `text_feats` in detections
- Query: "Find objects similar to 'cup for drinking coffee'"

#### 6.2 **Compositional Understanding**
*Can robots understand "the small red chair NEAR the wooden table"?*

**Current Limitation:** Individual objects, no compositions

**Future Work:**
- Combine spatial + semantic + attributes
- "the SMALL RED chair" requires multi-attribute understanding

---

## 🎓 Pedagogical Experiments (Easy to Implement)

### **Experiment 1: Ablation Study** (30 mins)
Turn off features one by one:
```yaml
# No semantic labels
class_agnostic: true

# No merge/filtering  
merge_interval: -1
filter_interval: -1

# Fewer detection classes
# Edit scannet200_classes.txt - keep only 10 classes
```

**Question:** What breaks? What still works?

### **Experiment 2: Class Confusion Matrix** (1 hour)
```python
# In utils.py, log when objects merge
"Merged STOOL into CHAIR - are they semantically similar?"

# Analyze: which objects get confused?
# Deep question: Is the distinction meaningful?
```

### **Experiment 3: Simulated Robot Tasks** (2 hours)
Create simple task descriptions:
```python
tasks = [
    "Navigate to the chair",           # Needs: object + location
    "Pick up the mug from table",      # Needs: 2 objects + relationship
    "Sit on any sittable furniture",   # Needs: affordance reasoning
]

# Score: Can the scene understanding support this task?
```

---

## 📊 Quick Config Tweaks

### Process More/Fewer Frames
```yaml
# In rerun_simple_test.yaml
start: 0
end: 100   # More frames = better reconstruction
stride: 5  # Lower = more frames processed
```

### Change Detection Sensitivity
```yaml
mask_conf_threshold: 0.25  # Lower = more detections, more noise
mask_area_threshold: 25    # Min pixels for detection
sim_threshold: 1.2         # Object matching sensitivity
```

### Visualize Different Scenes
```yaml
scene_id: room0    # Try: room1, office0, office1, etc.
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Already using MobileSAM (smallest)
# Reduce batch processing:
obj_pcd_max_points: 2000  # Default: 5000
```

### "PyTorch3D import error"
Already handled - we disabled PyTorch3D-dependent features:
- Merge operations
- IOU-based overlap
- Advanced filtering

These are refinements, not core functionality!

### Rerun viewer doesn't open
```bash
# Manual launch
rerun ~/projects/neuro-nav/data/Replica/room0/exps/r_mapping_rerun_test/rerun*.rrd
```

---

## 🎯 Learning Goals

Through these experiments, explore:

1. **Representation**: What information format is actually useful?
2. **Abstraction**: How much detail helps vs hurts?
3. **Grounding**: How do symbols connect to geometry?
4. **Uncertainty**: How should robots handle incomplete knowledge?
5. **Pragmatics**: What matters for *doing* vs just *knowing*?

---

## 📚 Key Papers to Read

1. **ConceptGraphs** - The base system (Gu et al., 2023)
2. **CLIP** - Vision-language grounding (Radford et al., 2021)
3. **Segment Anything** - Universal segmentation (Kirillov et al., 2023)
4. **NeRF/3D Gaussians** - Scene representation alternatives

---

## 🤝 Getting Help

**System is working but want to understand more?**
- Read: `conceptgraph/slam/utils.py` - Core object tracking
- Read: `conceptgraph/slam/mapping.py` - How objects are matched
- Read: `conceptgraph/utils/ious.py` - Geometric reasoning

**Want to modify behavior?**
- Start with config files in `conceptgraph/hydra_configs/`
- Then modify `utils.py` functions
- Visualization in `optional_rerun_wrapper.py`

---

## 💡 Research Questions for Your Class

**Fundamental:**
- Is semantic meaning necessary for robots, or is it just useful?
- What's the difference between "understanding" and "representing"?
- Can robots have "meaning" without language?

**Practical:**
- What's the minimum information for pick-and-place?
- Do robots need object permanence?
- Should robots understand scenes compositionally?

**Philosophical:**
- Is "chair" real or just a human concept?
- Do affordances exist independent of the agent?
- What does "knowing" mean for an embodied AI?

---

## 🎬 Next Session Startup

```bash
# 1. Open terminal
cd ~/projects/neuro-nav

# 2. Checkout your working branch
git checkout jesse-dev

# 3. Activate environment  
source .venv/bin/activate
source ./use-cuda-126.sh

# 3. Run visualization
python conceptgraph/slam/rerun_realtime_mapping.py --config-name=rerun_simple_test

# 4. Explore the scene, think about meaning!
```

---

**Remember:** You're not just building a perception system - you're exploring what perception *means* for a robot. Every object detected, every spatial relationship inferred, every semantic label assigned is a hypothesis about what matters for an embodied agent to function in the world.

**Have fun exploring meaning! 🤖✨**

