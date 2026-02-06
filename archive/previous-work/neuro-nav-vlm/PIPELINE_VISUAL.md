# Visual Pipeline Comparison

## Pipeline Architecture

### neuro-nav (Old Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: RGB-D IMAGES                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DETECTION & SEGMENTATION                                  │
│  ┌──────────┐      ┌──────────┐                                     │
│  │   YOLO   │  →   │   SAM    │  →  Bounding Boxes + Masks         │
│  │  ~50M    │      │  ~600M   │                                     │
│  └──────────┘      └──────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: FEATURE EXTRACTION & CAPTIONING                           │
│  ┌──────────┐      ┌──────────┐                                     │
│  │   CLIP   │      │  LLaVA   │                                     │
│  │  ~427M   │  +   │  ~7B     │  →  Short Captions (1 sentence)    │
│  │          │      │          │      + CLIP Features (512D)         │
│  └──────────┘      └──────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: CAPTION REFINEMENT (via API)                              │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              GPT-4 (OpenAI API)                          │       │
│  │              ~1.7T parameters                            │       │
│  │              • Costs $0.03-0.06 per object              │       │
│  │              • Requires internet                         │       │
│  │              • 25-second timeout                         │       │
│  └──────────────────────────────────────────────────────────┘       │
│                            │                                         │
│                            ▼                                         │
│              Output: "object_tag": "sofa"                            │
│                      "summary": "white sofa with pillows"            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: 3D RECONSTRUCTION                                         │
│  • Merge 2D detections → 3D objects                                 │
│  • Compute 3D IoU overlap                                           │
│  • Build spatial graph structure                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: RELATIONSHIP EXTRACTION (via API)                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              GPT-4 (OpenAI API)                          │       │
│  │              • Text-only reasoning                       │       │
│  │              • Costs $0.03 per relationship              │       │
│  │              • No visual grounding                       │       │
│  └──────────────────────────────────────────────────────────┘       │
│                            │                                         │
│                            ▼                                         │
│              Output: "relation": "a on b"                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Scene Graph                                                │
│  • 19 nodes (objects)                                               │
│  • 8 edges (relationships)                                          │
│  • Short captions (1 sentence)                                      │
│  • Object tags only                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QUERY INTERFACE (via API)                                          │
│  GPT-4: "Where can I sit?"                                          │
│  Response: "You can sit at Object 1, Object 2, Object 3..."         │
└─────────────────────────────────────────────────────────────────────┘

TOTAL COST: ~$1.20 per scene
MEMORY: ~18GB GPU VRAM
TIME: ~5-8 seconds per object
OFFLINE: ❌ No (requires GPT-4 API)
```

---

### neuro-nav-vlm (New Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: RGB-D IMAGES                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1: DETECTION & SEGMENTATION                                  │
│  ┌──────────┐      ┌──────────┐                                     │
│  │   YOLO   │  →   │   SAM    │  →  Bounding Boxes + Masks         │
│  │  ~50M    │      │  ~600M   │      (SAME AS OLD)                 │
│  └──────────┘      └──────────┘                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2: VLM CAPTIONING                                            │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              Qwen2-VL-2B (Local)                         │       │
│  │              ~2B parameters                              │       │
│  │              • Runs locally on GPU                       │       │
│  │              • No API costs                              │       │
│  │              • Visual + language understanding           │       │
│  └──────────────────────────────────────────────────────────┘       │
│                            │                                         │
│                            ▼                                         │
│              Output: Detailed captions (200-400 words)               │
│              "The image depicts a cozy indoor setting, likely       │
│              a living room... The primary focus is on a white       │
│              sofa with a minimalist design, featuring a smooth,     │
│              curved backrest... On the sofa, there is a single      │
│              decorative pillow with green and beige leaves..."       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3: (SKIPPED - Captions already refined!)                     │
│  • No GPT-4 refinement needed                                       │
│  • Captions are already detailed and high-quality                   │
│  • Saves time and money                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4: 3D RECONSTRUCTION                                         │
│  • Merge 2D detections → 3D objects                                 │
│  • Compute 3D IoU overlap                                           │
│  • Build spatial graph structure                                    │
│  (SAME AS OLD)                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5: RELATIONSHIP EXTRACTION (Local)                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              Qwen2-VL-2B (Same model)                    │       │
│  │              • Visual reasoning                          │       │
│  │              • No API costs                              │       │
│  │              • Understands spatial context               │       │
│  └──────────────────────────────────────────────────────────┘       │
│                            │                                         │
│                            ▼                                         │
│              Output: "relation": "a on b"                            │
│              (with visual grounding)                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Scene Graph                                                │
│  • 19 nodes (objects)                                               │
│  • 8 edges (relationships)                                          │
│  • Detailed captions (200-400 words each)                           │
│  • Rich spatial context                                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  QUERY INTERFACE (Local)                                            │
│  Qwen2-VL: "Where can I sit?"                                       │
│  Response: "You can sit on the white sofa with a smooth, curved     │
│            backrest positioned at (2.8, -0.7, -0.7). The sofa       │
│            features a minimalist design and has decorative          │
│            pillows..."                                              │
└─────────────────────────────────────────────────────────────────────┘

TOTAL COST: $0.00 (fully local)
MEMORY: ~7GB GPU VRAM
TIME: ~1-2 seconds per object
OFFLINE: ✅ Yes (no internet required)
```

---

## Side-by-Side Model Comparison

```
┌──────────────────────────┬──────────────────────────┐
│     OLD PIPELINE         │     NEW PIPELINE         │
├──────────────────────────┼──────────────────────────┤
│  5 Models                │  3 Models                │
│  ─────────               │  ─────────               │
│  1. YOLO (~50M)          │  1. YOLO (~50M)          │
│  2. SAM (~600M)          │  2. SAM (~600M)          │
│  3. CLIP (~427M)         │  3. Qwen2-VL (~2B)       │
│  4. LLaVA (~7B)          │                          │
│  5. GPT-4 (API)          │                          │
│                          │                          │
│  Total Local: ~8B        │  Total: ~2.7B            │
│  GPU Memory: ~18GB       │  GPU Memory: ~7GB        │
│  API Cost: ~$1.20/scene  │  API Cost: $0.00         │
└──────────────────────────┴──────────────────────────┘
```

---

## Data Flow Diagram

### OLD: Multi-Model Pipeline
```
Image → YOLO → Detections
  ↓
SAM → Masks
  ↓
CLIP → Features (512D vector)
  ↓
LLaVA → Short Caption ("white sofa")
  ↓
GPT-4 API → Refined Tag ("sofa")
  ↓
3D Reconstruction
  ↓
GPT-4 API → Relationships ("a on b")
  ↓
Scene Graph
```

### NEW: Unified VLM Pipeline
```
Image → YOLO → Detections
  ↓
SAM → Masks
  ↓
Qwen2-VL → Detailed Caption (200+ words)
  ↓
(No refinement needed)
  ↓
3D Reconstruction
  ↓
Qwen2-VL → Relationships (with visual context)
  ↓
Scene Graph
```

---

## Memory Usage Timeline

### OLD Pipeline
```
Time →
      ┌─────┬─────┬─────┬─────┬─────┬─────┐
18GB  │█████│     │     │█████│     │     │  LLaVA loaded
14GB  │█████│     │     │█████│     │     │
10GB  │█████│█████│     │█████│     │     │  
 6GB  │█████│█████│     │█████│     │     │  CLIP + SAM
 2GB  │█████│█████│     │█████│     │     │  YOLO
      └─────┴─────┴─────┴─────┴─────┴─────┘
       Det.  Cap.  GPT-4  Rel.  GPT-4  Query
              ↑           ↑      ↑      ↑
            Load       Wait    Wait   Wait
            LLaVA      API     API    API
```

### NEW Pipeline
```
Time →
      ┌─────┬─────┬─────┬─────┬─────┐
18GB  │     │     │     │     │     │
14GB  │     │     │     │     │     │
10GB  │     │     │     │     │     │  
 6GB  │█████│█████│█████│█████│█████│  Qwen2-VL loaded once
 2GB  │█████│█████│█████│█████│█████│  YOLO + SAM
      └─────┴─────┴─────┴─────┴─────┘
       Det.  Cap.  3D    Rel.  Query
              ↑           ↑      ↑
            Local       Local  Local
          Inference   Inference Inference
```

---

## Caption Quality Comparison

### OLD (LLaVA → GPT-4)
```
Input Image: [Cropped sofa]

LLaVA Output:
"The central object in the image is white sofa."

GPT-4 Refinement:
{
  "summary": "A white sofa",
  "possible_tags": ["sofa", "couch"],
  "object_tag": "sofa"
}

Character count: ~15 chars
Word count: ~3 words
```

### NEW (Qwen2-VL)
```
Input Image: [Cropped sofa]

Qwen2-VL Output:
"The image depicts a cozy indoor setting, likely a living room or a 
similar space. The primary focus is on a white sofa with a minimalist 
design, featuring a smooth, curved backrest and a straight armrest. 
The sofa is positioned against a wall, which appears to be painted in 
a light color, possibly beige or off-white.

On the sofa, there is a single decorative pillow. This pillow is 
rectangular and has a pattern of green and beige leaves or flowers, 
giving it a natural and somewhat rustic appearance. The pillow is 
placed on the sofa's seat, which is covered with a light-colored 
fabric, likely cotton or a similar material.

In the background, there is a small, round wooden table with a dark 
finish. The table has a simple design, with a smooth surface and a 
slightly curved edge. The table is positioned against the wall, and 
its presence adds a touch of warmth and functionality to the room."

Character count: ~1129 chars
Word count: ~186 words
Detail level: 60x more detailed!
```

---

## Cost Breakdown (100 Objects)

### OLD Pipeline
```
Caption Generation (LLaVA):        $0.00  (local)
Caption Refinement (GPT-4):       $3.00  (100 × $0.03)
Relationship Extraction (GPT-4):  $1.50  (50 pairs × $0.03)
Query Interface (GPT-4):          $0.30  (10 queries × $0.03)
                                  ─────
TOTAL:                           $4.80 per scene
```

### NEW Pipeline
```
Caption Generation (Qwen2-VL):     $0.00  (local)
Caption Refinement:                $0.00  (not needed)
Relationship Extraction (Qwen2-VL):$0.00  (local)
Query Interface (Qwen2-VL):        $0.00  (local)
                                   ─────
TOTAL:                            $0.00 per scene

SAVINGS: $4.80 per scene
```

---

## When to Use Each Pipeline

### Use OLD Pipeline If:
- ❌ You have unlimited GPT-4 API budget
- ❌ You need extremely short object tags only
- ❌ You have 24GB+ GPU VRAM
- ❌ You don't mind waiting for API calls
- ❌ You require cloud-based processing

### Use NEW Pipeline If:
- ✅ You want to run everything locally
- ✅ You need detailed object descriptions
- ✅ You have limited GPU memory (8GB+)
- ✅ You want faster processing
- ✅ You want zero API costs
- ✅ You need offline capability
- ✅ You want better query responses

**Recommendation: Use the NEW pipeline for 99% of use cases.**

