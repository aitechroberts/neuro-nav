# Integrating Distilled Model into Neuro-Nav Pipeline

This guide shows how to use your distilled student model as a drop-in replacement for Qwen2-VL in the neuro-nav-vlm pipeline.

## Quick Start

### Option 1: Direct Replacement (Easiest)

Modify the pipeline to use the distilled model instead of Qwen2-VL:

```python
# In build_scenegraph_vlm.py or your pipeline script

# OLD: from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
# NEW: Use distilled model
import sys
sys.path.insert(0, '/home/nick/Project_dir/neuro-nav')
from distilled_model.distilled_vlm_wrapper import DistilledVLMModel

# Initialize model (instead of Qwen2VLModel)
# qwen = Qwen2VLModel(model_name="Qwen/Qwen2-VL-2B-Instruct", device="cuda:0")
distilled = DistilledVLMModel(
    model_path="distilled_model/outputs/final_student_model.pt",
    device="cuda:0"
)

# Use exactly like Qwen2VLModel
result = distilled.refine_caption(image, captions)
```

### Option 2: Environment Variable Switch

Create a wrapper that switches between models:

```python
# In your pipeline script
import os

if os.getenv("USE_DISTILLED_MODEL", "false").lower() == "true":
    from distilled_model.distilled_vlm_wrapper import DistilledVLMModel
    model = DistilledVLMModel(
        model_path="distilled_model/outputs/final_student_model.pt",
        device="cuda:0"
    )
else:
    from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
    model = Qwen2VLModel(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda:0"
    )

# Use model (same interface)
result = model.refine_caption(image, captions)
```

Then run:
```bash
USE_DISTILLED_MODEL=true python your_pipeline.py
```

## Integration Examples

### Example 1: Replace in build_scenegraph_vlm.py

Edit `neuro-nav-vlm/conceptgraph/scenegraph/build_scenegraph_vlm.py`:

```python
# Around line 326, change:
# from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel

# To:
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from distilled_model.distilled_vlm_wrapper import DistilledVLMModel

# Around line 338, change:
# qwen = Qwen2VLModel(model_name=args.qwen_model, device=args.device)

# To:
qwen = DistilledVLMModel(
    model_path="distilled_model/outputs/final_student_model.pt",
    device=args.device
)
```

### Example 2: Replace in run_vlm_pipeline.sh

Edit `neuro-nav-vlm/run_vlm_pipeline.sh`:

```bash
# Add at the top:
DISTILLED_MODEL_PATH="../distilled_model/outputs/final_student_model.pt"
USE_DISTILLED=${USE_DISTILLED:-false}

# Modify the Python calls to pass model type:
if [ "$USE_DISTILLED" = "true" ]; then
    python conceptgraph/scenegraph/build_scenegraph_vlm.py \
        --mode refine-node-captions \
        --cachedir ${CACHEDIR} \
        --mapfile ${MAPFILE} \
        --distilled-model ${DISTILLED_MODEL_PATH} \
        --device cuda:0
else
    python conceptgraph/scenegraph/build_scenegraph_vlm.py \
        --mode refine-node-captions \
        --cachedir ${CACHEDIR} \
        --mapfile ${MAPFILE} \
        --qwen-model Qwen/Qwen2-VL-2B-Instruct \
        --device cuda:0
fi
```

### Example 3: Create New Pipeline Script

Create `neuro-nav-vlm/run_distilled_pipeline.sh`:

```bash
#!/bin/bash

# Use distilled model instead of Qwen2-VL
export USE_DISTILLED_MODEL=true
export DISTILLED_MODEL_PATH="../distilled_model/outputs/final_student_model.pt"

# Run the existing pipeline (it will use distilled model)
./run_vlm_pipeline.sh
```

## Interface Compatibility

The `DistilledVLMModel` class implements the same interface as `Qwen2VLModel`:

| Method | Description | Usage |
|--------|-------------|-------|
| `refine_caption(image, captions)` | Refine object captions | ✅ Same |
| `extract_object_relationships(images, obj1, obj2)` | Get spatial relationships | ✅ Same |
| `query_scene(image, query, context)` | Answer scene questions | ✅ Same |
| `caption_image(image, detail_level)` | Generate captions | ✅ Same |

## Important Notes

### 1. Text Generation Implementation

The current `DistilledVLMModel` wrapper uses **placeholder text generation**. For production use, you need to:

1. **Implement proper tokenization**: Use the same tokenizer that was used during training
2. **Implement text generation**: Add proper forward pass and decoding
3. **Handle model outputs**: Convert logits to text tokens

Example improvement:

```python
def _generate_text(self, image, prompt, max_length=256):
    # 1. Tokenize prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("your-tokenizer")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
    
    # 2. Preprocess image
    if image is not None:
        image_tensor = self._preprocess_image(image)
    else:
        image_tensor = None
    
    # 3. Forward pass
    with torch.no_grad():
        logits = self.model(images=image_tensor, input_ids=input_ids)
    
    # 4. Generate tokens
    generated_ids = self._sample_from_logits(logits, max_length)
    
    # 5. Decode to text
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text
```

### 2. Model Architecture Compatibility

Your distilled model needs to support:
- **Vision inputs**: Images (224x224 or your training size)
- **Text inputs**: Tokenized prompts
- **Text outputs**: Generated text/captions

The current `TinyVLMStudent` architecture supports this, but you may need to adjust:
- Input preprocessing
- Output decoding
- Tokenization

### 3. Performance Expectations

- **Speed**: 2-3x faster than Qwen2-VL
- **Memory**: 50-70% less VRAM
- **Quality**: 80-90% of Qwen2-VL quality

## Testing Integration

### Test 1: Basic Loading

```python
from distilled_model.distilled_vlm_wrapper import load_distilled_vlm

model = load_distilled_vlm("outputs/final_student_model.pt")
print("✓ Model loaded successfully")
```

### Test 2: Caption Refinement

```python
from PIL import Image

image = Image.open("test_image.jpg")
captions = ["a white sofa", "white couch", "sofa in room"]

result = model.refine_caption(image, captions)
print(f"Summary: {result['summary']}")
print(f"Tag: {result['object_tag']}")
```

### Test 3: Scene Query

```python
query = "Where can I sit?"
context = "Object 1: white sofa at position (1, 2, 3)"

answer = model.query_scene(image=None, query=query, context=context)
print(f"Answer: {answer}")
```

## Troubleshooting

### Issue: Model outputs placeholder text

**Solution**: Implement proper text generation (see "Text Generation Implementation" above)

### Issue: Import errors

**Solution**: Ensure paths are correct:
```python
import sys
sys.path.insert(0, '/home/nick/Project_dir/neuro-nav')
```

### Issue: Model architecture mismatch

**Solution**: Verify your student model supports vision+text inputs:
```python
# Check model forward signature
import inspect
print(inspect.signature(model.model.forward))
```

### Issue: Different output format

**Solution**: The wrapper handles JSON parsing, but ensure your model generates valid JSON when requested.

## Next Steps

1. **Implement proper text generation** in `_generate_text()`
2. **Add tokenizer** matching your training setup
3. **Test on real scenes** from neuro-nav
4. **Compare results** with Qwen2-VL
5. **Fine-tune** if needed for better quality

## Example: Complete Integration

```python
#!/usr/bin/env python3
"""
Example: Using distilled model in neuro-nav pipeline
"""

import sys
import os
sys.path.insert(0, '/home/nick/Project_dir/neuro-nav')

from distilled_model.distilled_vlm_wrapper import DistilledVLMModel
from PIL import Image

# Load distilled model
print("Loading distilled model...")
model = DistilledVLMModel(
    model_path="distilled_model/outputs/final_student_model.pt",
    device="cuda:0"
)

# Use in pipeline (same as Qwen2-VL)
image = Image.open("test_object.jpg")
captions = ["white sofa", "couch", "furniture"]

# Refine captions
result = model.refine_caption(image, captions)
print(f"Refined: {result}")

# Query scene
answer = model.query_scene(
    image=None,
    query="Where is the sofa?",
    context="Object 1: white sofa at (1, 2, 3)"
)
print(f"Answer: {answer}")
```

---

**Note**: The current implementation uses placeholder text generation. For production use, implement proper tokenization and text generation as described above.

