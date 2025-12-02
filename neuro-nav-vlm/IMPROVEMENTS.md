# VLM Query Improvements

## Summary
Enhanced the VLM scene query system to provide **descriptive, natural language responses** instead of just listing object IDs.

## What Changed

### 1. Enhanced Scene Context (`query_vlm_scene.py`)
**Before:**
- Only included captions shorter than 100 characters
- Most detailed Qwen2-VL captions were excluded
- Context was minimal: just object tags and positions

**After:**
- Includes first 1-3 sentences of each object's caption
- Provides concise descriptions for quick reference
- Optionally includes full detailed descriptions
- Much richer context for the VLM to reason with

### 2. Improved Prompt Engineering (`qwen2vl_model.py`)
**Before:**
```
Context: {context}
Question: {query}
Answer the question based on the image and context provided.
```

**After:**
```
You are a helpful assistant that provides clear, descriptive answers about a 3D scene.

Scene Information:
{context}

Question: {query}

Instructions:
- Provide a natural, descriptive answer using the object descriptions and locations
- Describe what objects are relevant to the question
- Include spatial information (where things are located)
- Be specific and helpful
- Use complete sentences, not just object IDs
```

### 3. Text-Only Query Support
- Fixed `query_scene()` to work without images
- Allows pure text-based reasoning over the scene graph
- No need to provide a scene image for queries

## Results Comparison

### Query: "Where can I do some work?"

**Before (Bad):**
```
Based on the objects in the image, it seems like you can do some work at the 
positions of Object 1, Object 2, Object 3, Object 5, Object 7, Object 8, 
Object 10, Object 11, Object 12, Object 13, Object 14, Object 16, Object 17, 
Object 19, Object 21, Object 23, Object 24, and Object 27.
```

**After (Good):**
```
You can do some work in the office or small room located at Object 27, which 
is an interior space that appears to be an office or a small room. The wall 
is plain and painted a light color, possibly white or off-white. On the wall, 
there is a rectangular window with a grid of small, square panes, allowing 
some natural light to enter the room.
```

### Query: "What furniture is available to sit on?"

**After (Good):**
```
There is a white sofa with a smooth, curved backrest and a straight armrest 
positioned against a wall painted in a light color, possibly beige or off-white. 
The sofa is the primary furniture available to sit on.
```

## Technical Details

### Files Modified:
1. **`query_vlm_scene.py`**
   - `build_scene_context()`: Enhanced to include detailed captions
   - Added parameter `include_full_captions` for controlling detail level

2. **`conceptgraph/vlm/qwen2vl_model.py`**
   - `query_scene()`: Improved prompt with clear instructions
   - Added text-only mode support (no image required)

### Configuration:
Add to your environment setup:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # For GPU memory
export HF_HUB_OFFLINE=1                                  # Use cached models
export TRANSFORMERS_OFFLINE=1                            # Avoid network delays
```

## Benefits

1. **More Useful Responses**: Natural language descriptions instead of object IDs
2. **Better Reasoning**: VLM has access to full object descriptions
3. **Task-Oriented**: Can answer "where can I..." type questions effectively
4. **Spatial Understanding**: Includes location information in responses
5. **User-Friendly**: Responses are helpful for robot navigation tasks

## Usage

```bash
# Setup
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Query
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I do some work?"
```

## Future Enhancements

Possible improvements:
- Add image support for visual grounding
- Cache model between queries (persistent server)
- Support for multi-turn conversations
- Integration with robot navigation planner
- Voice interface for natural queries

