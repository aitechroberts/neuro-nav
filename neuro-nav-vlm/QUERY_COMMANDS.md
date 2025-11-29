# Quick Query Commands

The VLM pipeline is working! Here are the commands you need:

## Setup (Run Once Per Terminal)
```bash
cd /home/nick/Project_dir/neuro-nav-vlm
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Query the Scene (Single Question)
```bash
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "YOUR QUESTION HERE"
```

### Example Queries:
```bash
# Find objects
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "What objects are in this room?"

# Locate specific items
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where is the sofa located?"

# Task-oriented queries (NEW!)
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Where can I do some work?"
# Response: "You can do some work in the office or small room located at Object 27..."

python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "What furniture is available to sit on?"
# Response: "There is a white sofa with a smooth, curved backrest..."

# Get descriptions
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "Describe the furniture in the room"

# Spatial reasoning
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
  --query "What is next to the ottoman?"
```

## Interactive Mode (Multiple Questions)
```bash
python query_vlm_scene.py \
  --scene-graph data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json
```
Then type your questions and press Enter. Type `quit` to exit.

## Generated Files
Your scene graph is located at:
```
data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json
```

This contains:
- 19 detected objects
- Detailed Qwen2-VL captions for each object
- 3D bounding box positions (x, y, z)
- Spatial relationships between objects

## Tips
- The first query loads the model (~3 seconds)
- Subsequent queries are much faster
- Use `HF_HUB_OFFLINE=1` to avoid network delays
- Scene context is automatically included in queries
- **NEW:** Responses are now descriptive with natural language (not just object IDs!)
- **NEW:** Each object's full caption is included in the context for detailed reasoning
- Add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid GPU memory fragmentation

## View Raw Scene Graph
```bash
cat data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json | jq .
```
(Install `jq` for pretty printing: `sudo apt install jq`)

