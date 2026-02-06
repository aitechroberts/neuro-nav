#!/bin/bash
# Complete workflow to run InternVL2 and compare with Qwen2-VL

echo "=================================================="
echo "Run InternVL2 and Compare with Qwen2-VL"
echo "=================================================="
echo ""

# Step 1: Run InternVL2
echo "Step 1: Running InternVL2 pipeline..."
cd /home/nick/Project_dir/neuro-nav-internVL
source /home/nick/Project_dir/neuro-nav/.venv/bin/activate
source /home/nick/Project_dir/neuro-nav/use-cuda-126.sh
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-internVL:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

bash run_internvl_pipeline.sh

if [ $? -ne 0 ]; then
    echo "❌ InternVL2 pipeline failed!"
    exit 1
fi

# Step 2: Backup InternVL2 scene graph
echo ""
echo "Step 2: Backing up InternVL2 scene graph..."
cp data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
   data/Replica/room0/exps/r_mapping_with_llm/scene_graph_internvl.json

# Step 3: Run Qwen2-VL
echo ""
echo "Step 3: Running Qwen2-VL pipeline..."
cd /home/nick/Project_dir/neuro-nav-vlm
export PYTHONPATH=/home/nick/Project_dir/neuro-nav-vlm:$PYTHONPATH

bash run_vlm_pipeline.sh

if [ $? -ne 0 ]; then
    echo "❌ Qwen2-VL pipeline failed!"
    exit 1
fi

# Step 4: Backup Qwen2-VL scene graph
echo ""
echo "Step 4: Backing up Qwen2-VL scene graph..."
cp data/Replica/room0/exps/r_mapping_with_llm/scene_graph.json \
   data/Replica/room0/exps/r_mapping_with_llm/scene_graph_qwen.json

# Step 5: Run comparison
echo ""
echo "Step 5: Running comparison..."
cd /home/nick/Project_dir
python compare_vlms.py

echo ""
echo "=================================================="
echo "✅ Complete! Both pipelines run and compared."
echo "=================================================="
echo ""
echo "Scene graphs saved:"
echo "  - InternVL2: data/.../scene_graph_internvl.json"
echo "  - Qwen2-VL:  data/.../scene_graph_qwen.json"
echo ""
echo "To query InternVL2:"
echo "  cd neuro-nav-internVL"
echo "  python query_internvl_scene.py --scene-graph data/.../scene_graph_internvl.json --query 'Your question?'"
echo ""
echo "To query Qwen2-VL:"
echo "  cd neuro-nav-vlm"
echo "  python query_vlm_scene.py --scene-graph data/.../scene_graph_qwen.json --query 'Your question?'"
echo ""

