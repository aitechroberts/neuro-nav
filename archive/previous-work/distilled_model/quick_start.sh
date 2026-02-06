#!/bin/bash
# Quick start script for knowledge distillation

set -e

echo "=========================================="
echo "Neuro-Nav Knowledge Distillation Setup"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "train.py" ]; then
    echo "Error: Please run this script from the distilled_model directory"
    exit 1
fi

# Activate environment if available
if [ -f "../.venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Check for CUDA
if [ -f "../use-cuda-126.sh" ]; then
    echo "Setting up CUDA..."
    source ../use-cuda-126.sh
fi

# Check if teacher models are available
echo "Checking for teacher models..."
if [ ! -d "../neuro-nav-vlm" ]; then
    echo "Warning: neuro-nav-vlm directory not found"
    echo "Please ensure neuro-nav-vlm is set up for teacher models"
fi

# Create output directory
mkdir -p outputs

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  python train.py --data_root ../data --output_dir outputs"
echo ""
echo "Or with custom settings:"
echo "  python train.py \\"
echo "    --data_root ../data \\"
echo "    --output_dir outputs \\"
echo "    --teacher_type qwen2vl \\"
echo "    --student_type tiny \\"
echo "    --num_epochs 10 \\"
echo "    --batch_size 4"
echo ""

