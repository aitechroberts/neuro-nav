#!/bin/bash
#
# Launch script for Exploration 1: Uncertainty Visualization
# CMU 11851 - Talking to Robots
#

echo "=========================================="
echo "EXPLORATION 1: UNCERTAINTY VISUALIZATION"
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/projects/neuro-nav

# Activate environment
echo "Activating environment..."
source .venv/bin/activate
source ./use-cuda-126.sh

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current git branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "jesse-dev" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH', not 'jesse-dev'"
    echo "   It's recommended to use jesse-dev for this exploration"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Show environment info
echo ""
echo "Environment:"
echo "  Python: $(python --version)"
echo "  CUDA: $CUDA_HOME"
echo "  Working directory: $(pwd)"
echo ""

# Run the exploration
echo "Starting uncertainty visualization..."
echo "This will:"
echo "  1. Process 30 frames from room0"
echo "  2. Run object detection and mapping"
echo "  3. Visualize uncertainty in Rerun"
echo ""
echo "Press Ctrl+C to stop at any time"
echo ""

sleep 2

python exploration1_run_with_uncertainty.py

echo ""
echo "Done!"

