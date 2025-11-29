#!/bin/bash
# Installation script for Phi-3-Vision dependencies

echo "Installing Phi-3-Vision dependencies..."

# Install core requirements
pip install transformers>=4.40.0
pip install torch>=2.0.0 torchvision>=0.15.0
pip install pillow>=9.0.0
pip install accelerate>=0.20.0
pip install sentencepiece protobuf

# Install scene graph dependencies
pip install numpy scipy opencv-python matplotlib tqdm rich tyro open3d

echo "✅ Core dependencies installed!"
echo ""
echo "Next steps:"
echo "1. Install the package: pip install -e ."
echo "2. Download models: python download_models.py"
echo "3. Run the pipeline: bash run_phi3_pipeline.sh"

