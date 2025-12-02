#!/bin/bash
# Installation script for InternVL2 dependencies

echo "Installing InternVL2 dependencies..."

# Install core requirements
pip install transformers>=4.37.2
pip install torch>=2.0.0 torchvision>=0.15.0
pip install pillow>=9.0.0
pip install timm>=0.9.0
pip install accelerate>=0.20.0
pip install sentencepiece protobuf

# Install scene graph dependencies
pip install numpy scipy opencv-python matplotlib tqdm rich tyro open3d

# Optional: flash-attention (requires compilation)
# Uncomment if you want faster inference
# pip install flash-attn --no-build-isolation

echo "✅ Core dependencies installed!"
echo ""
echo "Next steps:"
echo "1. Install the package: pip install -e ."
echo "2. Download models: python download_models.py"
echo "3. Run the pipeline: ./run_internvl_pipeline.sh"

