#!/bin/bash

# VLM dependencies installer for neuro-nav-vlm
# This script installs the required packages, skipping optional ones that might fail

set -e

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║            Installing VLM Dependencies                           ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "Please activate the environment first:"
    echo "  source ../neuro-nav/.venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) [n]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "→ Installing core VLM packages..."
echo ""

# Install required packages
pip install \
    transformers>=4.37.0 \
    timm>=0.9.0 \
    qwen-vl-utils>=0.0.1 \
    accelerate>=0.26.0 \
    einops>=0.7.0 \
    torchvision>=0.15.0 \
    bitsandbytes>=0.41.0

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Core packages installed successfully!"
else
    echo ""
    echo "✗ Installation failed. Check the error above."
    exit 1
fi

# Ask about flash-attn (optional)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Optional: Flash Attention"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Flash-attention provides faster inference but:"
echo "  - Requires compilation (takes 5-10 minutes)"
echo "  - May fail on some systems"
echo "  - Is NOT required for the VLM pipeline to work"
echo ""
read -p "Install flash-attn? (y/n) [n]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "→ Installing flash-attn (this may take several minutes)..."
    pip install packaging wheel
    pip install flash-attn --no-build-isolation
    
    if [ $? -eq 0 ]; then
        echo "✓ flash-attn installed successfully!"
    else
        echo "⚠️  flash-attn installation failed (this is okay, it's optional)"
        echo "   The VLM pipeline will work without it."
    fi
else
    echo ""
    echo "⊳ Skipping flash-attn (you can install it later if needed)"
fi

echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                  Installation Complete! ✓                        ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Test setup:      python test_vlm_setup.py"
echo "  2. Download models: python download_models.py"
echo "  3. Setup data:      ./setup_data_link.sh"
echo "  4. Run pipeline:    ./run_vlm_pipeline.sh"
echo ""

