#!/bin/bash
# Setup script for neuro-nav-vlm

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         Neuro-Nav VLM Setup Script                          ║"
echo "║   Setting up Florence-2 + Qwen2-VL pipeline                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Check Python version
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python version..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if python -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo -e "${GREEN}✓ Python version OK${NC}"
else
    echo -e "${RED}✗ Python 3.10+ required${NC}"
    exit 1
fi
echo ""

# Step 2: Check CUDA
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking CUDA..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ CUDA available${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not detected. VLMs will run on CPU (very slow)${NC}"
fi
echo ""

# Step 3: Check if virtual environment exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Setting up virtual environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d ".venv" ]; then
    echo "Creating new virtual environment..."
    python -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo "Virtual environment activated"
echo ""

# Step 4: Install base dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Installing base dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Upgrade pip
pip install --upgrade pip

# Check if neuro-nav is installed
if pip show neuro-nav &> /dev/null; then
    echo -e "${GREEN}✓ neuro-nav base package already installed${NC}"
else
    echo "Installing neuro-nav base package..."
    if [ -f "../neuro-nav/pyproject.toml" ]; then
        pip install -e ../neuro-nav
        echo -e "${GREEN}✓ Installed neuro-nav from ../neuro-nav${NC}"
    else
        echo -e "${YELLOW}⚠ neuro-nav not found. You may need to install it manually.${NC}"
    fi
fi
echo ""

# Step 5: Install VLM dependencies
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Installing VLM-specific dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pip install -r requirements_vlm.txt
echo -e "${GREEN}✓ VLM dependencies installed${NC}"
echo ""

# Step 6: Test imports
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Testing imports..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python -c "from conceptgraph.vlm import Florence2Model, Qwen2VLModel; print('✓ VLM modules imported successfully')"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All imports working${NC}"
else
    echo -e "${RED}✗ Import test failed${NC}"
    exit 1
fi
echo ""

# Step 7: Download models (optional)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Download VLM models (optional)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Models will be downloaded automatically on first use."
echo "Total download size: ~5GB (cached in ~/.cache/huggingface/)"
echo ""
read -p "Download models now? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading Florence-2-large..."
    python -c "from transformers import AutoProcessor, AutoModelForCausalLM; \
               AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); \
               AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True); \
               print('✓ Florence-2 downloaded')"
    
    echo "Downloading Qwen2-VL-2B-Instruct..."
    python -c "from transformers import Qwen2VLForConditionalGeneration, AutoProcessor; \
               Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct'); \
               AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct'); \
               print('✓ Qwen2-VL downloaded')"
    
    echo -e "${GREEN}✓ Models downloaded${NC}"
else
    echo "Skipping model download. Models will download on first use."
fi
echo ""

# Step 8: Create symlink to data (if exists)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: Setting up data directory..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d "../neuro-nav/data" ] && [ ! -d "data" ]; then
    ln -s ../neuro-nav/data data
    echo -e "${GREEN}✓ Created symlink to ../neuro-nav/data${NC}"
elif [ -d "data" ]; then
    echo -e "${GREEN}✓ Data directory already exists${NC}"
else
    mkdir -p data
    echo -e "${YELLOW}⚠ Created empty data directory${NC}"
fi
echo ""

# Final summary
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   Setup Complete! ✓                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Read the documentation:"
echo "   cat README_VLM.md"
echo ""
echo "3. Test the VLM models:"
echo "   python test_vlm_models.py"
echo ""
echo "4. Run the scene graph pipeline:"
echo "   python conceptgraph/scenegraph/build_scenegraph_vlm.py --help"
echo ""
echo -e "${GREEN}Happy VLM-ing! 🤖✨${NC}"

