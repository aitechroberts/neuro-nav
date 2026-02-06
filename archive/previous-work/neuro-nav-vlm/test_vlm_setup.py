#!/usr/bin/env python3
"""
Test VLM Setup - Verify that all VLM components are properly installed and working

This script tests:
1. Import of VLM modules
2. Model loading (without downloading if already cached)
3. Basic inference
4. GPU availability
"""

import sys
import os
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_test(name, status, message=""):
    """Print test result"""
    if status:
        print(f"{GREEN}✓{RESET} {name}")
        if message:
            print(f"  {message}")
    else:
        print(f"{RED}✗{RESET} {name}")
        if message:
            print(f"  {RED}{message}{RESET}")


def test_imports():
    """Test that all required modules can be imported"""
    print(f"\n{BOLD}Testing Imports...{RESET}")
    
    tests = []
    
    # Test basic imports
    try:
        import torch
        tests.append(("PyTorch", True, f"Version {torch.__version__}"))
    except ImportError as e:
        tests.append(("PyTorch", False, str(e)))
    
    try:
        import transformers
        tests.append(("Transformers", True, f"Version {transformers.__version__}"))
    except ImportError as e:
        tests.append(("Transformers", False, str(e)))
    
    try:
        import PIL
        tests.append(("Pillow", True, ""))
    except ImportError as e:
        tests.append(("Pillow", False, str(e)))
    
    try:
        import numpy
        tests.append(("NumPy", True, ""))
    except ImportError as e:
        tests.append(("NumPy", False, str(e)))
    
    try:
        import qwen_vl_utils
        tests.append(("qwen-vl-utils", True, ""))
    except ImportError as e:
        tests.append(("qwen-vl-utils", False, str(e)))
    
    try:
        from conceptgraph.vlm.florence2_model import Florence2Model
        tests.append(("Florence2Model", True, ""))
    except ImportError as e:
        tests.append(("Florence2Model", False, str(e)))
    
    try:
        from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
        tests.append(("Qwen2VLModel", True, ""))
    except ImportError as e:
        tests.append(("Qwen2VLModel", False, str(e)))
    
    for name, status, message in tests:
        print_test(name, status, message)
    
    return all(status for _, status, _ in tests)


def test_gpu():
    """Test GPU availability"""
    print(f"\n{BOLD}Testing GPU...{RESET}")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print_test("CUDA available", cuda_available)
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print_test("GPU count", device_count > 0, f"{device_count} GPU(s) found")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            return True
        else:
            print(f"{YELLOW}Warning: No GPU available. Models will run slowly on CPU.{RESET}")
            return False
            
    except Exception as e:
        print_test("GPU test", False, str(e))
        return False


def test_model_cache():
    """Test if models are already downloaded"""
    print(f"\n{BOLD}Checking Model Cache...{RESET}")
    
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    hub_cache = Path(hf_home) / 'hub'
    
    print(f"  HuggingFace cache: {hub_cache}")
    
    models_to_check = [
        ("Florence-2-large", "models--microsoft--Florence-2-large"),
        ("Florence-2-base", "models--microsoft--Florence-2-base"),
        ("Qwen2-VL-2B", "models--Qwen--Qwen2-VL-2B-Instruct"),
        ("Qwen2-VL-7B", "models--Qwen--Qwen2-VL-7B-Instruct"),
    ]
    
    found_any = False
    for name, cache_dir in models_to_check:
        model_path = hub_cache / cache_dir
        cached = model_path.exists()
        if cached:
            found_any = True
        print_test(name, cached, "Cached" if cached else "Not downloaded")
    
    if not found_any:
        print(f"\n{YELLOW}No models cached. Run: python download_models.py{RESET}")
    
    return found_any


def test_florence2_inference():
    """Test Florence-2 model inference"""
    print(f"\n{BOLD}Testing Florence-2 Inference...{RESET}")
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        from conceptgraph.vlm.florence2_model import Florence2Model
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Try to load model (will download if not cached)
        print("  Loading model (may take time if downloading)...")
        model = Florence2Model(
            model_name="microsoft/Florence-2-base",  # Use base for faster testing
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Test captioning
        print("  Testing captioning...")
        caption = model.caption_image(dummy_image)
        
        print_test("Florence-2 inference", True, f"Generated caption: {caption[:50]}...")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print_test("Florence-2 inference", False, str(e))
        return False


def test_qwen2vl_inference():
    """Test Qwen2-VL model inference"""
    print(f"\n{BOLD}Testing Qwen2-VL Inference...{RESET}")
    
    try:
        import torch
        from PIL import Image
        import numpy as np
        from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
        
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Try to load model
        print("  Loading model (may take time if downloading)...")
        model = Qwen2VLModel(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        # Test captioning
        print("  Testing captioning...")
        caption = model.caption_image(dummy_image)
        
        print_test("Qwen2-VL inference", True, f"Generated caption: {caption[:50]}...")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print_test("Qwen2-VL inference", False, str(e))
        return False


def main():
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           VLM Setup Verification Test                            ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("GPU", test_gpu()))
    results.append(("Model Cache", test_model_cache()))
    
    # Ask before running inference tests (they download models)
    print(f"\n{BOLD}Inference Tests{RESET}")
    print("These tests will download models if not cached (~5GB total).")
    response = input("Run inference tests? (y/n) [n]: ").strip().lower()
    
    if response == 'y':
        results.append(("Florence-2 Inference", test_florence2_inference()))
        results.append(("Qwen2-VL Inference", test_qwen2vl_inference()))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"{BOLD}Summary:{RESET}")
    print(f"{'='*70}")
    
    for name, status in results:
        print_test(name, status)
    
    all_passed = all(status for _, status in results)
    
    print(f"\n{'='*70}")
    if all_passed:
        print(f"{GREEN}{BOLD}✓ All tests passed! You're ready to use VLMs.{RESET}")
        print(f"\nNext steps:")
        print(f"  1. Download models: python download_models.py")
        print(f"  2. Run pipeline:    ./run_vlm_pipeline.sh")
        print(f"  3. Query scene:     python query_vlm_scene.py")
    else:
        print(f"{RED}{BOLD}✗ Some tests failed. Check the output above.{RESET}")
        print(f"\nTroubleshooting:")
        print(f"  1. Install dependencies: pip install -r requirements_vlm.txt")
        print(f"  2. Check GPU: nvidia-smi")
        print(f"  3. See: SETUP_GUIDE.md")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

