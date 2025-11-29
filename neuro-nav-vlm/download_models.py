#!/usr/bin/env python3
"""
Download VLM models for neuro-nav-vlm
This script downloads Florence-2 and Qwen2-VL models to HuggingFace cache
"""

import os
import sys
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration
import torch

def download_florence2(model_name="microsoft/Florence-2-large"):
    """Download Florence-2 model"""
    print(f"\n{'='*70}")
    print(f"Downloading Florence-2: {model_name}")
    print(f"{'='*70}")
    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        print("✓ Processor downloaded")
        
        print("Loading model (this may take a few minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        print(f"✓ Florence-2 model downloaded successfully!")
        
        # Clean up
        del model, processor
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"✗ Error downloading Florence-2: {e}")
        return False

def download_qwen2vl(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    """Download Qwen2-VL model"""
    print(f"\n{'='*70}")
    print(f"Downloading Qwen2-VL: {model_name}")
    print(f"{'='*70}")
    
    try:
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        print("✓ Processor downloaded")
        
        print("Loading model (this may take several minutes)...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print(f"✓ Qwen2-VL model downloaded successfully!")
        
        # Clean up
        del model, processor
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        print(f"✗ Error downloading Qwen2-VL: {e}")
        return False

def main():
    print("╔═════════════════════════════════════════════════════════════════════╗")
    print("║          VLM Model Downloader for neuro-nav-vlm                    ║")
    print("╚═════════════════════════════════════════════════════════════════════╝")
    
    # Check HuggingFace cache
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    print(f"\nHuggingFace cache directory: {hf_home}")
    
    # Ask user which models to download
    print("\nAvailable models:")
    print("1. Florence-2-base (230MB, faster)")
    print("2. Florence-2-large (770MB, better quality) [RECOMMENDED]")
    print("3. Qwen2-VL-2B-Instruct (~4GB) [RECOMMENDED]")
    print("4. Qwen2-VL-7B-Instruct (~14GB, best quality)")
    print("5. All recommended models (Florence-2-large + Qwen2-VL-2B)")
    print("6. All models")
    
    choice = input("\nEnter your choice (1-6) [default: 5]: ").strip() or "5"
    
    results = []
    
    if choice in ["1", "5", "6"]:
        results.append(("Florence-2-base", download_florence2("microsoft/Florence-2-base")))
    
    if choice in ["2", "5", "6"]:
        results.append(("Florence-2-large", download_florence2("microsoft/Florence-2-large")))
    
    if choice in ["3", "5", "6"]:
        results.append(("Qwen2-VL-2B", download_qwen2vl("Qwen/Qwen2-VL-2B-Instruct")))
    
    if choice in ["4", "6"]:
        results.append(("Qwen2-VL-7B", download_qwen2vl("Qwen/Qwen2-VL-7B-Instruct")))
    
    # Summary
    print(f"\n{'='*70}")
    print("Download Summary:")
    print(f"{'='*70}")
    for model_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {model_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 All models downloaded successfully!")
        print("\nYou can now run the VLM pipeline:")
        print("  ./run_vlm_pipeline.sh")
        print("\nOr manually:")
        print("  python conceptgraph/scenegraph/build_scenegraph_vlm.py --mode extract-node-captions ...")
        return 0
    else:
        print("\n⚠️  Some models failed to download. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Check GPU memory with: nvidia-smi")
        print("4. Try downloading individual models instead of all at once")
        return 1

if __name__ == "__main__":
    sys.exit(main())

