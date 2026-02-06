#!/usr/bin/env python3
"""
Download Phi-3-Vision model from HuggingFace (Microsoft)
"""

import os
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Disable FlashAttention if not available
os.environ["DISABLE_FLASH_ATTN"] = "1"

def download_phi3():
    """Download Phi-3-Vision model and processor"""
    model_name = "microsoft/Phi-3-vision-128k-instruct"
    
    print(f"Downloading {model_name}...")
    print("This may take a while (model is ~3-4GB)")
    
    try:
        # Download processor (tokenizer + image processor)
        print("\nDownloading processor...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print("✓ Processor downloaded successfully")
        
        # Download model with eager attention (disable FlashAttention)
        print("\nDownloading model...")
        from transformers import AutoConfig
        
        # Load and modify config to disable FlashAttention
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Set attention implementation to eager (both attributes)
        config._attn_implementation = "eager"
        if hasattr(config, '_attn_implementation_internal'):
            config._attn_implementation_internal = "eager"
        
        # Load model with explicit eager attention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Explicit parameter
        )
        print("✓ Model downloaded successfully")
        
        print(f"\n✅ Phi-3-Vision model downloaded and cached!")
        print(f"Cache location: ~/.cache/huggingface/hub/")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~5GB)")
        print("3. Try: pip install --upgrade transformers")

if __name__ == "__main__":
    download_phi3()

