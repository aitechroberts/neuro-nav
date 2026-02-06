#!/usr/bin/env python3
"""
Download PaliGemma-3B model from HuggingFace (Google)
"""

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

def download_paligemma():
    """Download PaliGemma-3B model and processor"""
    model_name = "google/paligemma-3b-mix-224"
    
    print(f"Downloading {model_name}...")
    print("This may take a while (model is ~2.9GB)")
    
    try:
        # Download processor
        print("\nDownloading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        print("✓ Processor downloaded successfully")
        
        # Download model
        print("\nDownloading model...")
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        print("✓ Model downloaded successfully")
        
        print(f"\n✅ PaliGemma-3B model downloaded and cached!")
        print(f"Cache location: ~/.cache/huggingface/hub/")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~3GB)")
        print("3. Try: pip install --upgrade transformers")

if __name__ == "__main__":
    download_paligemma()

