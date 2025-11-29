#!/usr/bin/env python3
"""
Download Phi-3-Vision model from HuggingFace (Microsoft)
"""

from transformers import AutoProcessor, AutoModelForCausalLM
import torch

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
        
        # Download model
        print("\nDownloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
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

