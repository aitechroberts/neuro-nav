#!/usr/bin/env python3
"""
Download InternVL2-2B model from HuggingFace
"""

from transformers import AutoTokenizer, AutoModel
import torch

def download_internvl2():
    """Download InternVL2-2B model and tokenizer"""
    model_name = "OpenGVLab/InternVL2-2B"
    
    print(f"Downloading {model_name}...")
    print("This may take a while (model is ~4-5GB)")
    
    try:
        # Download tokenizer
        print("\nDownloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        print("✓ Tokenizer downloaded successfully")
        
        # Download model
        print("\nDownloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("✓ Model downloaded successfully")
        
        print(f"\n✅ InternVL2-2B model downloaded and cached!")
        print(f"Cache location: ~/.cache/huggingface/hub/")
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space (~5GB)")
        print("3. Try: pip install --upgrade transformers")

if __name__ == "__main__":
    download_internvl2()

