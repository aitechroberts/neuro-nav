"""
Extract and save the trained model from checkpoint
"""

import torch
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from distilled_model.student_models import create_student_model

def extract_model(checkpoint_path: str, output_path: str):
    """Extract model from checkpoint and save cleanly."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("✓ Checkpoint loaded successfully!")
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
    print(f"  Keys: {list(checkpoint.keys())}")
    
    # Create model and load
    print("\nCreating model...")
    student = create_student_model(model_type='tiny')
    student.load_state_dict(checkpoint['student_state_dict'])
    print("✓ Model loaded successfully!")
    
    # Count parameters
    params = sum(p.numel() for p in student.parameters())
    print(f"Model parameters: {params:,}")
    
    # Save cleanly
    print(f"\nSaving to {output_path}...")
    torch.save({
        'student_state_dict': student.state_dict(),
        'epoch': checkpoint.get('epoch', 5),
        'loss': checkpoint.get('loss'),
        'config': {
            'student_type': 'tiny',
            'teacher_type': 'qwen2vl',
        }
    }, output_path)
    print(f"✓ Final model saved successfully!")
    
    # Check file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    checkpoint_path = "outputs/checkpoint_epoch_5.pt"
    output_path = "outputs/final_student_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found")
        sys.exit(1)
    
    extract_model(checkpoint_path, output_path)

