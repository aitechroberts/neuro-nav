"""
Script to load and use the distilled student model
"""

import torch
import os
import sys
from PIL import Image

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from distilled_model.student_models import create_student_model

def load_distilled_model(checkpoint_path: str, device: str = "cuda:0"):
    """
    Load a distilled student model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded student model
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model type from checkpoint or default
    if 'config' in checkpoint:
        student_type = checkpoint['config'].get('student_type', 'tiny')
    else:
        # Try to infer from checkpoint size or use default
        student_type = 'tiny'
        print("Warning: Could not determine student type from checkpoint, using 'tiny'")
    
    print(f"Student model type: {student_type}")
    
    # Create model
    student = create_student_model(model_type=student_type)
    student.load_state_dict(checkpoint['student_state_dict'])
    student = student.to(device)
    student.eval()
    
    # Print info
    if 'epoch' in checkpoint:
        print(f"Model trained for {checkpoint['epoch']} epochs")
    if 'loss' in checkpoint:
        print(f"Final loss: {checkpoint['loss']:.4f}")
    
    params = sum(p.numel() for p in student.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Model loaded successfully on {device}")
    
    return student


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load distilled model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoint_epoch_10.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        print("\nAvailable checkpoints:")
        output_dir = os.path.dirname(args.checkpoint) or "outputs"
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith('.pt'):
                    print(f"  - {os.path.join(output_dir, f)}")
        sys.exit(1)
    
    model = load_distilled_model(args.checkpoint, args.device)
    print("\n✓ Model loaded successfully!")
    print("You can now use this model for inference.")

