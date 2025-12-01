"""
Training script for knowledge distillation
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from pathlib import Path
from typing import Optional
import shutil

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distilled_model.distillation import VLMKnowledgeDistillation, TaskSpecificDistillation
from distilled_model.student_models import create_student_model
from distilled_model.data_loader import create_distillation_dataloader

# Try to import teacher models
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neuro-nav-vlm'))
    from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
    from conceptgraph.vlm.florence2_model import Florence2Model
    TEACHER_AVAILABLE = True
except ImportError:
    TEACHER_AVAILABLE = False
    logging.warning("Teacher models not available. Training will use placeholder teacher.")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_teacher_model(teacher_type: str = "qwen2vl", model_name: str = None, device: str = "cuda:0", use_cpu_offload: bool = True):
    """
    Load teacher model.
    
    Args:
        teacher_type: Type of teacher model
        model_name: Model name
        device: Device for student (teacher may be on CPU if use_cpu_offload=True)
        use_cpu_offload: If True, load teacher on CPU to save GPU memory
    """
    if not TEACHER_AVAILABLE:
        logger.warning("Teacher models not available. Returning None.")
        return None
    
    # For memory efficiency, use CPU for teacher if requested
    teacher_device = "cpu" if use_cpu_offload else device
    
    if teacher_type.lower() == "qwen2vl":
        if model_name is None:
            model_name = "Qwen/Qwen2-VL-2B-Instruct"
        logger.info(f"Loading Qwen2-VL teacher: {model_name} on {teacher_device}")
        # Fix deprecation warning: use dtype instead of torch_dtype
        try:
            teacher = Qwen2VLModel(model_name=model_name, device=teacher_device)
        except TypeError:
            # Fallback if the model doesn't support dtype parameter yet
            teacher = Qwen2VLModel(model_name=model_name, device=teacher_device)
        return teacher
    
    elif teacher_type.lower() == "florence2":
        if model_name is None:
            model_name = "microsoft/Florence-2-large"
        logger.info(f"Loading Florence-2 teacher: {model_name} on {teacher_device}")
        teacher = Florence2Model(model_name=model_name, device=teacher_device)
        return teacher
    
    else:
        raise ValueError(f"Unknown teacher type: {teacher_type}")


def train_distillation(
    data_root: str,
    output_dir: str,
    teacher_type: str = "qwen2vl",
    student_type: str = "tiny",
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    device: str = "cuda:0",
    scene_ids: Optional[list] = None,
    task: str = "caption_refinement",
    max_samples: Optional[int] = None,
    save_freq: int = 5,
    use_cpu_offload: bool = True,
    use_gradient_checkpointing: bool = True,
    use_mixed_precision: bool = True,
):
    """
    Main training function for knowledge distillation.
    
    Args:
        data_root: Root directory for neuro-nav data
        output_dir: Directory to save trained models
        teacher_type: Type of teacher model ('qwen2vl' or 'florence2')
        student_type: Type of student model ('tiny' or 'phi2')
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        scene_ids: List of scene IDs to use
        task: Task type
        max_samples: Maximum number of samples
        save_freq: Save checkpoint every N epochs
    """
    logger.info("=" * 60)
    logger.info("Starting Knowledge Distillation Training")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check available disk space
    try:
        stat = shutil.disk_usage(output_dir)
        free_gb = stat.free / (1024**3)
        logger.info(f"Available disk space: {free_gb:.2f} GB")
        if free_gb < 1.0:
            logger.warning("Low disk space! Model saving may fail.")
    except Exception:
        pass  # Ignore if we can't check disk space
    
    # Clear GPU cache
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        logger.info(f"GPU memory before loading: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    
    # Load teacher model (potentially on CPU to save GPU memory)
    logger.info("Loading teacher model...")
    if use_cpu_offload:
        logger.info("Using CPU offloading for teacher model to save GPU memory")
        logger.info("Teacher will run on CPU (slower but saves ~4-8GB GPU memory)")
    teacher = load_teacher_model(teacher_type, device=device, use_cpu_offload=use_cpu_offload)
    
    if teacher is None:
        logger.warning("Teacher model not available. Training will be limited.")
        logger.warning("Please ensure neuro-nav-vlm is properly set up.")
        return
    
    # Create student model
    logger.info(f"Creating student model: {student_type}")
    student = create_student_model(model_type=student_type)
    
    # Enable gradient checkpointing to save memory
    if use_gradient_checkpointing and hasattr(student, 'transformer'):
        if hasattr(student.transformer, 'gradient_checkpointing_enable'):
            student.transformer.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for memory efficiency")
    
    # Move student to device
    try:
        student = student.to(device)
        if device.startswith("cuda"):
            logger.info(f"GPU memory after loading student: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"Out of memory loading student model. Try:")
        logger.error("  1. Use --use_cpu_offload (teacher on CPU)")
        logger.error("  2. Reduce --batch_size (e.g., --batch_size 2)")
        logger.error("  3. Use smaller student model (--student_type phi2)")
        logger.error("  4. Use gradient checkpointing (already enabled)")
        raise
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.model.parameters() if p.requires_grad) if hasattr(teacher, 'model') else 0
    student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info(f"Teacher parameters: {teacher_params:,}")
    logger.info(f"Student parameters: {student_params:,}")
    logger.info(f"Compression ratio: {teacher_params / student_params:.2f}x")
    
    # Create distillation framework
    distiller = VLMKnowledgeDistillation(
        teacher_model=teacher,
        student_model=student,
        device=device,
        temperature=4.0,
        alpha=0.7,
    )
    
    # Create data loader
    logger.info("Loading training data...")
    dataloader = create_distillation_dataloader(
        data_root=data_root,
        batch_size=batch_size,
        scene_ids=scene_ids,
        task=task,
        max_samples=max_samples,
    )
    
    logger.info(f"Training on {len(dataloader.dataset)} samples")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )
    
    # Mixed precision training scaler
    scaler = None
    if use_mixed_precision and device.startswith("cuda"):
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using mixed precision training (FP16/BF16)")
    
    # Training loop
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 60)
        
        student.train()
        epoch_losses = []
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['images']
            captions = batch['captions']
            
            # Create prompts
            prompts = []
            for cap_list in captions:
                if len(cap_list) > 0:
                    prompts.append(f"Describe this object: {cap_list[0]}")
                else:
                    prompts.append("Describe this object in detail.")
            
            # Distill batch
            try:
                # Clear cache before each batch
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                
                # Use mixed precision if enabled
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss_dict = distiller.distill_captioning_batch(
                            images=images,
                            prompts=prompts,
                            optimizer=optimizer,
                            scaler=scaler,
                        )
                else:
                    loss_dict = distiller.distill_captioning_batch(
                        images=images,
                        prompts=prompts,
                        optimizer=optimizer,
                    )
                
                epoch_losses.append(loss_dict['loss'])
                progress_bar.set_postfix({
                    'loss': f"{loss_dict['loss']:.4f}",
                    'dist_loss': f"{loss_dict['distillation_loss']:.4f}",
                })
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Out of memory in batch {batch_idx}")
                logger.error("Try reducing batch_size or using CPU offload")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                continue
        
        # Update learning rate
        scheduler.step()
        
        # Calculate epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        logger.info(f"Average loss: {avg_loss:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'learning_rate': scheduler.get_last_lr()[0],
        })
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or avg_loss < best_loss:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            try:
                # Save with compression to reduce file size
                torch.save({
                    'epoch': epoch + 1,
                    'student_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'training_history': training_history,
                }, checkpoint_path, _use_new_zipfile_serialization=False)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
                logger.warning("Trying to save with compression...")
                try:
                    # Try saving just the model state dict
                    torch.save({
                        'epoch': epoch + 1,
                        'student_state_dict': student.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_path, _use_new_zipfile_serialization=False)
                    logger.info(f"Saved minimal checkpoint to {checkpoint_path}")
                except Exception as e2:
                    logger.error(f"Failed to save checkpoint even with minimal data: {e2}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_checkpoint = os.path.join(output_dir, "best_model.pt")
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'student_state_dict': student.state_dict(),
                        'loss': avg_loss,
                    }, best_checkpoint, _use_new_zipfile_serialization=False)
                    logger.info(f"New best model saved (loss: {avg_loss:.4f})")
                except Exception as e:
                    logger.warning(f"Failed to save best model: {e}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_student_model.pt")
    try:
        torch.save({
            'student_state_dict': student.state_dict(),
            'training_history': training_history,
            'config': {
                'student_type': student_type,
                'teacher_type': teacher_type,
                'task': task,
                'num_epochs': num_epochs,
            },
        }, final_model_path, _use_new_zipfile_serialization=False)
        logger.info(f"Final model saved to {final_model_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
        logger.info("Trying to save minimal model...")
        try:
            # Try saving just the state dict
            minimal_path = os.path.join(output_dir, "final_student_model_minimal.pt")
            torch.save({
                'student_state_dict': student.state_dict(),
            }, minimal_path, _use_new_zipfile_serialization=False)
            logger.info(f"Minimal model saved to {minimal_path}")
        except Exception as e2:
            logger.error(f"Failed to save even minimal model: {e2}")
            logger.error("Model training completed but could not save. State dict is in memory.")
    
    # Save training history (separate file, smaller)
    history_path = os.path.join(output_dir, "training_history.json")
    try:
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.warning(f"Failed to save training history: {e}")
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train distilled VLM model")
    
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for neuro-nav data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="distilled_model/outputs",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--teacher_type",
        type=str,
        default="qwen2vl",
        choices=["qwen2vl", "florence2"],
        help="Type of teacher model"
    )
    parser.add_argument(
        "--student_type",
        type=str,
        default="tiny",
        choices=["tiny", "phi2"],
        help="Type of student model"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use"
    )
    parser.add_argument(
        "--scene_ids",
        type=str,
        nargs="+",
        default=None,
        help="Scene IDs to use (e.g., room0 room1)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="caption_refinement",
        choices=["caption_refinement", "relationships", "querying"],
        help="Task type"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use"
    )
    # CPU offloading is enabled by default to save GPU memory
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Load teacher model on CPU to save GPU memory (default: enabled)"
    )
    parser.add_argument(
        "--no_cpu_offload",
        action="store_false",
        dest="use_cpu_offload",
        help="Disable CPU offloading (teacher on GPU, uses more memory)"
    )
    # Set default to True (CPU offload enabled by default)
    parser.set_defaults(use_cpu_offload=True)
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--use_mixed_precision",
        action="store_true",
        default=True,
        help="Use mixed precision training (FP16/BF16)"
    )
    
    args = parser.parse_args()
    
    train_distillation(
        data_root=args.data_root,
        output_dir=args.output_dir,
        teacher_type=args.teacher_type,
        student_type=args.student_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        scene_ids=args.scene_ids,
        task=args.task,
        max_samples=args.max_samples,
        use_cpu_offload=args.use_cpu_offload,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_mixed_precision=args.use_mixed_precision,
    )


if __name__ == "__main__":
    main()

