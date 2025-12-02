"""
Knowledge Distillation Framework for Vision-Language Models
Distills larger VLMs (teacher) to smaller models (student) for neuro-nav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import logging
import os
import sys

# Add parent directory to path to import neuro-nav modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from neuro_nav_vlm.conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
    from neuro_nav_vlm.conceptgraph.vlm.florence2_model import Florence2Model
except ImportError:
    # Try alternative import path
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neuro-nav-vlm'))
        from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel
        from conceptgraph.vlm.florence2_model import Florence2Model
    except ImportError:
        Qwen2VLModel = None
        Florence2Model = None
        logging.warning("Could not import teacher models. Make sure neuro-nav-vlm is available.")

logger = logging.getLogger(__name__)


class VLMKnowledgeDistillation:
    """
    Knowledge distillation framework for distilling large VLMs to smaller models.
    
    Supports:
    - Qwen2-VL → Smaller VLM student
    - Florence-2 → Smaller detection/captioning model
    - Task-specific distillation (captioning, relationships, etc.)
    """
    
    def __init__(
        self,
        teacher_model: Union[Qwen2VLModel, Florence2Model, None],
        student_model: nn.Module,
        device: str = "cuda:0",
        temperature: float = 4.0,
        alpha: float = 0.7,  # Weight for distillation loss vs hard labels
    ):
        """
        Initialize distillation framework.
        
        Args:
            teacher_model: Pre-trained teacher VLM (Qwen2VLModel or Florence2Model)
            student_model: Student model (nn.Module)
            device: Device to run on
            temperature: Temperature for softmax in distillation
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        self.teacher = teacher_model
        self.student = student_model
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        
        if self.teacher is not None:
            if hasattr(self.teacher, 'model'):
                self.teacher.model.eval()  # Teacher in eval mode
            else:
                self.teacher.eval()
        
        self.student.train()  # Student in train mode
        
        # Move models to device
        self.student = self.student.to(device)
        if self.teacher is not None:
            if hasattr(self.teacher, 'device'):
                # Teacher should already be on device
                pass
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model logits [batch, seq_len, vocab_size]
            teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
            labels: Optional ground truth labels for hard loss
            
        Returns:
            Dictionary with 'loss', 'distillation_loss', 'hard_loss'
        """
        # Ensure logits are on same device and have compatible shapes
        if student_logits.shape != teacher_logits.shape:
            # Handle shape mismatches (e.g., different sequence lengths)
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # Soft targets from teacher (with temperature)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        
        # KL divergence loss (distillation loss)
        distillation_loss = F.kl_div(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_probs.view(-1, teacher_probs.size(-1)),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard label loss (if labels provided)
        hard_loss = None
        if labels is not None:
            # Ensure labels match sequence length
            if labels.size(1) != student_logits.size(1):
                min_len = min(labels.size(1), student_logits.size(1))
                labels = labels[:, :min_len]
                student_logits = student_logits[:, :min_len, :]
            
            hard_loss = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
        
        # Combined loss
        if hard_loss is not None:
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        else:
            total_loss = distillation_loss
        
        return {
            'loss': total_loss,
            'distillation_loss': distillation_loss,
            'hard_loss': hard_loss if hard_loss is not None else torch.tensor(0.0).to(self.device),
        }
    
    def distill_captioning_batch(
        self,
        images: List[Image.Image],
        prompts: List[str],
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """
        Distill a single batch for captioning task.
        
        Args:
            images: List of PIL Images
            prompts: List of text prompts
            optimizer: Optimizer for student model
            
        Returns:
            Dictionary with loss values
        """
        if self.teacher is None:
            raise ValueError("Teacher model is not available")
        
        # Get teacher outputs (soft targets)
        teacher_outputs = []
        teacher_logits_list = []
        
        with torch.no_grad():
            for img, prompt in zip(images, prompts):
                if isinstance(self.teacher, Qwen2VLModel):
                    # For Qwen2-VL, we need to get logits
                    # This is a simplified version - you may need to adapt based on model internals
                    try:
                        # Try to get logits from teacher
                        teacher_output = self._get_teacher_logits(img, prompt)
                        teacher_logits_list.append(teacher_output)
                    except:
                        # Fallback: get text output and convert to logits approximation
                        text_output = self.teacher.caption_image(img, detail_level="detailed")
                        # For now, we'll use a placeholder - you'd need to implement proper logit extraction
                        teacher_outputs.append(text_output)
                elif isinstance(self.teacher, Florence2Model):
                    text_output = self.teacher.caption_image(img, detail_level="detailed")
                    teacher_outputs.append(text_output)
        
        # Get student outputs
        student_outputs = self._get_student_outputs(images, prompts)
        
        # For now, if we don't have logits, we'll use a simplified approach
        # In practice, you'd want to extract actual logits from both models
        if len(teacher_logits_list) > 0 and isinstance(student_outputs, torch.Tensor):
            # We have logits from both
            teacher_logits = torch.stack(teacher_logits_list).to(self.device)
            loss_dict = self.compute_distillation_loss(
                student_outputs,
                teacher_logits,
            )
        else:
            # Fallback: use a simple reconstruction loss
            # This is a placeholder - implement proper loss based on your needs
            loss_dict = {
                'loss': torch.tensor(0.0, requires_grad=True).to(self.device),
                'distillation_loss': torch.tensor(0.0).to(self.device),
                'hard_loss': torch.tensor(0.0).to(self.device),
            }
            logger.warning("Using placeholder loss - implement proper logit extraction for full distillation")
        
        # Backward pass
        optimizer.zero_grad()
        
        if scaler is not None:
            # Mixed precision backward
            scaler.scale(loss_dict['loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward
            loss_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            optimizer.step()
        
        return {
            'loss': loss_dict['loss'].item(),
            'distillation_loss': loss_dict['distillation_loss'].item(),
            'hard_loss': loss_dict['hard_loss'].item(),
        }
    
    def _get_teacher_logits(self, image: Image.Image, prompt: str) -> torch.Tensor:
        """
        Extract logits from teacher model.
        This needs to be implemented based on your specific teacher model architecture.
        """
        # Placeholder - implement based on your teacher model
        # For Qwen2-VL, you'd need to access the model's forward pass
        raise NotImplementedError(
            "Implement based on your teacher model architecture. "
            "You need to access the model's logits, not just text outputs."
        )
    
    def _get_student_outputs(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        """
        Get student model outputs (logits).
        Implement based on your student architecture.
        """
        # Placeholder - implement based on your student model
        raise NotImplementedError("Implement based on your student model architecture")
    
    def save_student_model(self, save_path: str):
        """Save the distilled student model."""
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'model_config': getattr(self.student, 'config', {}),
        }, os.path.join(save_path, 'student_model.pt'))
        logger.info(f"Student model saved to {save_path}")


class TaskSpecificDistillation:
    """
    Distill specific tasks (caption refinement, relationships) separately.
    More focused and potentially more effective than full model distillation.
    """
    
    def __init__(
        self,
        teacher: Union[Qwen2VLModel, None],
        task: str = "caption_refinement",  # or "relationships", "querying"
    ):
        """
        Initialize task-specific distillation.
        
        Args:
            teacher: Qwen2-VL teacher model
            task: Which task to distill ('caption_refinement', 'relationships', 'querying')
        """
        self.teacher = teacher
        self.task = task
    
    def create_training_dataset(
        self,
        scene_images: List[Image.Image],
        object_captions: List[List[str]],
    ) -> List[Dict]:
        """
        Create training dataset from neuro-nav scenes.
        
        Args:
            scene_images: Images from neuro-nav scenes
            object_captions: Captions for objects in each scene
            
        Returns:
            List of training examples
        """
        if self.teacher is None:
            raise ValueError("Teacher model is not available")
        
        dataset = []
        
        logger.info(f"Creating training dataset for task: {self.task}")
        logger.info(f"Processing {len(scene_images)} scenes...")
        
        for idx, (img, captions) in enumerate(zip(scene_images, object_captions)):
            if idx % 10 == 0:
                logger.info(f"Processing scene {idx}/{len(scene_images)}")
            
            # Get teacher outputs
            try:
                if self.task == "caption_refinement":
                    teacher_output = self.teacher.refine_caption(img, captions)
                elif self.task == "relationships":
                    # Would need object pairs - placeholder
                    teacher_output = None
                elif self.task == "querying":
                    # Would need queries - placeholder
                    teacher_output = None
                else:
                    teacher_output = None
                
                if teacher_output is not None:
                    dataset.append({
                        'image': img,
                        'input': captions,
                        'teacher_output': teacher_output,
                    })
            except Exception as e:
                logger.warning(f"Error processing scene {idx}: {e}")
                continue
        
        logger.info(f"Created {len(dataset)} training examples")
        return dataset

