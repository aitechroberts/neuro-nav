"""
Student model architectures for knowledge distillation
Smaller models that learn from teacher VLMs
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TinyVLMStudent(nn.Module):
    """
    Small VLM student model based on TinyLlama architecture.
    Designed to be distilled from Qwen2-VL or Florence-2.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        num_layers: int = 22,
        num_attention_heads: int = 32,
        vision_embed_dim: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
    ):
        """
        Initialize TinyVLM student model.
        
        Args:
            vocab_size: Vocabulary size for language model
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            vision_embed_dim: Vision encoder embedding dimension
            image_size: Input image size
            patch_size: Vision patch size
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.vision_embed_dim = vision_embed_dim
        
        # Vision encoder (simplified ViT)
        self.vision_encoder = self._build_vision_encoder(
            image_size, patch_size, vision_embed_dim
        )
        
        # Projection layer to match hidden_size
        self.vision_proj = nn.Linear(vision_embed_dim, hidden_size)
        
        # Language model (simplified transformer)
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, 1024, hidden_size))
    
    def _build_vision_encoder(self, image_size: int, patch_size: int, embed_dim: int) -> nn.Module:
        """Build a simple vision encoder."""
        num_patches = (image_size // patch_size) ** 2
        
        return nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),  # [B, C, H*W]
            nn.LayerNorm(embed_dim),
        )
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Image tensor [B, C, H, W]
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Logits [B, seq_len, vocab_size]
        """
        batch_size = input_ids.size(0) if input_ids is not None else images.size(0)
        
        # Process vision
        vision_features = None
        if images is not None:
            # Vision encoder
            vision_out = self.vision_encoder(images)  # [B, embed_dim, num_patches]
            vision_out = vision_out.permute(0, 2, 1)  # [B, num_patches, embed_dim]
            vision_features = self.vision_proj(vision_out)  # [B, num_patches, hidden_size]
        
        # Process text
        text_features = None
        if input_ids is not None:
            text_features = self.text_embedding(input_ids)  # [B, seq_len, hidden_size]
        
        # Concatenate vision and text
        if vision_features is not None and text_features is not None:
            # Combine vision and text
            combined = torch.cat([vision_features, text_features], dim=1)
            seq_len = combined.size(1)
        elif vision_features is not None:
            combined = vision_features
            seq_len = vision_features.size(1)
        elif text_features is not None:
            combined = text_features
            seq_len = text_features.size(1)
        else:
            raise ValueError("Either images or input_ids must be provided")
        
        # Add positional embeddings
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # Interpolate if needed
            pos_emb = nn.functional.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)
        
        combined = combined + pos_emb
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Extend mask for vision tokens if present
            if vision_features is not None:
                vision_mask = torch.ones(
                    batch_size, vision_features.size(1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # Transformer
        if attention_mask is not None:
            # Convert to attention mask format (invert for transformer)
            attn_mask = (1 - attention_mask).bool()
            output = self.transformer(combined, src_key_padding_mask=attn_mask)
        else:
            output = self.transformer(combined)
        
        # Language modeling head
        logits = self.lm_head(output)
        
        return logits


class Phi2VLMStudent(nn.Module):
    """
    Student model based on Phi-2 architecture with vision capabilities.
    Smaller alternative to TinyVLMStudent.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 2560,
        num_layers: int = 32,
        num_attention_heads: int = 32,
        vision_embed_dim: int = 512,
    ):
        """Initialize Phi2-based VLM student."""
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Simpler vision encoder
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, vision_embed_dim, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(1),
            nn.Linear(vision_embed_dim * 14 * 14, hidden_size),
        )
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Simplified transformer (using standard PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.lm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass."""
        features = []
        
        if images is not None:
            vision_feat = self.vision_encoder(images)  # [B, hidden_size]
            vision_feat = vision_feat.unsqueeze(1)  # [B, 1, hidden_size]
            features.append(vision_feat)
        
        if input_ids is not None:
            text_feat = self.text_embedding(input_ids)  # [B, seq_len, hidden_size]
            features.append(text_feat)
        
        if len(features) == 0:
            raise ValueError("Either images or input_ids must be provided")
        
        combined = torch.cat(features, dim=1)  # [B, total_seq_len, hidden_size]
        
        # Transformer
        if attention_mask is not None:
            attn_mask = (1 - attention_mask).bool()
            output = self.transformer(combined, src_key_padding_mask=attn_mask)
        else:
            output = self.transformer(combined)
        
        logits = self.lm_head(output)
        return logits


def create_student_model(
    model_type: str = "tiny",
    **kwargs
) -> nn.Module:
    """
    Factory function to create student models.
    
    Args:
        model_type: Type of student model ('tiny' or 'phi2')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Student model instance
    """
    if model_type.lower() == "tiny":
        return TinyVLMStudent(**kwargs)
    elif model_type.lower() == "phi2":
        return Phi2VLMStudent(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

