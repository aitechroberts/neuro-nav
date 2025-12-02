"""
Vision-Language Models for ConceptGraph
Replaces YOLO+CLIP+LLaVA+GPT-4 with modern VLMs
"""

from .florence2_model import Florence2Model
from .qwen2vl_model import Qwen2VLModel

__all__ = ["Florence2Model", "Qwen2VLModel"]

