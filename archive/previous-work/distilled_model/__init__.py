"""
Knowledge Distillation for Neuro-Nav VLMs
Distills larger VLM models to smaller, more efficient student models
"""

from .distillation import VLMKnowledgeDistillation, TaskSpecificDistillation
from .student_models import TinyVLMStudent, Phi2VLMStudent
from .data_loader import NeuroNavDistillationDataset

__all__ = [
    "VLMKnowledgeDistillation",
    "TaskSpecificDistillation",
    "TinyVLMStudent",
    "Phi2VLMStudent",
    "NeuroNavDistillationDataset",
]

