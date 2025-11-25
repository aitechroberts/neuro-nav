"""
Query System for ConceptGraphs Scene Understanding

This module implements 5 different query engines to test the research question:
"What computational mechanisms are necessary for semantic understanding?"

Setups:
- Setup 1: GPT-4V (CLIP + cloud text-based reasoning)
- Setup 2: Qwen2.5-3B (CLIP + local text-based reasoning)
- Setup 3: CLIP-only (visual grounding, no reasoning)
- Setup 4: YOLO labels (symbolic matching only)
- Setup 5: GPT-4V with Vision (CLIP + cloud vision-language reasoning)
"""

from .base_query_engine import BaseQueryEngine, QueryResult
from .setup1_gpt4_query import GPT4QueryEngine
from .setup2_local_vlm_query import LocalVLMQueryEngine
from .setup3_clip_query import CLIPQueryEngine
from .setup4_yolo_query import YOLOQueryEngine
from .setup5_gpt4_vision import GPT4VisionQueryEngine

__all__ = [
    'BaseQueryEngine',
    'QueryResult',
    'GPT4QueryEngine',
    'LocalVLMQueryEngine',
    'CLIPQueryEngine',
    'YOLOQueryEngine',
    'GPT4VisionQueryEngine',
]
