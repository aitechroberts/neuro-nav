"""
Setup 3: CLIP-Only Query Engine

This setup tests visual grounding without reasoning.

Components:
- CLIP visual features (from mapping)
- CLIP text encoder for queries
- Text-to-image matching (cosine similarity)

What it tests:
- Does visual grounding improve over symbolic reasoning?
- Can retrieval alone solve queries without reasoning?

"""

from typing import List, Dict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .base_query_engine import BaseQueryEngine, QueryResult


class CLIPQueryEngine(BaseQueryEngine):
    """
    Query engine using CLIP for visual-semantic matching.

    Uses pre-computed CLIP features from objects + CLIP text encoder for queries.
    """

    def __init__(self, objects: List[Dict], config: Dict = None):
        super().__init__(objects, config)
        self.setup_name = "Setup 3: CLIP (Visual Grounding)"

        # Load CLIP model (same as used during mapping)
        print(f"  Loading CLIP model...")
        model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"  ✓ Loaded CLIP on {self.device}")

        # Extract and stack all object CLIP features
        self._prepare_object_features()

    def _prepare_object_features(self):
        """
        Extract CLIP features from all objects and stack them.
        """
        self.object_features = []

        for obj in self.objects:
            clip_ft = obj.get('clip_ft')
            if clip_ft is None:
                # If no CLIP feature, create zero vector
                clip_ft = torch.zeros(1024)
            elif isinstance(clip_ft, np.ndarray):
                clip_ft = torch.from_numpy(clip_ft)

            self.object_features.append(clip_ft)

        # Stack into (N, 1024) tensor
        self.object_features = torch.stack(self.object_features).to(self.device)

        # Normalize for cosine similarity
        self.object_features = F.normalize(self.object_features, dim=-1)

        print(f"  Prepared features for {len(self.object_features)} objects")

    def answer(self, query: str) -> QueryResult:
        """
        Answer query using CLIP text-to-image matching.

        Process:
        1. Encode query text with CLIP text encoder
        2. Compute cosine similarity to all object image features
        3. Return object with highest similarity
        """
        if len(self.objects) == 0:
            return QueryResult(
                object_id=None,
                object_index=-1,
                confidence=0.0,
                reasoning="No objects in scene",
                class_name="none"
            )

        # 1. Encode query text
        with torch.no_grad():
            inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            query_features = self.model.get_text_features(**inputs)
            query_features = F.normalize(query_features, dim=-1)  # (1, 1024)

        # 2. Compute cosine similarity to all objects
        similarities = torch.mm(query_features, self.object_features.T).squeeze()  # (N,)

        # Handle single object case
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        # 3. Find best match
        best_idx = torch.argmax(similarities).item()
        confidence = float(similarities[best_idx])
        best_obj = self.objects[best_idx]

        # Get top-3 matches for comparison
        top_k = min(3, len(similarities))
        top_similarities, top_indices = torch.topk(similarities, top_k)

        matched_objects = []
        for sim, idx in zip(top_similarities, top_indices):
            obj = self.objects[int(idx)]
            matched_objects.append({
                'index': int(idx),
                'class_name': obj.get('class_name', 'unknown'),
                'similarity': float(sim)
            })

        return QueryResult(
            object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
            object_index=best_idx,
            confidence=confidence,
            reasoning=f"CLIP visual similarity (retrieval, no reasoning)",
            class_name=best_obj.get('class_name', 'unknown'),
            matched_objects=matched_objects
        )

    def encode_text(self, text: str) -> torch.Tensor:
        """Helper to encode arbitrary text with CLIP."""
        with torch.no_grad():
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            features = self.model.get_text_features(**inputs)
            features = F.normalize(features, dim=-1)

        return features

    def __str__(self):
        return f"{self.setup_name}\n  Device: {self.device}\n  Objects: {len(self.objects)}"
