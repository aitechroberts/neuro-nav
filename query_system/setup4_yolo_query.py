"""
Setup 4: YOLO Labels + Text Similarity

This is the MINIMAL setup that tests pure symbolic reasoning.

Components:
- YOLO labels only (no visual features)
- Sentence-transformers for text similarity
- Text-to-text matching

What it tests:
- Can symbolic/linguistic reasoning alone solve queries?
- How much does visual grounding contribute?

"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from .base_query_engine import BaseQueryEngine, QueryResult


class YOLOQueryEngine(BaseQueryEngine):
    """
    Query engine using only YOLO text labels + semantic text matching.

    No visual information - purely symbolic reasoning.
    """

    def __init__(self, objects: List[Dict], config: Dict = None):
        super().__init__(objects, config)
        self.setup_name = "Setup 4: YOLO Labels (Text-Only)"

        # Load lightweight text embedding model (~80MB)
        print(f"  Loading sentence-transformers model...")
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"  ✓ Loaded (384-dim embeddings)")

        # Pre-compute label embeddings for efficiency
        self._precompute_label_embeddings()

    def _precompute_label_embeddings(self):
        """
        Pre-compute embeddings for all unique labels in the scene.
        This makes queries faster.
        """
        # Extract all unique labels
        self.labels = []
        self.label_to_objects = {}  # Map label -> list of object indices

        for idx, obj in enumerate(self.objects):
            label = obj.get('class_name', 'unknown')
            if label not in self.label_to_objects:
                self.label_to_objects[label] = []
                self.labels.append(label)
            self.label_to_objects[label].append(idx)

        # Encode all labels
        if self.labels:
            self.label_embeddings = self.text_model.encode(
                self.labels,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        else:
            self.label_embeddings = None

        print(f"  Pre-computed embeddings for {len(self.labels)} unique labels: {self.labels}")

    def answer(self, query: str) -> QueryResult:
        """
        Answer query using text-to-text matching.

        Process:
        1. Encode query as text embedding
        2. Compare to all label embeddings (cosine similarity)
        3. Pick label with highest similarity
        4. Return an object with that label
        """
        if not self.labels:
            return QueryResult(
                object_id=None,
                object_index=-1,
                confidence=0.0,
                reasoning="No objects in scene",
                class_name="none"
            )

        # 1. Encode query
        query_embedding = self.text_model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False
        )

        # 2. Compute cosine similarity to all labels
        # Normalize embeddings
        query_embedding = query_embedding / torch.norm(query_embedding)
        label_embeddings_norm = self.label_embeddings / torch.norm(
            self.label_embeddings, dim=1, keepdim=True
        )

        # Cosine similarity
        similarities = torch.mm(
            query_embedding.unsqueeze(0),
            label_embeddings_norm.T
        ).squeeze()

        # Handle single label case
        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        # 3. Find best matching label
        best_idx = torch.argmax(similarities).item()
        best_label = self.labels[best_idx]
        confidence = float(similarities[best_idx])

        # 4. Get an object with that label (pick one with highest intrinsic confidence)
        candidate_indices = self.label_to_objects[best_label]
        candidate_objects = [self.objects[i] for i in candidate_indices]

        # Pick object with highest intrinsic confidence
        best_obj_idx = candidate_indices[0]
        best_obj_conf = self.compute_object_confidence(candidate_objects[0])

        for i, obj in zip(candidate_indices, candidate_objects):
            obj_conf = self.compute_object_confidence(obj)
            if obj_conf > best_obj_conf:
                best_obj_idx = i
                best_obj_conf = obj_conf

        best_obj = self.objects[best_obj_idx]

        # Build top-3 matches for comparison
        top_k = min(3, len(similarities))
        top_indices = torch.topk(similarities, top_k).indices.tolist()
        matched_objects = [
            {
                'label': self.labels[i],
                'similarity': float(similarities[i])
            }
            for i in top_indices
        ]

        return QueryResult(
            object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
            object_index=best_obj_idx,
            confidence=confidence,
            reasoning=f"Text similarity: '{query}' → '{best_label}' (no visual grounding)",
            class_name=best_label,
            matched_objects=matched_objects
        )

    def __str__(self):
        return f"{self.setup_name}\n  Labels: {self.labels}\n  Objects: {len(self.objects)}"
