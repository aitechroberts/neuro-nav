"""
Setup 1: GPT-4V (CLIP + Cloud Reasoning)

This is the FULL-STACK setup with best performance.

Components:
- CLIP for candidate retrieval
- GPT-4V for advanced reasoning
- Two-stage: retrieval → reasoning

What it tests:
- Best possible performance (baseline)
- What does advanced reasoning contribute?

"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import os
import base64
from io import BytesIO

from .base_query_engine import BaseQueryEngine, QueryResult


class GPT4QueryEngine(BaseQueryEngine):
    """
    Query engine using CLIP + GPT-4V for reasoning.

    Two-stage process:
    1. CLIP retrieves top-K candidates
    2. GPT-4V reasons about candidates (with text descriptions)
    """

    def __init__(self, objects: List[Dict], config: Dict = None):
        super().__init__(objects, config)
        self.setup_name = "Setup 1: GPT-4V (Cloud Reasoning)"

        # Configuration
        self.top_k = config.get('top_k', 3) if config else 3
        self.use_gpt4 = True

        # Check for OpenAI API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print(f"  ⚠ Warning: OPENAI_API_KEY not found in environment")
            print(f"  GPT-4V reasoning will be disabled, falling back to CLIP-only")
            self.use_gpt4 = False

        # Load CLIP for candidate retrieval
        print(f"  Loading CLIP model...")
        model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = self.clip_model.to(self.device)
        print(f"  ✓ Loaded CLIP on {self.device}")

        # Prepare object features
        self._prepare_object_features()

        # Initialize OpenAI client
        if self.use_gpt4:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"  ✓ OpenAI client initialized")
            except ImportError:
                print(f"  ⚠ OpenAI library not installed. Run: pip install openai")
                self.use_gpt4 = False
            except Exception as e:
                print(f"  ⚠ Failed to initialize OpenAI client: {e}")
                self.use_gpt4 = False

    def _prepare_object_features(self):
        """Extract and normalize CLIP features."""
        self.object_features = []

        for obj in self.objects:
            clip_ft = obj.get('clip_ft')
            if clip_ft is None:
                clip_ft = torch.zeros(1024)
            elif isinstance(clip_ft, np.ndarray):
                clip_ft = torch.from_numpy(clip_ft)

            self.object_features.append(clip_ft)

        self.object_features = torch.stack(self.object_features).to(self.device)
        self.object_features = F.normalize(self.object_features, dim=-1)

        print(f"  Prepared CLIP features for {len(self.object_features)} objects")

    def _retrieve_candidates_with_clip(self, query: str, top_k: int = 3) -> tuple:
        """
        Use CLIP to retrieve top-K candidate objects.

        Returns:
            (candidate_indices, candidate_scores)
        """
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[query],
                return_tensors="pt",
                padding=True
            ).to(self.device)

            query_features = self.clip_model.get_text_features(**inputs)
            query_features = F.normalize(query_features, dim=-1)

        similarities = torch.mm(query_features, self.object_features.T).squeeze()

        if similarities.dim() == 0:
            similarities = similarities.unsqueeze(0)

        # Get top-K
        actual_k = min(top_k, len(similarities))
        top_similarities, top_indices = torch.topk(similarities, actual_k)

        return top_indices.cpu().tolist(), top_similarities.cpu().tolist()

    def _reason_with_gpt4(
        self,
        query: str,
        candidate_indices: List[int],
        candidate_scores: List[float]
    ) -> QueryResult:
        """
        Use GPT-4V to reason about candidate objects.

        Since we don't have RGB images readily available, we'll use
        text-based reasoning with object metadata.
        """
        if not self.use_gpt4:
            # Fallback: return top CLIP match
            best_idx = candidate_indices[0]
            best_obj = self.objects[best_idx]

            return QueryResult(
                object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
                object_index=best_idx,
                confidence=candidate_scores[0],
                reasoning="GPT-4V not available, using CLIP-only",
                class_name=best_obj.get('class_name', 'unknown')
            )

        # Build candidate descriptions
        candidate_info = []
        for i, (idx, score) in enumerate(zip(candidate_indices, candidate_scores)):
            obj = self.objects[idx]
            class_name = obj.get('class_name', 'unknown')
            num_detections = obj.get('num_detections', 1)
            obj_conf = self.compute_object_confidence(obj)

            # Get class distribution (alternative hypotheses)
            class_ids = obj.get('class_id', [])
            if class_ids:
                from collections import Counter
                counter = Counter(class_ids)
                # Get class names if available
                alternatives = counter.most_common(3)
                alt_str = ", ".join([f"{self.objects[idx].get('class_name', 'unknown')}({cnt})"
                                    for val, cnt in alternatives[:3]])
            else:
                alt_str = class_name

            candidate_info.append({
                'index': i,
                'object_index': idx,
                'class_name': class_name,
                'clip_score': score,
                'detection_confidence': obj_conf,
                'num_detections': num_detections,
                'alternatives': alt_str
            })

        # Build prompt for GPT-4
        prompt = f"""You are a robot analyzing a 3D scene. A user has asked: "{query}"

You have identified {len(candidate_info)} candidate objects using visual similarity (CLIP). Please analyze which object best answers the user's query.

Candidates:
"""
        for c in candidate_info:
            prompt += f"\n{c['index'] + 1}. {c['class_name'].upper()}"
            prompt += f"\n   - Visual similarity to query: {c['clip_score']:.3f}"
            prompt += f"\n   - Detection confidence: {c['detection_confidence']:.3f}"
            prompt += f"\n   - Number of detections: {c['num_detections']}"
            prompt += f"\n   - Alternative classifications: {c['alternatives']}"

        prompt += f"""

Based on the user's query and the candidate information:
1. Which object (1-{len(candidate_info)}) best answers the query?
2. What is your confidence (0.0-1.0)?
3. Briefly explain your reasoning (1-2 sentences).

Respond in this exact format:
OBJECT: <number>
CONFIDENCE: <0.0-1.0>
REASONING: <explanation>
"""

        try:
            # Call GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using GPT-4o (latest, faster, cheaper)
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant analyzing 3D scenes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Extract fields
            object_num = None
            confidence = None
            reasoning = None

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('OBJECT:'):
                    try:
                        object_num = int(line.split('OBJECT:')[1].strip()) - 1  # 0-indexed
                    except:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split('CONFIDENCE:')[1].strip())
                    except:
                        pass
                elif line.startswith('REASONING:'):
                    reasoning = line.split('REASONING:')[1].strip()

            # Validate and get result
            if object_num is not None and 0 <= object_num < len(candidate_indices):
                best_idx = candidate_indices[object_num]
                best_obj = self.objects[best_idx]
                final_confidence = confidence if confidence else candidate_scores[object_num]
                final_reasoning = reasoning if reasoning else "GPT-4V reasoning"
            else:
                # Fallback to top CLIP match
                best_idx = candidate_indices[0]
                best_obj = self.objects[best_idx]
                final_confidence = candidate_scores[0]
                final_reasoning = "GPT-4V parsing failed, using top CLIP match"

            return QueryResult(
                object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
                object_index=best_idx,
                confidence=final_confidence,
                reasoning=final_reasoning,
                class_name=best_obj.get('class_name', 'unknown'),
                matched_objects=[
                    {
                        'index': idx,
                        'class_name': self.objects[idx].get('class_name', 'unknown'),
                        'similarity': score
                    }
                    for idx, score in zip(candidate_indices, candidate_scores)
                ]
            )

        except Exception as e:
            print(f"  ⚠ GPT-4V error: {e}")
            # Fallback to CLIP
            best_idx = candidate_indices[0]
            best_obj = self.objects[best_idx]

            return QueryResult(
                object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
                object_index=best_idx,
                confidence=candidate_scores[0],
                reasoning=f"GPT-4V failed ({str(e)}), using CLIP-only",
                class_name=best_obj.get('class_name', 'unknown')
            )

    def answer(self, query: str) -> QueryResult:
        """
        Answer query using CLIP + GPT-4V.

        Process:
        1. CLIP retrieves top-K candidates
        2. GPT-4V reasons about candidates
        3. Return best match with reasoning
        """
        if len(self.objects) == 0:
            return QueryResult(
                object_id=None,
                object_index=-1,
                confidence=0.0,
                reasoning="No objects in scene",
                class_name="none"
            )

        # Stage 1: CLIP retrieval
        candidate_indices, candidate_scores = self._retrieve_candidates_with_clip(
            query, top_k=self.top_k
        )

        # Stage 2: GPT-4V reasoning
        result = self._reason_with_gpt4(query, candidate_indices, candidate_scores)

        return result

    def __str__(self):
        gpt4_status = "enabled" if self.use_gpt4 else "disabled"
        return f"{self.setup_name}\n  CLIP: loaded\n  GPT-4V: {gpt4_status}\n  Objects: {len(self.objects)}"
