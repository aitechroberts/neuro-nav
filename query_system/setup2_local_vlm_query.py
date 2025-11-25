"""
Setup 2: Local LLM (CLIP + Qwen2.5-3B)

This setup tests local reasoning capabilities.

Components:
- CLIP for initial candidate retrieval
- Qwen2.5-3B-Instruct (local LLM) for reasoning
- Two-stage: retrieval → reasoning

What it tests:
- Can local models match cloud performance?
- Cost vs quality tradeoff

"""

from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import io
import base64
import os
import sys

from .base_query_engine import BaseQueryEngine, QueryResult


class LocalVLMQueryEngine(BaseQueryEngine):
    """
    Query engine using CLIP + local LLM (Qwen2.5-3B-Instruct).

    Two-stage process:
    1. CLIP retrieves top-K candidates
    2. Qwen2.5 reasons about candidates to pick best
    """

    def __init__(self, objects: List[Dict], config: Dict = None):
        super().__init__(objects, config)
        self.setup_name = "Setup 2: Qwen2.5-3B (Local LLM)"

        # Configuration
        self.top_k = config.get('top_k', 3) if config else 3

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

        # Load Qwen2.5 (lazy loading - only when needed)
        self.llm_model = None
        self.llm_tokenizer = None
        print(f"  Qwen2.5-3B will be loaded on first query (lazy loading)")

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

    def _load_llm(self):
        """
        Lazy load Qwen2.5-3B-Instruct for text reasoning.

        This is a lightweight local LLM (3B parameters) that provides
        good reasoning capabilities without requiring heavy resources.
        """
        if self.llm_model is not None:
            return

        print(f"  Loading Qwen2.5-3B-Instruct...")

        try:
            model_name = "Qwen/Qwen2.5-3B-Instruct"

            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Load model (fp16 for efficiency)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.llm_model.eval()

            print(f"  ✓ Loaded Qwen2.5-3B-Instruct on {self.llm_model.device}")

        except Exception as e:
            print(f"  ⚠ Failed to load Qwen2.5: {e}")
            print(f"  Falling back to CLIP-only mode")
            self.llm_model = None
            self.llm_tokenizer = None

    def _retrieve_candidates_with_clip(self, query: str, top_k: int = 3) -> List[int]:
        """
        Use CLIP to retrieve top-K candidate objects.

        Returns:
            List of object indices (top-K matches)
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

    def _reason_with_llm(
        self,
        query: str,
        candidate_indices: List[int],
        candidate_scores: List[float]
    ) -> QueryResult:
        """
        Use Qwen2.5 to reason about top candidates.

        Uses text-based reasoning to select the best match from CLIP candidates.
        """
        if self.llm_model is None:
            # Fallback: just return top CLIP match
            best_idx = candidate_indices[0]
            best_obj = self.objects[best_idx]

            return QueryResult(
                object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
                object_index=best_idx,
                confidence=candidate_scores[0],
                reasoning="Local LLM not available, using CLIP-only",
                class_name=best_obj.get('class_name', 'unknown')
            )

        # Build text prompt for Qwen2.5
        candidate_descriptions = []
        for i, (idx, score) in enumerate(zip(candidate_indices, candidate_scores), 1):
            obj = self.objects[idx]
            class_name = obj.get('class_name', 'unknown')
            conf = self.compute_object_confidence(obj)
            num_detections = obj.get('num_detections', 1)
            candidate_descriptions.append(
                f"{i}. {class_name.upper()} (visual similarity: {score:.2f}, detection confidence: {conf:.2f}, detections: {num_detections})"
            )

        system_prompt = "You are a helpful assistant analyzing a 3D scene. Given a user query and candidate objects, select the best match and explain your reasoning."

        user_prompt = f"""User query: "{query}"

Top candidate objects from the scene:
{chr(10).join(candidate_descriptions)}

Which object (1-{len(candidate_indices)}) best answers the user's query? Consider the query semantics and object properties.

Respond in this exact format:
OBJECT: <number>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>"""

        try:
            # Format chat prompt for Qwen2.5
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize and generate
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True
                )

            # Decode response
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Parse response
            object_num = None
            confidence = None
            reasoning = None

            for line in response.split('\n'):
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
                final_reasoning = reasoning if reasoning else "Qwen2.5 text-based reasoning"
            else:
                # Fallback to top CLIP match if parsing failed
                best_idx = candidate_indices[0]
                best_obj = self.objects[best_idx]
                final_confidence = candidate_scores[0]
                final_reasoning = f"Qwen2.5 response (parsed): {response[:100]}..."

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
            print(f"  ⚠ Qwen2.5 reasoning failed: {e}")
            # Fallback to CLIP
            best_idx = candidate_indices[0]
            best_obj = self.objects[best_idx]

            return QueryResult(
                object_id=best_obj.get('id', best_obj.get('curr_obj_num')),
                object_index=best_idx,
                confidence=candidate_scores[0],
                reasoning=f"LLM failed ({str(e)}), using CLIP-only",
                class_name=best_obj.get('class_name', 'unknown')
            )

    def answer(self, query: str) -> QueryResult:
        """
        Answer query using CLIP + Qwen2.5.

        Process:
        1. CLIP retrieves top-K candidates
        2. Qwen2.5 reasons about candidates
        3. Return best match
        """
        if len(self.objects) == 0:
            return QueryResult(
                object_id=None,
                object_index=-1,
                confidence=0.0,
                reasoning="No objects in scene",
                class_name="none"
            )

        # Load Qwen2.5 if not loaded
        if self.llm_model is None:
            self._load_llm()

        # Stage 1: CLIP retrieval
        candidate_indices, candidate_scores = self._retrieve_candidates_with_clip(
            query, top_k=self.top_k
        )

        # Stage 2: Qwen2.5 reasoning
        result = self._reason_with_llm(query, candidate_indices, candidate_scores)

        return result

    def __str__(self):
        llm_status = "loaded" if self.llm_model else "not loaded"
        return f"{self.setup_name}\n  CLIP: loaded\n  Qwen2.5: {llm_status}\n  Objects: {len(self.objects)}"
