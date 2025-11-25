"""
Setup 5: GPT-4V with Vision (CLIP + Vision-Language Reasoning)

This setup tests TRUE multimodal vision-language capabilities.

Components:
- CLIP for candidate retrieval
- GPT-4V with ACTUAL RGB images for visual reasoning
- Two-stage: retrieval → vision+text reasoning

What it tests:
- Does seeing actual images improve over text-only reasoning (Setup 1)?
- How much does visual grounding contribute to semantic understanding?
- Can GPT-4V leverage visual details (color, shape, context) that text can't capture?

Key Difference from Setup 1:
- Setup 1: GPT-4 receives only text descriptions (labels, scores, metadata)
- Setup 5: GPT-4V receives actual RGB crops of objects for visual analysis
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os
import base64
from io import BytesIO
from pathlib import Path

from .base_query_engine import BaseQueryEngine, QueryResult


class GPT4VisionQueryEngine(BaseQueryEngine):
    """
    Query engine using CLIP + GPT-4V with actual RGB images.

    Two-stage process:
    1. CLIP retrieves top-K candidates
    2. GPT-4V analyzes actual RGB crops to pick best match
    """

    def __init__(self, objects: List[Dict], config: Dict = None):
        super().__init__(objects, config)
        self.setup_name = "Setup 5: GPT-4V (Vision+Language)"

        # Configuration
        self.top_k = config.get('top_k', 3) if config else 3
        self.num_images_per_object = config.get('num_images_per_object', 3) if config else 3
        self.use_gpt4v = True

        # Check for OpenAI API key
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print(f"  ⚠ Warning: OPENAI_API_KEY not found in environment")
            print(f"  GPT-4V will be disabled, falling back to CLIP-only")
            self.use_gpt4v = False

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
        if self.use_gpt4v:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"  ✓ OpenAI client initialized")
            except ImportError:
                print(f"  ⚠ OpenAI library not installed. Run: pip install openai")
                self.use_gpt4v = False
            except Exception as e:
                print(f"  ⚠ Failed to initialize OpenAI client: {e}")
                self.use_gpt4v = False

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

    def _retrieve_candidates_with_clip(self, query: str, top_k: int = 3) -> Tuple[List[int], List[float]]:
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

    def _select_best_image_crops(self, obj: Dict, num_images: int = 3) -> List[Image.Image]:
        """
        Select the best image crops for an object.

        Strategy:
        1. Highest confidence detection
        2. Largest bounding box (clearest view)
        3. Middle temporal observation (diversity)

        Args:
            obj: Object dictionary with multiple observations
            num_images: Number of images to return (default: 3)

        Returns:
            List of PIL Images
        """
        color_paths = obj.get('color_path', [])
        xyxys = obj.get('xyxy', [])
        confs = obj.get('conf', [])

        if not color_paths or not xyxys:
            return []

        num_detections = len(color_paths)
        if num_detections == 0:
            return []

        # Calculate scores for each detection
        detection_scores = []
        for i in range(num_detections):
            xyxy = xyxys[i]
            conf = confs[i] if i < len(confs) else 0.5

            # Bbox area (larger = clearer view)
            bbox_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

            # Combined score: confidence * bbox_area
            score = conf * bbox_area

            detection_scores.append({
                'index': i,
                'score': score,
                'conf': conf,
                'bbox_area': bbox_area
            })

        # Sort by score
        detection_scores.sort(key=lambda x: x['score'], reverse=True)

        # Select indices
        selected_indices = []

        # Strategy 1: Highest confidence
        selected_indices.append(detection_scores[0]['index'])

        # Strategy 2: Middle temporal (if we have enough detections)
        if num_detections >= 3 and num_images >= 2:
            mid_idx = num_detections // 2
            if mid_idx not in selected_indices:
                selected_indices.append(mid_idx)

        # Strategy 3: Fill remaining with next best scores
        for det in detection_scores[1:]:
            if len(selected_indices) >= num_images:
                break
            if det['index'] not in selected_indices:
                selected_indices.append(det['index'])

        # Load and crop images
        crops = []
        for idx in selected_indices[:num_images]:
            try:
                img_path = color_paths[idx]
                xyxy = xyxys[idx]

                # Load image
                image = Image.open(img_path).convert('RGB')

                # Crop to bbox
                x1, y1, x2, y2 = xyxy
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))

                crops.append(crop)
            except Exception as e:
                print(f"  ⚠ Failed to load crop {idx}: {e}")
                continue

        return crops

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL Image to base64 string for OpenAI API."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _reason_with_gpt4v(
        self,
        query: str,
        candidate_indices: List[int],
        candidate_scores: List[float]
    ) -> QueryResult:
        """
        Use GPT-4V with actual RGB images to reason about candidates.

        Each candidate gets 1-3 representative image crops.
        """
        if not self.use_gpt4v:
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

        # Build message content with images
        content = [
            {
                "type": "text",
                "text": f"""You are a robot with vision analyzing a 3D scene. A user has asked: "{query}"

You have identified {len(candidate_indices)} candidate objects using visual similarity. I'm showing you actual RGB images of each candidate object (1-{self.num_images_per_object} views per object).

Please analyze the visual appearance of each candidate and determine which object best answers the user's query."""
            }
        ]

        # Add images for each candidate
        for i, idx in enumerate(candidate_indices):
            obj = self.objects[idx]
            class_name = obj.get('class_name', 'unknown')

            # Add text header for this candidate
            content.append({
                "type": "text",
                "text": f"\n\nCandidate {i+1}: {class_name.upper()} (CLIP similarity: {candidate_scores[i]:.3f})"
            })

            # Load and add images
            crops = self._select_best_image_crops(obj, self.num_images_per_object)

            if not crops:
                content.append({
                    "type": "text",
                    "text": "[No images available for this object]"
                })
            else:
                for j, crop in enumerate(crops):
                    base64_image = self._encode_image_to_base64(crop)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"  # Use "low" for cost efficiency, "high" for better quality
                        }
                    })

        # Add final prompt
        content.append({
            "type": "text",
            "text": f"""
Based on the user's query and what you see in the images:
1. Which candidate (1-{len(candidate_indices)}) best answers the query?
2. What is your confidence (0.0-1.0)?
3. Briefly explain your reasoning based on what you observe in the images (1-2 sentences).

Respond in this exact format:
OBJECT: <number>
CONFIDENCE: <0.0-1.0>
REASONING: <visual explanation>"""
        })

        try:
            # Call GPT-4V
            response = self.client.chat.completions.create(
                model="gpt-4o",  # gpt-4o has vision capabilities
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful robot assistant with vision capabilities analyzing 3D scenes."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.3,
                max_tokens=300
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
                final_reasoning = reasoning if reasoning else "GPT-4V visual reasoning"
            else:
                # Fallback to top CLIP match
                best_idx = candidate_indices[0]
                best_obj = self.objects[best_idx]
                final_confidence = candidate_scores[0]
                final_reasoning = f"GPT-4V parsing failed, using top CLIP match. Response: {response_text[:100]}"

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
            import traceback
            traceback.print_exc()

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
        Answer query using CLIP + GPT-4V with vision.

        Process:
        1. CLIP retrieves top-K candidates
        2. Load RGB crops for each candidate
        3. GPT-4V analyzes images and reasons
        4. Return best match with visual reasoning
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

        # Stage 2: GPT-4V visual reasoning
        result = self._reason_with_gpt4v(query, candidate_indices, candidate_scores)

        return result

    def __str__(self):
        gpt4v_status = "enabled" if self.use_gpt4v else "disabled"
        return f"{self.setup_name}\n  CLIP: loaded\n  GPT-4V: {gpt4v_status}\n  Images/object: {self.num_images_per_object}\n  Objects: {len(self.objects)}"
