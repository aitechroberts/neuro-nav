"""
Qwen2-VL Vision-Language Model Wrapper
Replaces GPT-4 for caption refinement and scene querying

Qwen2-VL is a powerful vision-language model from Alibaba that excels at:
- Understanding images with high detail
- Multi-image reasoning
- Spatial relationship understanding
- Complex question answering
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)


class Qwen2VLModel:
    """
    Wrapper for Qwen2-VL model for caption refinement and scene querying.
    
    This model can:
    - Refine captions from Florence-2
    - Answer questions about images
    - Understand spatial relationships
    - Extract structured information (JSON)
    - Compare multiple images
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Qwen2-VL model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "Qwen/Qwen2-VL-2B-Instruct" (2B params, efficient)
                - "Qwen/Qwen2-VL-7B-Instruct" (7B params, best quality)
            device: Device to run model on
            torch_dtype: Data type for model weights (bfloat16 recommended)
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        
        logger.info(f"Loading Qwen2-VL model: {model_name}")
        
        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        logger.info(f"Qwen2-VL model loaded successfully on {device}")
    
    def refine_caption(
        self,
        image: Image.Image,
        raw_captions: List[str],
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Refine multiple captions of the same object from different views.
        This replaces the GPT-4 caption refinement step.
        
        Args:
            image: Representative PIL Image of the object
            raw_captions: List of captions from different views
            prompt_template: Optional custom prompt template
            
        Returns:
            Dictionary with:
                - 'summary': Refined caption/description
                - 'object_tag': Short object tag
                - 'possible_tags': List of alternative tags
        """
        if prompt_template is None:
            prompt_template = """You are analyzing an object from a 3D scene reconstruction. 
Here are multiple captions of the same object from different viewpoints:

{captions}

Based on these captions and the image, provide a JSON response with:
1. "summary": A concise but detailed description of the object
2. "object_tag": A short 1-3 word label (e.g., "wooden chair", "laptop")
3. "possible_tags": A list of 2-3 alternative labels

Output ONLY valid JSON, no other text."""
        
        # Format captions
        captions_text = "\n".join([f"- {cap}" for cap in raw_captions])
        prompt = prompt_template.format(captions=captions_text)
        
        # Prepare messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response
        try:
            result = json.loads(output_text)
            # Ensure all required fields exist
            if 'object_tag' not in result:
                result['object_tag'] = raw_captions[0].split()[0] if raw_captions else "object"
            if 'summary' not in result:
                result['summary'] = raw_captions[0] if raw_captions else "unknown object"
            if 'possible_tags' not in result:
                result['possible_tags'] = [result['object_tag']]
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Qwen2-VL output: {output_text}")
            # Fallback to using the first caption
            return {
                'summary': raw_captions[0] if raw_captions else "unknown object",
                'object_tag': raw_captions[0].split()[0] if raw_captions else "object",
                'possible_tags': [raw_captions[0].split()[0]] if raw_captions else ["object"]
            }
    
    def extract_object_relationships(
        self,
        images: List[Image.Image],
        object1_info: Dict,
        object2_info: Dict,
    ) -> Dict[str, str]:
        """
        Determine spatial relationship between two objects.
        This replaces the GPT-4 relationship extraction step.
        
        Args:
            images: List of PIL Images showing both objects
            object1_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            object2_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            
        Returns:
            Dictionary with:
                - 'object_relation': Relation type ('a on b', 'b on a', 'a in b', 'b in a', 'none of these')
                - 'reason': Explanation for the relationship
        """
        prompt = f"""You are analyzing the spatial relationship between two objects in a 3D scene:

Object 1: {object1_info['object_tag']}
- Bounding box center: {object1_info['bbox_center']}
- Bounding box extent: {object1_info['bbox_extent']}

Object 2: {object2_info['object_tag']}
- Bounding box center: {object2_info['bbox_center']}
- Bounding box extent: {object2_info['bbox_extent']}

Determine the spatial relationship. Output ONLY a JSON with:
- "object_relation": Must be EXACTLY one of: "a on b", "b on a", "a in b", "b in a", "none of these"
- "reason": Brief explanation (1-2 sentences)

Consider:
- "a on b": object a is typically placed on top of object b
- "b on a": object b is typically placed on top of object a
- "a in b": object a is typically contained within object b
- "b in a": object b is typically contained within object a
- "none of these": no clear containment or support relationship

Output ONLY valid JSON."""
        
        # For now, use the first image (could be extended to use multiple)
        image = images[0] if images else None
        
        if image is None:
            # No image available, use spatial reasoning only
            content = [{"type": "text", "text": prompt}]
        else:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        
        messages = [{"role": "user", "content": content}]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse JSON response
        try:
            result = json.loads(output_text)
            if 'object_relation' not in result:
                result['object_relation'] = 'none of these'
            if 'reason' not in result:
                result['reason'] = 'Unable to determine relationship'
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from Qwen2-VL output: {output_text}")
            return {
                'object_relation': 'none of these',
                'reason': 'Unable to determine relationship'
            }
    
    def query_scene(
        self,
        image: Image.Image,
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Answer a question about a scene.
        
        Args:
            image: PIL Image of the scene
            query: Question to answer
            context: Optional context (e.g., scene graph description)
            
        Returns:
            Answer string
        """
        if context:
            full_prompt = f"""You are a helpful assistant that provides clear, descriptive answers about a 3D scene.

Scene Information:
{context}

Question: {query}

Instructions:
- Provide a natural, descriptive answer using the object descriptions and locations
- Describe what objects are relevant to the question
- Include spatial information (where things are located)
- Be specific and helpful
- Use complete sentences, not just object IDs

Answer:"""
        else:
            full_prompt = query
        
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": full_prompt}
                    ],
                }
            ]
        else:
            # Text-only mode (no image provided)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt}
                    ],
                }
            ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image is not None:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Text-only inputs
            inputs = self.processor(
                text=[text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def caption_image(
        self,
        image: Image.Image,
        detail_level: str = "detailed"
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image
            detail_level: 'simple' or 'detailed'
            
        Returns:
            Caption string
        """
        if detail_level == "simple":
            prompt = "Briefly describe what you see in this image in one sentence."
        else:
            prompt = "Describe this image in detail."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        
        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text
    
    def __del__(self):
        """Clean up model from memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()

