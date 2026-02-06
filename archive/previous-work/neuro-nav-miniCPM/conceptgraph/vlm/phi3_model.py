"""
Phi-3-Vision Model Wrapper for Scene Graph Construction

Phi-3-Vision is a 4.2B parameter multimodal model from Microsoft.
It provides strong vision-language understanding capabilities.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phi3VisionModel:
    """
    Wrapper for Phi-3-Vision model with unified interface for scene graph tasks.
    
    Phi-3-Vision is a 4.2B parameter vision-language model from Microsoft.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-vision-128k-instruct",
        device: str = "cuda:0",
        load_in_8bit: bool = False,
    ):
        """
        Initialize the Phi-3-Vision model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading Phi-3-Vision model: {model_name}")
        
        # Load processor (handles both text and images)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        
        logger.info(f"Phi-3-Vision model loaded on {self.device}")
    
    def caption_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Generate a detailed caption for an image.
        
        Args:
            image: PIL Image
            prompt: Optional custom prompt (default: detailed description)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated caption string
        """
        if prompt is None:
            prompt = "Please provide a detailed description of this image, including objects, their properties, positions, and relationships."
        
        try:
            # Phi-3-Vision uses <|image_1|> token for image
            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{prompt}"
                }
            ]
            
            # Apply chat template
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self.processor(
                prompt_text,
                [image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return "Error generating caption"
    
    def caption_region(
        self,
        image: Image.Image,
        bbox: Optional[List[float]] = None,
    ) -> str:
        """
        Caption a specific region of an image (cropped to bbox).
        
        Args:
            image: PIL Image (should be pre-cropped to region)
            bbox: Bounding box (not used, image should already be cropped)
            
        Returns:
            Caption for the region
        """
        prompt = "Describe the main object in this image in detail."
        return self.caption_image(image, prompt=prompt)
    
    def refine_caption(
        self,
        captions: List[str],
        bbox_info: Optional[dict] = None,
    ) -> dict:
        """
        Refine multiple captions of the same object into a single coherent description.
        
        Args:
            captions: List of caption strings from different views
            bbox_info: Optional dict with 'bbox_extent' and 'bbox_center'
            
        Returns:
            Dict with 'object_tag', 'summary', and 'possible_tags'
        """
        if not captions:
            return {
                "object_tag": "unknown",
                "summary": "No captions available",
                "possible_tags": []
            }
        
        # Build refinement prompt
        captions_text = "\n".join([f"- {cap}" for cap in captions])
        
        prompt = f"""Given multiple descriptions of the same object from different viewpoints:

{captions_text}

Please provide:
1. A single, concise object tag (1-3 words, like "white sofa" or "wooden table")
2. A brief summary (1 sentence)
3. A list of 2-3 possible alternative tags

Respond in JSON format:
{{
  "object_tag": "...",
  "summary": "...",
  "possible_tags": ["...", "..."]
}}"""
        
        try:
            # Use a minimal dummy image for text-only task
            dummy_image = Image.new('RGB', (10, 10), color='white')
            
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(prompt_text, [dummy_image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Parse JSON response
            import json
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"Error refining caption: {e}")
            # Fallback: use first caption as tag
            first_words = captions[0].split()[:3]
            return {
                "object_tag": " ".join(first_words),
                "summary": captions[0],
                "possible_tags": [" ".join(captions[0].split()[:2])]
            }
    
    def extract_object_relationships(
        self,
        obj1_info: dict,
        obj2_info: dict,
    ) -> dict:
        """
        Extract spatial relationships between two objects.
        
        Args:
            obj1_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            obj2_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            
        Returns:
            Dict with 'object_relation' and 'reason'
        """
        prompt = f"""Given two objects in a 3D scene:

Object 1:
- Tag: {obj1_info.get('object_tag', 'unknown')}
- Center: {obj1_info.get('bbox_center', [])}
- Size: {obj1_info.get('bbox_extent', [])}

Object 2:
- Tag: {obj2_info.get('object_tag', 'unknown')}
- Center: {obj2_info.get('bbox_center', [])}
- Size: {obj2_info.get('bbox_extent', [])}

Determine the spatial relationship. Choose one:
- "a on b": if object 1 is typically placed on object 2
- "b on a": if object 2 is typically placed on object 1
- "a in b": if object 1 is typically inside object 2
- "b in a": if object 2 is typically inside object 1
- "none of these": if none apply

Respond in JSON format:
{{
  "object_relation": "...",
  "reason": "..."
}}"""
        
        try:
            dummy_image = Image.new('RGB', (10, 10), color='white')
            messages = [{"role": "user", "content": f"<|image_1|>\n{prompt}"}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(prompt_text, [dummy_image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Parse JSON
            import json
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            return result
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return {
                "object_relation": "none of these",
                "reason": "Unable to determine relationship"
            }
    
    def query_scene(
        self,
        query: str,
        scene_context: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """
        Answer a question about the scene using scene graph context.
        
        Args:
            query: User's question
            scene_context: Text description of scene graph (objects, positions, etc.)
            image: Optional image for visual grounding
            
        Returns:
            Natural language answer
        """
        full_prompt = f"""You are a helpful assistant that provides clear, descriptive answers about a 3D scene.

Scene Information:
{scene_context}

Question: {query}

Instructions:
- Provide a natural, descriptive answer using the object descriptions and locations
- Describe what objects are relevant to the question
- Include spatial information (where things are located)
- Be specific and helpful
- Use complete sentences, not just object IDs

Answer:"""
        
        # Use dummy image if none provided
        if image is None:
            image = Image.new('RGB', (10, 10), color='white')
        
        try:
            messages = [{"role": "user", "content": f"<|image_1|>\n{full_prompt}"}]
            prompt_text = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(prompt_text, [image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error querying scene: {e}")
            return f"Error processing query: {str(e)}"
    
    def __del__(self):
        """Cleanup model from GPU memory"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
