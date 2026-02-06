"""
PaliGemma-3B Model Wrapper for Scene Graph Construction

PaliGemma is a 3B parameter vision-language model from Google.
It's efficient, fast, and designed for image-text understanding.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaliGemmaModel:
    """
    Wrapper for PaliGemma-3B model with unified interface for scene graph tasks.
    
    PaliGemma is a 3B parameter vision-language model from Google.
    """
    
    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-224",
        device: str = "cuda:0",
        load_in_8bit: bool = False,
    ):
        """
        Initialize the PaliGemma model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading PaliGemma model: {model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()
        
        logger.info(f"PaliGemma model loaded on {self.device}")
    
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
            prompt = "caption en"  # PaliGemma uses short prompts
        
        try:
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            # Decode (skip prompt tokens)
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
        prompt = "describe"
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
        captions_text = " ".join(captions[:3])  # Use first 3 captions
        
        # PaliGemma works best with short, direct prompts
        prompt = f"summarize: {captions_text}"
        
        try:
            # Use a minimal dummy image for text-focused task
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            inputs = self.processor(
                text=prompt,
                images=dummy_image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Extract object tag (first few words of summary)
            words = response.split()
            object_tag = " ".join(words[:3]) if len(words) >= 3 else response
            
            return {
                "object_tag": object_tag,
                "summary": response,
                "possible_tags": [" ".join(words[:2]) if len(words) >= 2 else object_tag]
            }
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
        # Build concise prompt for PaliGemma
        obj1_tag = obj1_info.get('object_tag', 'object1')
        obj2_tag = obj2_info.get('object_tag', 'object2')
        
        prompt = f"relation: {obj1_tag} and {obj2_tag}"
        
        try:
            dummy_image = Image.new('RGB', (224, 224), color='white')
            
            inputs = self.processor(
                text=prompt,
                images=dummy_image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            
            response = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip().lower()
            
            # Parse response for spatial relations
            if "on" in response and obj1_tag.lower() in response:
                relation = "a on b"
            elif "on" in response and obj2_tag.lower() in response:
                relation = "b on a"
            elif "in" in response and obj1_tag.lower() in response:
                relation = "a in b"
            elif "in" in response and obj2_tag.lower() in response:
                relation = "b in a"
            else:
                relation = "none of these"
            
            return {
                "object_relation": relation,
                "reason": response
            }
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
        # PaliGemma works best with concise prompts
        # Truncate scene context if too long
        if len(scene_context) > 500:
            scene_context = scene_context[:500] + "..."
        
        full_prompt = f"answer: {query} context: {scene_context}"
        
        # Use dummy image if none provided
        if image is None:
            image = Image.new('RGB', (224, 224), color='white')
        
        try:
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
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

