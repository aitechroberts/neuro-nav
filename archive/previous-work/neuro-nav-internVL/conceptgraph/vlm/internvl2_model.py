"""
InternVL2-2B Model Wrapper for Scene Graph Construction

This module provides a unified interface for using InternVL2-2B for:
- Image captioning (detailed descriptions)
- Caption refinement
- Spatial relationship extraction
- Scene querying (visual question answering)
"""

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InternVL2Model:
    """
    Wrapper for InternVL2-2B model with unified interface for scene graph tasks.
    
    InternVL2-2B is a 2B parameter vision-language model from OpenGVLab/Shanghai AI Lab.
    It provides strong performance on captioning, VQA, and spatial reasoning tasks.
    """
    
    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-2B",
        device: str = "cuda:0",
        load_in_8bit: bool = False,
    ):
        """
        Initialize the InternVL2 model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading InternVL2 model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Load model
        if load_in_8bit:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device).eval()
        
        # Set generation config
        self.generation_config = {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.0,
        }
        
        logger.info(f"InternVL2 model loaded on {self.device}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL image to tensor format expected by InternVL2.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed pixel_values tensor
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self.device)
        return pixel_values
    
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
        
        # InternVL2 uses a specific chat format
        question = f"<image>\n{prompt}"
        
        # Generate response
        try:
            pixel_values = self._preprocess_image(image)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=self.generation_config,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        # Create a dummy white image (InternVL2 requires an image for chat)
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        try:
            pixel_values = self._preprocess_image(dummy_image)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.generation_config,
            )
            
            # Try to parse JSON response
            import json
            # Extract JSON from response (may have extra text)
            response = response.strip()
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
        
        # Create dummy image
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        try:
            pixel_values = self._preprocess_image(dummy_image)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config=self.generation_config,
            )
            
            # Parse JSON
            import json
            response = response.strip()
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
            image = Image.new('RGB', (224, 224), color='white')
        
        try:
            pixel_values = self._preprocess_image(image)
            
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                question=full_prompt,
                generation_config={
                    "max_new_tokens": 512,
                    "do_sample": False,
                },
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

