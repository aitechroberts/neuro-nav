"""
Phi-3-Vision Model Wrapper for Scene Graph Construction

Phi-3-Vision is a 4.2B parameter multimodal model from Microsoft.
It provides strong vision-language understanding capabilities.
"""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from typing import Optional, List
import logging

# Disable FlashAttention if not available
os.environ["DISABLE_FLASH_ATTN"] = "1"

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
        
        # Load config and disable FlashAttention
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Override FlashAttention setting - set both attributes
        config._attn_implementation = "eager"
        if hasattr(config, '_attn_implementation_internal'):
            config._attn_implementation_internal = "eager"
        
        # Prepare model loading kwargs
        model_kwargs = {
            "config": config,
            "trust_remote_code": True,
            "attn_implementation": "eager",  # Explicit override
            "low_cpu_mem_usage": True,  # Reduce peak memory during loading
        }
        
        # Add quantization if requested
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # Let transformers handle device placement
                logger.info("Using 8-bit quantization to reduce memory usage")
            except ImportError:
                logger.warning("bitsandbytes not available, loading in bfloat16 instead")
                model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map (8-bit uses device_map="auto")
        if not load_in_8bit and "device_map" not in model_kwargs:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
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
            )
            
            # Move inputs to model device (for 8-bit, model might be on different device)
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif hasattr(self.model, 'hf_device_map'):
                # For device_map="auto", find the device of the first parameter
                device = next(self.model.parameters()).device
            else:
                device = self.device
            
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=False,  # Disable cache to avoid DynamicCache compatibility issues
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.processor.tokenizer.decode(
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
            
            inputs = self.processor(prompt_text, [dummy_image], return_tensors="pt")
            
            # Move inputs to model device
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif hasattr(self.model, 'hf_device_map'):
                device = next(self.model.parameters()).device
            else:
                device = self.device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            response = self.processor.tokenizer.decode(
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
            
            response = self.processor.tokenizer.decode(
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
            
            inputs = self.processor(prompt_text, [image], return_tensors="pt")
            
            # Move inputs to model device
            if hasattr(self.model, 'device'):
                device = self.model.device
            elif hasattr(self.model, 'hf_device_map'):
                device = next(self.model.parameters()).device
            else:
                device = self.device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=False,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            response = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Error querying scene: {e}")
            return f"Error processing query: {str(e)}"
    
    def __del__(self):
        """Cleanup model from GPU memory"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass  # Ignore errors during cleanup
