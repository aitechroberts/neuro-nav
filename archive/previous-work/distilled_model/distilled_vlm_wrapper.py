"""
Wrapper for Distilled Student Model to match Qwen2VLModel interface
Allows using distilled model as drop-in replacement in neuro-nav-vlm pipeline
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Optional, Union
import logging
import json
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from distilled_model.student_models import create_student_model

logger = logging.getLogger(__name__)


class DistilledVLMModel:
    """
    Wrapper for distilled student model that matches Qwen2VLModel interface.
    
    This allows the distilled model to be used as a drop-in replacement
    for Qwen2-VL in the neuro-nav-vlm pipeline.
    
    Methods match Qwen2VLModel:
    - refine_caption()
    - extract_object_relationships()
    - query_scene()
    - caption_image()
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        student_type: str = "tiny",
    ):
        """
        Initialize distilled VLM model.
        
        Args:
            model_path: Path to trained student model checkpoint
            device: Device to run model on
            student_type: Type of student model ('tiny' or 'phi2')
        """
        self.device = device
        self.model_path = model_path
        
        logger.info(f"Loading distilled student model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model type
        if 'config' in checkpoint:
            student_type = checkpoint['config'].get('student_type', student_type)
        
        # Create and load model
        self.model = create_student_model(model_type=student_type)
        self.model.load_state_dict(checkpoint['student_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image preprocessing
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Simple tokenizer (for basic text processing)
        # Note: For production, you'd want a proper tokenizer
        self.vocab_size = 32000  # Default vocab size
        
        logger.info(f"Distilled model loaded successfully on {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0)  # [1, 3, H, W]
        return image_tensor.to(self.device)
    
    def _simple_text_encode(self, text: str) -> torch.Tensor:
        """
        Simple text encoding (placeholder).
        For production, use a proper tokenizer matching your training.
        """
        # This is a simplified version - you'd need to use the actual tokenizer
        # that was used during training
        # For now, return a placeholder
        # In practice, you'd do: tokenizer.encode(text, return_tensors="pt")
        return torch.zeros(1, 10, dtype=torch.long).to(self.device)
    
    def _generate_text(self, image: Optional[Image.Image], prompt: str, max_length: int = 256) -> str:
        """
        Generate text response from model.
        
        Note: This is a simplified implementation. For full functionality,
        you'd need to implement proper tokenization and text generation.
        """
        # Preprocess inputs
        if image is not None:
            image_tensor = self._preprocess_image(image)
        else:
            image_tensor = None
        
        # Encode prompt (simplified - use actual tokenizer in production)
        # For now, this is a placeholder that returns a simple response
        # You would need to:
        # 1. Tokenize the prompt
        # 2. Run model forward pass
        # 3. Decode output tokens to text
        
        # Placeholder response (replace with actual model inference)
        logger.warning("Using placeholder text generation. Implement proper inference for production use.")
        
        # Simple rule-based response for demonstration
        if "JSON" in prompt or "json" in prompt:
            return json.dumps({
                "summary": "Object description from distilled model",
                "object_tag": "object",
                "possible_tags": ["item", "thing"]
            })
        else:
            return "Response from distilled model (implement proper text generation)"
    
    def refine_caption(
        self,
        image: Image.Image,
        raw_captions: List[str],
        prompt_template: Optional[str] = None,
    ) -> Dict[str, Union[str, List[str]]]:
        """
        Refine multiple captions of the same object from different views.
        Matches Qwen2VLModel.refine_caption() interface.
        
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
        
        # Generate response
        response_text = self._generate_text(image, prompt, max_length=512)
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            # Ensure all required fields exist
            if 'object_tag' not in result:
                result['object_tag'] = raw_captions[0].split()[0] if raw_captions else "object"
            if 'summary' not in result:
                result['summary'] = raw_captions[0] if raw_captions else "unknown object"
            if 'possible_tags' not in result:
                result['possible_tags'] = [result['object_tag']]
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from distilled model output: {response_text}")
            # Fallback
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
        Matches Qwen2VLModel.extract_object_relationships() interface.
        
        Args:
            images: List of PIL Images showing both objects
            object1_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            object2_info: Dict with 'object_tag', 'bbox_center', 'bbox_extent'
            
        Returns:
            Dictionary with:
                - 'object_relation': Relation type
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

Output ONLY valid JSON."""
        
        image = images[0] if images else None
        response_text = self._generate_text(image, prompt, max_length=256)
        
        try:
            result = json.loads(response_text)
            if 'object_relation' not in result:
                result['object_relation'] = 'none of these'
            if 'reason' not in result:
                result['reason'] = 'Unable to determine relationship'
            return result
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {response_text}")
            return {
                'object_relation': 'none of these',
                'reason': 'Unable to determine relationship'
            }
    
    def query_scene(
        self,
        image: Optional[Image.Image],
        query: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Answer a question about a scene.
        Matches Qwen2VLModel.query_scene() interface.
        
        Args:
            image: PIL Image of the scene (optional)
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
        
        response_text = self._generate_text(image, full_prompt, max_length=512)
        return response_text
    
    def caption_image(
        self,
        image: Image.Image,
        detail_level: str = "detailed"
    ) -> str:
        """
        Generate a caption for an image.
        Matches Qwen2VLModel.caption_image() interface.
        
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
        
        response_text = self._generate_text(image, prompt, max_length=256)
        return response_text
    
    def __del__(self):
        """Clean up model from memory"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


# Factory function for easy loading
def load_distilled_vlm(
    model_path: str = "distilled_model/outputs/final_student_model.pt",
    device: str = "cuda:0",
) -> DistilledVLMModel:
    """
    Convenience function to load distilled VLM model.
    
    Usage:
        from distilled_model.distilled_vlm_wrapper import load_distilled_vlm
        
        # Load model
        model = load_distilled_vlm("outputs/final_student_model.pt")
        
        # Use like Qwen2VLModel
        result = model.refine_caption(image, captions)
    """
    return DistilledVLMModel(model_path=model_path, device=device)

