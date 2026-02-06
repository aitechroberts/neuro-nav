"""
Florence-2 Vision-Language Model Wrapper
Replaces YOLO detection and provides captioning capabilities

Florence-2 is a lightweight but powerful VLM from Microsoft that can:
- Detect objects
- Generate captions
- Do OCR
- Understand spatial relationships
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Florence2Model:
    """
    Wrapper for Florence-2 model for object detection and captioning.
    
    Modes:
    - '<CAPTION>': Generate a simple caption
    - '<DETAILED_CAPTION>': Generate a detailed caption  
    - '<MORE_DETAILED_CAPTION>': Generate a very detailed caption
    - '<OD>': Object detection with bounding boxes
    - '<DENSE_REGION_CAPTION>': Dense region captioning
    - '<CAPTION_TO_PHRASE_GROUNDING>': Phrase grounding given a caption
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize Florence-2 model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "microsoft/Florence-2-base" (0.23B params, faster)
                - "microsoft/Florence-2-large" (0.77B params, better quality)
            device: Device to run model on
            torch_dtype: Data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        
        logger.info(f"Loading Florence-2 model: {model_name}")
        
        # Load model and processor
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device)
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        logger.info(f"Florence-2 model loaded successfully on {device}")
    
    def detect_objects(
        self,
        image: Image.Image,
        text_input: Optional[str] = None,
    ) -> Dict:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image
            text_input: Optional text prompt for phrase grounding
            
        Returns:
            Dictionary with:
                - 'bboxes': List of bounding boxes [[x1, y1, x2, y2], ...]
                - 'labels': List of object labels
        """
        if text_input is not None:
            task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
            prompt = task_prompt + text_input
        else:
            task_prompt = '<OD>'
            prompt = task_prompt
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer.get(task_prompt, {'bboxes': [], 'labels': []})
    
    def caption_image(
        self,
        image: Image.Image,
        detail_level: str = "detailed"
    ) -> str:
        """
        Generate a caption for an image.
        
        Args:
            image: PIL Image
            detail_level: 'simple', 'detailed', or 'more_detailed'
            
        Returns:
            Caption string
        """
        task_map = {
            'simple': '<CAPTION>',
            'detailed': '<DETAILED_CAPTION>',
            'more_detailed': '<MORE_DETAILED_CAPTION>',
        }
        
        task_prompt = task_map.get(detail_level, '<DETAILED_CAPTION>')
        
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer.get(task_prompt, "")
    
    def caption_region(
        self,
        image: Image.Image,
        bbox: List[int],
    ) -> str:
        """
        Generate a caption for a specific region (cropped from bbox).
        
        Args:
            image: PIL Image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Caption string
        """
        # Crop the image to the bounding box
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1, y1, x2, y2))
        
        # Caption the cropped region
        return self.caption_image(cropped_image, detail_level='detailed')
    
    def dense_caption(
        self,
        image: Image.Image,
    ) -> Dict:
        """
        Generate dense region captions (multiple regions with captions).
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with:
                - 'bboxes': List of bounding boxes
                - 'labels': List of captions for each region
        """
        task_prompt = '<DENSE_REGION_CAPTION>'
        
        inputs = self.processor(
            text=task_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )
        
        return parsed_answer.get(task_prompt, {'bboxes': [], 'labels': []})
    
    def get_embeddings(
        self,
        image: Image.Image,
        text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Get image (and optionally text) embeddings from the model.
        This can replace CLIP features.
        
        Args:
            image: PIL Image
            text: Optional text prompt
            
        Returns:
            Embedding tensor
        """
        if text is None:
            text = '<CAPTION>'
        
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.torch_dtype)
        
        with torch.no_grad():
            outputs = self.model.model.encoder(
                inputs["pixel_values"],
                output_hidden_states=True
            )
            # Use the last hidden state as embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def __del__(self):
        """Clean up model from memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        torch.cuda.empty_cache()

