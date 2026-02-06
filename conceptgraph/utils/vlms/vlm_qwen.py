import json
import os
import re
import ast
import base64
from typing import List, Dict, Optional, Tuple, Any, Union
from PIL import Image
import torch
import numpy as np
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Response Parsing Utilities
# =============================================================================

def _clean_markdown_json(text: str) -> str:
    """Extracts JSON content from Markdown code blocks if present."""
    text = text.strip()
    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text

def _safe_eval_llm_output(text: str) -> Any:
    """Robustly parses LLM output into Python objects (Dict/List)."""
    cleaned_text = _clean_markdown_json(text)
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned_text)
    except (ValueError, SyntaxError):
        pass

    # Fallback: Heuristic extraction
    try:
        start_idx = -1
        end_idx = -1
        if '[' in cleaned_text and ']' in cleaned_text:
            start_idx = cleaned_text.find('[')
            end_idx = cleaned_text.rfind(']') + 1
        elif '{' in cleaned_text and '}' in cleaned_text:
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
        if start_idx != -1 and end_idx != -1:
            candidate = cleaned_text[start_idx:end_idx]
            try:
                return json.loads(candidate)
            except:
                return ast.literal_eval(candidate)
    except:
        pass
    return text

# =============================================================================
# Qwen3-VL Client
# =============================================================================

class Qwen3VLClient:
    """Client for local inference with Qwen3-VL family models."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = None,
        prompts: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # 1. Store Prompts
        if prompts is None:
            self.prompts = {}
        else:
            try:
                self.prompts = OmegaConf.to_container(prompts, resolve=True)
            except Exception:
                self.prompts = dict(prompts)

        # 2. Device Setup
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 3. Determine Attention Implementation
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("[Qwen3-VL] Flash Attention 2 is available and enabled.")
        except ImportError:
            attn_impl = "sdpa"
            logger.warning("[Qwen3-VL] Flash Attention 2 not found. Using 'sdpa' instead.")

        logger.info(f"[Qwen3-VL] Loading model: {model_name}...")

        # 4. Load Model
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        
        # 5. Load Processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.model.eval()
        logger.info(f"[Qwen3-VL] Model loaded successfully on {self.device}.")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        tokens = max_new_tokens or self.max_new_tokens
        
        # Construct Message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Process Inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=0.9 if self.temperature > 0 else None,
                )
            
            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            return output_text[0]
        except Exception as e:
            logger.error(f"[Qwen3-VL] Generation Error: {e}")
            return ""

    def caption_objects_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        caption_system_prompt: str,
        captions_with_labels_template: str,
    ) -> List[Dict[str, str]]:
        """Generates captions for a list of objects in a single VLM call."""
        
        # Format the list of labels
        labels_str = "\n".join(labels)
        
        # === THE FIX IS HERE ===
        # Use .replace() instead of .format() to preserve JSON braces in the template
        user_content = captions_with_labels_template.replace("{labels}", labels_str)
        # =======================
        
        full_prompt = f"{caption_system_prompt}\n\n{user_content}"

        response = self.generate(image, full_prompt)
        parsed = _safe_eval_llm_output(response)
        
        results = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    results.append({
                        "id": str(item.get("id", "")),
                        "name": str(item.get("name", "")),
                        "caption": str(item.get("caption", item.get("description", "")))
                    })
        
        # Fallback if parsing failed
        if not results:
            for label in labels:
                parts = label.split(":", 1)
                obj_id = parts[0].strip() if len(parts) > 0 else "0"
                obj_name = parts[1].strip() if len(parts) > 1 else label
                results.append({
                    "id": obj_id,
                    "name": obj_name,
                    "caption": f"A {obj_name}."
                })

        return results

    def infer_relations_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        relation_system_prompt: str,
        relations_with_labels_template: str,
    ) -> List[Tuple[str, str, str]]:
        """Generates spatial relationships between objects."""
        
        labels_str = ", ".join(labels)
        
        # === THE FIX IS HERE ALSO ===
        user_content = relations_with_labels_template.replace("{labels}", labels_str)
        # ============================
        
        full_prompt = f"{relation_system_prompt}\n\n{user_content}"

        response = self.generate(image, full_prompt)
        parsed = _safe_eval_llm_output(response)
        
        edges = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    edges.append((str(item[0]), str(item[1]), str(item[2])))
                elif isinstance(item, dict):
                    subj = item.get("subject", item.get("object1", ""))
                    rel = item.get("relation", item.get("predicate", ""))
                    obj = item.get("object", item.get("object2", ""))
                    if subj and rel and obj:
                        edges.append((str(subj), str(rel), str(obj)))
        return edges

    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[Qwen3-VL] Client cleaned up.")


# =============================================================================
# Global Access
# =============================================================================

_qwen3vl_client: Optional[Qwen3VLClient] = None

def get_qwen3vl_client(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "cuda",
    prompts: Optional[Any] = None,
    force_new: bool = False,
    **kwargs
) -> Qwen3VLClient:
    global _qwen3vl_client
    if _qwen3vl_client is None or force_new:
        _qwen3vl_client = Qwen3VLClient(
            model_name=model_name,
            device=device,
            prompts=prompts,
            **kwargs
        )
    return _qwen3vl_client

def consolidate_captions(client: Qwen3VLClient, captions: List[Dict]) -> str:
    valid_caps = [c['caption'] for c in captions if c.get('caption')]
    if not valid_caps:
        return "Unknown object."
    
    captions_block = "\n".join([f"- {c}" for c in valid_caps[:10]])
    template = client.prompts.get("consolidate_prompt", "Summarize these: {captions}")
    
    # Safe replace
    prompt = template.replace("{captions}", captions_block)
    
    dummy_image = Image.new('RGB', (28, 28), color=(0, 0, 0))
    response = client.generate(dummy_image, prompt, max_new_tokens=128)
    
    if "{" in response and "}" in response:
        parsed = _safe_eval_llm_output(response)
        if isinstance(parsed, dict) and "consolidated_caption" in parsed:
            return parsed["consolidated_caption"]
            
    return _clean_markdown_json(response)