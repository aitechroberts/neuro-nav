"""
Gemma 3 VLM Module

A standalone client for Google's Gemma 3 Multimodal models (4B, 12B, 27B).
Note: The 1B variant is text-only and cannot be used here.

Designed to handle Set-of-Mark visual prompting (bounding boxes + IDs) via
chat-based interaction, similar to the Qwen implementation.
"""

import json
import os
import re
import ast
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
import torch
from omegaconf import OmegaConf

# =============================================================================
# Response Parsing Utilities
# =============================================================================

def _clean_markdown_json(text: str) -> str:
    """
    Extracts JSON content from Markdown code blocks if present.
    """
    text = text.strip()
    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text

def _safe_eval_llm_output(text: str) -> Any:
    """
    Robustly parses LLM output into Python objects (Dict/List).
    """
    cleaned_text = _clean_markdown_json(text)
    
    # Try Standard JSON
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Try Python Literal Eval
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
# Gemma 3 Client
# =============================================================================

class Gemma3Client:
    """
    Client for local inference with Google Gemma 3 Multimodal models.
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it", # Defaulting to 4B as 1B is text-only
        device: str = None,
        prompts: Optional[Any] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        if prompts is None:
            self.prompts = {}
        else:
            try:
                self.prompts = OmegaConf.to_container(prompts, resolve=True)
            except Exception:
                self.prompts = dict(prompts)

        print(f"[Gemma 3] Loading model: {model_name}...")
        
        # Load Processor and Model
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Gemma 3 is optimized for bfloat16
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # We use AutoModelForVision2Seq which maps to Gemma3ForConditionalGeneration
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        if device != "cuda":
            self.model.to(device)
            
        self.model.eval()
        print(f"[Gemma 3] Model loaded successfully.")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate response using Gemma's chat template.
        """
        tokens = max_new_tokens or self.max_new_tokens
        
        # 1. Construct Message
        # Gemma 3 processor handles images via the 'images' argument, 
        # but the chat template needs to know where the image goes.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 2. Apply Chat Template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            # 3. Process Inputs
            # Gemma 3 expects inputs in a specific way; inputs["pixel_values"] 
            # will be generated from the `images` argument.
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            ).to(self.device)
            
            # 4. Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                )
            
            # 5. Decode
            # Strip input tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            return output_text[0]

        except Exception as e:
            print(f"[Gemma 3] Generation Error: {e}")
            return ""

    def caption_objects_with_labels(
        self,
        image: Image.Image,
        labels: List[str],
        caption_system_prompt: str,
        captions_with_labels_template: str,
    ) -> List[Dict[str, str]]:
        """
        Generates captions for a list of objects in a single VLM call.
        """
        labels_str = "\n".join(labels)
        full_prompt = (
            f"{caption_system_prompt}\n\n"
            f"{captions_with_labels_template.format(labels=labels_str)}"
        )

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
        
        if not results:
            # Fallback
            print(f"[Gemma 3] Failed to parse structured output. Raw response: {response[:100]}...")
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
        """
        Generates spatial relationships between objects.
        """
        labels_str = ", ".join(labels)
        full_prompt = (
            f"{relation_system_prompt}\n\n"
            f"{relations_with_labels_template.format(labels=labels_str)}"
        )

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
        print("[Gemma 3] Client cleaned up.")

# =============================================================================
# Factory / Global Access
# =============================================================================

_gemma3_client: Optional[Gemma3Client] = None

def get_gemma3_client(
    model_name: str = "google/gemma-3-4b-it",
    device: str = "cuda",
    prompts: Optional[Any] = None,
    force_new: bool = False,
    **kwargs
) -> Gemma3Client:
    global _gemma3_client
    if _gemma3_client is None or force_new:
        _gemma3_client = Gemma3Client(
            model_name=model_name,
            device=device,
            prompts=prompts,
            **kwargs
        )
    return _gemma3_client

# =============================================================================
# Helper for Consolidation
# =============================================================================

def consolidate_captions(client: Gemma3Client, captions: List[Dict]) -> str:
    """
    Consolidates a list of captions into a single summary.
    """
    valid_caps = [c['caption'] for c in captions if c.get('caption')]
    if not valid_caps:
        return "Unknown object."
    
    captions_block = "\n".join([f"- {c}" for c in valid_caps[:10]])
    
    template = client.prompts.get("consolidate_prompt", "Summarize these: {captions}")
    prompt = template.format(captions=captions_block)
    
    # Gemma needs image input, use placeholder
    dummy_image = Image.new('RGB', (64, 64), color=(0, 0, 0))
    
    response = client.generate(dummy_image, prompt, max_new_tokens=128)
    
    if "{" in response and "}" in response:
        parsed = _safe_eval_llm_output(response)
        if isinstance(parsed, dict) and "consolidated_caption" in parsed:
            return parsed["consolidated_caption"]
            
    return _clean_markdown_json(response)