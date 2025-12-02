"""
PaliGemma VLM Module - Drop-in replacement for OpenAI-based vlm.py

Provides the same function signatures as vlm.py but uses PaliGemma-3B for local inference.
This allows direct comparison between GPT-4 and PaliGemma outputs.

ABLATION STUDY NOTE: All prompts are defined at module level for easy modification.
"""

import json
import os
import base64
import ast
import re
from typing import List, Dict, Optional, Tuple, Any
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
import torch



# =============================================================================
# PROMPTS FOR ABLATION STUDIES - MODIFY THESE
# =============================================================================

# Prompt for extracting spatial relationships between objects
# Used in: get_obj_rel_from_image_paligemma()
SYSTEM_PROMPT_RELATIONS = "describe spatial relations"

# Prompt for generating object captions
# Used in: get_obj_captions_from_image_paligemma()
SYSTEM_PROMPT_CAPTIONS = "caption en"

# Prompt for consolidating multiple captions into one
# Used in: consolidate_captions()
SYSTEM_PROMPT_CONSOLIDATE = "summarize:"

# Prompt template for describing specific labeled objects
# {labels} will be replaced with the actual label list
PROMPT_TEMPLATE_RELATIONS = "objects: {labels}. describe spatial relations between them"
PROMPT_TEMPLATE_CAPTIONS = "objects: {labels}. caption each object"


CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*|```$", re.MULTILINE)

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove leading ```lang and trailing ```
        text = CODE_FENCE_RE.sub("", text).strip()
    return text


def _safe_eval_llm_output(text: str) -> Any:
    """
    Try to parse LLM output as JSON or Python literal.
    If everything fails, just return the raw text.
    """
    text = _strip_code_fences(text)

    # Some models wrap the result in backticks or 'json'
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    # Try JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Then Python literal
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


# =============================================================================
# PaliGemma Client Class (replaces OpenAI client)
# =============================================================================

class PaliGemmaClient:
    """
    PaliGemma client that provides a similar interface to OpenAI client.
    
    Usage:
        client = get_paligemma_client()
        # or
        client = get_paligemma_client(model_name="google/paligemma-3b-mix-448")
    """
    
    def __init__(
        self,
        model_name: str = "google/paligemma-3b-mix-224",
        device: str = None,
        prompts: Optional[Any] = None,
        load_model: bool = True,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        # ---- NEW: store prompts from Hydra config ----
        if prompts is None:
            self.prompts = {}
        else:
            # Allow DictConfig OR plain dict
            try:
                self.prompts = OmegaConf.to_container(prompts, resolve=True)
            except Exception:
                # Already a dict-like
                self.prompts = dict(prompts)
        # ------------------------------------------------

        self.processor = None
        self.model = None

        if load_model:
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

            print(f"[PaliGemma] Loading model: {model_name}")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device).eval()

            print(f"[PaliGemma] Model loaded on {self.device}")
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate text from image and prompt.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError(
                "PaliGemma processor/model not initialized; set load_model=True when constructing the client."
            )
        try:
            # Ensure PaliGemma sees an image token when we pass both text + image
            # but do NOT force this if the prompt already has one.
            if "<image>" not in prompt and "<img>" not in prompt:
                prompt = "<image> " + prompt

            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            response = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            print(f"[PaliGemma] Generation error: {e}")
            return ""
    
    def caption_objects_with_labels(
        self,
        image: "Image.Image",
        labels: List[str],
        caption_system_prompt: str,
        captions_with_labels_template: str,
        max_new_tokens: int = 512,
    ) -> List[Dict[str, str]]:
        """
        Mimics get_obj_captions_from_image_gpt4v: one call per frame.

        Args
        ----
        image:            PIL image for the *whole* frame.
        labels:           List of label strings like ['0: chair', '1: table', ...].
        caption_system_prompt:
                          cfg.prompts.caption (system-style description of the task).
        captions_with_labels_template:
                          cfg.prompts.captions_with_labels, with "{labels}" placeholder.
        """
        labels_str = "[" + ", ".join(f'"{lab}"' for lab in labels) + "]"
        user_prompt = captions_with_labels_template.format(labels=labels_str)
        full_prompt = caption_system_prompt.strip() + "\n\n" + user_prompt.strip()

        raw = self.generate(image, full_prompt, max_new_tokens=max_new_tokens)
        parsed = _safe_eval_llm_output(raw)

        # Normalize to list[dict]
        if isinstance(parsed, dict):
            # common patterns: {"captions": [...]}, {"objects": [...]}
            if "captions" in parsed:
                parsed = parsed["captions"]
            elif "objects" in parsed:
                parsed = parsed["objects"]
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            # Absolute worst case: treat the whole thing as one caption
            return [
                {
                    "id": str(i),
                    "name": labels[i] if i < len(labels) else "",
                    "caption": str(parsed),
                }
                for i in range(len(labels))
            ]

        result: List[Dict[str, str]] = []
        for i, item in enumerate(parsed):
            if isinstance(item, str):
                # no structure, just text
                result.append(
                    {
                        "id": str(i),
                        "name": labels[i] if i < len(labels) else "",
                        "caption": item,
                    }
                )
            elif isinstance(item, dict):
                id_raw = item.get("id", i)
                id_str = "".join(ch for ch in str(id_raw) if ch.isdigit()) or str(i)

                # recover name from labels if missing
                name = item.get("name")
                if not name and i < len(labels):
                    # handle formats like "0: chair" or "chair 0"
                    lab = labels[i]
                    if ":" in lab:
                        name = lab.split(":", 1)[1].strip()
                    else:
                        name = lab.rsplit(" ", 1)[0].strip()

                caption = item.get("caption") or item.get("description") or ""
                result.append(
                    {"id": id_str, "name": str(name or ""), "caption": str(caption)}
                )

        return result
    
    def infer_relations_with_labels(
        self,
        image: "Image.Image",
        labels: List[str],
        relation_system_prompt: str,
        relations_with_labels_template: str,
        max_new_tokens: int = 512,
    ) -> List[Tuple[str, str, str]]:
        """
        Mimics get_obj_rel_from_image_gpt4v: one call per frame that returns a
        list of (obj1_id, relation, obj2_id) triples as strings.
        """
        labels_str = "[" + ", ".join(f'"{lab}"' for lab in labels) + "]"
        user_prompt = relations_with_labels_template.format(labels=labels_str)
        full_prompt = relation_system_prompt.strip() + "\n\n" + user_prompt.strip()

        raw = self.generate(image, full_prompt, max_new_tokens=max_new_tokens)
        parsed = _safe_eval_llm_output(raw)

        if isinstance(parsed, dict):
            # common wrappers: {"relations": [...]}, {"relationships": [...]}, {"edges": [...]}
            for key in ("relations", "relationships", "edges"):
                if key in parsed:
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]

        if not isinstance(parsed, list):
            return []

        edges: List[Tuple[str, str, str]] = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                obj1, rel, obj2 = item[0], item[1], item[2]
            elif isinstance(item, dict):
                # handle dict-shaped outputs
                obj1 = (
                    item.get("subject")
                    or item.get("obj1")
                    or item.get("from")
                    or item.get("source")
                    or ""
                )
                rel = item.get("relation") or item.get("predicate") or ""
                obj2 = (
                    item.get("object")
                    or item.get("obj2")
                    or item.get("to")
                    or item.get("target")
                    or ""
                )
            else:
                continue

            edges.append((str(obj1), str(rel), str(obj2)))

        return edges
    
    def cleanup(self):
        """Free GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _get_prompt(client: "PaliGemmaClient", key: str, default: str) -> str:
    prompts = getattr(client, "prompts", {}) or {}
    return prompts.get(key, default)

# Global client instance (lazy initialization)
_paligemma_client: Optional[PaliGemmaClient] = None



class Qwen3VLClient(PaliGemmaClient):
    """
    Lightweight client for Qwen3-VL that mirrors the PaliGemmaClient interface.

    The core difference is the chat-style prompt construction that relies on the
    processor's chat template to insert the correct vision start/end tokens so
    that image features align with image tokens during generation.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prompts: Optional[Any] = None,
    ):
        from transformers import AutoProcessor, AutoModelForVision2Seq

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Store prompts (keeps compatibility with PaliGemma helper methods)
        super().__init__(
            model_name=model_name,
            device=device,
            prompts=prompts,
            load_model=False,
        )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        # Qwen-specific generation defaults
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[Qwen3VL] Model loaded on {self.device}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Qwen3-VL specific generate.

        We build chat-style messages with a dedicated image content block, then
        let the processor's chat_template inject the right special tokens so
        that image tokens and image features match.
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p

        # 1) Build chat-style messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 2) Let the processor build the actual text prompt with the correct
        #    image tokens via its chat template, if available.
        if hasattr(self.processor, "apply_chat_template"):
            chat_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: explicitly include the vision start/end tokens that
            # Qwen expects around the image placeholder.
            chat_prompt = f"<|vision_start|><|image|><|vision_end|>\n{prompt}"

        # 3) Prepare multimodal inputs (always pass lists so image order matches
        #    placeholders in the chat prompt)
        inputs = self.processor(
            text=[chat_prompt],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.0,
                    temperature=temperature,
                    top_p=top_p,
                )
        except Exception as e:
            print(f"[Qwen3VL] Generation error: {e}")
            return ""

        # 4) Decode
        decoded = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded[0].strip() if len(decoded) > 0 else ""

_qwen3vl_client: Optional[Qwen3VLClient] = None

def get_paligemma_client(
    model_name: str = "google/paligemma2-3b-mix-224",
    device: str = None,
    prompts: Optional[Any] = None,
    force_new: bool = False,
) -> PaliGemmaClient:
    global _paligemma_client
    
    if _paligemma_client is None or force_new:
        _paligemma_client = PaliGemmaClient(
            model_name=model_name,
            device=device,
            prompts=prompts,
        )
    
    return _paligemma_client


def get_qwen3vl_client(
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "cuda",
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prompts: Optional[Any] = None,
    force_new: bool = False,
) -> Qwen3VLClient:
    """
    Lazy singleton accessor for Qwen3-VL.

    Usage:
        qwen_client = get_qwen3vl_client(
            model_name=cfg.qwen_model,
            device=cfg.device,
            prompts=cfg.prompts,
        )
    """
    global _qwen3vl_client

    if _qwen3vl_client is None or force_new:
        _qwen3vl_client = Qwen3VLClient(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompts=prompts,
        )
    return _qwen3vl_client

# Alias to match OpenAI interface
def get_openai_client():
    """
    Compatibility alias - returns PaliGemma client instead of OpenAI.
    Drop-in replacement for code that calls get_openai_client().
    """
    return get_paligemma_client()


# =============================================================================
# Image Encoding (kept for compatibility, though PaliGemma uses PIL directly)
# =============================================================================

def encode_image_for_openai(image_path: str, resize: bool = False, target_size: int = 512) -> str:
    """
    Encode image as base64 (kept for compatibility with existing code).
    PaliGemma actually uses PIL images directly, but this maintains the interface.
    """
    print(f"Checking if image exists at path: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not resize:
        print(f"Opening image from path: {image_path}")
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        return encoded_image
    
    print(f"Opening image from path: {image_path}")
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width} x {original_height}")
        
        if original_width > original_height:
            scale = target_size / original_width
            new_width = target_size
            new_height = int(original_height * scale)
        else:
            scale = target_size / original_height
            new_height = target_size
            new_width = int(original_width * scale)

        print(f"Resized image dimensions: {new_width} x {new_height}")
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        print("Image resized successfully.")
        
        with open("temp_resized_image.jpg", "wb") as temp_file:
            img_resized.save(temp_file, format="JPEG")
            print("Resized image saved temporarily for encoding.")
        
        with open("temp_resized_image.jpg", "rb") as temp_file:
            encoded_image = base64.b64encode(temp_file.read()).decode('utf-8')
            print("Image encoded in base64 format.")
        
        os.remove("temp_resized_image.jpg")
        print("Temporary file removed.")

    return encoded_image


def load_image_for_paligemma(image_path: str, target_size: int = 224) -> Image.Image:
    """
    Load and optionally resize image for PaliGemma.
    
    Args:
        image_path: Path to image file
        target_size: Target size for the shorter dimension
        
    Returns:
        PIL Image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")
    return img


# =============================================================================
# Response Parsing Utilities (same as vlm.py)
# =============================================================================

def extract_list_of_tuples(text: str) -> List[Tuple]:
    """
    Extract list of tuples from text response.
    Same implementation as vlm.py for compatibility.
    """
    text = text.replace('\n', ' ')
    pattern = r'\[.*?\]'
    
    match = re.search(pattern, text)
    if match:
        list_str = match.group(0)
        try:
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        print("No list of tuples found in the text.")
        return []


def vlm_extract_object_captions(text: str) -> List[Dict]:
    """
    Extract list of object caption dictionaries from text.
    Same implementation as vlm.py for compatibility.
    """
    text = text.replace('\n', ' ')
    pattern = r'\[(.*?)\]'
    
    match = re.search(pattern, text)
    if match:
        list_str = match.group(0)
        try:
            result = ast.literal_eval(list_str)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            elements = re.findall(r'{.*?}', list_str)
            result = []
            for element in elements:
                try:
                    obj = ast.literal_eval(element)
                    if isinstance(obj, dict):
                        result.append(obj)
                except (ValueError, SyntaxError):
                    print(f"Error processing element: {element}")
            return result
    else:
        print("No list of objects found in the text.")
        return []


def parse_paligemma_relations(response: str, label_list: List[str]) -> List[Tuple[str, str, str]]:
    """
    Parse PaliGemma response into list of (obj1, relation, obj2) tuples.
    
    PaliGemma outputs free-form text, so we need to extract relationships.
    """
    relations = []
    response_lower = response.lower()
    
    # Common spatial relation keywords
    relation_keywords = {
        "on": "on",
        "on top of": "on",
        "above": "above",
        "below": "below",
        "under": "under",
        "beneath": "under",
        "next to": "next_to",
        "beside": "next_to",
        "near": "near",
        "in front of": "in_front_of",
        "behind": "behind",
        "inside": "inside",
        "in": "inside",
        "left of": "left_of",
        "right of": "right_of",
    }
    
    # Try to find relationships between labeled objects
    for i, label1 in enumerate(label_list):
        label1_base = label1.split()[0].lower() if label1 else ""
        for j, label2 in enumerate(label_list):
            if i == j:
                continue
            label2_base = label2.split()[0].lower() if label2 else ""
            
            # Check for relation patterns
            for keyword, relation in relation_keywords.items():
                # Pattern: "label1 is keyword label2" or "label1 keyword label2"
                patterns = [
                    f"{label1_base} is {keyword} {label2_base}",
                    f"{label1_base} {keyword} {label2_base}",
                    f"{label1_base} is {keyword} the {label2_base}",
                ]
                for pattern in patterns:
                    if pattern in response_lower:
                        relations.append((label1, relation, label2))
                        break
    
    return relations


def parse_paligemma_captions(response: str, label_list: List[str]) -> List[Dict[str, str]]:
    """
    Parse PaliGemma response into list of caption dictionaries.
    
    Returns:
        List of {"label": str, "caption": str} dicts
    """
    captions = []
    
    # Split response into sentences/segments
    segments = re.split(r'[.;]', response)
    
    for label in label_list:
        label_base = label.split()[0].lower() if label else ""
        
        # Find caption for this label
        for segment in segments:
            if label_base in segment.lower():
                caption = segment.strip()
                if caption:
                    captions.append({
                        "label": label,
                        "caption": caption
                    })
                    break
        else:
            # No specific caption found, use generic
            captions.append({
                "label": label,
                "caption": f"a {label_base}"
            })
    
    return captions


# =============================================================================
# Main API Functions (matching vlm.py signatures)
# =============================================================================

def consolidate_captions(client: PaliGemmaClient, captions: List[Dict]) -> str:
    """
    Consolidate multiple captions into a single coherent caption.
    
    MATCHES vlm.py SIGNATURE: consolidate_captions(client, captions)
    
    Args:
        client: PaliGemmaClient instance
        captions: List of caption dicts with 'caption' key
        
    Returns:
        Consolidated caption string
    """
    # Extract caption text
    caption_texts = [cap.get('caption', '') for cap in captions if cap.get('caption')]
    if not caption_texts:
        return "unknown object"
    if len(caption_texts) == 1:
        return caption_texts[0]

    captions_text = " ".join(caption_texts[:5])

    template = _get_prompt(client, "consolidate", SYSTEM_PROMPT_CONSOLIDATE)

    if "{captions}" in template:
        prompt = template.format(captions=captions_text)
    else:
        prompt = f"{template} {captions_text}"

    dummy_image = Image.new('RGB', (224, 224), color='white')

    try:
        consolidated = client.generate(dummy_image, prompt, max_new_tokens=128)
        
        # Try to extract JSON if the model returns it
        if '{' in consolidated and '}' in consolidated:
            try:
                json_match = re.search(r'\{.*?\}', consolidated, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    if 'consolidated_caption' in parsed:
                        return parsed['consolidated_caption']
            except json.JSONDecodeError:
                pass
        
        print(f"Consolidated Caption: {consolidated}")
        return consolidated if consolidated else caption_texts[0]
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return caption_texts[0] if caption_texts else ""


def get_obj_rel_from_image_gpt4v(
    client: PaliGemmaClient,
    image_path: str,
    label_list: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Extract spatial relationships between objects in an image.
    
    MATCHES vlm.py SIGNATURE: get_obj_rel_from_image_gpt4v(client, image_path, label_list)
    
    Args:
        client: PaliGemmaClient instance
        image_path: Path to the annotated image
        label_list: List of object labels in the image
        
    Returns:
        List of (object1, relation, object2) tuples
    """
    # Load image
    try:
        image = load_image_for_paligemma(image_path)
    except FileNotFoundError as e:
        print(f"An error occurred: {str(e)}")
        return []
    
    # Build prompt
    labels_str = ", ".join(label_list)
    template = _get_prompt(client, "relations_with_labels", PROMPT_TEMPLATE_RELATIONS)
    prompt = template.format(labels=labels_str)
    
    vlm_answer = []
    try:
        response = client.generate(image, prompt, max_new_tokens=256)
        print(f"Line 113, vlm_answer_str: {response}")
        
        # First try to extract structured list
        vlm_answer = extract_list_of_tuples(response)
        
        # If no structured output, parse free-form text
        if not vlm_answer:
            vlm_answer = parse_paligemma_relations(response, label_list)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer = []
    
    print(f"Line 68, user_query: {prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer}")
    
    return vlm_answer


def get_obj_captions_from_image_gpt4v(
    client: PaliGemmaClient,
    image_path: str,
    label_list: List[str]
) -> List[Dict[str, str]]:
    """
    Generate captions for labeled objects in an image.
    
    MATCHES vlm.py SIGNATURE: get_obj_captions_from_image_gpt4v(client, image_path, label_list)
    
    Args:
        client: PaliGemmaClient instance
        image_path: Path to the annotated image
        label_list: List of object labels in the image
        
    Returns:
        List of {"label": str, "caption": str} dictionaries
    """
    # Load image
    try:
        image = load_image_for_paligemma(image_path)
    except FileNotFoundError as e:
        print(f"An error occurred: {str(e)}")
        return []
    
    # Build prompt
    labels_str = ", ".join(label_list)
    template = _get_prompt(client, "captions_with_labels", PROMPT_TEMPLATE_CAPTIONS)
    prompt = template.format(labels=labels_str)

    
    vlm_answer_captions = []
    try:
        response = client.generate(image, prompt, max_new_tokens=256)
        print(f"Line 113, vlm_answer_str: {response}")
        
        # First try to extract structured list
        vlm_answer_captions = vlm_extract_object_captions(response)
        
        # If no structured output, parse free-form text
        if not vlm_answer_captions:
            vlm_answer_captions = parse_paligemma_captions(response, label_list)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        vlm_answer_captions = []
    
    print(f"Line 68, user_query: {prompt}")
    print(f"Line 97, vlm_answer: {vlm_answer_captions}")
    
    return vlm_answer_captions


# =============================================================================
# Additional PaliGemma-specific functions
# =============================================================================

def caption_single_object(
    client: PaliGemmaClient,
    image: Image.Image,
    bbox: Optional[np.ndarray] = None,
    padding: int = 10,
) -> str:
    """
    Generate caption for a single object (optionally cropped by bbox).
    
    Args:
        client: PaliGemmaClient instance
        image: PIL Image
        bbox: Optional [x1, y1, x2, y2] bounding box
        padding: Padding around bbox when cropping
        
    Returns:
        Caption string
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = image.size
        
        # Apply padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        image = image.crop((x1, y1, x2, y2))
    
    return client.generate(image, SYSTEM_PROMPT_CAPTIONS, max_new_tokens=128)


def extract_relationship(
    client: PaliGemmaClient,
    obj1_tag: str,
    obj2_tag: str,
) -> Tuple[str, str]:
    """
    Extract spatial relationship between two named objects.
    
    Args:
        client: PaliGemmaClient instance
        obj1_tag: First object tag/name
        obj2_tag: Second object tag/name
        
    Returns:
        Tuple of (relation, reason/explanation)
    """
    base = _get_prompt(client, "relation", "relation: {obj1} and {obj2}")
    if "{obj1" in base or "{obj2" in base:
        prompt = base.format(obj1=obj1_tag, obj2=obj2_tag)
    else:
        prompt = f"{base} {obj1_tag} and {obj2_tag}"

    
    dummy_image = Image.new('RGB', (224, 224), color='white')
    response = client.generate(dummy_image, prompt, max_new_tokens=64).lower()
    
    # Parse response for spatial relations
    if "on" in response and obj1_tag.lower() in response:
        relation = "on"
    elif "on" in response and obj2_tag.lower() in response:
        relation = "under"
    elif "in" in response:
        relation = "inside"
    elif "next" in response or "beside" in response:
        relation = "next_to"
    elif "above" in response:
        relation = "above"
    elif "below" in response:
        relation = "below"
    else:
        relation = "near"
    
    return relation, response


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_paligemma():
    """Release PaliGemma model from GPU memory."""
    global _paligemma_client
    if _paligemma_client is not None:
        _paligemma_client.cleanup()
        _paligemma_client = None
        print("[PaliGemma] Client cleaned up.")