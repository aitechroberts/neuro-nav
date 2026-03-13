"""
VLM Vision Encoder Extractor.

Loads only the vision encoder component of a VLM for local embedding extraction,
allowing comparison between fine-tuned VLM encoder representations and standalone
pretrained embedding models (TinyCLIP, SigLIP, etc.).

The full model is loaded in float16, the vision encoder is extracted, and all
other components (LLM decoder, projector) are deleted to save VRAM.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


# =============================================================================
# Model Family Detection
# =============================================================================

# Maps lowercased substrings in HuggingFace model IDs to a family key.
_FAMILY_PATTERNS = [
    ("qwen3-vl", "qwen3vl"),
    ("qwen2.5-vl", "qwen25vl"),
    ("qwen2-vl", "qwen2vl"),
    ("internvl", "internvl"),
    ("cogvlm2", "cogvlm2"),
    ("cogvlm", "cogvlm"),
    ("ovis", "ovis"),
    ("gemma-3", "gemma3"),
    ("llava-onevision", "llava_ov"),
    ("llava", "llava"),
    ("minicpm-v", "minicpm"),
    ("minicpm_v", "minicpm"),
    ("smolvlm", "smolvlm"),
    ("idefics3", "idefics3"),
    ("idefics2", "idefics2"),
]


def _detect_family(model_name: str) -> str:
    lower = model_name.lower().replace("/", "-").replace("_", "-")
    for pattern, family in _FAMILY_PATTERNS:
        if pattern in lower:
            return family
    return "unknown"


# =============================================================================
# Per-Family Encoder Extraction Strategies
# =============================================================================

def _load_qwen3vl(model_name, device, dtype):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    encoder = model.visual
    processor = AutoProcessor.from_pretrained(model_name)
    del model.model  # drop the LLM decoder
    torch.cuda.empty_cache()
    return encoder, processor, "qwen3vl"


def _load_qwen25vl(model_name, device, dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    encoder = model.visual
    processor = AutoProcessor.from_pretrained(model_name)
    del model.model
    torch.cuda.empty_cache()
    return encoder, processor, "qwen25vl"


def _load_qwen2vl(model_name, device, dtype):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    encoder = model.visual
    processor = AutoProcessor.from_pretrained(model_name)
    del model.model
    torch.cuda.empty_cache()
    return encoder, processor, "qwen2vl"


def _load_internvl(model_name, device, dtype):
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    encoder = model.vision_model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model, "language_model"):
        del model.language_model
    torch.cuda.empty_cache()
    return encoder, processor, "internvl"


def _load_generic_vision_tower(model_name, device, dtype, attr="vision_tower"):
    """Covers LLaVA-OneVision, Gemma 3, and similar architectures."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    encoder = getattr(model, attr, None)
    if encoder is None:
        for candidate in ["vision_tower", "vision_model", "visual", "vpm"]:
            encoder = getattr(model, candidate, None)
            if encoder is not None:
                break
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model, "language_model"):
        del model.language_model
    torch.cuda.empty_cache()
    return encoder, processor, "generic"


_LOADERS = {
    "qwen3vl": _load_qwen3vl,
    "qwen25vl": _load_qwen25vl,
    "qwen2vl": _load_qwen2vl,
    "internvl": _load_internvl,
    "cogvlm": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_model"),
    "cogvlm2": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_model"),
    "ovis": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "visual_tokenizer"),
    "gemma3": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_tower"),
    "llava_ov": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_tower"),
    "llava": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_tower"),
    "minicpm": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vpm"),
    "smolvlm": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_model"),
    "idefics3": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_model"),
    "idefics2": lambda n, d, dt: _load_generic_vision_tower(n, d, dt, "vision_model"),
}


# =============================================================================
# VLM Encoder Extractor
# =============================================================================

class VLMEncoderExtractor:
    """
    Loads just the vision encoder from a VLM for embedding extraction.

    Usage:
        extractor = VLMEncoderExtractor("Qwen/Qwen3-VL-2B-Instruct", "cuda")
        feats = extractor.encode_crops([crop1, crop2, ...])
        extractor.cleanup()
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.family = _detect_family(model_name)

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.dtype = dtype

        logger.info(
            f"[VLM-Encoder] Loading vision encoder from {model_name} "
            f"(family={self.family})..."
        )

        loader = _LOADERS.get(self.family)
        if loader is None:
            logger.warning(
                f"[VLM-Encoder] Unknown family '{self.family}' for {model_name}, "
                f"attempting generic loader."
            )
            loader = lambda n, d, dt: _load_generic_vision_tower(n, d, dt)

        self.encoder, self.processor, self._loader_tag = loader(
            model_name, device, dtype
        )

        if self.encoder is None:
            raise RuntimeError(
                f"[VLM-Encoder] Could not extract vision encoder from {model_name}. "
                f"No known attribute (vision_model, visual, vision_tower, vpm) found."
            )

        self.encoder.eval()

        param_count = sum(p.numel() for p in self.encoder.parameters())
        vram_mb = param_count * (2 if dtype == torch.float16 else 2) / 1e6
        logger.info(
            f"[VLM-Encoder] Loaded: {param_count/1e6:.1f}M params (~{vram_mb:.0f}MB)"
        )

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> np.ndarray:
        """
        Encode a single PIL image through the vision encoder.
        Returns an L2-normalized 1-D feature vector (numpy).
        """
        inputs = self.processor(
            images=pil_image, return_tensors="pt"
        )
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise ValueError(
                "[VLM-Encoder] Processor did not return pixel_values."
            )
        pixel_values = pixel_values.to(self.device, dtype=self.dtype)

        out = self.encoder(pixel_values)

        if hasattr(out, "last_hidden_state"):
            feats = out.last_hidden_state.mean(dim=1)
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif isinstance(out, torch.Tensor):
            if out.dim() == 3:
                feats = out.mean(dim=1)
            elif out.dim() == 2:
                feats = out
            else:
                feats = out.reshape(1, -1)
        else:
            raise ValueError(
                f"[VLM-Encoder] Unexpected encoder output type: {type(out)}"
            )

        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy().squeeze()

    @torch.no_grad()
    def encode_crops(
        self,
        crops: List[Image.Image],
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode a batch of PIL image crops. Returns (N, D) numpy array, L2-normalized.
        Falls back to sequential encoding if batched processing fails.
        """
        if not crops:
            return np.zeros((0, 1), dtype=np.float32)

        try:
            inputs = self.processor(images=crops, return_tensors="pt")
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                raise ValueError("No pixel_values from processor")
            pixel_values = pixel_values.to(
                device or self.device, dtype=self.dtype
            )

            out = self.encoder(pixel_values)

            if hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state.mean(dim=1)
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            elif isinstance(out, torch.Tensor):
                if out.dim() == 3:
                    feats = out.mean(dim=1)
                else:
                    feats = out
            else:
                raise ValueError(f"Unexpected output type: {type(out)}")

            feats = F.normalize(feats.float(), dim=-1)
            return feats.cpu().numpy()

        except Exception as e:
            logger.warning(
                f"[VLM-Encoder] Batched encoding failed ({e}), falling back to sequential."
            )
            results = []
            for crop in crops:
                results.append(self.encode_image(crop))
            return np.stack(results, axis=0)

    def cleanup(self):
        """Free GPU memory held by the vision encoder."""
        if hasattr(self, "encoder"):
            del self.encoder
        if hasattr(self, "processor"):
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[VLM-Encoder] Cleanup complete.")
