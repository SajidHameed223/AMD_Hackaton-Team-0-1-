"""
Model Loading & Caching for Multi-Model Routing

Phase 4A: Dynamic Model Selection
- Supports loading multiple Gemma models on demand
- Caches loaded models to avoid reloading
- Tracks model memory usage and device placement
- Backward compatible: default model loaded at startup
"""

import os
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================================
# Global Model Manager
# ============================================================================

class ModelManager:
    """Manages dynamic model loading and caching."""

    def __init__(self):
        self.loaded_models = {}  # model_id → (model, tokenizer)
        self.default_model_id = os.getenv(
            "MODEL_NAME", "google/gemma-4-26b-a4b-it"
        )
        self._print_system_info()

    def _print_system_info(self):
        """Print GPU/system info once at startup."""
        print(f"[Model] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Model] GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"[Model] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )

    def get_model_and_tokenizer(
        self, model_id: str = None
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Get a model and tokenizer, loading if not already cached.

        Args:
            model_id: HuggingFace model ID. If None, uses default.

        Returns:
            (model, tokenizer) tuple
        """
        if model_id is None:
            model_id = self.default_model_id

        # Return cached if available
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        # Load new model
        print(f"[Model] Loading: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
            trust_remote_code=True,
        )

        print(f"[Model] Loaded successfully on {model.device}")
        print(
            f"[Model] Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
        )

        # Cache it
        self.loaded_models[model_id] = (model, tokenizer)

        return model, tokenizer

    def unload_model(self, model_id: str):
        """Unload a model to free memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[Model] Unloaded: {model_id}")

    def get_loaded_models(self):
        """Get list of currently loaded models."""
        return list(self.loaded_models.keys())

    def get_memory_usage(self):
        """Get GPU memory currently used by loaded models."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            }
        return {"allocated_gb": 0.0, "reserved_gb": 0.0}


# Global manager instance
_manager = ModelManager()

# Load default model at startup
print(f"[Model] Initializing with default: {_manager.default_model_id}")
model, tokenizer = _manager.get_model_and_tokenizer()


# ============================================================================
# Public API (for backward compatibility)
# ============================================================================


def get_model_and_tokenizer(model_id: str = None) -> Tuple:
    """Get model and tokenizer (may load if not cached)."""
    return _manager.get_model_and_tokenizer(model_id)


def unload_model(model_id: str):
    """Unload a model to free memory."""
    _manager.unload_model(model_id)


def get_loaded_models():
    """List currently loaded models."""
    return _manager.get_loaded_models()


def get_memory_usage():
    """Get current GPU memory usage."""
    return _manager.get_memory_usage()