''' 
Utility functions for Taivium.
'''
import os
import logging
from functools import lru_cache
from typing import Any
import onnxruntime as rt
import spacy

logger = logging.getLogger("taivium.utility")


def _verify_onnx_provider(model: Any) -> str:  # pylint: disable=unused-argument
    """Verify which ONNX execution provider is actually in use.

    Args:
        model: Loaded model instance (not introspected directly).

    Returns:
        Name of the likely active ONNX execution provider.
    """
    try:
        available = rt.get_available_providers()
        logger.debug("ONNX Runtime available providers: %s", available)

        if "CoreMLExecutionProvider" in available:
            return "CoreMLExecutionProvider"
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to verify ONNX provider: %s", e)
        return "CPUExecutionProvider (fallback)"


@lru_cache(maxsize=1)
def get_gliner_model() -> Any:
    """Backward-compatible wrapper for GLiNER model loading.

    Delegates to ``taivium.backend.transformer_gliner.get_gliner_model``.
    """
    from .backend.transformer_gliner import get_gliner_model as _backend_get_gliner_model

    return _backend_get_gliner_model()

# -----------------------------
# Lazy-load spaCy model
# -----------------------------

@lru_cache(maxsize=8)
def get_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Lazy-load and return a spaCy model with only NER enabled.

    Args:
        model_name: spaCy model package name to load (default: ``en_core_web_sm``).

    Returns:
        Loaded spaCy pipeline instance.

    Raises:
        OSError: If the requested spaCy model is not installed.
    """
    try:
        # Disable unused components (tagger, parser, lemmatizer) for faster
        # inference
        return spacy.load(
            model_name,
            exclude=[
                "tagger",
                "parser",
                "lemmatizer",
                "attribute_ruler"])
    except OSError as exc:
        # Raise an error instead of falling back to a blank pipeline.
        error_text = f"spaCy model '{model_name}' not found."
        error_text += f" Please install it with 'python -m spacy download {model_name}'."
        logger.error(error_text, exc_info=True)
        raise OSError(error_text) from exc
