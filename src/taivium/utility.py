''' 
Utility functions for Taivium.
'''
import logging
from functools import lru_cache
from typing import Any
import spacy

logger = logging.getLogger("taivium.utility")
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
