"""
Transformer-based OpenAI privacy filter NER evidence collector.
https://openai.com/index/introducing-openai-privacy-filter/

OpenAI privacy filter detects sensitive entities in text.
This module provides a consistent interface for privacy-focused entity detection
using the OpenAI privacy filter, being an alternative to the GLiNER backend 
for the long-text detection route.
"""
# pylint: disable=import-outside-toplevel
import logging
from typing import Any, List
from functools import lru_cache
import warnings
from ..defs import Evidence, normalize_label
logger = logging.getLogger("taivium.backend.transformer_openai_privacy_filter")

# Mapping from dslim/bert-base-NER entity groups to internal labels.
_LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": "UNKNOWN",  # too ambiguous for privacy use-cases; skipped
    "private_person": "PERSON",
    "account_number": "ORG",
    "private_url": "LOCATION",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "LOCATION",
    "secret": "API",
    "private_date": "DATE"   
}

# Labels that map directly without going through normalize_label.

@lru_cache(maxsize=1)
def _get_openai_ner_pipeline() -> Any:
    """Lazy-loads and caches the HuggingFace NER pipeline.

    Returns ``None`` when ``transformers`` is not installed or the model
    cannot be loaded, so the evidence layer degrades gracefully.
    """
    try:
        from huggingface_hub import snapshot_download
        from openmed.mlx.inference import PrivacyFilterMLXPipeline
        # Downloads the Apple Silicon optimized 8-bit model weights
        model_path = snapshot_download("OpenMed/privacy-filter-multilingual-mlx-8bit")
        return PrivacyFilterMLXPipeline(model_path)
    except (ModuleNotFoundError, ImportError, OSError, RuntimeError):
        warning_text = (
            "Transformer NER pipeline unavailable. "+
            "Install 'transformers' and 'torch' for transformer-based evidence."
        )  # noqa: E501
        warnings.warn(
            warning_text,
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(warning_text, exc_info=True)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        # Catch-all for unexpected errors (model download, config, etc.)
        logger.error(
            "Unexpected error in transformer NER pipeline: %s", exc, exc_info=True
        )
        warnings.warn(
            "Transformer NER pipeline unavailable due to unexpected error.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

def openai_privacy_filter_evidence(text: str) -> List[Evidence]:
    """Collects NER evidence from the OpenAI privacy filter.

    Uses the OpenAI privacy filter to detect sensitive entities in text.
    Returns an empty list when the filter is not available or fails to load.

    Label mapping:
        - ``PER``  → ``PERSON``
        - ``ORG``  → ``ORG``
        - ``LOC``  → ``LOCATION``
        - ``MISC`` → skipped (ambiguous)

    Args:
        text: The input text to run NER over.

    Returns:
        A list of :class:`~taivium.engine.Evidence` records with
        ``source="openai_privacy_filter"``.
    """

    ner = _get_openai_ner_pipeline()
    if ner is None:
        return []
    try:
        predictions = ner(text)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("OpenAI privacy filter evidence extraction failed: %s", exc, exc_info=True)
        return []

    evidence: List[Evidence] = []
    for pred in predictions:
        raw_label: str = pred.get("entity_group", "")
        label = normalize_label(_LABEL_MAP.get(raw_label, "UNKNOWN"))
        if label == "UNKNOWN":
            continue
        start: int = pred["start"]
        end: int = pred["end"]
        if not 0 <= start < end <= len(text):
            continue
        evidence.append(Evidence(
            start=start,
            end=end,
            label=label,
            source="openai_privacy_filter",
            confidence=float(pred["score"]),
        ))
    return evidence
