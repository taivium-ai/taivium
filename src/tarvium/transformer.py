"""
Transformer-based NER evidence collector.
------------------------------------------
Uses ``dslim/bert-base-NER`` (BERT fine-tuned on CoNLL-2003) via the
HuggingFace ``transformers`` library.  The dependency is optional: when
``transformers`` (or a required backend such as PyTorch) is not installed,
:func:`transformer_evidence` degrades gracefully to an empty list.

"""
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import warnings

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List

logger = logging.getLogger("tarvium.transformer")

if TYPE_CHECKING:
    from .engine import Evidence

# Mapping from dslim/bert-base-NER entity groups to internal labels.
_LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": "UNKNOWN",  # too ambiguous for privacy use-cases; skipped
}

# Labels that map directly without going through normalize_label.
_DIRECT_LABELS = {"PERSON", "ORG", "LOCATION"}


@lru_cache(maxsize=1)
def _get_ner_pipeline() -> Any:
    """Lazy-loads and caches the HuggingFace NER pipeline.

    Returns ``None`` when ``transformers`` is not installed or the model
    cannot be loaded, so the evidence layer degrades gracefully.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
        return hf_pipeline(
            "ner",  # type: ignore[arg-type]
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
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


def transformer_evidence(text: str) -> List[Evidence]:
    """Collects NER evidence from a BERT-based NER pipeline.

    Uses ``dslim/bert-base-NER`` (BERT fine-tuned on CoNLL-2003) via
    HuggingFace ``transformers`` when available.  Returns an empty list
    when the package is not installed or the model fails to load.

    Label mapping:
        - ``PER``  → ``PERSON``
        - ``ORG``  → ``ORG``
        - ``LOC``  → ``LOCATION``
        - ``MISC`` → skipped (ambiguous)

    Args:
        text: The input text to run NER over.

    Returns:
        A list of :class:`~tarvium.engine.Evidence` records with
        ``source="transformer"``.
    """
    # Deferred import to avoid circular dependency with engine.py.
    from .engine import Evidence, normalize_label  # pylint: disable=import-outside-toplevel


    ner = _get_ner_pipeline()
    if ner is None:
        return []
    try:
        predictions = ner(text)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Transformer evidence extraction failed: %s", exc, exc_info=True)
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
            source="transformer",
            confidence=float(pred["score"]),
        ))
    return evidence
