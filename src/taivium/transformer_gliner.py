"""GLiNER transformer for multi-token NER detection.

GLiNER is a state-of-the-art Named Entity Recognition model that can handle
variable-length text inputs. This module handles texts longer than GLiNER's
384-token window limit by chunking with overlap and processing in mini-batches.
"""

import logging
from typing import Any, cast, Dict, List, Optional, Tuple
from functools import lru_cache

import spacy

from .utility import get_gliner_model

# Import Evidence class and normalize_label from engine
# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This is only for type hints, won't cause circular import at runtime
    from .engine import Evidence
else:
    # At runtime, we'll get Evidence from the caller's context
    Evidence = Any

logger = logging.getLogger("taivium.engine")

# GLiNER inference configuration constants
_GLINER_MAX_TOKENS = 384
_GLINER_OVERLAP_TOKENS = 32
_GLINER_BATCH_SIZE = 8


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Get cached blank spaCy tokenizer for fast, accurate token counting.
    
    Uses spaCy's blank "en" model which provides accurate tokenization
    without the overhead of full NLP pipeline (POS, dependencies, etc.).
    """
    return spacy.blank("en")


def _chunk_text_for_gliner(
    text: str,
    max_tokens: int = _GLINER_MAX_TOKENS,
    overlap_tokens: int = _GLINER_OVERLAP_TOKENS,
) -> List[Tuple[int, str]]:
    """Split text into overlapping token chunks for GLiNER window-limited inference.

    Uses spaCy's blank tokenizer for accurate token counting without NLP overhead.
    Returns a list of ``(offset, chunk_text)`` pairs where ``offset`` is the character 
    start in the original text. Offsets allow chunk-local predictions to be remapped 
    back to global coordinates.
    """
    if not text:
        return []

    try:
        nlp = _get_tokenizer()
        doc = nlp(text)
        tokens = list(doc)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.debug("spaCy tokenization failed: %s, falling back to text as single chunk", e)
        return [(0, text)]

    if not tokens:
        return [(0, text)]
    if len(tokens) <= max_tokens:
        return [(0, text)]

    effective_overlap = max(0, min(overlap_tokens, max_tokens - 1))
    step = max_tokens - effective_overlap
    chunks: List[Tuple[int, str]] = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunk_start_char = tokens[start_idx].idx
        chunk_end_char = tokens[end_idx - 1].idx + len(tokens[end_idx - 1].text)
        chunks.append((chunk_start_char, text[chunk_start_char:chunk_end_char]))

        if end_idx >= len(tokens):
            break
        start_idx += step

    return chunks


def _predict_gliner_chunks(
    model: Any,
    chunks: List[Tuple[int, str]],
    target_labels: List[str],
    threshold: float,
) -> List[List[Dict[str, Any]]]:
    """Run GLiNER inference across chunks, using the fastest stable API first."""
    texts = [chunk_text for _, chunk_text in chunks]

    inference = getattr(model, "inference", None)
    if callable(inference):
        try:
            all_predictions: List[List[Dict[str, Any]]] = []
            for i in range(0, len(texts), _GLINER_BATCH_SIZE):
                batch_texts = texts[i:i + _GLINER_BATCH_SIZE]
                batch_preds = cast(
                    List[List[Dict[str, Any]]],
                    inference(
                        batch_texts,
                        target_labels,
                        threshold=threshold,
                        batch_size=_GLINER_BATCH_SIZE,
                    ),
                )
                if len(batch_preds) != len(batch_texts):
                    raise ValueError("GLiNER inference prediction shape mismatch")
                all_predictions.extend(batch_preds)
            return all_predictions
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "GLiNER inference API unavailable; trying legacy batch mode: %s",
                exc,
            )

    batch_predict = getattr(model, "batch_predict_entities", None)
    if callable(batch_predict):
        try:
            all_predictions = []
            for i in range(0, len(texts), _GLINER_BATCH_SIZE):
                batch_texts = texts[i:i + _GLINER_BATCH_SIZE]
                batch_preds = cast(
                    List[List[Dict[str, Any]]],
                    batch_predict(batch_texts, target_labels, threshold=threshold),
                )
                if len(batch_preds) != len(batch_texts):
                    raise ValueError("GLiNER batch prediction shape mismatch")
                all_predictions.extend(batch_preds)
            return all_predictions
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "GLiNER batch inference unavailable; falling back to per-chunk mode: %s",
                exc,
            )

    return [
        cast(List[Dict[str, Any]], model.predict_entities(chunk_text, target_labels, threshold=threshold))
        for chunk_text in texts
    ]


def gliner_evidence(text: str, targets: Optional[List[str]] = None) -> List[Any]:
    """Collect evidence from GLiNER for PERSON, LOCATION, and ORGANIZATION entities.

    GLiNER is optimized for high-precision detection: uses 0.55 threshold to filter
    out uncertain predictions, prioritizing correct detections over recall. Inputs
    longer than 384 tokens are chunked with overlap and processed in batches.

    Args:
        text: Input text to analyze.
        targets: List of entity labels to detect (default: ["PERSON", "LOCATION", "ORGANIZATION"]).

    Returns:
        List of GLiNER-origin `Evidence` records with high-confidence threshold.
    """
    # Import Evidence and normalize_label here to avoid circular import at module load time
    from .engine import Evidence, normalize_label  # pylint: disable=import-outside-toplevel

    evidence: List[Any] = []

    # Use provided targets or default to PERSON, LOCATION, and ORGANIZATION
    target_labels = targets if targets is not None else ["PERSON", "LOCATION", "ORGANIZATION"]
    if not target_labels:
        return evidence

    try:
        model = get_gliner_model()
        chunks = _chunk_text_for_gliner(text)
        if not chunks:
            return evidence

        chunk_predictions = _predict_gliner_chunks(
            model,
            chunks,
            target_labels,
            threshold=0.55,
        )
        seen_spans: set[Tuple[int, int, str]] = set()

        for (chunk_offset, _), predictions in zip(chunks, chunk_predictions):
            for pred in predictions:
                # Normalize the label from GLiNER output (e.g., "name" → "PERSON")
                label = normalize_label(pred.get("label", "UNKNOWN"))
                if label == "UNKNOWN":
                    continue

                start = chunk_offset + int(pred["start"])
                end = chunk_offset + int(pred["end"])
                key = (start, end, label)
                if key in seen_spans:
                    continue
                seen_spans.add(key)

                # Use GLiNER's confidence score directly for precision-focused filtering
                gliner_confidence = pred.get("score", 0.55)
                evidence.append(Evidence(
                    start=start,
                    end=end,
                    label=label,
                    source="gliner",
                    confidence=gliner_confidence,  # High-confidence GLiNER scores
                ))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("GLiNER detection failed: %s", e)

    return evidence
