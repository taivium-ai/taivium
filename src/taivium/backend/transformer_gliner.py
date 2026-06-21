"""GLiNER transformer for multi-token NER detection.

GLiNER is a state-of-the-art Named Entity Recognition model that can handle
variable-length text inputs. This module handles texts longer than GLiNER's
384-token window limit by chunking with overlap and processing in mini-batches.
"""

import logging
from functools import lru_cache
from typing import Any, cast, Dict, List, Optional, Tuple
import warnings
import os
from huggingface_hub import snapshot_download
from gliner import GLiNER
import onnxruntime as rt

from transformers.utils import logging as hf_logging
from ..defs import Evidence, normalize_label
from ..utility import _verify_onnx_provider

def suppress_gliner_warnings_veified_by_tests():
    """Suppress specific warnings from transformers and ONNX Runtime during GLiNER loading."""
    warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*fix_mistral_regex.*")
    warnings.filterwarnings("ignore", message=".*no maximum length is provided.*")
    hf_logging.set_verbosity_error()
suppress_gliner_warnings_veified_by_tests()

logger = logging.getLogger("taivium.backend.gliner")

# GLiNER inference configuration constants
_GLINER_MAX_TOKENS = 384
_GLINER_OVERLAP_TOKENS = 32
_GLINER_BATCH_SIZE = 8
# Tuned default from evaluation sweeps on privacy dataset subsets.
# Keeps precision and recall near a balanced operating point.
DEFAULT_GLINER_THRESHOLD = 0.47


@lru_cache(maxsize=1)
def get_gliner_model():
    """Lazy-load and cache GLiNER ONNX quantized model for faster inference."""
    try:
        try:
            available_providers = rt.get_available_providers()
            logger.info("Available ONNX providers: %s", available_providers)
        except ImportError:
            available_providers = ["CPUExecutionProvider"]
            logger.warning("onnxruntime not installed, using CPU only")

        preferred_providers = []
        if "CoreMLExecutionProvider" in available_providers:
            preferred_providers.append("CoreMLExecutionProvider")
            logger.info("CoreML provider available (M2/M3 GPU acceleration)")
        if "CUDAExecutionProvider" in available_providers:
            preferred_providers.append("CUDAExecutionProvider")
            logger.info("CUDA provider available (NVIDIA GPU acceleration)")
        if "CPUExecutionProvider" in available_providers:
            preferred_providers.append("CPUExecutionProvider")

        selected_provider = (
            preferred_providers[0]
            if preferred_providers
            else "CPUExecutionProvider"
        )
        logger.info("Selected ONNX provider: %s", selected_provider)

        repo_id = "onnx-community/gliner_small-v2.1"
        default_revision = "8142fb00740ccea973e64b1272949ff48653df5e"
        revision = os.getenv("TAIVIUM_GLINER_REVISION", default_revision)
        logger.info(
            "Downloading ONNX GLiNER model from %s at revision %s",
            repo_id,
            revision,
        )
        local_dir = snapshot_download(repo_id=repo_id, revision=revision)

        onnx_model_file = os.path.join("onnx", "model_quantized.onnx")

        logger.info(
            "Loading GLiNER from %s with ONNX quantization on %s",
            local_dir,
            selected_provider,
        )

        model = GLiNER.from_pretrained(
            local_dir,
            load_onnx_model=True,
            load_tokenizer=True,
            onnx_model_file=onnx_model_file,
            trust_remote_code=True,
            providers=preferred_providers,
        )

        actual_provider = _verify_onnx_provider(model)
        logger.info(
            "ONNX GLiNER loaded successfully on %s (5-10x faster inference)",
            actual_provider,
        )

        if actual_provider != selected_provider:
            logger.warning(
                "Provider mismatch: requested %s, but using %s. "
                "This may indicate GPU unavailability.",
                selected_provider,
                actual_provider,
            )

        return model

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "ONNX GLiNER load failed (%s), falling back to standard weights. "
            "Inference will be slower. Install onnxruntime for speedup: "
            "pip install onnxruntime",
            e,
        )
        return GLiNER.from_pretrained("knowledgator/gliner-pii-small-v1.0")


def _resolve_gliner_threshold(threshold: Optional[float]) -> float:
    """Resolve GLiNER confidence threshold from arg/env with validation."""
    raw: Any = threshold
    if raw is None:
        raw = os.getenv("TAIVIUM_GLINER_THRESHOLD", str(DEFAULT_GLINER_THRESHOLD))

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid GLiNER threshold=%r; using default %.2f",
            raw,
            DEFAULT_GLINER_THRESHOLD,
        )
        return DEFAULT_GLINER_THRESHOLD

    if not 0.0 <= value <= 1.0:
        logger.warning(
            "Out-of-range GLiNER threshold=%s; using default %.2f",
            value,
            DEFAULT_GLINER_THRESHOLD,
        )
        return DEFAULT_GLINER_THRESHOLD

    return value


def _chunk_text_for_gliner(
    text: str,
    max_tokens: int = _GLINER_MAX_TOKENS,
    overlap_tokens: int = _GLINER_OVERLAP_TOKENS,
    tokenizer: Any = None,
) -> List[Tuple[int, str]]:
    """Split text into overlapping token chunks for GLiNER window-limited inference.

    REQUIRES the GLiNER's actual DeBERTa tokenizer 
    (obtained from model.data_processor.transformer_tokenizer).
    Chunks are sized using exact subword token counts to eliminate truncation.

    Args:
        text: Input text to chunk.
        max_tokens: Maximum tokens per chunk (default: 384, GLiNER's hard limit).
        overlap_tokens: Overlap between consecutive chunks (default: 32 tokens).
        tokenizer: REQUIRED. GLiNER's DeBERTa tokenizer with offset_mapping support.
                   Must not be None.

    Returns:
        List of ``(offset, chunk_text)`` pairs where ``offset`` 
            is the character start in original text.

    Raises:
        ValueError: If tokenizer is None (no safe chunking without exact token counts).
        RuntimeError: If tokenization fails (DeBERTa tokenizer error).
    """
    if not text:
        return []

    # Tokenizer is REQUIRED — no unsafe fallback
    if tokenizer is None:
        raise ValueError(
            "GLiNER tokenizer is required for safe chunking. "
            "Pass tokenizer=model.data_processor.transformer_tokenizer " \
            "to _chunk_text_for_gliner(). "
            "No fallback tokenizer is available; truncation risk is unacceptable."
        )

    # Reserve 2 positions for CLS + SEP special tokens added at inference time
    max_content_tokens = max_tokens - 2
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_offsets: List[Tuple[int, int]] = enc["offset_mapping"]

        if len(token_offsets) <= max_content_tokens:
            return [(0, text)]

        effective_overlap = max(0, min(overlap_tokens, max_content_tokens - 1))
        step = max_content_tokens - effective_overlap
        chunks: List[Tuple[int, str]] = []
        start_idx = 0

        while start_idx < len(token_offsets):
            end_idx = min(start_idx + max_content_tokens, len(token_offsets))
            chunk_start_char = token_offsets[start_idx][0]
            chunk_end_char = token_offsets[end_idx - 1][1]
            chunks.append((chunk_start_char, text[chunk_start_char:chunk_end_char]))

            if end_idx >= len(token_offsets):
                break
            start_idx += step

        return chunks
    except (AttributeError, TypeError, KeyError) as e:
        raise RuntimeError(
            f"DeBERTa tokenizer failed unexpectedly: {e}. "
            "Ensure tokenizer is model.data_processor.transformer_tokenizer " \
            "with offset_mapping support."
        ) from e


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
        cast(List[Dict[str, Any]], model.predict_entities(chunk_text,
                                                target_labels, threshold=threshold))
        for chunk_text in texts
    ]

# pylint: disable=too-many-locals
def gliner_evidence(
    text: str,
    targets: Optional[List[str]] = None,
    threshold: Optional[float] = None,
) -> List[Any]:
    """Collect evidence from GLiNER for PERSON, LOCATION, and ORGANIZATION entities.

    GLiNER detection threshold is configurable via function argument or
    ``TAIVIUM_GLINER_THRESHOLD`` environment variable. Inputs
    longer than 384 tokens are chunked with overlap and processed in batches.

    Args:
        text: Input text to analyze.
        targets: List of entity labels to detect (default: ["PERSON", "LOCATION", "ORGANIZATION"]).
        threshold: Optional confidence threshold in [0.0, 1.0].

    Returns:
        List of GLiNER-origin `Evidence` records with high-confidence threshold.
    """
    evidence: List[Any] = []

    # Use provided targets or default to PERSON, LOCATION, and ORGANIZATION
    target_labels = targets if targets is not None else ["PERSON", "LOCATION", "ORGANIZATION"]
    if not target_labels:
        return evidence
    gliner_threshold = _resolve_gliner_threshold(threshold)

    try:
        model = get_gliner_model()
        gliner_tokenizer = getattr(getattr(model, "data_processor", None),
                                   "transformer_tokenizer", None)
        chunks = _chunk_text_for_gliner(text, tokenizer=gliner_tokenizer)
        if not chunks:
            return evidence

        chunk_predictions = _predict_gliner_chunks(
            model,
            chunks,
            target_labels,
            threshold=gliner_threshold,
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
                evidence.append(Evidence(
                    start=start,
                    end=end,
                    label=label,
                    source="gliner",
                    confidence=pred.get("score", 0.55),  # High-confidence GLiNER scores
                ))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("GLiNER detection failed: %s", e)

    return evidence
