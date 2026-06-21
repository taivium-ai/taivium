"""
LLM-assisted NER evidence collector.
--------------------------------------
Uses a local LLM model via llama.cpp (GGUF format) to extract named entities
from text.  The detector is opt-in: it only runs when ``use_llm=True`` is
passed to :class:`~taivium.engine.Taivium` and a ``LLM_MODEL_PATH`` environment
variable is set pointing to a GGUF model file.

Model configuration::

    LLM_MODEL_PATH=/path/to/model.gguf  # set in environment to use a local GGUF model
    LLM_N_GPU_LAYERS=35  # optional: number of layers to offload to GPU (default: 0)
"""
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import json
import os
import re
import warnings
from typing import List, Any
import logging
from .defs import Evidence, normalize_label

logger = logging.getLogger("taivium.llm")


# Only warn once per process about missing API key (no global)
class _WarnState:
    warned_no_api_key = False

    @classmethod
    def is_warned(cls):
        """Returns True if the API key warning has been issued."""
        return cls.warned_no_api_key

    @classmethod
    def reset_warning(cls):
        """Resets the API key warning state (for testing or re-initialization)."""
        cls.warned_no_api_key = False

_SYSTEM_PROMPT = """\
You are a precise named-entity recognizer for privacy protection.
Extract every sensitive entity from the user text.

Return ONLY valid JSON — a flat list of objects with exactly two keys:
  "text"  — the exact surface form as it appears in the input
  "type"  — one of: PERSON, ORG, LOCATION, EMAIL, PHONE, API_KEY

Rules:
- Preserve the exact casing and whitespace of the matched text.
- Do not merge separate mentions; list each distinct surface form once.
- If no entities are found, return [].
- Output no other text, markdown, or explanation.

Example output:
[{"text": "Alex Smith", "type": "PERSON"}, {"text": "user@example.com", "type": "EMAIL"}]
"""

# Labels the LLM is permitted to emit; anything else is skipped.
# Unlike spaCy/regex/transformer detectors (whose output labels are constrained
# by their trained taxonomy), the LLM returns free-form strings and can
# hallucinate types (e.g. DATE, PRODUCT) that the rest of the pipeline has no
# policy or canonicalization logic for.  This allowlist is the hard gate.
_VALID_LABELS = {"PERSON", "ORG", "LOCATION", "EMAIL", "PHONE", "API_KEY"}

# Helper to clean LLM JSON output
def _clean_llm_json(raw: str | None) -> str:
    raw = (raw or "[]").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()

    return raw

# Helper to extract evidence from entities
def _extract_evidence(
        entities: list[dict[str, Any]],
        text: str
    ) -> list[Evidence]:
    evidence = []
    seen = set()
    for item in entities:
        if not isinstance(item, dict):
            continue
        surface = item.get("text", "")
        label = item.get("type", "")
        if not surface.strip() or not label.strip():
            continue
        label = normalize_label(label)
        if label not in _VALID_LABELS:
            continue
        key = (surface, label)
        if key in seen:
            continue
        seen.add(key)

        for match in re.finditer(re.escape(surface), text):
            start = match.start()
            end = match.end()
            evidence.append(Evidence(
                start=start,
                end=end,
                label=label,
                source="llm",
                confidence=0.85,
            ))
    return evidence

def llm_evidence(text: str) -> List[Evidence]:
    """Collects NER evidence by querying a local LLM model via llama.cpp.

    Loads a GGUF model (specified in ``LLM_MODEL_PATH`` environment variable)
    and runs entity extraction with a structured prompt. Returns a JSON list
    of entity surface forms and types. Character offsets are recovered by
    scanning the source text for each returned surface form.

    Returns an empty list when:

    * the ``LLM_MODEL_PATH`` environment variable is not set

    Raises:
        Exception: If the model cannot be loaded, inference fails, or JSON
                   parsing fails. All errors are logged before raising.

    Args:
        text: The input text to run entity extraction over.

    Returns:
        A list of :class:`~taivium.engine.Evidence` records with
        ``source="llm"``.
    """
    model_path = os.getenv("LLM_MODEL_PATH")
    if not model_path:
        if not _WarnState.warned_no_api_key:
            warning_text = (
                "LLM evidence layer skipped: LLM_MODEL_PATH is not set. "
                "Set the environment variable to a GGUF model file path to enable LLM-assisted NER."
            )
            warnings.warn(
                warning_text,
                RuntimeWarning,
                stacklevel=2,
            )
            logger.warning(warning_text)
            _WarnState.warned_no_api_key = True
        return []

    try:
        from llama_cpp import Llama  # type: ignore[import]  # pylint: disable=import-outside-toplevel
        n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=2048,
            verbose=False,
        )
        prompt_text = f"{_SYSTEM_PROMPT}\n\nText to analyze:\n{text}\n\nJSON output:"
        response = llm(
            prompt_text,
            temperature=0,
            max_tokens=1024,
            stop=["\n\n"],
        )
        raw = response["choices"][0]["text"].strip()
        raw = _clean_llm_json(raw)
        entities = json.loads(raw)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("LLM evidence extraction failed: %s", exc, exc_info=True)
        raise

    if not isinstance(entities, list):
        return []

    return _extract_evidence(entities, text)
