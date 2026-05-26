"""
LLM-assisted NER evidence collector.
--------------------------------------
Uses the OpenAI chat completions API (``gpt-4o-mini`` by default) to extract
named entities from text.  The detector is opt-in: it only runs when
``use_llm=True`` is passed to :class:`~taivium.engine.Taivium` and an
``OPENAI_API_KEY`` environment variable is set.

Model override::

    PRIVACYZE_LLM_MODEL=gpt-4o  # set in environment to use a different model
"""
# pylint: disable=import-outside-toplevel
from __future__ import annotations

import json
import os
import re
import warnings
from typing import TYPE_CHECKING, List, Any
from collections.abc import Callable  # pylint: disable=import-error

# Logging system
import logging
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

if TYPE_CHECKING:
    from .engine import Evidence

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
[{"text": "Alice Smith", "type": "PERSON"}, {"text": "alice@example.com", "type": "EMAIL"}]
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
        text: str,
        normalize_label: Callable[[str], str],
        Evidence, # pylint: disable=invalid-name
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
    """Collects NER evidence by querying an OpenAI chat model.

    Sends *text* to ``gpt-4o-mini`` (or the model set in the
    ``PRIVACYZE_LLM_MODEL`` environment variable) with a structured prompt
    that requests a JSON list of entity surface forms and types.  Character
    offsets are recovered by scanning the source text for each returned
    surface form.

    Returns an empty list when:

    * the ``OPENAI_API_KEY`` environment variable is not set
    * the ``openai`` package is not installed
    * the API call fails for any reason

    Args:
        text: The input text to run entity extraction over.

    Returns:
        A list of :class:`~taivium.engine.Evidence` records with
        ``source="llm"``.
    """
    # Deferred import to avoid circular dependency with engine.py.
    from .engine import Evidence, normalize_label  # pylint: disable=import-outside-toplevel

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if not _WarnState.warned_no_api_key:
            warning_text = (
                "LLM evidence layer skipped: OPENAI_API_KEY is not set. "
                "Set the environment variable to enable LLM-assisted NER."
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
        from openai import OpenAI  # type: ignore[import]  # pylint: disable=import-outside-toplevel
        client = OpenAI(api_key=api_key)
        model = os.getenv("PRIVACYZE_LLM_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=1024,
        )
        raw = _clean_llm_json(response.choices[0].message.content)
        entities = json.loads(raw)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("LLM evidence extraction failed: %s", exc, exc_info=True)
        return []

    if not isinstance(entities, list):
        return []

    return _extract_evidence(entities, text, normalize_label, Evidence)
