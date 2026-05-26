"""
Semantic Identity Transformation Engine
----------------------------------------------------------------------
Deterministic, privacy-preserving NLP pipeline for evidence collection,
span canonicalization, identity resolution, and anonymization.
"""

# pylint: disable=invalid-name,too-few-public-methods,unused-argument,too-many-lines


import bisect
import hashlib
import logging
import re
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

import spacy
from .transformer import transformer_evidence
from .session_store import InMemorySessionStore, SessionStore
from .llm import llm_evidence

logger = logging.getLogger("tarvium.engine")

# -----------------------------
# Policy Action and Risk Level Enums
# -----------------------------


class PolicyAction(str, Enum):
    """Defines possible actions for detected entities based on policy evaluation."""
    ALLOW = "allow"
    ANONYMIZE = "anonymize"
    BLOCK = "block"


class RiskLevel(str, Enum):
    """Defines risk levels for detected entities based on policy evaluation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# -----------------------------
# Lazy-load spaCy model
# -----------------------------

@lru_cache(maxsize=1)
def get_spacy_model() -> Any:
    """Lazy-loads and returns the spaCy model with only the NER component enabled."""
    try:
        # Disable unused components (tagger, parser, lemmatizer) for faster
        # inference
        return spacy.load(
            "en_core_web_sm",
            disable=[
                "tagger",
                "parser",
                "lemmatizer",
                "attribute_ruler"])
    except OSError as exc:
        # Raise an error instead of falling back to a blank pipeline.
        error_text = "spaCy model 'en_core_web_sm' not found."
        error_text += " Please install it with 'python -m spacy download en_core_web_sm'."
        logger.error(error_text, exc_info=True)
        raise OSError(error_text) from exc

# -----------------------------
# Evidence and Entity structures
# -----------------------------


@dataclass(frozen=True)
class Evidence:
    """Represents detector evidence before span canonicalization."""
    start: int
    end: int
    label: str
    source: str
    confidence: float


@dataclass(frozen=True)
class SpanCandidate:
    """Represents one span hypothesis for canonicalization optimization.

    Candidates are merged only by exact `(start, end, label)` equivalence so
    different span boundaries remain distinct competing hypotheses.
    """
    start: int
    end: int
    label: str
    score: float
    evidence: Tuple[Evidence, ...]


@dataclass(frozen=True)
class Entity:
    """Represents an entity span with retained provenance (immutable).

    Attributes:
        text: Surface form in the input text.
        label: Normalized entity type.
        start: Inclusive start offset in the input text.
        end: Exclusive end offset in the input text.
        source: Primary source tag for this entity instance.
        evidence_sources: Ordered detector/source lineage contributing to the
            entity decision (for canonical entities this can include multiple
            detectors; for direct detections this is typically one source).
        confidence: Confidence score retained with the entity. For canonical
            entities this represents normalized vote support within the overlap
            cluster. For direct detections it is the detector confidence.
    """
    text: str
    label: str
    start: int
    end: int
    source: str
    evidence_sources: Tuple[str, ...] = ()
    confidence: float = 0.0


# -----------------------------
# Regex detectors (PII / secrets)
# -----------------------------
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\+?\d[\d\s\-]{7,}\d")
API_KEY_REGEX = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|api[_-]?key\s*[:=]\s*[a-zA-Z0-9]+)", re.I)


# -----------------------------
# Label normalization
# -----------------------------
def normalize_label(label: str) -> str:
    """Normalizes entity labels to a consistent set
        (e.g., GPE and LOC → LOCATION)."""
    mapping = {
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "PERSON": "PERSON",
        "ORG": "ORG",
    }
    return mapping.get(label, label)


# -----------------------------
# Evidence detectors
# -----------------------------

def spacy_evidence(text: str) -> List[Evidence]:
    """Collects NER evidence from spaCy."""
    nlp = get_spacy_model()
    doc = nlp(text)
    evidence: List[Evidence] = []

    for ent in doc.ents:
        label = normalize_label(ent.label_)
        evidence.append(Evidence(
            start=ent.start_char,
            end=ent.end_char,
            label=label,
            source="spacy",
            confidence=0.75,
            ))

    return evidence

def regex_evidence(text: str) -> List[Evidence]:
    """
    Collects high-confidence evidence from regex-based PII/secret patterns.
    """
    evidence: List[Evidence] = []

    for m in EMAIL_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "EMAIL", "regex", 0.90))

    for m in PHONE_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "PHONE", "regex", 0.80))

    for m in API_KEY_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "API_KEY", "regex", 0.95))

    return evidence


# -----------------------------
# Evidence merge and canonicalization
# -----------------------------
SOURCE_WEIGHT: Dict[str, float] = {
    "spacy": 0.7,
    "regex": 1.0,
    "transformer": 0.8,
    "llm": 0.6,
    "recurrence": 0.5,  # Lower than detectors — fills gaps, doesn't override
}

# Labels safe for strict lexical recurrence by default.
RECURRENCE_ALLOWED = {
    "EMAIL",
    "PHONE",
    "API_KEY",
}


def collect_evidence(
    text: str,
    *,
    use_transformer: bool = False,
    use_llm: bool = False,
    transformer_fn: Optional[Callable[[str], List[Evidence]]] = None,
    llm_fn: Optional[Callable[[str], List[Evidence]]] = None,
) -> List[Evidence]:
    """Collects raw evidence from spaCy, regex, and optionally transformer/LLM layers.

    Args:
        text: Input text to run detectors over.
        use_transformer: Master switch for the transformer detector layer. Must be
            ``True`` for the layer to run. When ``True`` and no *transformer_fn* is
            provided, uses the built-in BERT NER detector (requires
            ``transformers`` + ``torch``). Disabled by default.
        use_llm: Master switch for the LLM detector layer. Must be ``True`` for
            the layer to run. When ``True`` and no *llm_fn* is provided, uses the
            built-in OpenAI detector (requires ``OPENAI_API_KEY``). Disabled by
            default.
        transformer_fn: Custom transformer detector callable. Replaces the
            built-in transformer when *use_transformer* is ``True``. Has no effect
            when *use_transformer* is ``False``.
        llm_fn: Custom LLM detector callable. Replaces the built-in LLM layer
            when *use_llm* is ``True``. Has no effect when *use_llm* is ``False``.
    """
    evidence = spacy_evidence(text) + regex_evidence(text)
    if use_transformer:
        evidence += (transformer_fn or transformer_evidence)(text)
    if use_llm:
        evidence += (llm_fn or llm_evidence)(text)
    return evidence


# -----------------------------
# Span Integrity Utilities
# -----------------------------
def assert_non_overlapping(entities: List[Entity]) -> None:
    """
    Raises AssertionError when entity-set invariants are violated:
    - each entity must satisfy start < end
    - entities must be sorted by ascending start offset
    - entities must be strictly non-overlapping

    This is a hard invariant for canonical, recurrence, and transform layers.
    """
    for i, curr in enumerate(entities):
        if curr.start >= curr.end:
            raise AssertionError(
                f"Invalid entity span: '{curr.text}' [{curr.start}:{curr.end}]"
            )
        if i == 0:
            continue

        prev = entities[i - 1]
        if curr.start < prev.start:
            raise AssertionError(
                f"Entities not sorted by start: '{prev.text}' [{prev.start}:{prev.end}] before "
                f"'{curr.text}' [{curr.start}:{curr.end}]"
            )
        if curr.start < prev.end:
            raise AssertionError(
                f"Overlapping entities: '{prev.text}' [{prev.start}:{prev.end}] and "
                f"'{curr.text}' [{curr.start}:{curr.end}]"
            )


def assert_text_span_integrity(text: str, entities: List[Entity]) -> None:
    """Raises AssertionError if any entity text does not match its source span."""
    for entity in entities:
        expected = text[entity.start:entity.end]
        if entity.text != expected:
            raise AssertionError(
                f"Text-span mismatch for '{entity.text}' [{entity.start}:{entity.end}]: "
                f"expected '{expected}'"
            )

def canonicalize_spans(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        text: str, evidence: List[Evidence]) -> List[Entity]:
    """
    Converts noisy overlapping evidence into a globally optimized canonical set.

    Architecture:
      1. Build explicit span candidates from evidence by exact-equivalence merge
         on `(start, end, label)` only.
      2. Score each candidate deterministically from detector confidence, source
         reliability, span-shape bonus, and label prior.
      3. Solve a weighted interval scheduling optimization to select the best
         non-overlapping candidate set globally.

    This avoids transitive-overlap collapse from connected clustering and keeps
    competing span hypotheses explicit until optimization.
    """
    valid: List[Evidence] = []
    for item in evidence:
        if not 0 <= item.start < item.end <= len(text):
            continue
        normalized = normalize_label(item.label)
        if normalized == "UNKNOWN":
            continue
        valid.append(
            Evidence(
                start=item.start,
                end=item.end,
                label=normalized,
                source=item.source,
                confidence=item.confidence,
            )
        )

    if not valid:
        return []

    # Step 1: merge only exact-equivalent hypotheses.
    grouped: Dict[Tuple[int, int, str], List[Evidence]] = defaultdict(list)
    for item in valid:
        grouped[(item.start, item.end, item.label)].append(item)

    label_prior: Dict[str, float] = {
        "PERSON": 0.05,
        "ORG": 0.05,
        "LOCATION": 0.05,
        "EMAIL": 0.1,
        "PHONE": 0.1,
        "API_KEY": 0.15,
    }

    def _score_candidate(candidate_evidence: Tuple[Evidence, ...], label: str) -> float:
        """
        Deterministic scoring for one exact span+label hypothesis.

        This function assigns a score to each span candidate during canonicalization.
        The score is used to select the globally optimal, non-overlapping set of entity spans.

        Scoring components:
        - Sums the confidence values from all supporting evidence sources.
        - Adds up reliability weights for each evidence source (from SOURCE_WEIGHT).
                - Adds a capped linear bonus based on span length
                    (`min(0.15 * span_len, 2.0)`) to discourage fragmentation while
                    avoiding over-preference for very long spans.
        - Adds a label prior (from label_prior) to favor certain entity types if needed.

        The total score is the sum of these components. Higher scores mean the
        candidate is more likely to be selected.

        Purpose:
        - Ensures deterministic, transparent, and tunable scoring for canonicalization.
                - Discourages over-fragmentation without letting span length dominate
                    detector evidence.
        - Makes the canonicalization process robust and auditable.
        """
        confidence_sum = sum(item.confidence for item in candidate_evidence)
        source_reliability = sum(
            SOURCE_WEIGHT.get(item.source, 0.5)
            for item in candidate_evidence
        )
        span_len = candidate_evidence[0].end - candidate_evidence[0].start
        # Capped linear bonus reduces fragmentation without over-biasing long spans.
        span_length_bonus = min(0.15 * span_len, 2.0)
        return (
            confidence_sum
            + source_reliability
            + span_length_bonus
            + label_prior.get(label, 0.0)
        )

    candidates: List[SpanCandidate] = []
    for (start, end, label), items in grouped.items():
        merged_evidence = tuple(sorted(items, key=lambda ev: (ev.source, ev.confidence)))
        candidates.append(
            SpanCandidate(
                start=start,
                end=end,
                label=label,
                score=_score_candidate(merged_evidence, label),
                evidence=merged_evidence,
            )
        )

    if not candidates:
        return []

    # Step 2: global non-overlap optimization (weighted interval scheduling).
    candidates.sort(key=lambda c: (c.end, c.start, c.label))
    ends = [candidate.end for candidate in candidates]

    prev_non_overlap: List[int] = []
    for candidate in candidates:
        idx = bisect.bisect_right(ends, candidate.start) - 1
        prev_non_overlap.append(idx)

    best_score: List[float] = [0.0] * len(candidates)
    best_coverage: List[int] = [0] * len(candidates)
    best_count: List[int] = [0] * len(candidates)
    # Stores deterministic terminal tie-break signature for the selected
    # solution up to each index: (-start, -end, label).
    best_terminal: List[Tuple[int, int, str]] = [(0, 0, "")] * len(candidates)
    take: List[bool] = [False] * len(candidates)

    for i, candidate in enumerate(candidates):
        p_idx = prev_non_overlap[i]
        include_score = candidate.score + (best_score[p_idx] if p_idx >= 0 else 0.0)
        include_cov = (candidate.end - candidate.start) + \
                        (best_coverage[p_idx] if p_idx >= 0 else 0)
        include_count = 1 + (best_count[p_idx] if p_idx >= 0 else 0)

        exclude_score = best_score[i - 1] if i > 0 else 0.0
        exclude_cov = best_coverage[i - 1] if i > 0 else 0
        exclude_count = best_count[i - 1] if i > 0 else 0

        include_key = (
            include_score,
            include_cov,
            -include_count,
            -candidate.start,
            -candidate.end,
            candidate.label,
        )
        exclude_terminal = best_terminal[i - 1] if i > 0 else (0, 0, "")
        exclude_key = (
            exclude_score,
            exclude_cov,
            -exclude_count,
            exclude_terminal[0],
            exclude_terminal[1],
            exclude_terminal[2],
        )

        if include_key > exclude_key:
            take[i] = True
            best_score[i] = include_score
            best_coverage[i] = include_cov
            best_count[i] = include_count
            best_terminal[i] = (-candidate.start, -candidate.end, candidate.label)
        else:
            best_score[i] = exclude_score
            best_coverage[i] = exclude_cov
            best_count[i] = exclude_count
            best_terminal[i] = exclude_terminal

    chosen: List[SpanCandidate] = []
    i = len(candidates) - 1
    while i >= 0:
        if take[i]:
            chosen.append(candidates[i])
            i = prev_non_overlap[i]
        else:
            i -= 1

    chosen.sort(key=lambda c: (c.start, c.end, c.label))

    canonical: List[Entity] = []
    for candidate in chosen:
        evidence_sources = tuple(sorted({item.source for item in candidate.evidence}))
        avg_confidence = (
            sum(item.confidence for item in candidate.evidence) / len(candidate.evidence)
        )

        canonical.append(
            Entity(
                text=text[candidate.start:candidate.end],
                label=candidate.label,
                start=candidate.start,
                end=candidate.end,
                source="canonical",
                evidence_sources=evidence_sources,
                confidence=avg_confidence,
            )
        )

    # Optimized selection is non-overlapping by construction; sort for stable output.
    canonical.sort(key=lambda entity: entity.start)
    assert_non_overlapping(canonical)
    assert_text_span_integrity(text, canonical)
    return canonical


def _is_recurrence_eligible(entity: Entity) -> bool:  # pylint: disable=too-many-return-statements
    """Returns True when lexical recurrence expansion is safe for this entity."""
    normalized_text = " ".join(entity.text.split())
    if not normalized_text:
        return False

    if entity.label in RECURRENCE_ALLOWED:
        return True

    if entity.label == "PERSON":
        # Single-token person names are often ambiguous (name/month/verb/etc.).
        return len(normalized_text.split()) >= 2

    if entity.label == "ORG":
        # Permit only reasonably specific org mentions; avoid short acronyms.
        if len(normalized_text) < 5:
            return False
        if normalized_text.isupper() and len(normalized_text) <= 5:
            return False
        return True

    # LOCATION and other labels are intentionally excluded unless explicitly allowed.
    return False


def recurrence_evidence(  # pylint: disable=too-many-locals
        text: str, canonical: List[Entity],
        max_recurrences_per_entity: int = 1000,
        min_recurrence_span_len: int = 3,
) -> List[Evidence]:
    """
    Generates Evidence records for eligible recurrences of canonical entity surface forms.

    This function implements the semantic recurrence layer of the privacy pipeline.
    After canonicalization, it scans the input text for repeated, non-overlapping
    surface-form matches of canonical entities that may have been missed by NER detectors.
    Recurrence evidence is used to increase recall for repeated sensitive entities
    (e.g., multiple mentions of the same email or name) while maintaining strict safety
    and auditability guarantees.

        Key properties:
        - Only entities that pass ``_is_recurrence_eligible`` are considered
            (default: EMAIL, PHONE, API_KEY; PERSON and ORG are gated by heuristics).
        - For each eligible canonical entity, scans for exact substring matches in the text,
            with strict boundary checks (word/non-word/whitespace) to avoid overmatching.
        - Skips spans already covered by canonical entities; ensures no overlap with 
            canonical spans.
        - Emits new Evidence records with ``source="recurrence"`` for each safe,
            non-overlapping recurrence found.
        - Recurrence evidence is merged with detector evidence and re-canonicalized,
            so all spans are globally optimized together in the second canonicalization pass.
        - All recurrence logic is deterministic, non-overlapping, and idempotent.

    Privacy rationale:
        Recurrence evidence increases recall for repeated sensitive entities while ensuring
        that no new labels, widened spans, or invented entities are introduced.
        All recurrence entities inherit provenance and confidence from their canonical source,
        and are fully auditable.

    Token-boundary rules:
        * Word-character surfaces require non-word neighbours (or string edges).
        * Non-word-character surfaces require whitespace neighbours (or string edges).

    Args:
        text:      The original input text.
        canonical: Non-overlapping canonical entities from the first
                   ``canonicalize_spans`` pass (detector evidence only).
        max_recurrences_per_entity: Safety cap — at most this many Evidence
            records are emitted per canonical entity.
        min_recurrence_span_len: Spans shorter than this are skipped.

    Returns:
        A (possibly empty) list of :class:`Evidence` records with
        ``source="recurrence"``.
    """
    if not canonical:
        return []

    covered: List[Tuple[int, int]] = sorted((e.start, e.end) for e in canonical)
    result: List[Evidence] = []

    def _overlaps(m_start: int, m_end: int) -> bool:
        i = bisect.bisect_left(covered, (m_start, m_end))
        if i > 0:
            _, c_end = covered[i - 1]
            if c_end > m_start:
                return True
        if i < len(covered):
            c_start, c_end = covered[i]
            if c_start < m_end and c_end > m_start:
                return True
        return False

    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == '_'

    def _left_boundary_ok(pos: int, starts_with_word: bool) -> bool:
        if pos == 0:
            return True
        left = text[pos - 1]
        return (not _is_word_char(left)) if starts_with_word else left.isspace()

    def _right_boundary_ok(pos: int, ends_with_word: bool) -> bool:
        if pos == len(text):
            return True
        right = text[pos]
        return (not _is_word_char(right)) if ends_with_word else right.isspace()

    for entity in canonical:
        if not _is_recurrence_eligible(entity):
            continue
        surface = entity.text
        if not surface or len(surface) < min_recurrence_span_len:
            continue
        starts_with_word = _is_word_char(surface[0])
        ends_with_word = _is_word_char(surface[-1])
        found = 0
        for match in re.finditer(re.escape(surface), text):
            if found >= max_recurrences_per_entity:
                logger.warning(
                    "Recurrence cap hit for entity (label=%s)", entity.label
                )
                break
            s, e = match.start(), match.end()
            if not _left_boundary_ok(s, starts_with_word):
                continue
            if not _right_boundary_ok(e, ends_with_word):
                continue
            if _overlaps(s, e):
                continue
            bisect.insort(covered, (s, e))
            result.append(Evidence(
                start=s,
                end=e,
                label=entity.label,
                source="recurrence",
                confidence=entity.confidence,
            ))
            found += 1
    return result


def find_recurrences(
        text: str, canonical: List[Entity],
        max_recurrences_per_entity: int = 1000,
        min_recurrence_span_len: int = 3,
) -> List[Entity]:
    """Compatibility shim — wraps :func:`recurrence_evidence` and returns Entity objects.

    Callers that previously consumed the recurrence layer directly (e.g. tests)
    continue to work unchanged.  In the main pipeline, ``process()`` calls
    ``recurrence_evidence`` and feeds the results back into ``canonicalize_spans``
    so that all spans are globally optimized together.
    """
    evs = recurrence_evidence(text, canonical, max_recurrences_per_entity, min_recurrence_span_len)
    if not evs:
        return []
    # Build a lookup so each recurrence inherits provenance from its canonical source.
    canon_by_key: Dict[Tuple[str, str], Entity] = {
        (e.label, e.text): e for e in canonical
    }
    entities: List[Entity] = []
    for ev in evs:
        span_text = text[ev.start:ev.end]
        canon = canon_by_key.get((ev.label, span_text))
        entities.append(Entity(
            text=span_text,
            label=ev.label,
            start=ev.start,
            end=ev.end,
            source="recurrence",
            evidence_sources=canon.evidence_sources if canon else (),
            confidence=canon.confidence if canon else ev.confidence,
        ))
    entities.sort(key=lambda e: e.start)
    return entities

def _text_span_integrity(text: str, entities: List[Entity]) -> bool:
    """Returns True if all entity.text matches text[entity.start:entity.end]."""
    for e in entities:
        if e.text != text[e.start:e.end]:
            return False
    return True

# -----------------------------
# Identity Engine (deterministic)
# -----------------------------
class IdentityEngine:
    """
    Deterministic entity → ID mapping system.

    Maps entities to unique, deterministic IDs based solely on their semantic
    identity (text + label). The same entity text and label always produce the
    same ID, regardless of where in the text the entity appears or how many
    times it is detected.

    Methods:
        generate_id(text: str, label: str) -> str:
            Generates a deterministic ID for a given entity text and label.

        resolve(entities: List[Entity]) -> List[Tuple[Entity, str]]:
            Deduplicates entities by (text, label) and returns one
            (Entity, id) tuple per unique semantic identity.
    """

    @staticmethod
    def normalize_identity_text(text: str) -> str:
        """Normalizes text for semantic identity hashing.

        Normalization steps:
        1. Unicode normalization (NFKC).
        2. Case folding for robust case-insensitive matching.
        3. Collapse internal whitespace runs to a single space.
        4. Strip punctuation only at text edges.
        """
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.casefold()
        normalized = " ".join(normalized.split())

        def _is_edge_punctuation(ch: str) -> bool:
            return unicodedata.category(ch).startswith("P")

        start = 0
        end = len(normalized)

        while start < end and _is_edge_punctuation(normalized[start]):
            start += 1
        while end > start and _is_edge_punctuation(normalized[end - 1]):
            end -= 1

        return normalized[start:end]

    def __init__(self, salt: Optional[str] = None, hash_len: int = 12):
        """
        Optionally provide a salt to scope IDs to a tenant, session, or namespace.
        If no salt is provided, IDs are globally stable (legacy behavior).
        hash_len: Number of hex digits to use from the hash (default 12 for legacy compatibility).
        """
        self.salt = salt or ""
        self.hash_len = hash_len

    def generate_id(self, text: str, label: str) -> str:
        """Generates a deterministic ID for a given entity text and label, 
            optionally scoped by salt.

        The same (text, label, salt, hash_len) tuple always produces the same ID.
        If salt is not set, IDs are globally stable (legacy behavior).
        hash_len controls the number of hex digits in the ID.
        """
        normalized = self.normalize_identity_text(text)
        key = f"{self.salt}:{label}:{normalized}"
        hash_id = hashlib.sha256(key.encode()).hexdigest()[:self.hash_len]
        return f"{label}_{hash_id}"

    def resolve(self, entities: List[Entity]) -> List[Tuple[Entity, str]]:
        """
        Resolves a list of entities to their deterministic IDs.
        All positional occurrences are retained so that every span in the
        document is replaced during transformation.  Entities sharing the
        same semantic identity (text, label) receive the same deterministic
        ID regardless of position.
        Returns a list of (Entity, id) tuples, one per input entity.
        """
        return [(e, self.generate_id(e.text, e.label)) for e in entities]


# -----------------------------
# Anonymization Engine
# -----------------------------
def transform(text: str, resolved_entities: List[Tuple[Entity, str]]) -> str:
    """Transforms the input text by replacing detected entities with their corresponding
        anonymized IDs. The function takes the original text and a list of tuples containing
        Entity objects and their assigned IDs. It returns the transformed text with all
        specified entities replaced by their anonymized placeholders.
    """
    sorted_entities = sorted(resolved_entities, key=lambda x: x[0].start)
    try:
        entities = [e for e, _ in sorted_entities]
        assert_non_overlapping(entities)
        assert_text_span_integrity(text, entities)
    except AssertionError as exc:
        raise ValueError(f"Overlapping spans in transform(): {exc}") from exc
    output: List[str] = []
    last_idx = 0
    for ent, eid in sorted_entities:
        output.append(text[last_idx:ent.start])
        output.append(eid)
        last_idx = ent.end
    output.append(text[last_idx:])
    return "".join(output)


def reverse_transform(text: str, mapping: Dict[str, Dict[str, Any]]) -> str:
    """Reverses anonymization by replacing every entity ID token in *text*
    with the original value from *mapping*.

    *mapping* is the ``"mapping"`` dict returned by ``Tarvium.process()``:
    ``{eid: {"text": <original>, "label": ..., ...}}``.

    Replacement is applied longest-token-first to avoid partial matches
    when one token is a prefix of another (unlikely given SHA-256 IDs, but safe).
    """
    result = text
    for eid in sorted(mapping, key=len, reverse=True):
        result = result.replace(eid, mapping[eid]["text"])
    return result


# -----------------------------
# Policy Engine
# -----------------------------

@dataclass
class PolicyRule:
    """Defines a policy rule for a specific entity label, including the
        action to take and the associated risk level."""
    label: str
    action: PolicyAction
    risk: RiskLevel


DEFAULT_POLICY: Dict[str, PolicyRule] = {
    "PERSON": PolicyRule("PERSON", PolicyAction.ANONYMIZE, RiskLevel.MEDIUM),
    "ORG": PolicyRule("ORG", PolicyAction.ANONYMIZE, RiskLevel.MEDIUM),
    "LOCATION": PolicyRule("LOCATION", PolicyAction.ANONYMIZE, RiskLevel.LOW),
    "EMAIL": PolicyRule("EMAIL", PolicyAction.ANONYMIZE, RiskLevel.HIGH),
    "PHONE": PolicyRule("PHONE", PolicyAction.ANONYMIZE, RiskLevel.HIGH),
    "API_KEY": PolicyRule("API_KEY", PolicyAction.ANONYMIZE, RiskLevel.CRITICAL),
}


DEFAULT_UNDEFINED_POLICY_RISK = RiskLevel.UNKNOWN


class PolicyDecisionReason(str, Enum):
    """Enumerates reasons for a policy decision (explicit rule or fallback)."""
    EXPLICIT = "explicit_rule"
    FALLBACK = "fallback_rule"


@dataclass(frozen=True)
class PolicyDecision:
    """Represents the decision made by the PolicyEngine for a specific entity.

    Includes the entity's label, the action to take, the associated risk level,
    and the reason for the decision.
    """
    label: str
    action: PolicyAction
    risk: RiskLevel
    reason: PolicyDecisionReason


@dataclass(frozen=True)
class PolicyContext:
    """Optional context payload for future policy decisions.

    The current PolicyEngine implementation remains label-only, but this
    structure is threaded through evaluation so future policies can use
    additional signals (context, confidence, detector source, etc.) without
    changing the public call shape.
    """
    text: str
    confidence: float
    source: str
    evidence_sources: Tuple[str, ...] = ()
    metadata: Optional[Dict[str, Any]] = None


class PolicyEngine:
    """PolicyEngine determines the action to take for each detected
    entity based on its label.

    Args:
        policy_table: Optional mapping of entity label to :class:`PolicyRule`.
            Defaults to :data:`DEFAULT_POLICY`.
        default_action: Action applied to labels not present in ``policy_table``.
            Defaults to ``PolicyAction.ANONYMIZE`` (strict — unknown labels are
            anonymized). Pass ``PolicyAction.ALLOW`` for permissive mode where
            unknown labels are passed through unchanged.
    """

    def __init__(
        self,
        policy_table: Optional[Dict[str, PolicyRule]] = None,
        default_action: PolicyAction = PolicyAction.ANONYMIZE,
    ):
        self.policy_table: Dict[str, PolicyRule] = (
            policy_table if policy_table is not None else DEFAULT_POLICY
        )
        self._default_action = default_action

    def _make_fallback_rule(self, label: str) -> PolicyRule:
        """Returns a fallback PolicyRule for labels not in the policy table."""
        return PolicyRule(label, self._default_action, DEFAULT_UNDEFINED_POLICY_RISK)

    def _evaluate_label_only(self, entity: Entity) -> PolicyDecision:
        """Evaluates a policy decision from label-only rules."""
        if entity.label in self.policy_table:
            rule = self.policy_table[entity.label]
            reason = PolicyDecisionReason.EXPLICIT
        else:
            rule = self._make_fallback_rule(entity.label)
            reason = PolicyDecisionReason.FALLBACK
        return PolicyDecision(
            label=entity.label,
            action=rule.action,
            risk=rule.risk,
            reason=reason
        )

    def evaluate(
        self,
        entity: Entity,
        context: Optional[PolicyContext] = None,
    ) -> PolicyDecision:
        """Evaluates policy for an entity with optional context signals.

        Current behavior is label-only. The optional *context* argument enables
        forward-compatible policy evolution without breaking callers. Subclasses
        can override this method to add context-aware policy decisions.
        """
        decision = self._evaluate_label_only(entity)
        if context is not None:
            logger.info(
                "Policy decision: label=%s, action=%s, risk=%s, reason=%s, text=%.40r",
                decision.label, decision.action, decision.risk, decision.reason, context.text[:40]
            )
        return decision


class Tarvium:  # pylint: disable=too-many-instance-attributes
    """
    Tarvium orchestrates the end-to-end semantic identity transformation process.
    It detects entities in text, normalizes and resolves overlaps, assigns deterministic IDs,
    and transforms the text by replacing sensitive entities with anonymized placeholders.

    Args:
        policy_engine (PolicyEngine, optional):
            A policy engine instance to determine actions for each entity.
            If None, uses the default policy engine.
        session_store (SessionStore, optional):
            Pluggable session store for persisting entity-ID → metadata mappings
            across pipeline calls.  Defaults to ``InMemorySessionStore`` (in-process
            only).  Any object satisfying the :class:`~tarvium.session_store.SessionStore`
            protocol is accepted (``RedisSessionStore``, custom backends, etc.).
        id_salt (str, optional):
            Optional salt to scope entity IDs to a tenant, session, or namespace. 
            If not provided, IDs are globally stable (legacy behavior).
        id_hash_len (int, optional):
            Number of hex digits to use from the hash (default 12 for legacy compatibility).

    Usage Examples:
        # Default (global, legacy-stable IDs)
        engine = Tarvium()

        # Tenant-scoped IDs (prevents cross-tenant linkage)
        engine = Tarvium(id_salt="tenant_1234")

        # Session-scoped IDs (prevents cross-session linkage)
        engine = Tarvium(id_salt="session_5678")

        # Custom hash length (longer IDs)
        engine = Tarvium(id_hash_len=24)

        # Both salt and custom hash length
        engine = Tarvium(id_salt="tenant_1234", id_hash_len=24)
    """

    def __init__(
        self,
        policy_engine: Optional[PolicyEngine] = None,
        session_store: Optional[SessionStore] = None,
        use_transformer: bool = False,
        use_llm: bool = False,
        transformer_fn: Optional[Callable[[str], List[Evidence]]] = None,
        llm_fn: Optional[Callable[[str], List[Evidence]]] = None,
        id_salt: Optional[str] = None,
        id_hash_len: int = 12,
    ):  # pylint: disable=too-many-arguments
        """
        id_salt: Optional string to scope entity IDs (tenant/session/namespace).
        id_hash_len: Number of hex digits to use from the hash (default 12 for 
        legacy compatibility).
        If not provided, IDs are globally stable (legacy behavior).
        """
        self.identity = IdentityEngine(salt=id_salt, hash_len=id_hash_len)
        self.policy = policy_engine or PolicyEngine()
        self.session_store = (
            session_store if session_store is not None else InMemorySessionStore()
        )
        self.use_transformer = use_transformer
        self.use_llm = use_llm
        self.transformer_fn = transformer_fn
        self.llm_fn = llm_fn
        self.latency_history: List[float] = []  # Stores recent processing latencies in milliseconds

    def process(self, text: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals
        """
        Process text through the privacy pipeline.

        Pipeline:
            Detectors (spaCy/regex/LLM/transformer)
            -> Evidence
            -> Canonical span resolver
            -> Identity resolver
            -> Policy engine
            -> Transform

        Detection is uncertain.
        Canonicalization defines truth.
        Identity is separate from spans.    
        
        Args:
            text (str): The input text to be processed.

        Returns:
            dict: A dictionary containing the original text, anonymized text,
                  entity mapping, and detailed entity information.
        """
        start = time.perf_counter()

        logger.info("Processing text: %.60r", text[:60])

        # Step 1: collect raw detector evidence.
        evidence = collect_evidence(
            text,
            use_transformer=self.use_transformer,
            use_llm=self.use_llm,
            transformer_fn=self.transformer_fn,
            llm_fn=self.llm_fn,
        )  # pylint: disable=line-too-long

        logger.info("Collected evidence: %d items", len(evidence))

        # Step 2: canonicalize to one entity per span.
        # Pass 1 — detector evidence only, to discover canonical surface forms.
        initial_canonical = canonicalize_spans(text, evidence)
        logger.info("Initial canonicalized entities: %d", len(initial_canonical))

        # Step 2b: generate recurrence Evidence from initial canonical entities
        # and re-run canonicalization so recurrence spans compete equally with
        # detector spans in the global interval-scheduling optimizer.
        recurrence_evs = recurrence_evidence(text, initial_canonical)
        if recurrence_evs:
            logger.info("Recurrence evidence generated: %d", len(recurrence_evs))
        all_evidence = evidence + recurrence_evs
        all_ents = canonicalize_spans(text, all_evidence)

        logger.info("Canonicalized entities (with recurrence): %d", len(all_ents))

        # Hard invariant before identity/policy/transform stages.
        assert_non_overlapping(all_ents)
        assert_text_span_integrity(text, all_ents)

        # Step 3: identity resolution
        resolved = self.identity.resolve(all_ents)

        logger.info("Resolved identities: %d", len(resolved))

        # Step 4: policy evaluation for each entity
        results: List[Tuple[Entity, str, PolicyDecision]] = []
        for e, eid in resolved:
            policy_decision = self.policy.evaluate(
                e,
                PolicyContext(
                    text="[REDACTED]",  # Do not log entity text
                    confidence=e.confidence,
                    source=e.source,
                    evidence_sources=e.evidence_sources,
                ),
            )
            if policy_decision.action == PolicyAction.BLOCK:
                logger.error("Blocked sensitive entity: label=%s, id=%s", e.label, eid)
                raise ValueError(
                    f"Blocked sensitive entity: {e.label} (id={eid})")
            if policy_decision.action == PolicyAction.ANONYMIZE:
                results.append((e, eid, policy_decision))
            elif policy_decision.action == PolicyAction.ALLOW:
                pass  # Do nothing
            else:  # this should never happen if policy engine is implemented correctly
                logger.error("Unknown policy action: %r for entity label=%s, id=%s",
                             policy_decision.action, e.label, eid)
                raise ValueError(
                    f"Unknown policy action: {policy_decision.action} "
                    f"for entity label={e.label} (id={eid})")

        anonymized_text = transform(text, [(e, eid) for e, eid, _ in results])

        logger.info("Anonymized text generated.")

        # structured id -> metadata mapping (with source, risk, action)
        mapping: Dict[str, Dict[str, Any]] = {}
        for e, eid, policy_decision in results:
            mapping[eid] = {
                "text": e.text,
                "label": e.label,
                "source": e.source,
                "evidence_sources": e.evidence_sources,
                "confidence": e.confidence,
                "risk": policy_decision.risk,
                "action": policy_decision.action,
                "reason": policy_decision.reason
            }

        # Persist new mapping entries to the session store.
        self.session_store.set_many(mapping)

        logger.info("Session mapping updated: %d entities", len(mapping))

        latency_ms = (time.perf_counter() - start) * 1000
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        logger.info("Processing latency: %.2f ms", latency_ms)
        return {
            "original": text,
            "anonymized": anonymized_text,
            "store_type": type(self.session_store).__name__,
            "mapping": mapping,
            "entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "id": eid,
                    "start": e.start,
                    "end": e.end,
                    "source": e.source,
                    "evidence_sources": e.evidence_sources,
                    "confidence": e.confidence,
                }
                for e, eid, _ in results
            ]
        }
