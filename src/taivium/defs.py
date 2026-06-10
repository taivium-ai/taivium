'''Defines data structures used across the codebase.'''
from dataclasses import dataclass
from typing import Tuple
from enum import Enum

@dataclass(frozen=True)
class Evidence:
    """Represents detector evidence before span canonicalization."""
    start: int
    end: int
    label: str
    source: str
    confidence: float

# -----------------------------
# Label normalization
# -----------------------------
def normalize_label(label: str) -> str:
    """Normalizes internal NLP/Regex entity labels to match the
    canonical evaluation targets output by eval_datasets.py.
    """
    if not label:
        return "UNKNOWN"

    # Force uppercase to eliminate string-casing mismatches
    lookup = label.strip().upper()

    mapping = {
        # --- SpaCy Base NER Mappings ---
        "GPE": "LOCATION",         # SpaCy Geopolitical Entities -> LOCATION
        "LOC": "LOCATION",         # SpaCy Locations -> LOCATION
        "PERSON": "PERSON",
        "ORG": "ORG",
        "ORGANIZATION": "ORG",     # GLiNER returns ORGANIZATION -> canonical ORG

        # --- Your Internal Regex Engine Mappings ---
        "EMAIL_ADDRESS": "EMAIL",  # Ensures internal variations map to 'EMAIL'
        "PHONE_NUMBER": "PHONE",   # Maps to evaluation canonical 'PHONE'
        "TEL": "PHONE",            # Backwards compatibility if engine catches TEL

        # --- Expanding Regex Infrastructure (Unlocks the fallback metrics) ---
        "IP_ADDRESS": "IP",        # Maps your internal regex label -> canonical 'IP'
        "DATE_TIME": "DATE",       # Maps your internal dates/times -> canonical 'DATE'
        "TIME": "DATE",            # Matches eval_datasets.py conversion: TIME -> DATE
        "BOD": "DATE",             # Matches eval_datasets.py conversion: BOD -> DATE

        # --- Document Identifier Grouping ---
        "US_SSN": "SOCIALNUMBER",  # Maps internal SSN -> canonical 'SOCIALNUMBER'
        "PASSPORT": "SOCIALNUMBER",
        "IDCARD": "SOCIALNUMBER",
        "DRIVER_LICENSE": "SOCIALNUMBER",
        "DRIVERLICENSE": "SOCIALNUMBER",
    }

    # Return the mapped evaluation token if found; otherwise, pass the raw token back
    # This ensures explicit inputs like 'USERNAME' or 'EMAIL' flow through cleanly.
    return mapping.get(lookup, label)


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
# Evidence and Entity structures
# -----------------------------

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
