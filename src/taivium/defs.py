'''Defines data structures used across the codebase.'''
from dataclasses import dataclass

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
