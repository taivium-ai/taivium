'''Module for loading and caching evaluation datasets, 
    converting them to a standardized format, and preparing ground 
    truth spans for performance evaluation.'''
import ast
import logging
from datasets import load_dataset #huggingface
import pickle
from pathlib import Path
from typing import Any
from .utility import conll_to_gold_spans
from .utility import cache_file_from_payload

logger = logging.getLogger(__name__)

# Cache dataset locally for fast reuse
CACHE_DIR = Path(__file__).parent / ".cache"

DATASET_LIST = {"tomaarsen/conll2003": {"source": "huggingface"},
                "ai4privacy/pii-masking-300k": {"source": "huggingface"},
                "beki/en_pvt": {"source": "huggingface"},
                }

LABEL_PROFILES = {
    "conll_ner": {
        "PERSON",
        "ORG",
        "LOCATION",
    },
    "privacy": {
        "PERSON",
        "ORG",
        "LOCATION",
        "EMAIL",
        "PHONE",
        "API_KEY",
        "DATE",
        "IP",
        "SOCIALNUMBER",
        "USERNAME"
    },
    "privacy_no_org": {
        "PERSON",
        "LOCATION",
        "EMAIL",
        "PHONE",
        "API_KEY",
        "DATE",
        "IP",
        "SOCIALNUMBER",
        "USERNAME"
    },
}

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": "UNKNOWN",
}

# Explicit mapping for non-CONLL privacy-style labels.
# Keep this conservative: only map labels with clear semantic equivalence.
PRIVACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "NAME": "PERSON",
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "BUILDING": "LOCATION",
    "POSTCODE": "LOCATION",
    "SECADDRESS": "LOCATION",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TEL": "PHONE",
    "MOBILE": "PHONE",
    "MOBILE_PHONE_NUMBER": "PHONE",
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "ACCESS_TOKEN": "API_KEY",
    "DATE": "DATE",
    "IP": "IP",
    "SOCIALNUMBER": "SOCIALNUMBER",
    "USERNAME": "USERNAME",
    "BOD": "DATE",
    "TIME": "DATE",
    "GEOCOORD": "LOCATION",
    "TITLE": "PERSON",
    "SEX": "PERSON",
    # Document identifiers grouped under SOCIALNUMBER for broader PII coverage.
    "PASSPORT": "SOCIALNUMBER",
    "IDCARD": "SOCIALNUMBER",
    "DRIVERLICENSE": "SOCIALNUMBER",
    "CARDISSUER": "SOCIALNUMBER",
    "PASS": "SOCIALNUMBER",
    "US_SSN": "SOCIALNUMBER",           # Presidio US SSN detection
    "CREDIT_CARD": "SOCIALNUMBER",      # Presidio credit card detection
    "CRYPTO": "SOCIALNUMBER",           # Presidio cryptocurrency address detection
}


# Annotation-noise tokens that should not be scored as PERSON in evaluation.
# These values encode demographic fields (sex/gender) rather than identities.
_GENDER_PERSON_NOISE = {
    "m",
    "f",
    "o",
    "male",
    "female",
    "femle",
    "prefer not to disclose",
    "not specified",
    "not disclosed",
    "unknown",
}


def _is_person_gender_noise(mapped_label: str, text: str, start: int, end: int) -> bool:
    """Return True when a PERSON span is actually a gender/demographic token."""
    if mapped_label != "PERSON":
        return False
    value = text[start:end].strip().lower()
    return value in _GENDER_PERSON_NOISE


def _map_raw_label(raw_label: str) -> str:
    """Map heterogeneous dataset labels to the project label schema."""
    value = str(raw_label).strip().upper()

    if value in PRIVACY_LABEL_MAP:
        return PRIVACY_LABEL_MAP[value]

    if value in LABEL_MAP:
        return LABEL_MAP[value]

    # Conservative fallbacks for common noisy variants.
    if "EMAIL" in value:
        return "EMAIL"
    if "PHONE" in value or "MOBILE" in value:
        return "PHONE"
    if "API" in value and "KEY" in value:
        return "API_KEY"
    if "DATE" in value or "TIME" in value:
        return "DATE"
    if value == "IP" or "IP" in value:
        return "IP"

    # Do not coerce weak identifiers (e.g. USERNAME, HANDLE, ACCOUNT_ID) to PERSON.
    return "UNKNOWN"


def _load_conll_style_split(ds_split, ner_tag_names, allowed_labels):
    """Build comparable gold spans for token-tag (BIO) style datasets."""
    comparable_golds = []
    for example in ds_split:
        text, gold_spans = conll_to_gold_spans(example, ner_tag_names, LABEL_MAP)
        comparable_gold = {s for s in gold_spans if s[2] in allowed_labels}
        comparable_golds.append((text, comparable_gold))
    return comparable_golds


def _extract_span_labels(example: dict[str, Any]) -> list[tuple[int, int, str]]:
    """Extract span labels from privacy datasets (list or serialized string)."""
    if isinstance(example.get("privacy_mask"), list):
        spans = []
        for item in example["privacy_mask"]:
            try:
                spans.append((int(item["start"]), int(item["end"]), str(item["label"])))
            except (KeyError, TypeError, ValueError):
                continue
        return spans

    raw = example.get("span_labels")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return []
        spans = []
        for item in parsed:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                try:
                    spans.append((int(item[0]), int(item[1]), str(item[2])))
                except (TypeError, ValueError):
                    continue
        return spans
    return []


def _load_span_style_split(ds_split, allowed_labels):
    """Build comparable gold spans for datasets providing explicit char spans."""
    comparable_golds = []
    for example in ds_split:
        text = str(example.get("source_text") or example.get("text") or "")
        spans = _extract_span_labels(example)
        comparable_gold = set()
        for start, end, raw_label in spans:
            mapped = _map_raw_label(raw_label)
            if mapped in allowed_labels and 0 <= start < end <= len(text):
                if _is_person_gender_noise(mapped, text, start, end):
                    continue
                comparable_gold.add((start, end, mapped))
        comparable_golds.append((text, comparable_gold))
    return comparable_golds

def load_cached_dataset(ds_name: str, allowed_labels, split_name: str = "train") -> Any:
    """Load dataset from cache or fetch and cache it, then build gold spans for one split.

    Args:
        ds_name: Dataset identifier from DATASET_LIST.
        allowed_labels: Canonical labels to keep in comparable gold spans.
        split_name: Split to evaluate (for example: train, validation, test).
    """
    dataset_file = cache_file_from_payload(__file__, 
                    {"dataset": ds_name, "allowed_labels": allowed_labels})

    if dataset_file.exists():
        logger.warning("Loading dataset from cache: %s", dataset_file)
        with open(dataset_file, "rb") as f:
            ds = pickle.load(f)
    else:
        if DATASET_LIST[ds_name]['source'] == "huggingface":
            try:
                ds = load_dataset(ds_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to load dataset '{ds_name}': {exc}") from exc
        else:
            raise ValueError(f"Unsupported dataset source: {ds_name}")

    if split_name not in ds:
        available_splits = ", ".join(sorted(ds.keys()))
        raise ValueError(
            f"Dataset '{ds_name}' does not contain split '{split_name}'. "
            f"Available splits: {available_splits}"
        )

    split = ds[split_name]
    features = split.features

    _ner_tag_names = []
    if "ner_tags" in features and "tokens" in features:
        _ner_tag_names = split.features["ner_tags"].feature.names
        comparable_golds = _load_conll_style_split(split, _ner_tag_names, allowed_labels)
    elif "privacy_mask" in features or "span_labels" in features:
        comparable_golds = _load_span_style_split(split, allowed_labels)
    else:
        feature_names = ", ".join(features.keys())
        raise ValueError(
            f"Dataset '{ds_name}' is not in a supported schema. "
            f"Found features: {feature_names}"
        )

    if not dataset_file.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        logger.warning("Saving dataset to cache: %s", dataset_file)
        with open(dataset_file, "wb") as f:
            pickle.dump(ds, f)

    return ds, comparable_golds
