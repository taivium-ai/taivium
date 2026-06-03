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
    },
    "privacy_no_org": {
        "PERSON",
        "LOCATION",
        "EMAIL",
        "PHONE",
        "API_KEY",
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
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ADDRESS": "LOCATION",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "MOBILE": "PHONE",
    "MOBILE_PHONE_NUMBER": "PHONE",
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "ACCESS_TOKEN": "API_KEY",
}


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
                comparable_gold.add((start, end, mapped))
        comparable_golds.append((text, comparable_gold))
    return comparable_golds

def load_cached_dataset(ds_name: str, allowed_labels) -> Any:
    """Load dataset from cache or fetch and cache it."""
    dataset_file = cache_file_from_payload(__file__, 
                    {"dataset": ds_name, "allowed_labels": allowed_labels})

    if dataset_file.exists():
        logger.warning(f"Loading dataset from cache: {dataset_file}")
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

    split_name = "validation" if "validation" in ds else "train"
    split = ds[split_name]
    features = split.features

    _ner_tag_names = []
    if "ner_tags" in features and "tokens" in features:
        _ner_tag_names = ds["train"].features["ner_tags"].feature.names
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
        logger.warning(f"Saving dataset to cache: {dataset_file}")
        with open(dataset_file, "wb") as f:
            pickle.dump(ds, f)

    return ds, comparable_golds
