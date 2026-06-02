'''Module for loading and caching evaluation datasets, 
    converting them to a standardized format, and preparing ground 
    truth spans for performance evaluation.'''
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

DATASET_LIST = {"tomaarsen/conll2003": {"source": "huggingface"}}

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
}

LABEL_MAP = {
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOCATION",
    "MISC": "UNKNOWN",
}

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
            ds = load_dataset(ds_name)
        else:
            raise ValueError(f"Unsupported dataset source: {ds_name}")

    _ner_tag_names = ds["train"].features["ner_tags"].feature.names

    #build ground truth
    comparable_golds = []
    for _, example in enumerate(ds["validation"]):
        # -------  prepare golden ground truth -------
        text, gold_spans = conll_to_gold_spans(example, _ner_tag_names, LABEL_MAP)
        # Strictly compare only the target label set.
        comparable_gold = {s for s in gold_spans if s[2] in allowed_labels}
        comparable_golds.append((text, comparable_gold))

    if not dataset_file.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        logger.warning(f"Saving dataset to cache: {dataset_file}")
        with open(dataset_file, "wb") as f:
            pickle.dump(ds, f)

    return ds, _ner_tag_names, comparable_golds
