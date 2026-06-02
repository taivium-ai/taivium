'''
Reference evaluation using spaCy NER models. Provides a benchmark for 
Taivium's performance on the same datasets and label profiles.
'''
from pathlib import Path
import logging
import pickle

from taivium import Taivium
from .utility import compute_prf, cache_file_from_payload, get_git_commit_hash

logger = logging.getLogger(__name__)
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def taivium_evaluation(dataset, comparable_golds, allowed_labels,
                     max_errors, model_name="en_core_web_sm"):
    '''Evaluate spaCy NER performance on the dataset. 
    Returns TP, FP, FN counts and error samples.'''

    cache_payload = {
        "dataset": dataset,
        "comparable_golds": comparable_golds,
        "max_errors": max_errors,
        "allowed_labels": allowed_labels,
        "model_name": model_name,
        "commit_hash": get_git_commit_hash('.')  # Returns: 3aee14c8d2ab26b56645a4f6ff5616ea5477870a
    }
    cache_file = cache_file_from_payload(__file__, cache_payload)

    # Check if cache exists
    if cache_file.exists():
        logger.warning(f"Loading Taivium evaluation from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            taivium_metrics, taivium_errors = pickle.load(f)
        return taivium_metrics, taivium_errors, cache_file


    # Initialize Taivium engine
    engine = Taivium()
    # Accumulators for span-based evaluation
    taivium_tp = taivium_fp = taivium_fn = 0
    taivium_errors = []

    # -------  start evaluation loop -------
    for idx, _ in enumerate(dataset["validation"]):
        # -------  Golden ground truth -------
        text, comparable_gold = comparable_golds[idx]

        # --- Taivium span-based evaluation ---
        result = engine.process(text)
        taivium_pred_spans = set()
        for ent in result.get("entities", []):
            if ent["label"] in allowed_labels:
                taivium_pred_spans.add((ent["start"], ent["end"], ent["label"]))

        taivium_fp_set = taivium_pred_spans - comparable_gold
        taivium_fn_set = comparable_gold - taivium_pred_spans
        taivium_tp += len(taivium_pred_spans & comparable_gold)
        taivium_fp += len(taivium_fp_set)
        taivium_fn += len(taivium_fn_set)
        if (taivium_fp_set or taivium_fn_set) and len(taivium_errors) < max_errors:
            taivium_errors.append(
                {
                    "index": idx,
                    "text": text,
                    "false_positives": sorted(taivium_fp_set),
                    "false_negatives": sorted(taivium_fn_set),
                }
            )

    taivium_p, taivium_r, taivium_f1 = compute_prf(taivium_tp, taivium_fp, taivium_fn)
    taivium_metrics = {"precision": taivium_p, "recall": taivium_r, "f1": taivium_f1}

    # Save to pickle cache
    logger.warning(f"Saving Taivium evaluation to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((taivium_metrics, taivium_errors), f)

    return taivium_metrics, taivium_errors, cache_file
