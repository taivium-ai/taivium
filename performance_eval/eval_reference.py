'''
Reference evaluation using spaCy NER models. Provides a benchmark for 
Taivium's performance on the same datasets and label profiles.
'''
import logging
import pickle
import spacy
from taivium.engine import normalize_label
from .utility import compute_prf, cache_file_from_payload

logger = logging.getLogger(__name__)


def spacy_evaluation(dataset, comparable_golds, allowed_labels,
                     max_errors, model_name="en_core_web_lg"):
    '''Evaluate spaCy NER performance on the dataset. 
    Returns TP, FP, FN counts and error samples.'''

    cache_payload = {
        "dataset": dataset,
        "comparable_golds": comparable_golds,
        "max_errors": max_errors,
        "allowed_labels": allowed_labels,
        "model_name": model_name,
    }
    cache_file = cache_file_from_payload(__file__, cache_payload)

    # Check if cache exists
    if cache_file.exists():
        logger.warning(
            f"Loading spaCy evaluation from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            spacy_metrics, spacy_errors = pickle.load(f)
        return spacy_metrics, spacy_errors, cache_file

    nlp = spacy.load(model_name)
    spacy_tp = spacy_fp = spacy_fn = 0
    spacy_errors = []
    # -------  start evaluation loop -------
    for idx, _ in enumerate(dataset["validation"]):
        # -------  Golden ground truth -------
        text, comparable_gold = comparable_golds[idx]

        # --- spaCy span-based evaluation (normalize labels to match gold) ---
        pred_doc = nlp(text)
        spacy_pred_spans = set()
        for ent in pred_doc.ents:
            normalized = normalize_label(ent.label_)
            if normalized in allowed_labels:
                spacy_pred_spans.add((ent.start_char, ent.end_char, normalized))

        spacy_fp_set = spacy_pred_spans - comparable_gold
        spacy_fn_set = comparable_gold - spacy_pred_spans
        spacy_tp += len(spacy_pred_spans & comparable_gold)
        spacy_fp += len(spacy_fp_set)
        spacy_fn += len(spacy_fn_set)
        if (spacy_fp_set or spacy_fn_set) and len(spacy_errors) < max_errors:
            spacy_errors.append(
                {
                    "index": idx,
                    "text": text,
                    "false_positives": sorted(spacy_fp_set),
                    "false_negatives": sorted(spacy_fn_set),
                }
            )
    spacy_p, spacy_r, spacy_f1 = compute_prf(spacy_tp, spacy_fp, spacy_fn)
    spacy_metrics = {"precision": spacy_p, "recall": spacy_r, "f1": spacy_f1}

    # Save to pickle cache
    logger.warning(f"Saving spaCy evaluation to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((spacy_metrics, spacy_errors), f)

    return spacy_metrics, spacy_errors, cache_file
