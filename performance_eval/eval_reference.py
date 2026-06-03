'''
Reference evaluation using NER models. Provides a benchmark for 
Taivium's performance on the same datasets and label profiles.
'''
import logging
import pickle
import spacy
from taivium import Taivium
from taivium.engine import normalize_label
from .utility import compute_prf, cache_file_from_payload, get_git_commit_hash

logger = logging.getLogger(__name__)

# Module-level cache for spaCy models
_spacy_models = {}
# Module-level cache for Taivium engine
_taivium_engine = None

def taivium_detection(text, allowed_labels, model_name="en_core_web_sm"):
    '''Detect entities in text using Taivium. Returns a set of (start, end, label) spans.'''
    # Load engine only once, reuse on subsequent calls
    global _taivium_engine
    if _taivium_engine is None:
        _taivium_engine = Taivium()
    engine = _taivium_engine
    result = engine.process(text)
    pred_spans = set()
    for ent in result.get("entities", []):
        if ent["label"] in allowed_labels:
            pred_spans.add((ent["start"], ent["end"], ent["label"]))
    return pred_spans

def spacy_detection(text, allowed_labels, model_name="en_core_web_lg"):
    '''Detect entities in text using spaCy NER model. Returns a set of (start, end, label) spans.'''
    # Load model only once, reuse on subsequent calls
    if model_name not in _spacy_models:
        _spacy_models[model_name] = spacy.load(model_name)
    nlp = _spacy_models[model_name]
    
    pred_doc = nlp(text)
    pred_spans = set()
    for ent in pred_doc.ents:
        normalized = normalize_label(ent.label_)
        if normalized in allowed_labels:
            pred_spans.add((ent.start_char, ent.end_char, normalized))
    return pred_spans

def evaluation(detection, dataset, comparable_golds, allowed_labels,
                     max_errors, model_name="en_core_web_lg"):
    '''Evaluate NER performance on the dataset. 
    Returns TP, FP, FN counts and error samples.'''

    cache_payload = {
        "detection": detection.__name__,
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
        logger.warning(
            f"Loading evaluation from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            metrics, errors = pickle.load(f)
        return metrics, errors, cache_file

    tp = fp = fn = 0
    errors = []
    # -------  start evaluation loop -------
    for idx, _ in enumerate(dataset["validation"]):
        # -------  Golden ground truth -------
        text, comparable_gold = comparable_golds[idx]

        # --- spaCy span-based evaluation (normalize labels to match gold) ---
        pred_spans = detection(text,allowed_labels, model_name=model_name)

        fp_set = pred_spans - comparable_gold
        fn_set = comparable_gold - pred_spans
        tp += len(pred_spans & comparable_gold)
        fp += len(fp_set)
        fn += len(fn_set)
        if (fp_set or fn_set) and len(errors) < max_errors:
            errors.append(
                {
                    "index": idx,
                    "text": text,
                    "false_positives": sorted(fp_set),
                    "false_negatives": sorted(fn_set),
                }
            )
    p, r, f1 = compute_prf(tp, fp, fn)
    metrics = {"precision": p, "recall": r, "f1": f1}

    # Save to pickle cache
    logger.warning(f"Saving spaCy evaluation to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((metrics, errors), f)

    return metrics, errors, cache_file
