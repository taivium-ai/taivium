'''
Reference evaluation using NER models. Provides a benchmark for 
Taivium's performance on the same datasets and label profiles.
'''
import logging
import os
import pickle
import time
import spacy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from presidio_analyzer import AnalyzerEngine
from taivium import Taivium
from taivium.engine import normalize_label
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from tqdm import tqdm
from .utility import compute_prf, cache_file_from_payload, get_git_commit_hash

logger = logging.getLogger(__name__)

# Module-level cache for spaCy models
_spacy_models = {}
# Module-level cache for Taivium engines (keyed by spaCy model)
_taivium_engines = {}
# Module-level cache for Presidio AnalyzerEngine
_presidio_engine = None

# Mapping from Presidio entity types to project label schema
_PRESIDIO_LABEL_MAP = {
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
}


def _evaluate_single_sample(task):
    """Worker task for multiprocessing evaluation."""
    idx, text, comparable_gold, detection_name, allowed_labels, model_name = task
    detection_fn = globals().get(detection_name)
    if detection_fn is None:
        raise ValueError(f"Unknown detection function: {detection_name}")

    pred_spans = detection_fn(text, allowed_labels, model_name=model_name)
    fp_set = pred_spans - comparable_gold
    fn_set = comparable_gold - pred_spans

    error = None
    if fp_set or fn_set:
        error = {
            "index": idx,
            "text": text,
            "false_positives": sorted(fp_set),
            "false_negatives": sorted(fn_set),
        }

    return (
        len(pred_spans & comparable_gold),
        len(fp_set),
        len(fn_set),
        error,
        mp.current_process().name,
    )

def taivium_detection(text, allowed_labels, model_name="en_core_web_sm"):
    '''Detect entities in text using Taivium. Returns a set of (start, end, label) spans.'''
    # Load engine per model only once, reuse on subsequent calls
    if model_name not in _taivium_engines:
        print(f"Loading Taivium engine for evaluation with spaCy model: {model_name}")
        _taivium_engines[model_name] = Taivium(spacy_model_name=model_name)
    engine = _taivium_engines[model_name]
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
        print(f"Loading spaCy model for evaluation: {model_name}")
        _spacy_models[model_name] = spacy.load(model_name, disable=[
        "tagger",
        "parser",
        "lemmatizer",
        "attribute_ruler"
    ])
    nlp = _spacy_models[model_name]
    
    pred_doc = nlp(text)
    pred_spans = set()
    for ent in pred_doc.ents:
        normalized = normalize_label(ent.label_)
        if normalized in allowed_labels:
            pred_spans.add((ent.start_char, ent.end_char, normalized))
    return pred_spans

@lru_cache(maxsize=8)
def get_optimized_presidio_engine(model_name: str = "en_core_web_lg") -> AnalyzerEngine:
    """
    Creates a thread-safe, cached Presidio Analyzer instance operating 
    on a stripped-down, high-performance spaCy pipeline.
    """
    # 1. Load spaCy explicitly with heavy, unused sub-components disabled
    # (Just like you did in your native spaCy wrapper)
    nlp = spacy.load(
        model_name,
        disable=["tagger", "parser", "lemmatizer", "attribute_ruler"]
    )
    
    # 2. Configure Presidio's underlying SpacyNlpEngine configuration manually
    # We pass the pre-loaded, stripped nlp instance as a pre-warmed model map
    nlp_engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": model_name}])
    nlp_engine.nlp = {"en": nlp}
    
    # 3. Supply the optimized engine configuration directly into the AnalyzerEngine
    return AnalyzerEngine(nlp_engine=nlp_engine)


def presidio_detection(text, allowed_labels, model_name="en_core_web_lg"):
    """
    Detect entities in text using Microsoft Presidio AnalyzerEngine.
    Returns a set of (start, end, label) spans.
    """
    # Retrieve our highly optimized and cached instance instantly
    engine = get_optimized_presidio_engine(model_name)

    # Request only Presidio types that map to our allowed labels
    presidio_entities = [
        presidio_type
        for presidio_type, project_label in _PRESIDIO_LABEL_MAP.items()
        if project_label in allowed_labels
    ]
    
    results = engine.analyze(text=text, entities=presidio_entities, language="en")

    pred_spans = set()
    for result in results:
        label = _PRESIDIO_LABEL_MAP.get(result.entity_type)
        if label and label in allowed_labels:
            pred_spans.add((result.start, result.end, label))
            
    return pred_spans



def evaluation(detection, dataset, comparable_golds, allowed_labels,
                     max_errors, model_name="en_core_web_lg", shared_cache_name=None,
                     workers=None, chunksize=64, show_worker_progress=False):
    '''Evaluate NER performance on the dataset. 
    Returns TP, FP, FN counts and error samples.'''

    base_payload = {
        "dataset": dataset,
        "comparable_golds": comparable_golds,
        "max_errors": max_errors,
        "allowed_labels": allowed_labels,
        "commit_hash": get_git_commit_hash('.')
    }
    run_cache_name = (
        str(shared_cache_name)
        if shared_cache_name
        else cache_file_from_payload(__file__, base_payload).stem
    )
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    safe_model_name = str(model_name).replace("/", "_")
    cache_file = cache_dir / f"{run_cache_name}__{detection.__name__}__{safe_model_name}.pkl"

    # Check if cache exists
    if cache_file.exists():
        logger.warning(
            f"Loading evaluation from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        # Support old cache format (metrics, errors) and new (metrics, errors, total_time, n)
        if len(cached) == 4:
            metrics, errors, total_time, n_samples = cached
        else:
            metrics, errors = cached
            total_time, n_samples = None, None
        return metrics, errors, cache_file, total_time, n_samples

    tp = fp = fn = 0
    all_errors = []
    t_start = time.perf_counter()

    n_samples = len(comparable_golds)
    max_workers = workers if isinstance(workers, int) and workers > 0 else max(1, (os.cpu_count() or 2) - 1)

    # Multiprocessing gives substantial speedup on large datasets.
    if max_workers > 1 and n_samples > 1:
        tasks = [
            (idx, text, comparable_gold, detection.__name__, allowed_labels, model_name)
            for idx, (text, comparable_gold) in enumerate(comparable_golds)
        ]
        overall_bar = None
        worker_bars = {}
        if show_worker_progress:
            overall_bar = tqdm(total=n_samples, desc=f"{detection.__name__} total", position=0)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            try:
                for tp_i, fp_i, fn_i, error, worker_name in executor.map(
                    _evaluate_single_sample,
                    tasks,
                    chunksize=max(1, int(chunksize)),
                ):
                    tp += tp_i
                    fp += fp_i
                    fn += fn_i
                    if error is not None:
                        all_errors.append(error)

                    if show_worker_progress:
                        if worker_name not in worker_bars:
                            worker_bars[worker_name] = tqdm(
                                total=0,
                                desc=worker_name,
                                position=len(worker_bars) + 1,
                                leave=False,
                                bar_format="{desc}: {n_fmt} samples",
                            )
                        worker_bars[worker_name].update(1)
                        if overall_bar is not None:
                            overall_bar.update(1)
            finally:
                if overall_bar is not None:
                    overall_bar.close()
                for bar in worker_bars.values():
                    bar.close()
    else:
        iter_rows = enumerate(comparable_golds)
        if show_worker_progress:
            iter_rows = enumerate(
                tqdm(comparable_golds, total=n_samples, desc=f"{detection.__name__} total")
            )

        for idx, (text, comparable_gold) in iter_rows:
            pred_spans = detection(text, allowed_labels, model_name=model_name)
            fp_set = pred_spans - comparable_gold
            fn_set = comparable_gold - pred_spans
            tp += len(pred_spans & comparable_gold)
            fp += len(fp_set)
            fn += len(fn_set)
            if fp_set or fn_set:
                all_errors.append(
                    {
                        "index": idx,
                        "text": text,
                        "false_positives": sorted(fp_set),
                        "false_negatives": sorted(fn_set),
                    }
                )

    errors = all_errors[:max_errors]
    p, r, f1 = compute_prf(tp, fp, fn)
    metrics = {"precision": p, "recall": r, "f1": f1}
    total_time = time.perf_counter() - t_start

    # Save to pickle cache
    logger.warning(f"Saving spaCy evaluation to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((metrics, errors, total_time, n_samples), f)

    return metrics, errors, cache_file, total_time, n_samples
