'''
Reference evaluation using NER models. Provides a benchmark for 
Taivium's performance on the same datasets and label profiles.
'''
import logging
import os
import pickle
import time
from typing import Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from taivium import Taivium
from taivium.engine import (
    _resolve_context_ner_backend,
)
from tqdm import tqdm
from .utility import compute_prf, cache_file_from_payload, get_git_commit_hash

logger = logging.getLogger(__name__)

# Module-level cache for Taivium engines (keyed by short_text_backend, long_text_backend)
_taivium_engines = {}


def _evaluate_single_sample(task):
    """Worker task for multiprocessing evaluation."""
    (
        idx,
        text,
        comparable_gold,
        detection_name,
        allowed_labels,
        short_text_backend,
        long_text_backend,
    ) = task
    detection_fn = globals().get(detection_name)
    if detection_fn is None:
        raise ValueError(f"Unknown detection function: {detection_name}")

    pred_spans = detection_fn(
        text,
        allowed_labels,
        short_text_backend=short_text_backend,
        long_text_backend=long_text_backend,
    )
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

def taivium_detection(
    text,
    allowed_labels,
    short_text_backend="en_core_web_sm",
    long_text_backend=None,
):
    '''Detect entities in text using Taivium. Returns a set of (start, end, label) spans.'''
    # Load engine per model/config once, reuse on subsequent calls.
    backend = _resolve_context_ner_backend(long_text_backend, None)
    engine_key = (short_text_backend, backend)
    if engine_key not in _taivium_engines:
        print(
            "Loading Taivium engine for evaluation: "
            f"short-text=spaCy({short_text_backend}), long-text={backend}"
        )
        _taivium_engines[engine_key] = Taivium(
            spacy_model_name=short_text_backend,
            context_ner_backend=backend,
        )
    engine = _taivium_engines[engine_key]
    result = engine.process(text)
    pred_spans = set()
    for ent in result.get("entities", []):
        if ent["label"] in allowed_labels:
            pred_spans.add((ent["start"], ent["end"], ent["label"]))
    return pred_spans


def evaluation(detection, dataset, comparable_golds, allowed_labels,
                     max_errors, short_text_backend="en_core_web_lg",
                     long_text_backend=None,
                     shared_cache_name=None,
                     use_cache=True,
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
    safe_model_name = str(short_text_backend).replace("/", "_")
    variant = ""
    if detection.__name__ == "taivium_detection":
        backend = _resolve_context_ner_backend(long_text_backend, None)
        variant = f"__backend_{backend}"
    cache_file = cache_dir / f"{run_cache_name}__{detection.__name__}__{safe_model_name}{variant}.pkl"

    # Check if cache exists
    if use_cache and cache_file.exists():
        logger.warning("Loading evaluation from cache: %s", cache_file)
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
            (
                idx,
                text,
                comparable_gold,
                detection.__name__,
                allowed_labels,
                short_text_backend,
                long_text_backend,
            )
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
            pred_spans = detection(text, allowed_labels, 
                                   short_text_backend=short_text_backend,
                                   long_text_backend=long_text_backend)
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
    if use_cache:
        logger.warning("Saving evaluation to cache: %s", cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump((metrics, errors, total_time, n_samples), f)

    return metrics, errors, cache_file, total_time, n_samples
