"""
Analyze distribution of correct vs incorrect detections from evaluation runs.
Shows per-label breakdown of true positives, false positives, false negatives.
"""

import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import with proper module paths
from taivium import Taivium
from performance_eval.eval_datasets import load_cached_dataset, LABEL_PROFILES

# Module-level engine singleton — loaded once per process (worker or main)
_engine: Taivium | None = None
_engine_model_name: str | None = None


def _init_worker(model_name: str) -> None:
    """Initializer for each worker process: loads the engine once."""
    global _engine, _engine_model_name
    _engine = Taivium(spacy_model_name=model_name)
    _engine_model_name = model_name


def _process_sample(task):
    """Worker task: run detection on one sample and return per-label TP/FP/FN counts."""
    text, gold, allowed_labels = task
    result = _engine.process(text)
    pred_spans = set()
    for ent in result.get("entities", []):
        if ent["label"] in allowed_labels:
            pred_spans.add((ent["start"], ent["end"], ent["label"]))

    per_label = {}
    pred_by_label = defaultdict(set)
    for s, e, lbl in pred_spans:
        pred_by_label[lbl].add((s, e))

    for label in allowed_labels:
        gold_spans = {(s, e) for s, e, l in gold if l == label}
        pred_spans_label = pred_by_label.get(label, set())
        tp_spans = gold_spans & pred_spans_label
        fn_spans = gold_spans - pred_spans_label
        fp_spans = pred_spans_label - gold_spans

        per_label[label] = {
            "tp": [{"span": text[s:e], "context": text[max(0,s-20):min(len(text),e+20)]} for s, e in tp_spans],
            "fp": [{"span": text[s:e], "context": text[max(0,s-20):min(len(text),e+20)]} for s, e in fp_spans],
            "fn": [{"span": text[s:e], "context": text[max(0,s-20):min(len(text),e+20)]} for s, e in fn_spans],
        }
    return per_label


def taivium_detection(text, allowed_labels, model_name="en_core_web_lg"):
    """Detect entities using the module-level singleton engine."""
    global _engine, _engine_model_name
    if _engine is None or _engine_model_name != model_name:
        _engine = Taivium(spacy_model_name=model_name)
        _engine_model_name = model_name
    result = _engine.process(text)
    pred_spans = set()
    for ent in result.get("entities", []):
        if ent["label"] in allowed_labels:
            pred_spans.add((ent["start"], ent["end"], ent["label"]))
    return pred_spans


def analyze_detections(dataset_name="ai4privacy/pii-masking-300k", 
                      profile="privacy",
                      model_name="en_core_web_lg",
                      max_samples=None,
                      workers=None):
    """
    Analyze detection performance and show distribution of errors.
    
    Args:
        dataset_name: Dataset to analyze
        profile: Label profile to use
        model_name: spaCy model to use
        max_samples: Limit number of samples (None = all)
        workers: Number of parallel worker processes (None = cpu_count - 1)
    """
    
    print("=" * 100)
    print(f"DETECTION ERROR ANALYSIS")
    print(f"Dataset: {dataset_name} | Profile: {profile} | Model: {model_name}")
    print("=" * 100)
    
    # Load dataset
    allowed_labels = LABEL_PROFILES[profile]
    dataset, golds = load_cached_dataset(dataset_name, allowed_labels)
    
    if max_samples:
        golds = golds[:max_samples]
    
    n_workers = workers if isinstance(workers, int) and workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    print(f"\nAnalyzing {len(golds)} samples using {n_workers} worker process(es)...")

    # Per-label statistics
    label_stats = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})

    tasks = [(text, gold, allowed_labels) for text, gold in golds]

    if n_workers > 1:
        import multiprocessing as mp
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(model_name,),
        ) as executor:
            for per_label in tqdm(executor.map(_process_sample, tasks, chunksize=32),
                                  total=len(tasks), desc="Analyzing", unit="sample"):
                for label, counts in per_label.items():
                    label_stats[label]["tp"].extend(counts["tp"])
                    label_stats[label]["fp"].extend(counts["fp"])
                    label_stats[label]["fn"].extend(counts["fn"])
    else:
        # Single-process fallback (still uses singleton engine)
        _init_worker(model_name)
        for task in tqdm(tasks, desc="Analyzing", unit="sample"):
            per_label = _process_sample(task)
            for label, counts in per_label.items():
                label_stats[label]["tp"].extend(counts["tp"])
                label_stats[label]["fp"].extend(counts["fp"])
                label_stats[label]["fn"].extend(counts["fn"])
    
    # Print summary statistics
    print("\n" + "=" * 100)
    print("DETECTION DISTRIBUTION BY LABEL")
    print("=" * 100)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    results = []
    for label in sorted(allowed_labels):
        stats = label_stats[label]
        tp_count = len(stats["tp"])
        fp_count = len(stats["fp"])
        fn_count = len(stats["fn"])
        
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count
        
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append((label, tp_count, fp_count, fn_count, precision, recall, f1))
    
    # Print sorted by F1
    print(f"\n{'Label':15} {'TP':8} {'FP':8} {'FN':8} {'Precision':10} {'Recall':10} {'F1':8}")
    print("-" * 100)
    
    for label, tp, fp, fn, prec, recall, f1 in sorted(results, key=lambda x: x[6], reverse=True):
        status = "✓" if f1 >= 0.5 else "⚠️ " if f1 >= 0.3 else "✗"
        print(f"{status} {label:13} {tp:8d} {fp:8d} {fn:8d} {prec:10.4f} {recall:10.4f} {f1:8.4f}")
    
    # Overall metrics
    print("-" * 100)
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"{'OVERALL':15} {total_tp:8d} {total_fp:8d} {total_fn:8d} {overall_precision:10.4f} {overall_recall:10.4f} {overall_f1:8.4f}")
    
    # Detailed examples of errors
    print("\n" + "=" * 100)
    print("DETAILED ERROR EXAMPLES")
    print("=" * 100)
    
    for label in sorted(allowed_labels):
        stats = label_stats[label]
        
        # Only show if there are errors
        if not stats["fn"] and not stats["fp"]:
            print(f"\n✓ {label:15} - No errors (all detections correct)")
            continue
        
        print(f"\n{label:15}")
        print("-" * 100)
        
        # False negatives (missed)
        if stats["fn"]:
            print(f"  FALSE NEGATIVES ({len(stats['fn'])} missed):")
            for i, error in enumerate(stats["fn"][:5], 1):
                span_repr = repr(error['span'])[:30]
                context = error['context'][-30:] if len(error['context']) > 30 else error['context']
                print(f"    {i}. {span_repr:30s} | context: ...{context}...")
            if len(stats["fn"]) > 5:
                print(f"    ... and {len(stats['fn']) - 5} more")
        
        # False positives (incorrectly detected)
        if stats["fp"]:
            print(f"  FALSE POSITIVES ({len(stats['fp'])} false alarms):")
            for i, error in enumerate(stats["fp"][:5], 1):
                span_repr = repr(error['span'])[:30]
                context = error['context'][-30:] if len(error['context']) > 30 else error['context']
                print(f"    {i}. {span_repr:30s} | context: ...{context}...")
            if len(stats["fp"]) > 5:
                print(f"    ... and {len(stats['fp']) - 5} more")
    
    print("\n" + "=" * 100)
    
    return label_stats, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze detection errors from evaluation")
    parser.add_argument("--dataset", default="ai4privacy/pii-masking-300k",
                       help="Dataset name")
    parser.add_argument("--profile", default="privacy",
                       help="Label profile (privacy, etc.)")
    parser.add_argument("--model", default="en_core_web_lg",
                       help="spaCy model name")
    parser.add_argument("--samples", type=int, default=None,
                       help="Limit number of samples to analyze")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel worker processes (default: cpu_count - 1)")
    
    args = parser.parse_args()
    
    analyze_detections(
        dataset_name=args.dataset,
        profile=args.profile,
        model_name=args.model,
        max_samples=args.samples,
        workers=args.workers,
    )
