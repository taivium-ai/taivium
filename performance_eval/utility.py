'''Utility functions for performance evaluation, including deterministic \
    cache file generation and results persistence.'''
import json
import numpy as np
import hashlib
import subprocess
import logging
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _stable_serialize(value):
    """Convert nested inputs into a deterministic JSON-serializable structure."""
    if isinstance(value, dict):
        return {str(k): _stable_serialize(v) 
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_serialize(v) for v in value]
    if isinstance(value, set):
        serialized_items = [_stable_serialize(v) for v in value]
        return sorted(
            serialized_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _stable_serialize(to_dict())
        except TypeError:
            pass
    return repr(value)


def cache_file_from_payload(module_file, payload, cache_subdir=".cache", suffix=".pkl"):
    """Return a deterministic cache file path for the given payload."""
    cache_dir = Path(module_file).parent / cache_subdir
    if not cache_dir.exists():
        logger.warning("Creating cache directory: %s", cache_dir)
    cache_dir.mkdir(exist_ok=True)
    stable_payload = _stable_serialize(payload)
    cache_key = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    cache_hash = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{cache_hash}{suffix}"


def get_git_commit_hash(repo_path="."):
    """Get the current git commit hash. Raises if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git commit hash: {e.stderr}") from e


def get_git_tag(repo_path="."):
    """Get comma-separated git tag(s) pointing at HEAD, or None if no tags exist."""
    try:
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return ", ".join(tags) if tags else None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get git tag(s): {e.stderr}") from e


def check_git_clean(repo_path="."):
    """Check if git working directory is clean (no staged or unstaged changes).
    Raises ValueError if there are uncommitted changes.
    Returns the commit hash if clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        status_output = result.stdout.strip()
        if status_output:
            raise ValueError(
                f"Git working directory is not clean. \
                    Uncommitted changes detected:\n{status_output}"
            )
        commit_hash = get_git_commit_hash(repo_path)
        return commit_hash
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to check git status: {e.stderr}") from e

def conll_to_gold_spans(example, ner_tag_names, label_map):
    """Returns (text, gold_spans) where gold_spans is a set of (start_char, end_char, label)."""
    words = example["tokens"]
    tags = example["ner_tags"]
    text = " ".join(words)

    # Build word -> char offset mapping using the simple space-joined text
    char_offsets = []
    pos = 0
    for word in words:
        char_offsets.append(pos)
        pos += len(word) + 1  # +1 for the space

    gold_spans = set()
    start_idx = None
    label = None

    for i, tag_id in enumerate(tags):
        tag = ner_tag_names[tag_id]
        mapped_label = label_map.get(tag[2:] if tag != "O" else "", None)

        if tag.startswith("B-"):
            if start_idx is not None and label is not None:
                end_char = char_offsets[i - 1] + len(words[i - 1])
                gold_spans.add((char_offsets[int(start_idx)], end_char, label))
            start_idx = i
            label = mapped_label

        elif tag.startswith("I-"):
            pass  # continue current entity

        else:  # "O"
            if start_idx is not None and label is not None:
                end_char = char_offsets[i - 1] + len(words[i - 1])
                gold_spans.add((char_offsets[int(start_idx)], end_char, label))
            start_idx = None
            label = None

    if start_idx is not None and label is not None:
        end_char = char_offsets[len(words) - 1] + len(words[-1])
        gold_spans.add((char_offsets[int(start_idx)], end_char, label))

    return text, gold_spans

def compute_prf(tp, fp, fn):
    '''Compute precision, recall, and F1 score from true positives, \
        false positives, and false negatives.'''
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def plot_label_distribution(comparable_golds, dataset_name, profile_name, save_path):
    """Plot and save a bar chart of gold label counts from the evaluation split.

    Args:
        comparable_golds: List of (text, gold_spans_set) where each span is (start, end, label).
        dataset_name: Dataset name string (used in title).
        profile_name: Label profile name (used in title).
        save_path: Path object where the PNG will be saved.
    """
    counts = Counter()
    for _, spans in comparable_golds:
        for _, _, label in spans:
            counts[label] += 1

    labels = sorted(counts.keys())
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_title(f"Label Distribution — {dataset_name} / {profile_name}")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count (validation split)")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def calculate_delta(metrics1, metrics2):
    """Calculate the delta between two metrics."""
    delta = {
        "precision": metrics1["precision"] - metrics2["precision"],
        "recall": metrics1["recall"] - metrics2["recall"],
        "f1": metrics1["f1"] - metrics2["f1"],
    }
    return delta


def get_delta_matrix_tables(settings_results):
    """Get delta metrics as matrix tables (like confusion matrices).
    
    Creates separate tables for precision, recall, F1 score, and timing (ms/sample).
    Rows: baseline model (i), Columns: comparison model (j)
    Cell value: delta of metric when j - i
    Returns a dictionary with matrices data.
    """
    n = len(settings_results)
    short_names = [f"{s['detection_func'].__name__} ({s['spacy_model_name']})" 
                   for s in settings_results]
    full_names = [
        f"{s['detection_func'].__name__} ({s['spacy_model_name']}) [{s['cache_file'].stem if 'cache_file' in s and s['cache_file'] else 'no-cache'}]"
        for s in settings_results
    ]

    # metric_name -> (values_per_model_fn, highlight_positive)
    # For timing: lower is better, so negative delta = improvement → highlight_positive=False
    def _avg_ms(s):
        t = s.get("total_time")
        n_s = s.get("n_samples")
        return t / n_s * 1000 if (t is not None and n_s) else None

    metrics_config = [
        ("precision",   lambda s: s["metrics"]["precision"], True),
        ("recall",      lambda s: s["metrics"]["recall"],    True),
        ("f1",          lambda s: s["metrics"]["f1"],        True),
        ("timing_ms",   _avg_ms,                             False),
    ]
    matrices_data = {}

    for metric_name, value_fn, highlight_positive in metrics_config:
        values = [value_fn(s) for s in settings_results]
        # Skip timing matrix if any model has no timing data
        if metric_name == "timing_ms" and any(v is None for v in values):
            continue

        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = values[j] - values[i]

        mask = ~np.eye(n, dtype=bool)
        max_val = np.max(np.abs(matrix[mask])) if np.any(mask) else 0

        matrices_data[metric_name] = {
            "model_names": short_names,
            "model_legend": full_names,
            "matrix": matrix.tolist(),
            "max_absolute_value": float(max_val),
            "highlight_positive": highlight_positive,
        }

    return matrices_data


def print_delta_matrix_tables(settings_results):
    """Print delta metrics as matrix tables (like confusion matrices) with highlighting.
    
    Creates separate tables for precision, recall, and F1 score.
    Rows: baseline model (i), Columns: comparison model (j)
    Cell value: delta of metric when j - i
    Highest values in each metric are highlighted.
    Each detection is labeled with its detection function name and spacy model name.
    """

    matrices_data = get_delta_matrix_tables(settings_results)

    # Print legend (cache mapping) once before all tables
    first_key = next(iter(matrices_data))
    legend = matrices_data[first_key]["model_legend"]
    print(f"\n{'='*80}")
    print("Model Legend:")
    for entry in legend:
        print(f"  {entry}")

    for metric_name, data in matrices_data.items():
        model_names = data["model_names"]
        matrix = np.array(data["matrix"])
        max_val = data["max_absolute_value"]
        highlight_positive = data.get("highlight_positive", True)
        n = len(model_names)

        # Print table header
        print(f"\n{'='*80}")
        print(f"DELTA {metric_name.upper()} - Rows (baseline) vs Columns (comparison)")
        print(f"{'='*80}")

        # Print column headers
        header = f"{'Baseline':<15} |"
        for col_name in model_names:
            header += f" {col_name:>12} |"
        print(header)
        print("-" * len(header))

        # Print rows
        for i, row_name in enumerate(model_names):
            row_str = f"{row_name:<15} |"
            for j in range(n):
                val = matrix[i][j]
                if highlight_positive:
                    is_best = (abs(val) == max_val and val > 0) if max_val > 0 else False
                else:
                    is_best = (abs(val) == max_val and val < 0) if max_val > 0 else False
                marker = "⭐" if is_best else "  "
        
                row_str += f" {val:>10.4f}{marker} |"
            print(row_str)

        print("=" * len(header))


def format_delta_matrix_tables_as_text(settings_results):
    """Format delta metrics as matrix tables text (like confusion matrices).
    
    Returns a formatted string with separate tables for precision, recall, and F1.
    Rows: baseline model (i), Columns: comparison model (j)
    Cell value: delta of metric when j - i
    Highest values in each metric are highlighted with ⭐.
    Each detection is labeled with its detection function name and spacy model name.
    """

    matrices_data = get_delta_matrix_tables(settings_results)

    output_lines = []

    # Print legend (cache mapping) once before all tables
    first_key = next(iter(matrices_data))
    legend = matrices_data[first_key]["model_legend"]
    output_lines.append(f"\n{'='*80}")
    output_lines.append("Model Legend:")
    for entry in legend:
        output_lines.append(f"  {entry}")

    for metric_name, data in matrices_data.items():
        model_names = data["model_names"]
        matrix = np.array(data["matrix"])
        max_val = data["max_absolute_value"]
        highlight_positive = data.get("highlight_positive", True)
        n = len(model_names)

        # Table header
        output_lines.append(f"\n{'='*80}")
        output_lines.append(
            f"DELTA {metric_name.upper()} - Rows (baseline) vs Columns (comparison)")
        output_lines.append(f"{'='*80}")

        # Column headers
        header = f"{'Baseline':<15} |"
        for col_name in model_names:
            header += f" {col_name:>12} |"
        output_lines.append(header)
        output_lines.append("-" * len(header))

        # Rows
        for i, row_name in enumerate(model_names):
            row_str = f"{row_name:<15} |"
            for j in range(n):
                val = matrix[i][j]
                if highlight_positive:
                    is_best = (abs(val) == max_val and val > 0) if max_val > 0 else False
                else:
                    is_best = (abs(val) == max_val and val < 0) if max_val > 0 else False
                marker = "⭐" if is_best else "  "

                row_str += f" {val:>10.4f}{marker} |"
            output_lines.append(row_str)

        output_lines.append("=" * len(header))

    return "\n".join(output_lines)


def save_evaluation_results(cache_file, detection_func_name, dataset, label_profile,
                           allowed_labels, model_name, metrics, errors):
    """Save evaluation results (report, metrics JSON, and errors JSON).
    
    Args:
        cache_file: Path object for the cache file
        detection_func_name: Name of detection function
        dataset: Dataset name
        label_profile: Label profile name
        allowed_labels: Set/list of allowed labels
        model_name: Name of the model used
        metrics: Dict with 'precision', 'recall', 'f1' keys
        errors: List of error dicts with 'index', 'text', 'false_positives', 'false_negatives'
    """
    # Report text file
    report_path = cache_file.parent / f"{cache_file.stem}_{detection_func_name}_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Report for {detection_func_name}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Label Profile: {label_profile}\n")
        f.write(f"Allowed Labels: {', '.join(sorted(allowed_labels))}\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Git Commit Hash: {get_git_commit_hash()}\n")
        f.write("\nMetrics:\n")
        f.write(f"Precision: {metrics['precision']:.12f}\n")
        f.write(f"Recall:    {metrics['recall']:.12f}\n")
        f.write(f"F1 Score:  {metrics['f1']:.12f}\n")

    # Metrics JSON
    json_path = cache_file.parent / f"{cache_file.stem}_{detection_func_name}_report.json"
    json_data = {
        "detection_func": detection_func_name,
        "dataset": dataset,
        "label_profile": label_profile,
        "allowed_labels": sorted(allowed_labels),
        "model_name": model_name,
        "git_commit_hash": get_git_commit_hash(),
        "metrics": {
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1']
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Errors JSON (both timestamped and latest versions)
    errors_json_path = cache_file.parent / f"{detection_func_name}_errors.json"
    latest_errors_json_path = cache_file.parent / f"latest_{detection_func_name}_errors.json"

    errors_json_data = {
        "detection_func": detection_func_name,
        "cache_name": cache_file.stem,
        "model_name": model_name,
        "dataset": dataset,
        "label_profile": label_profile,
        "total_errors": len(errors),
        "errors": errors
    }

    for path in (errors_json_path, latest_errors_json_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(errors_json_data, f, indent=2)

    # Errors text file
    errors_txt_path = cache_file.parent / f"{cache_file.stem}_{detection_func_name}_errors.txt"
    latest_errors_txt_path = cache_file.parent / f"{cache_file.stem}_latest_{detection_func_name}_errors.txt"

    lines = [
        f"Errors Report for {detection_func_name}",
        f"Cache Name: {cache_file.stem}",
        f"Model Name: {model_name}",
        f"Dataset: {dataset}",
        f"Label Profile: {label_profile}",
        f"Total Errors: {len(errors)}",
        "=" * 80,
    ]
    for err in errors:
        lines.append(f"\n[#{err['index']}] {err['text']}")
        if err.get("false_positives"):
            lines.append("  False Positives:")
            for fp in err["false_positives"]:
                lines.append(f"    [{fp[0]}:{fp[1]}] {fp[2]!r}  \"{err['text'][fp[0]:fp[1]]}\"")
        if err.get("false_negatives"):
            lines.append("  False Negatives:")
            for fn in err["false_negatives"]:
                lines.append(f"    [{fn[0]}:{fn[1]}] {fn[2]!r}  \"{err['text'][fn[0]:fn[1]]}\"")
    errors_text = "\n".join(lines)

    for path in (errors_txt_path, latest_errors_txt_path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(errors_text)


def print_timing_summary(settings_results):
    """Print a timing summary table for all evaluated models.

    Args:
        settings_results: List of dicts with 'detection_func', 'spacy_model_name',
            'total_time' (seconds or None), and 'n_samples' (int or None).
    """
    print(f"\n{'='*80}")
    print("Timing Summary:")
    print(f"  {'Model':<45} {'Total (s)':>10} {'Avg (ms/sample)':>17}")
    print(f"  {'-'*45} {'-'*10} {'-'*17}")
    for s in settings_results:
        label = f"{s['detection_func'].__name__} ({s['spacy_model_name']})"
        t = s.get("total_time")
        n = s.get("n_samples")
        if t is not None and n:
            print(f"  {label:<45} {t:>10.2f} {t / n * 1000:>17.2f}")
        else:
            print(f"  {label:<45} {'(cached, no time)':>29}")
    print(f"{'='*80}")


def save_delta_matrices(cache_file, settings_results):
    """Save delta matrices in both JSON and text formats.
    
    Args:
        cache_file: Path object for the cache file
        settings_results: List of dicts with evaluation results and metrics
    
    Returns:
        Tuple of (delta_json_path, delta_txt_path) Path objects
    """

    matrices_data = get_delta_matrix_tables(settings_results)

    delta_matrices_json = {
        "precision": matrices_data["precision"],
        "recall": matrices_data["recall"],
        "f1": matrices_data["f1"],
        "note": "Rows are baseline models, columns are comparison models. "
                    + "Cell value = delta (column - row)"
    }

    # Save JSON
    delta_json_path = cache_file.parent / f"{cache_file.stem}_delta_matrices.json"
    with open(delta_json_path, "w", encoding="utf-8") as f:
        json.dump(delta_matrices_json, f, indent=2)

    # Save text format
    delta_text = format_delta_matrix_tables_as_text(settings_results)
    delta_txt_path = cache_file.parent / f"{cache_file.stem}_delta_matrices.txt"
    with open(delta_txt_path, "w", encoding="utf-8") as f:
        f.write(delta_text)

    return delta_json_path, delta_txt_path
