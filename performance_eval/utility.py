'''Utility functions for performance evaluation, including deterministic cache file generation and results persistence.'''
import json
import hashlib
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path

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
        logger.warning(f"Creating cache directory: {cache_dir}")
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
                f"Git working directory is not clean. Uncommitted changes detected:\n{status_output}"
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
                gold_spans.add((char_offsets[start_idx], end_char, label))
            start_idx = i
            label = mapped_label

        elif tag.startswith("I-"):
            pass  # continue current entity

        else:  # "O"
            if start_idx is not None and label is not None:
                end_char = char_offsets[i - 1] + len(words[i - 1])
                gold_spans.add((char_offsets[start_idx], end_char, label))
            start_idx = None
            label = None

    if start_idx is not None and label is not None:
        end_char = char_offsets[len(words) - 1] + len(words[-1])
        gold_spans.add((char_offsets[start_idx], end_char, label))

    return text, gold_spans

def compute_prf(tp, fp, fn):
    '''Compute precision, recall, and F1 score from true positives, \
        false positives, and false negatives.'''
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def calculate_delta(metrics1, metrics2):
    """Calculate the delta between two metrics."""
    delta = {
        "precision": metrics1["precision"] - metrics2["precision"],
        "recall": metrics1["recall"] - metrics2["recall"],
        "f1": metrics1["f1"] - metrics2["f1"],
    }
    return delta

def save_results(profile, labels, spacy_metrics, taivium_metrics,
                 spacy_errors, taivium_errors, taivium_cache_file,
                 spacy_model, taivium_spacy_model):
    """Persist metrics/deltas and wrong-detection samples to files."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{profile}_{timestamp}"

    delta = calculate_delta(taivium_metrics, spacy_metrics)

    commit_hash = get_git_commit_hash()
    git_tag = get_git_tag()

    payload = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "spaCy Model": spacy_model,
        "Taivium spaCy Model": taivium_spacy_model,
        "Git Commit Hash": commit_hash,
        "profile": profile,
        "labels": sorted(labels),
        "spacy": spacy_metrics,
        "taivium": taivium_metrics,
        "delta_taivium_minus_spacy": delta,
    }
    if git_tag:
        payload["Git Tag"] = git_tag

    # Use cache filename (without extension) as folder name
    cache_folder = taivium_cache_file.parent / taivium_cache_file.stem
    results_dir = cache_folder / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_json = results_dir / f"latest_{profile}.json"
    latest_txt = results_dir / f"latest_{profile}.txt"
    run_json = results_dir / f"{run_id}.json"
    run_txt = results_dir / f"{run_id}.txt"

    for path in (latest_json, run_json):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    errors_payload = {
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "profile": profile,
        "labels": sorted(labels),
        "spacy_wrong_detections": spacy_errors,
        "taivium_wrong_detections": taivium_errors,
    }

    tag_line = f"Git Tag: {git_tag}\n" if git_tag else ""

    report = (
        f"Run ID: {run_id}\n"
        f"Timestamp (UTC): {timestamp}\n"
        f"spaCy Model: {spacy_model}\n"
        f"Taivium spaCy Model: {taivium_spacy_model}\n"
        f"Git Commit Hash: {commit_hash}\n"
        f"{tag_line}"
        f"Profile: {profile}\n"
        f"Labels: {', '.join(sorted(labels))}\n\n"
        f"spaCy\n"
        f"  Precision: {spacy_metrics['precision']:.12f}\n"
        f"  Recall:    {spacy_metrics['recall']:.12f}\n"
        f"  F1:        {spacy_metrics['f1']:.12f}\n\n"
        f"Taivium\n"
        f"  Precision: {taivium_metrics['precision']:.12f}\n"
        f"  Recall:    {taivium_metrics['recall']:.12f}\n"
        f"  F1:        {taivium_metrics['f1']:.12f}\n\n"
        f"Delta (Taivium - spaCy)\n"
        f"  Precision: {delta['precision']:+.12f}\n"
        f"  Recall:    {delta['recall']:+.12f}\n"
        f"  F1:        {delta['f1']:+.12f}\n"
    )

    for path in (latest_txt, run_txt):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)

    latest_errors_json = results_dir / f"latest_{profile}_errors.json"
    run_errors_json = results_dir / f"{run_id}_errors.json"
    latest_errors_txt = results_dir / f"latest_{profile}_errors.txt"
    run_errors_txt = results_dir / f"{run_id}_errors.txt"

    for path in (latest_errors_json, run_errors_json):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(errors_payload, fh, indent=2)

    lines = [
        f"Run ID: {run_id}",
        f"Profile: {profile}",
        f"Labels: {', '.join(sorted(labels))}",
        "",
        f"spaCy wrong-detection samples: {len(spacy_errors)}",
        "",
    ]
    for i, sample in enumerate(spacy_errors, start=1):
        lines.extend([
            f"[spaCy sample #{i}] index={sample['index']}",
            f"text: {sample['text']}",
            f"false_positives: {sample['false_positives']}",
            f"false_negatives: {sample['false_negatives']}",
            "",
        ])

    lines.extend([
        f"Taivium wrong-detection samples: {len(taivium_errors)}",
        "",
    ])
    for i, sample in enumerate(taivium_errors, start=1):
        lines.extend([
            f"[Taivium sample #{i}] index={sample['index']}",
            f"text: {sample['text']}",
            f"false_positives: {sample['false_positives']}",
            f"false_negatives: {sample['false_negatives']}",
            "",
        ])

    errors_report = "\n".join(lines)
    for path in (latest_errors_txt, run_errors_txt):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(errors_report)

    print("\nSaved files:")
    print(latest_json)
    print(latest_txt)
    print(run_json)
    print(run_txt)
    print(latest_errors_json)
    print(latest_errors_txt)
    print(run_errors_json)
    print(run_errors_txt)
