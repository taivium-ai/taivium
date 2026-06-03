'''Main evaluation script for running entity detection evaluations on specified datasets and label profiles.
This script loads the specified dataset and label profile, runs evaluations using both spaCy and Taivium, and saves the results and error samples. It also computes and prints delta matrices comparing the models'''
import argparse
import logging
import os
import sys
from pathlib import Path

import tqdm


def _load_dotenv(path: Path) -> None:
    """Load .env variables without overriding already-exported environment values."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as dotenv_file:
        for raw_line in dotenv_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _resolve_log_level(level_name: str | None, default_level: int) -> int:
    """Resolve a logging level name to its numeric value, with fallback."""
    if not level_name:
        return default_level
    resolved = getattr(logging, level_name.strip().upper(), None)
    if isinstance(resolved, int):
        return resolved
    return default_level


def _configure_logging_from_env() -> None:
    """Configure root and audit logger levels from environment variables.

    Supported variables:
        LOG_LEVEL: Root logger level (default: WARNING)
        TAIVIUM_AUDIT_LOG_LEVEL: taivium.audit logger level (default: LOG_LEVEL)
    """
    root_level = _resolve_log_level(os.getenv("LOG_LEVEL"), logging.WARNING)
    logging.basicConfig(level=root_level)

    audit_level = _resolve_log_level(os.getenv("TAIVIUM_AUDIT_LOG_LEVEL"), root_level)
    logging.getLogger("taivium.audit").setLevel(audit_level)


# Ensure project root is importable when run as a script (e.g., via VS Code debugger)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_load_dotenv(Path(PROJECT_ROOT) / ".env")

from performance_eval.utility import print_delta_matrix_tables, \
                                    save_evaluation_results, save_delta_matrices, \
                                    plot_label_distribution, print_timing_summary, \
                                    cache_file_from_payload, get_git_commit_hash
from performance_eval.eval_datasets import DATASET_LIST, load_cached_dataset, LABEL_PROFILES
from performance_eval.eval_reference import spacy_detection, taivium_detection, \
    presidio_anonymization_detection, evaluation

def main() -> None:
    _configure_logging_from_env()

    parser = argparse.ArgumentParser(
        description="Evaluate framework, choose dataset and label profile and error saving.")
    parser.add_argument(
        "--profile",
        choices=["privacy", "conll_ner", "privacy_no_org"],
        default="privacy",
        help="Label profile for comparison: conll_ner \
            (PERSON/ORG/LOCATION) or privacy (adds EMAIL/PHONE/API_KEY).",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LIST.keys()),
        default=list(DATASET_LIST.keys())[1],
        help="Dataset for evaluation:" + ", ".join([f"{name} ({info['source']})" \
                                                    for name, info in DATASET_LIST.items()]),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Maximum number of wrong-detection text samples to save per model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of worker processes for evaluation. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--no-worker-progress",
        action="store_true",
        help="Disable per-worker progress bars during multiprocessing evaluation.",
    )

    args, _ = parser.parse_known_args()

    allowed_labels = LABEL_PROFILES[args.profile]
    print(f"Evaluating on dataset '{args.dataset}' with label profile '{args.profile}' \
          (allowed labels: {', '.join(sorted(allowed_labels))})")
    print(f"Using worker processes: {args.workers}")

    dataset, comparable_golds = load_cached_dataset(args.dataset, allowed_labels)

    shared_cache_payload = {
        "dataset": args.dataset,
        "comparable_golds": comparable_golds,
        "max_errors": args.max_errors,
        "allowed_labels": allowed_labels,
        "commit_hash": get_git_commit_hash('.'),
    }
    run_cache_name = cache_file_from_payload(__file__, shared_cache_payload).stem
    print(f"Shared run cache key: {run_cache_name}")

    # Plot and save label distribution for the evaluation split
    _dist_path = __import__('pathlib').Path(__file__).parent / ".cache" / \
        f"{args.dataset.replace('/', '_')}_{args.profile}_label_distribution.png"
    plot_label_distribution(comparable_golds, args.dataset, args.profile, _dist_path)
    print(f"Label distribution saved to: {_dist_path}")

    settings_results = [
        {"metrics": {}, "detection_func": spacy_detection, "spacy_model_name": "en_core_web_sm"},
        {"metrics": {}, "detection_func": spacy_detection, "spacy_model_name": "en_core_web_md"},
        {"metrics": {}, "detection_func": spacy_detection, "spacy_model_name": "en_core_web_lg"},
        {"metrics": {}, "detection_func": presidio_anonymization_detection, "spacy_model_name": "en_core_web_lg"},
        {"metrics": {}, "detection_func": taivium_detection, "spacy_model_name": "en_core_web_sm"},
        {"metrics": {}, "detection_func": taivium_detection, "spacy_model_name": "en_core_web_md"},
        {"metrics": {}, "detection_func": taivium_detection, "spacy_model_name": "en_core_web_lg"},
    ]

    for _, detection_settings_result in tqdm.tqdm(
        enumerate(settings_results), total=len(settings_results)
    ):
        detection = detection_settings_result["detection_func"]
        model_name = detection_settings_result["spacy_model_name"]
        metrics, errors, cache_file, total_time, n_samples = evaluation(
            detection,
            dataset,
            comparable_golds,
            allowed_labels,
            args.max_errors,
            model_name=model_name,
            shared_cache_name=run_cache_name,
            workers=args.workers,
            show_worker_progress=not args.no_worker_progress,
        )
        detection_settings_result["metrics"] = metrics
        detection_settings_result["errors"] = errors
        detection_settings_result["cache_file"] = cache_file
        detection_settings_result["run_cache_name"] = run_cache_name
        detection_settings_result["total_time"] = total_time
        detection_settings_result["n_samples"] = n_samples
        print(f"Results for {detection.__name__} (cache: {cache_file}):")
        print("Precision:", metrics["precision"])
        print("Recall:", metrics["recall"])

        # Save evaluation results (reports, metrics, errors)
        save_evaluation_results(
            cache_file,
            detection.__name__,
            args.dataset,
            args.profile,
            allowed_labels,
            model_name,
            metrics,
            errors,
        )

    # Print delta matrices
    print_delta_matrix_tables(settings_results)

    # Print timing summary
    print_timing_summary(settings_results)

    # Save delta matrices to JSON and text formats
    if settings_results:
        first_cache_file = settings_results[0]["cache_file"]
        delta_json_path, delta_txt_path = save_delta_matrices(first_cache_file, settings_results)
        print(f"\nDelta matrices saved to: {delta_json_path}")
        print(f"Delta matrices saved to: {delta_txt_path}")


if __name__ == "__main__":
    main()

