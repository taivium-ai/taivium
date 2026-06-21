'''Main evaluation script for entity detection benchmarks.

This script loads the selected dataset and label profile, runs Taivium
evaluations with long-text context backends (gliner, openai-privacy-filter),
saves metrics and error samples, and computes delta matrices.
'''
import argparse
import datetime as dt
import importlib
import json
import os
import sys
from pathlib import Path

import tqdm


# Ensure project root is importable when run as a script (e.g., via VS Code debugger)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure local src tree is importable before importing evaluation modules.
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _assert_local_taivium_import() -> None:
    """Fail fast if taivium resolves outside local src/ tree."""
    mod = importlib.import_module("taivium")
    mod_file = Path(getattr(mod, "__file__", "")).resolve()
    expected_root = (Path(SRC_ROOT) / "taivium").resolve()
    if expected_root not in mod_file.parents:
        raise RuntimeError(
            "main_evaluation.py is not using local source tree. "
            f"Resolved taivium from: {mod_file}; expected under: {expected_root}"
        )


_assert_local_taivium_import()

# Imports intentionally come after sys.path bootstrapping above.
# pylint: disable=wrong-import-position
from performance_eval.utility import print_delta_matrix_tables, \
                                    save_evaluation_results, save_delta_matrices, \
                                    plot_label_distribution, print_timing_summary, \
                                    cache_file_from_payload, get_git_commit_hash, \
                                    load_dotenv, configure_logging_from_env, \
                                    get_label_distribution_path, \
                                    update_performance_history, \
                                    print_performance_trend, \
                                    get_performance_trend_plot_path, \
                                    save_performance_trend_plot

load_dotenv(Path(PROJECT_ROOT) / ".env")
from performance_eval.eval_datasets import DATASET_LIST, load_cached_dataset, LABEL_PROFILES
from performance_eval.eval_reference import taivium_detection, evaluation

_BACKEND_ALIASES = {
    "openai": "openai-privacy-filter",
    "openai_privacy_filter": "openai-privacy-filter",
    "openai-privacy-filter": "openai-privacy-filter",
}


def _resolve_backend_from_settings(result: dict) -> str | None:
    """Resolve long-text backend label from settings."""
    backend = result.get("long_text_backend")
    if backend is not None:
        return str(backend)
    return None


# pylint: disable-next=too-many-locals
def _save_latest_report_json(
    project_root: Path,
    dataset: str,
    profile: str,
    commit_hash: str,
    settings_results: list[dict],
) -> Path:
    """Save one canonical performance report JSON per run.

    The file is overwritten each run and meant to be tracked by git.
    """
    results_dir = project_root / "performance_eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    latest_report_path = results_dir / "latest_report.json"

    models: dict[str, dict[str, float | str | None]] = {}
    for result in settings_results:
        detection_func = result.get("detection_func")
        detector = detection_func.__name__ if callable(detection_func) else "unknown_detection"
        short_text_backend = result.get("short_text_backend", "unknown")
        backend = _resolve_backend_from_settings(result)
        route_suffix = ""
        if backend is not None:
            route_suffix = f", context_ner={backend}"
        label = f"{detector} ({short_text_backend}{route_suffix})"
        metrics = result.get("metrics", {})
        total_time = result.get("total_time")
        n_samples = result.get("n_samples")

        timing_ms: float | None = None
        if isinstance(total_time, (int, float)) and isinstance(n_samples, int) and n_samples > 0:
            timing_ms = (float(total_time) / n_samples) * 1000.0

        models[label] = {
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "timing_ms": timing_ms,
            "long_text_backend": backend,
        }

    report_payload = {
        "timestamp": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset,
        "profile": profile,
        "commit_hash": commit_hash,
        "hardware": {
            "device": "MacBook Pro",
            "chip": "Apple M2 Pro",
            "cores": 10,
        },
        "models": models,
    }
    with open(latest_report_path, "w", encoding="utf-8") as latest_report_file:
        json.dump(report_payload, latest_report_file, indent=2)

    return latest_report_path


def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    """Parse CLI args, run evaluation backends, and persist reports/artifacts."""
    configure_logging_from_env()

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
        default=1,#max(1, (os.cpu_count() or 2) - 1), # For mac os using onnx runtime using gpu
        help="Number of worker processes for evaluation. Use 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--no-worker-progress",
        action="store_true",
        help="Disable per-worker progress bars during multiprocessing evaluation.",
    )
    parser.add_argument(
        "--no-trend",
        action="store_true",
        help="Disable performance history updates and trend chart generation.",
    )
    parser.add_argument(
        "--context-ner-backends",
        default="gliner,openai-privacy-filter",
        help=(
            "Comma-separated backends for Taivium long-text route "
            "(for example: gliner,openai-privacy-filter). "
            "Short-text detection always uses spaCy."
        ),
    )
    parser.add_argument(
        "--short-test-run",
        action="store_true",
        help=(
            "Run a short smoke evaluation without saving reports, plots, "
            "history, or cache results."
        ),
    )
    parser.add_argument(
        "--short-test-samples",
        type=int,
        default=50,
        help="Number of samples to use with --short-test-run (default: 50).",
    )

    args, _ = parser.parse_known_args()

    allowed_labels = LABEL_PROFILES[args.profile]
    dataset_source = DATASET_LIST[args.dataset]["source"]
    print(f"Evaluating on dataset '{args.dataset}' with label profile '{args.profile}' \
          (allowed labels: {', '.join(sorted(allowed_labels))})")
    print(f"Dataset used: {args.dataset} (source: {dataset_source})")
    print(f"Using worker processes: {args.workers}")

    dataset, comparable_golds = load_cached_dataset(args.dataset, allowed_labels)
    if args.short_test_run:
        sample_limit = max(1, int(args.short_test_samples))
        comparable_golds = comparable_golds[:sample_limit]
        print(
            "Short test run enabled: using "
            f"{len(comparable_golds)} samples and skipping saved artifacts."
        )

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
    if not args.short_test_run:
        _dist_path = get_label_distribution_path(__file__, args.dataset, args.profile)
        plot_label_distribution(comparable_golds, args.dataset, args.profile, _dist_path)
        print(f"Label distribution saved to: {_dist_path}")

    backend_options = [
        _BACKEND_ALIASES.get(item.strip().lower(), item.strip().lower())
        for item in str(args.context_ner_backends).split(",")
        if item.strip()
    ]
    supported_backends = {"gliner", "openai-privacy-filter"}
    invalid_backends = [opt for opt in backend_options if opt not in supported_backends]
    if invalid_backends:
        raise ValueError(
            "Unsupported --context-ner-backends values: "
            f"{invalid_backends}. Supported values: {sorted(supported_backends)}"
        )
    if not backend_options:
        raise ValueError("--context-ner-backends must include at least one backend")

    settings_results = [
        {
            "metrics": {},
            "detection_func": taivium_detection,
            "short_text_backend": "en_core_web_sm",
            "long_text_backend": backend,
        }
        for backend in backend_options
    ]

    # Determine which detections to run
    detections_to_run = settings_results

    for _, detection_settings_result in tqdm.tqdm(
        enumerate(detections_to_run), total=len(detections_to_run)
    ):
        detection = detection_settings_result["detection_func"]
        short_text_backend = detection_settings_result["short_text_backend"]
        long_text_backend = detection_settings_result.get("long_text_backend")
        metrics, errors, cache_file, total_time, n_samples = evaluation(
            detection,
            dataset,
            comparable_golds,
            allowed_labels,
            args.max_errors,
            short_text_backend=short_text_backend,
            long_text_backend=long_text_backend,
            shared_cache_name=run_cache_name,
            use_cache=not args.short_test_run,
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
        if not args.short_test_run:
            save_evaluation_results(
                cache_file,
                detection.__name__,
                args.dataset,
                args.profile,
                allowed_labels,
                short_text_backend,
                metrics,
                errors,
            )

    # Print delta matrices
    print_delta_matrix_tables(settings_results)

    # Print timing summary
    print_timing_summary(settings_results)

    if not args.short_test_run:
        latest_report_path = _save_latest_report_json(
            Path(PROJECT_ROOT),
            args.dataset,
            args.profile,
            shared_cache_payload["commit_hash"],
            settings_results,
        )
        print(f"Latest report saved: {latest_report_path}")

    if args.short_test_run:
        print("Performance trend recording skipped in short test run")
    elif args.no_trend:
        print("Performance trend recording disabled via --no-trend")
    else:
        # Update persistent history and print trend deltas vs previous run.
        history_path, history = update_performance_history(
            __file__, args.dataset, args.profile, settings_results
        )
        print(f"Performance history updated: {history_path}")
        print_performance_trend(history)

        trend_plot_path = get_performance_trend_plot_path(__file__, args.dataset, args.profile)
        if save_performance_trend_plot(history, trend_plot_path):
            print(f"Performance trend plot saved to: {trend_plot_path}")


    # Save delta matrices to JSON and text formats
    if settings_results and not args.short_test_run:
        first_cache_file = settings_results[0]["cache_file"]
        delta_json_path, delta_txt_path = save_delta_matrices(first_cache_file, settings_results)
        print(f"\nDelta matrices saved to: {delta_json_path}")
        print(f"Delta matrices saved to: {delta_txt_path}")


if __name__ == "__main__":
    main()
