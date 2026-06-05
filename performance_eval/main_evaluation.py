'''Main evaluation script for running entity detection evaluations on specified datasets and label profiles.
This script loads the specified dataset and label profile, runs evaluations using both spaCy and Taivium, and saves the results and error samples. It also computes and prints delta matrices comparing the models'''
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import tqdm


# Ensure project root is importable when run as a script (e.g., via VS Code debugger)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
from performance_eval.eval_reference import taivium_detection, \
    presidio_detection, evaluation
from performance_eval.generate_report_html import generate_html


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

    models: dict[str, dict[str, float | None]] = {}
    for result in settings_results:
        detector = result.get("detection_func").__name__
        model = result.get("spacy_model_name", "unknown")
        label = f"{detector} ({model})"
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


def main() -> None:
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
        "--skip-baselines",
        action="store_true",
        default=False,
        help="Skip spaCy and Presidio evaluation; reuse cached results from previous run.",
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
    _dist_path = get_label_distribution_path(__file__, args.dataset, args.profile)
    plot_label_distribution(comparable_golds, args.dataset, args.profile, _dist_path)
    print(f"Label distribution saved to: {_dist_path}")

    settings_results = [
        # {"metrics": {}, "detection_func": presidio_detection, "spacy_model_name": "en_core_web_lg"},
        {"metrics": {}, "detection_func": taivium_detection, "spacy_model_name": "en_core_web_sm"},
        # {"metrics": {}, "detection_func": taivium_detection, "spacy_model_name": "en_core_web_lg"},
    ]

    # Determine which detections to run
    detections_to_run = settings_results
    if args.skip_baselines:
        print("\n[--skip-baselines] Loading cached Presidio and Taivium(sm) results...")
        cache_dir = Path(__file__).parent / ".cache"

        # Load cached metrics for the first two configured models.
        baseline_targets = [
            (0, "presidio_detection", "en_core_web_lg"),
            (1, "taivium_detection", "en_core_web_sm"),
        ]
        for i, detection_name, model_name in baseline_targets:
            pattern = f"*{detection_name}*{model_name.replace('/', '_')}*_{detection_name}_report.json"
            matches = list(cache_dir.glob(pattern))
            if not matches:
                # Fallback if model-specific naming differs.
                matches = list(cache_dir.glob(f"*{detection_name}*_report.json"))
            if matches:
                # Sort by modification time and use newest
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                report_file = matches[0]
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    metrics = report.get("metrics", {})
                    settings_results[i]["metrics"] = metrics
                    settings_results[i]["cache_file"] = report_file
                    settings_results[i]["total_time"] = 0  # Cached, no actual runtime
                    settings_results[i]["n_samples"] = 0
                    precision = metrics.get("precision", "N/A")
                    recall = metrics.get("recall", "N/A")
                    if isinstance(precision, (int, float)):
                        precision = f"{precision:.3f}"
                    if isinstance(recall, (int, float)):
                        recall = f"{recall:.3f}"
                    print(f"  ✓ Loaded cached {detection_name}: P={precision}, R={recall}")
                except (OSError, json.JSONDecodeError) as e:
                    print(f"  ⚠ Could not load cached results for {detection_name}: {e}")
            else:
                print(f"  ⚠ No cached results found for {detection_name}")

        # Only run Taivium detection
        detections_to_run = [settings_results[2]]
        print("  Running only: taivium_detection\n")
    
    for _, detection_settings_result in tqdm.tqdm(
        enumerate(detections_to_run), total=len(detections_to_run)
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

    latest_report_path = _save_latest_report_json(
        Path(PROJECT_ROOT),
        args.dataset,
        args.profile,
        shared_cache_payload["commit_hash"],
        settings_results,
    )
    print(f"Latest report saved: {latest_report_path}")

    if args.no_trend:
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
    if settings_results:
        first_cache_file = settings_results[0]["cache_file"]
        delta_json_path, delta_txt_path = save_delta_matrices(first_cache_file, settings_results)
        print(f"\nDelta matrices saved to: {delta_json_path}")
        print(f"Delta matrices saved to: {delta_txt_path}")

    # Generate HTML report
    print("\nGenerating HTML report...")
    try:
        cache_dir = Path(__file__).parent / ".cache"
        output_file = Path(__file__).parent.parent / "web" / "index.html"
        generate_html(cache_dir, output_file)
        print("HTML report generated successfully at: web/index.html")
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"Warning: Failed to generate HTML report: {e}")


if __name__ == "__main__":
    main()

