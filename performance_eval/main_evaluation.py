'''Main evaluation script for running entity detection evaluations on specified datasets and label profiles.
This script loads the specified dataset and label profile, runs evaluations using both spaCy and Taivium, and saves the results and error samples. It also computes and prints delta matrices comparing the models'''
import argparse
import os
import sys

import tqdm
# Ensure project root is importable when run as a script (e.g., via VS Code debugger)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from performance_eval.utility import print_delta_matrix_tables, \
                                    save_evaluation_results, save_delta_matrices, \
                                    plot_label_distribution, print_timing_summary
from performance_eval.eval_datasets import DATASET_LIST, load_cached_dataset, LABEL_PROFILES
from performance_eval.eval_reference import spacy_detection, taivium_detection, \
    presidio_anonymization_detection, evaluation

parser = argparse.ArgumentParser(
    description="Evaluate framework, choose dataset and label profile and error saving.")
parser.add_argument(
    "--profile",
    choices=["privacy", "conll_ner", "privacy_no_org"],
    default="privacy_no_org",
    help="Label profile for comparison: conll_ner \
        (PERSON/ORG/LOCATION) or privacy (adds EMAIL/PHONE/API_KEY).",
)
parser.add_argument(
    "--dataset",
    choices=list(DATASET_LIST.keys()),
    default=list(DATASET_LIST.keys())[0],
    help="Dataset for evaluation:" + ", ".join([f"{name} ({info['source']})" \
                                                for name, info in DATASET_LIST.items()]),
)
parser.add_argument(
    "--max-errors",
    type=int,
    default=200,
    help="Maximum number of wrong-detection text samples to save per model.",
)

args, _ = parser.parse_known_args()

allowed_labels = LABEL_PROFILES[args.profile]
print(f"Evaluating on dataset '{args.dataset}' with label profile '{args.profile}' \
      (allowed labels: {', '.join(sorted(allowed_labels))})")

dataset, ner_tag_names, comparable_golds = load_cached_dataset(args.dataset, allowed_labels)

# Plot and save label distribution for the evaluation split
_dist_path = __import__('pathlib').Path(__file__).parent / ".cache" / \
    f"{args.dataset.replace('/', '_')}_{args.profile}_label_distribution.png"
plot_label_distribution(comparable_golds, args.dataset, args.profile, _dist_path)
print(f"Label distribution saved to: {_dist_path}")

settings_results = \
    [
        {"metrics":{},"detection_func": spacy_detection, "spacy_model_name": "en_core_web_sm"},
        {"metrics":{},"detection_func": spacy_detection, "spacy_model_name": "en_core_web_md"},
        {"metrics":{},"detection_func": spacy_detection, "spacy_model_name": "en_core_web_lg"},
        {"metrics":{},"detection_func": presidio_anonymization_detection, "spacy_model_name": "en_core_web_lg"},
        {"metrics":{},"detection_func": taivium_detection, "spacy_model_name": "en_core_web_sm"},
        {"metrics":{},"detection_func": taivium_detection, "spacy_model_name": "en_core_web_md"},
        {"metrics":{},"detection_func": taivium_detection, "spacy_model_name": "en_core_web_lg"},
    ]

for i, detection_settings_result in tqdm.tqdm(enumerate(settings_results), 
                                              total=len(settings_results)):
    detection = detection_settings_result["detection_func"]
    model_name = detection_settings_result["spacy_model_name"]
    metrics, errors, cache_file, total_time, n_samples = evaluation(
        detection, dataset, comparable_golds,
        allowed_labels, args.max_errors,
        model_name=model_name)
    detection_settings_result["metrics"] = metrics
    detection_settings_result["errors"] = errors
    detection_settings_result["cache_file"] = cache_file
    detection_settings_result["total_time"] = total_time
    detection_settings_result["n_samples"] = n_samples
    print(f"Results for {detection.__name__} (cache: {cache_file}):")
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])

    # Save evaluation results (reports, metrics, errors)
    save_evaluation_results(cache_file, detection.__name__, args.dataset, args.profile,
                           allowed_labels, model_name, metrics, errors)

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

