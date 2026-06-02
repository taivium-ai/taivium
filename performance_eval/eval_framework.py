import argparse
import os
import sys
# Ensure project root is importable when run as a script (e.g., via VS Code debugger)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from performance_eval.utility import save_results, calculate_delta
from performance_eval.eval_datasets import DATASET_LIST, load_cached_dataset, LABEL_PROFILES
from performance_eval.eval_reference import spacy_evaluation
from performance_eval.eval_taivium import taivium_evaluation

parser = argparse.ArgumentParser(
    description="Evaluate framework, choose dataset and label profile and error saving.")
parser.add_argument(
    "--profile",
    choices=["privacy", "conll_ner"],
    default="privacy",
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

parser.add_argument(
    "--spacy-model-name",
    type=str,
    default="en_core_web_lg",
    help="spaCy model name for evaluation.",
)

parser.add_argument(
    "--taivium-spacy-model-name",
    type=str,
    default="en_core_web_lg",
    help="spaCy model in Taivium for evaluation.",
)

args, _ = parser.parse_known_args()

allowed_labels = LABEL_PROFILES[args.profile]

dataset, ner_tag_names, comparable_golds = load_cached_dataset(args.dataset, allowed_labels)
spacy_metrics, spacy_errors, spacy_cache_file = spacy_evaluation(dataset, comparable_golds,
                                               allowed_labels, args.max_errors,
                                               model_name=args.spacy_model_name)

# Evaluate spaCy baseline
print(f"\nProfile: {args.profile}")
print("Labels:", ", ".join(sorted(allowed_labels)))
print("\n=== spaCy Baseline Performance ===")
print("Precision:", spacy_metrics["precision"])
print("Recall:", spacy_metrics["recall"])
print("F1:", spacy_metrics["f1"])

# --- Taivium span-based evaluation ---
# Evaluate Taivium pipeline
taivium_metrics, taivium_errors, taivium_cache_file = taivium_evaluation(dataset,
                                                                         comparable_golds,
                                                                         allowed_labels,
                     args.max_errors, model_name=args.taivium_spacy_model_name)
print("\n=== Taivium Pipeline Performance ===")
print("Precision:", taivium_metrics["precision"])
print("Recall:", taivium_metrics["recall"])
print("F1:", taivium_metrics["f1"])

delta = calculate_delta(taivium_metrics, spacy_metrics)
print("\n=== Delta (Taivium - spaCy) ===")
print("Precision:", delta["precision"])
print("Recall:", delta["recall"])
print("F1:", delta["f1"])

save_results(
    args.profile,
    allowed_labels,
    spacy_metrics,
    taivium_metrics,
    spacy_errors,
    taivium_errors,
    taivium_cache_file,
    args.spacy_model_name,
    args.taivium_spacy_model_name,
)
