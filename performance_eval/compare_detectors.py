#!/usr/bin/env python3
"""
Comparative analysis: GLiNER-only vs spaCy-only vs Hybrid detection.
Tests which detector performs better for PERSON, LOCATION, ORG.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

# Add parent dirs to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from taivium.engine import (
    collect_evidence, regex_evidence, spacy_evidence, gliner_evidence,
    canonicalize_spans, Evidence
)
from datasets import load_dataset

# Label mappings (copied from eval_datasets.py to avoid import issues)
PRIVACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "NAME": "PERSON",
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "BUILDING": "LOCATION",
    "POSTCODE": "LOCATION",
    "SECADDRESS": "LOCATION",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TEL": "PHONE",
    "MOBILE": "PHONE",
    "MOBILE_PHONE_NUMBER": "PHONE",
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "ACCESS_TOKEN": "API_KEY",
    "DATE": "DATE",
    "IP": "IP",
    "SOCIALNUMBER": "SOCIALNUMBER",
    "USERNAME": "USERNAME",
    "BOD": "DATE",
    "TIME": "DATE",
    "GEOCOORD": "LOCATION",
    "TITLE": "PERSON",
    "SEX": "PERSON",
    "PASSPORT": "SOCIALNUMBER",
    "IDCARD": "SOCIALNUMBER",
    "DRIVERLICENSE": "SOCIALNUMBER",
    "CARDISSUER": "SOCIALNUMBER",
    "PASS": "SOCIALNUMBER",
    "US_SSN": "SOCIALNUMBER",
    "CREDIT_CARD": "SOCIALNUMBER",
    "CRYPTO": "SOCIALNUMBER",
}

def _map_raw_label(raw_label: str) -> str:
    """Map heterogeneous dataset labels to the project label schema."""
    value = str(raw_label).strip().upper()
    if value in PRIVACY_LABEL_MAP:
        return PRIVACY_LABEL_MAP[value]
    if "EMAIL" in value:
        return "EMAIL"
    if "PHONE" in value or "MOBILE" in value:
        return "PHONE"
    if "API" in value and "KEY" in value:
        return "API_KEY"
    if "DATE" in value or "TIME" in value:
        return "DATE"
    if value == "IP" or "IP" in value:
        return "IP"
    return "UNKNOWN"

LABEL_PROFILES = {
    "privacy": {
        "PERSON",
        "ORG",
        "LOCATION",
        "EMAIL",
        "PHONE",
        "API_KEY",
        "DATE",
        "IP",
        "SOCIALNUMBER",
        "USERNAME"
    },
}


def evaluate_detector_config(texts, true_entities, config_name, evidence_fn):
    """Evaluate a specific detector configuration."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    start_time = time.time()
    
    for text, true_ents in zip(texts, true_entities):
        # Run detection
        evidence_list = evidence_fn(text)
        detected = canonicalize_spans(text, evidence_list)
        
        # Extract labels
        detected_labels = {(e.start, e.end, e.label) for e in detected}
        true_labels = {(e["start"], e["end"], e["label"]) for e in true_ents}
        
        # Compute metrics per label
        for label in {"PERSON", "LOCATION", "ORG", "EMAIL", "PHONE", "DATE", "IP", "SOCIALNUMBER"}:
            true_label_set = {(s, e) for s, e, l in true_labels if l == label}
            detected_label_set = {(s, e) for s, e, l in detected_labels if l == label}
            
            tp[label] += len(true_label_set & detected_label_set)
            fp[label] += len(detected_label_set - true_label_set)
            fn[label] += len(true_label_set - detected_label_set)
    
    elapsed = time.time() - start_time
    
    # Compute metrics
    results = {}
    for label in {"PERSON", "LOCATION", "ORG", "EMAIL", "PHONE", "DATE", "IP", "SOCIALNUMBER"}:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[label] = {
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    return results, elapsed


def main():
    """Main comparison runner."""
    print("=" * 100)
    print("DETECTOR COMPARISON: GLiNER-only vs spaCy-only vs Hybrid")
    print("=" * 100)
    
    # Load dataset
    print("\nLoading dataset (first 500 samples for speed)...")
    dataset = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
    
    texts = []
    all_entities = []
    allowed_labels = LABEL_PROFILES["privacy"]
    count = 0
    for sample in dataset:
        if count >= 500:
            break
        texts.append(sample["source_text"])
        # Convert privacy_mask format to entities with start, end, label (with mapping)
        entities = []
        for mask_item in sample["privacy_mask"]:
            raw_label = mask_item["label"]
            mapped_label = _map_raw_label(raw_label)
            # Only include mapped labels that are in our allowed set
            if mapped_label in allowed_labels:
                entities.append({
                    "start": mask_item["start"],
                    "end": mask_item["end"],
                    "label": mapped_label,
                })
        all_entities.append(entities)
        count += 1
    
    print(f"✓ Loaded {len(texts)} samples")
    
    # Define detector configurations
    configs = {
        "spacy_only": lambda text: regex_evidence(text) + spacy_evidence(text),
        "gliner_only": lambda text: regex_evidence(text) + gliner_evidence(text, targets=["PERSON", "LOCATION"]),
        "hybrid": lambda text: collect_evidence(text),  # Current implementation
    }
    
    results_by_config = {}
    
    for config_name, evidence_fn in configs.items():
        print(f"\n{'='*100}")
        print(f"Testing: {config_name.upper()}")
        print(f"{'='*100}")
        
        results, elapsed = evaluate_detector_config(texts, all_entities, config_name, evidence_fn)
        results_by_config[config_name] = results
        
        print(f"\nResults ({elapsed:.1f}s for {len(texts)} samples):\n")
        print(f"{'Label':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 100)
        
        for label in ["PERSON", "LOCATION", "ORG", "EMAIL", "PHONE", "DATE", "IP", "SOCIALNUMBER"]:
            r = results[label]
            marker = "✓" if r["f1"] > 0.5 else "✗"
            print(f"{marker} {label:<13} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
                  f"{r['precision']:>9.1%} {r['recall']:>9.1%} {r['f1']:>9.1%}")
        
        # Overall
        all_tp = sum(r["tp"] for r in results.values())
        all_fp = sum(r["fp"] for r in results.values())
        all_fn = sum(r["fn"] for r in results.values())
        overall_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        overall_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
                     if (overall_precision + overall_recall) > 0 else 0.0
        
        print("-" * 100)
        print(f"{'OVERALL':<15} {all_tp:>6} {all_fp:>6} {all_fn:>6} "
              f"{overall_precision:>9.1%} {overall_recall:>9.1%} {overall_f1:>9.1%}")
    
    # Comparison table
    print(f"\n\n{'='*100}")
    print("COMPARISON: F1 Score by Label")
    print(f"{'='*100}\n")
    
    print(f"{'Label':<15} {'spaCy-only':>15} {'GLiNER-only':>15} {'Hybrid':>15} {'Best':>15}")
    print("-" * 100)
    
    for label in ["PERSON", "LOCATION", "ORG", "EMAIL", "PHONE", "DATE", "IP", "SOCIALNUMBER"]:
        scores = {
            "spacy": results_by_config["spacy_only"][label]["f1"],
            "gliner": results_by_config["gliner_only"][label]["f1"],
            "hybrid": results_by_config["hybrid"][label]["f1"],
        }
        best_config = max(scores, key=scores.get)
        best_score = scores[best_config]
        
        print(f"{label:<15} {scores['spacy']:>14.1%} {scores['gliner']:>14.1%} "
              f"{scores['hybrid']:>14.1%} {best_config.upper():>14} ({best_score:.1%})")
    
    # Overall comparison
    print("-" * 100)
    spacy_f1 = sum(results_by_config["spacy_only"][l]["tp"] for l in results_by_config["spacy_only"]) / \
               (sum(results_by_config["spacy_only"][l]["tp"] + results_by_config["spacy_only"][l]["fp"] 
                   for l in results_by_config["spacy_only"]) or 1)
    gliner_f1 = sum(results_by_config["gliner_only"][l]["tp"] for l in results_by_config["gliner_only"]) / \
               (sum(results_by_config["gliner_only"][l]["tp"] + results_by_config["gliner_only"][l]["fp"] 
                   for l in results_by_config["gliner_only"]) or 1)
    hybrid_f1 = sum(results_by_config["hybrid"][l]["tp"] for l in results_by_config["hybrid"]) / \
               (sum(results_by_config["hybrid"][l]["tp"] + results_by_config["hybrid"][l]["fp"] 
                   for l in results_by_config["hybrid"]) or 1)
    
    best_overall = max(
        [("spacy", spacy_f1), ("gliner", gliner_f1), ("hybrid", hybrid_f1)],
        key=lambda x: x[1]
    )
    
    print(f"{'OVERALL':<15} {spacy_f1:>14.1%} {gliner_f1:>14.1%} "
          f"{hybrid_f1:>14.1%} {best_overall[0].upper():>14} ({best_overall[1]:.1%})")
    
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
