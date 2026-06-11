#!/usr/bin/env python
"""Validate label mappings in ai4privacy dataset."""
import logging
from collections import Counter
from pathlib import Path
from datasets import load_dataset

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from performance_eval.eval_datasets import PRIVACY_LABEL_MAP, _map_raw_label, LABEL_PROFILES

logging.basicConfig(level=logging.WARNING)

def validate_labels():
    """Load dataset and show before/after label mapping."""
    print("Loading ai4privacy/pii-masking-300k dataset...")
    ds = load_dataset("ai4privacy/pii-masking-300k")
    split = ds["validation"]
    
    raw_labels = Counter()
    mapped_labels = Counter()
    mapping_examples = {}
    
    # Collect all labels
    for example in split:
        if isinstance(example.get("privacy_mask"), list):
            for item in example["privacy_mask"]:
                try:
                    raw_label = str(item.get("label", "")).upper().strip()
                    if raw_label:
                        raw_labels[raw_label] += 1
                        mapped = _map_raw_label(raw_label)
                        mapped_labels[mapped] += 1
                        
                        # Store first example of each mapping
                        if raw_label not in mapping_examples:
                            mapping_examples[raw_label] = mapped
                except (KeyError, TypeError, ValueError):
                    continue
    
    print("\n" + "="*80)
    print("LABEL MAPPING VALIDATION: ai4privacy/pii-masking-300k (validation split)")
    print("="*80)
    
    print("\n📊 BEFORE MAPPING (Raw Labels in Dataset):")
    print("-" * 80)
    total_raw = sum(raw_labels.values())
    for label in sorted(raw_labels.keys()):
        count = raw_labels[label]
        pct = 100 * count / total_raw
        print(f"  {label:20s}  {count:6d}  ({pct:5.2f}%)")
    print(f"  {'TOTAL':20s}  {total_raw:6d}  (100.00%)")
    
    print("\n📊 AFTER MAPPING (Canonical Labels):")
    print("-" * 80)
    total_mapped = sum(mapped_labels.values())
    for label in sorted(mapped_labels.keys()):
        count = mapped_labels[label]
        pct = 100 * count / total_mapped
        print(f"  {label:20s}  {count:6d}  ({pct:5.2f}%)")
    print(f"  {'TOTAL':20s}  {total_mapped:6d}  (100.00%)")
    
    print("\n🔗 MAPPING RULES APPLIED:")
    print("-" * 80)
    
    # Group by target label
    mapping_by_target = {}
    for raw_label in sorted(raw_labels.keys()):
        mapped = mapping_examples[raw_label]
        if mapped not in mapping_by_target:
            mapping_by_target[mapped] = []
        mapping_by_target[mapped].append(raw_label)
    
    for target in sorted(mapping_by_target.keys()):
        sources = mapping_by_target[target]
        print(f"\n  {target}:")
        for source in sorted(sources):
            count = raw_labels[source]
            total = sum(raw_labels.values())
            pct = 100 * count / total
            in_map = "✓ explicit" if source in PRIVACY_LABEL_MAP else "⚡ fallback"
            print(f"    {source:25s}  {count:6d}  ({pct:5.2f}%)  [{in_map}]")
    
    # Check for unknowns
    unknown_count = mapped_labels.get("UNKNOWN", 0)
    if unknown_count > 0:
        print(f"\n⚠️  UNKNOWN labels: {unknown_count} (check mapping!)")
    else:
        print(f"\n✅ No UNKNOWN labels found")
    
    print("\n" + "="*80)
    print(f"Profile: privacy")
    print(f"Expected labels: {LABEL_PROFILES['privacy']}")
    print(f"Actual mapped labels: {set(mapped_labels.keys())}")
    print("="*80 + "\n")

if __name__ == "__main__":
    validate_labels()
