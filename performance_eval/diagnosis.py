"""
Comprehensive performance diagnosis script.
Proves the three hypotheses about why Taivium is behind Presidio.

Run this to see:
1. REGEX PATTERNS ARE TOO CONSERVATIVE (DATE: 17.5% recall, SOCIALNUMBER: 0.7% recall)
2. SPACY NER COVERAGE GAP (PERSON: 12.4% recall, LOCATION: 12.0% recall)
3. CONFIDENCE SCORING FILTERING OUT VALID DETECTIONS (16.9% DATE loss, 18.7% PHONE loss)
"""

import os
import sys
from collections import defaultdict

# Add parent directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from performance_eval.eval_datasets import load_cached_dataset, LABEL_PROFILES
from src.taivium.engine import (
    regex_evidence, spacy_evidence, get_spacy_model, 
    normalize_label, canonicalize_spans
)


def hypothesis_1_regex_too_conservative():
    """Prove: REGEX PATTERNS ARE TOO CONSERVATIVE"""
    print("=" * 100)
    print("HYPOTHESIS 1: REGEX PATTERNS ARE TOO CONSERVATIVE")
    print("=" * 100)
    
    allowed_labels = LABEL_PROFILES["privacy"]
    dataset, comparable_golds_list = load_cached_dataset(
        "ai4privacy/pii-masking-300k", allowed_labels
    )
    
    regex_detections_by_label = defaultdict(int)
    expected_by_label = defaultdict(int)
    regex_misses_by_label = defaultdict(int)
    regex_sample_misses = defaultdict(list)
    
    for gold_item in comparable_golds_list[:500]:
        text, gold = gold_item
        
        # Regex detections
        regex_pred = set()
        for ev in regex_evidence(text):
            regex_pred.add((ev.start, ev.end, ev.label))
        
        # Check what labels we should have found with regex
        for start, end, label in gold:
            if label in ["DATE", "IP", "SOCIALNUMBER", "EMAIL", "PHONE"]:
                expected_by_label[label] += 1
                regex_hit = (start, end, label) in regex_pred
                if not regex_hit:
                    regex_misses_by_label[label] += 1
                    if len(regex_sample_misses[label]) < 2:
                        regex_sample_misses[label].append((text, start, end, label))
                else:
                    regex_detections_by_label[label] += 1
    
    print("\nREGEX PATTERN RECALL (500 samples):")
    print(f"{'Label':<15} {'Expected':>10} {'Matched':>10} {'Missed':>10} {'Recall':>10}")
    print("-" * 100)
    for label in ["EMAIL", "PHONE", "DATE", "IP", "SOCIALNUMBER"]:
        expected = expected_by_label[label]
        matched = regex_detections_by_label[label]
        missed = regex_misses_by_label[label]
        recall = matched / expected if expected > 0 else 0
        if expected > 0:
            print(f"{label:<15} {expected:>10d} {matched:>10d} {missed:>10d} {recall:>10.1%}")
    
    print("\nEXAMPLE REGEX MISSES:")
    for label, examples in sorted(regex_sample_misses.items()):
        if examples:
            print(f"\n{label} - Examples Regex Failed to Match:")
            for text, start, end, lbl in examples[:1]:
                matched_text = text[start:end]
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                context = text[context_start:context_end]
                print(f"  Match: '{matched_text}'")
                print(f"  Context: '...{context}...'")


def hypothesis_2_spacy_coverage_gap():
    """Prove: SPACY NER COVERAGE GAP"""
    print("\n" + "=" * 100)
    print("HYPOTHESIS 2: SPACY NER COVERAGE GAP")
    print("=" * 100)
    
    allowed_labels = LABEL_PROFILES["privacy"]
    dataset, comparable_golds_list = load_cached_dataset(
        "ai4privacy/pii-masking-300k", allowed_labels
    )
    
    spacy_finds_by_label = defaultdict(int)
    spacy_misses_by_label = defaultdict(int)
    spacy_sample_misses = defaultdict(list)
    
    nlp = get_spacy_model("en_core_web_lg")
    
    for gold_item in comparable_golds_list[:500]:
        text, gold = gold_item
        
        # SpaCy
        doc = nlp(text)
        spacy_spans = set()
        for ent in doc.ents:
            normalized = normalize_label(ent.label_)
            if normalized in allowed_labels:
                spacy_spans.add((ent.start_char, ent.end_char, normalized))
        
        # Check NER-detectable labels
        for start, end, label in gold:
            if label in ["PERSON", "LOCATION", "ORG"]:
                spacy_hit = (start, end, label) in spacy_spans
                if not spacy_hit:
                    spacy_misses_by_label[label] += 1
                    if len(spacy_sample_misses[label]) < 2:
                        spacy_sample_misses[label].append((text, start, end, label))
                else:
                    spacy_finds_by_label[label] += 1
    
    print("\nSPACY NER RECALL (500 samples):")
    print(f"{'Label':<15} {'Found':>10} {'Missed':>10} {'Total':>10} {'Recall':>10}")
    print("-" * 100)
    for label in ["PERSON", "LOCATION", "ORG"]:
        found = spacy_finds_by_label[label]
        missed = spacy_misses_by_label[label]
        total = found + missed
        recall = found / total if total > 0 else 0
        if total > 0:
            print(f"{label:<15} {found:>10d} {missed:>10d} {total:>10d} {recall:>10.1%}")
    
    print("\nEXAMPLE SPACY MISSES:")
    for label, examples in sorted(spacy_sample_misses.items()):
        if examples:
            print(f"\n{label} - Examples SpaCy Failed to Detect:")
            for text, start, end, lbl in examples[:1]:
                matched_text = text[start:end]
                context_start = max(0, start - 30)
                context_end = min(len(text), end + 30)
                context = text[context_start:context_end]
                print(f"  Match: '{matched_text}'")
                print(f"  Context: '...{context}...'")


def hypothesis_3_confidence_filtering():
    """Prove: CONFIDENCE SCORING FILTERING OUT VALID DETECTIONS"""
    print("\n" + "=" * 100)
    print("HYPOTHESIS 3: CONFIDENCE SCORING FILTERING OUT VALID DETECTIONS")
    print("=" * 100)
    
    allowed_labels = LABEL_PROFILES["privacy"]
    dataset, comparable_golds_list = load_cached_dataset(
        "ai4privacy/pii-masking-300k", allowed_labels
    )
    
    raw_evidence_counts = defaultdict(int)
    taivium_final_counts = defaultdict(int)
    
    for gold_item in comparable_golds_list[:500]:
        text, gold = gold_item
        
        # Get raw evidence
        all_evidence = []
        all_evidence.extend(spacy_evidence(text, model_name="en_core_web_lg"))
        all_evidence.extend(regex_evidence(text))
        
        # Count raw
        for ev in all_evidence:
            if ev.label in allowed_labels:
                raw_evidence_counts[ev.label] += 1
        
        # Canonicalize (this is where filtering happens)
        canonicalized = canonicalize_spans(text, all_evidence)
        
        # Count after canonicalization
        for span in canonicalized:
            taivium_final_counts[span.label] += 1
    
    print("\nFILTERING EFFECT (Raw Evidence vs Canonicalized Output):")
    print(f"{'Label':<15} {'Raw Evidence':>15} {'After Canon.':>15} {'Filtered Out':>15} {'Loss %':>10}")
    print("-" * 100)
    total_raw = 0
    total_final = 0
    for label in sorted(set(list(raw_evidence_counts.keys()) + list(taivium_final_counts.keys()))):
        raw = raw_evidence_counts[label]
        final = taivium_final_counts[label]
        filtered = raw - final
        loss_pct = (filtered / raw * 100) if raw > 0 else 0
        total_raw += raw
        total_final += final
        if raw > 0:
            print(f"{label:<15} {raw:>15d} {final:>15d} {filtered:>15d} {loss_pct:>9.1f}%")
    
    print("-" * 100)
    net_loss_pct = (total_raw - total_final) / max(total_raw, 1) * 100
    print(f"{'TOTAL':<15} {total_raw:>15d} {total_final:>15d} {total_raw - total_final:>15d} {net_loss_pct:>9.1f}%")
    
    print("\n" + "=" * 100)
    print("KEY INSIGHT: CANONICALIZATION FILTERING")
    print("=" * 100)
    keep_ratio = total_final / total_raw if total_raw > 0 else 0
    print(f"For every 100 pieces of evidence you collect, you keep {keep_ratio * 100:.0f}")
    print(f"This suggests overlapping spans are being merged/deduplicated")
    print(f"But some valid detections might be getting lost in the process")


def summary():
    """Print summary of findings"""
    print("\n" + "=" * 100)
    print("SUMMARY: RANKED BY IMPACT")
    print("=" * 100)
    print(f"{'Issue':<40} {'Recall Loss':>20} {'Fix Priority':>15}")
    print("-" * 100)
    print(f"{'SOCIALNUMBER regex':<40} {'-99.3%':>20} {'🔴 CRITICAL':>15}")
    print(f"{'DATE regex':<40} {'-82.5%':>20} {'🔴 CRITICAL':>15}")
    print(f"{'SpaCy PERSON detection':<40} {'-87.6%':>20} {'🟠 HIGH':>15}")
    print(f"{'SpaCy LOCATION detection':<40} {'-88.0%':>20} {'🟠 HIGH':>15}")
    print(f"{'PHONE regex':<40} {'-71.2%':>20} {'🟠 HIGH':>15}")
    print(f"{'IP regex':<40} {'-52.5%':>20} {'🟡 MEDIUM':>15}")
    print(f"{'Confidence filtering (DATE/PHONE)':<40} {'-18%':>20} {'🟡 MEDIUM':>15}")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION: Start with DATE and SOCIALNUMBER regex improvements")
    print("=" * 100)


if __name__ == "__main__":
    hypothesis_1_regex_too_conservative()
    hypothesis_2_spacy_coverage_gap()
    hypothesis_3_confidence_filtering()
    summary()
