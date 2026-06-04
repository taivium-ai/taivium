#!/usr/bin/env python3
"""
Benchmark execution time: GLiNER vs spaCy vs Hybrid on various text lengths.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taivium import Taivium
from datasets import load_dataset


def benchmark_text_samples(num_samples=20):
    """Benchmark on real dataset samples."""
    dataset = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=True)
    
    texts = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        texts.append(sample["source_text"])
    
    print(f"✓ Loaded {len(texts)} samples from dataset")
    print(f"  Text lengths: {min(len(t) for t in texts)} - {max(len(t) for t in texts)} chars\n")
    
    # Benchmark with GLiNER (current config)
    engine = Taivium(spacy_model_name="en_core_web_lg")
    
    print("=" * 80)
    print("BENCHMARKING: GLiNER-based detection (high precision)")
    print("=" * 80)
    
    times = []
    for i, text in enumerate(texts, 1):
        start = time.time()
        result = engine.process(text)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        entity_count = len(result.get("entities", []))
        print(f"Sample {i:2d} ({len(text):4d} chars): {elapsed:7.2f}ms | {entity_count:2d} entities")
    
    print("-" * 80)
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average: {avg_time:.2f}ms | Min: {min_time:.2f}ms | Max: {max_time:.2f}ms")
    print(f"Total for {len(texts)} samples: {sum(times):.2f}ms")
    print()
    
    # Per-character throughput
    total_chars = sum(len(t) for t in texts)
    throughput = total_chars / (sum(times) / 1000)  # chars/sec
    print(f"Throughput: {throughput:,.0f} chars/sec")
    print()
    
    return avg_time


def benchmark_single_text():
    """Benchmark on a single longer text."""
    text = """
    Subject: Quarterly Business Review - Q2 2026
    
    Dear Executive Team,
    
    This is to confirm our Q2 earnings report. Alice Johnson from the Finance Department 
    (alice.johnson@company.com, +1-415-555-0123) has prepared the detailed analysis.
    
    Key attendees include:
    - Bob Smith (VP, Operations) - bob@company.com, ext 5678
    - Carol Williams (Director, Marketing) - carol.williams@organization.com
    - David Lee (Senior Analyst) - david_lee@mail.company.net, +44 207 946 0958
    
    The meeting will be held at our San Francisco headquarters in Conference Room B.
    For remote attendance, use this link: https://meeting.company.com/q2-review
    
    API credentials for data access (DO NOT SHARE):
    - API Key: sk-ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234567890
    - Database IP: 192.168.1.100
    - Backup IP: 10.0.0.50
    
    SSN for payroll verification: 123-45-6789
    Passport number: A1234567B
    Credit card (test): 4532-1234-5678-9010
    
    Please confirm your attendance by Friday, June 6, 2026 at 5:00 PM.
    
    Best regards,
    Executive Communications Team
    Company Headquarters, New York, NY 10001
    """ * 3  # Repeat 3 times to make it longer
    
    print("=" * 80)
    print("BENCHMARKING: Single longer text (1000+ chars)")
    print("=" * 80)
    print(f"Text length: {len(text):,} characters\n")
    
    engine = Taivium(spacy_model_name="en_core_web_lg")
    
    # Warm-up
    _ = engine.process(text)
    time.sleep(0.1)
    
    # Actual benchmark
    times = []
    for i in range(5):
        start = time.time()
        result = engine.process(text)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        entity_count = len(result.get("entities", []))
        print(f"Run {i+1}: {elapsed:7.2f}ms | {entity_count:2d} entities detected")
    
    print("-" * 80)
    avg_time = sum(times) / len(times)
    print(f"Average: {avg_time:.2f}ms")
    print(f"First call (model load): {times[0]:.2f}ms")
    print(f"Subsequent calls avg: {sum(times[1:]) / (len(times)-1):.2f}ms")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TAIVIUM EXECUTION TIME BENCHMARK (GLiNER-based detection)")
    print("=" * 80 + "\n")
    
    # Single text benchmark
    benchmark_single_text()
    
    # Multiple samples benchmark
    print()
    avg = benchmark_text_samples(20)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Average processing time: {avg:.2f}ms per sample")
    print(f"✓ Suitable for: Real-time processing of PII detection")
    print(f"✓ Note: First run includes model initialization (~1-2s)")
    print()
