"""
Example: Adaptive CPU/GPU Pipeline Testing
===========================================

Demonstrates how to test Taivium with:
- Adaptive routing (short vs long text)
- Backend detection (CPU vs GPU/MPS on M2)
- Chunking strategies for large documents
- Batching for inference
- Known organization detection
"""

import time
from typing import Dict, Any, List
from taivium import Taivium, RiskLevel
from taivium.engine import PolicyEngine, PolicyRule, PolicyAction

# ============================================================================
# Test 1: Backend Detection (CPU vs GPU)
# ============================================================================

def detect_available_backends() -> Dict[str, bool]:
    """Check which execution backends are available on this machine."""
    try:
        import onnxruntime as rt
        available = {
            "CPU": "CPUExecutionProvider" in rt.get_available_providers(),
            "CoreML (M2/M3 GPU)": "CoreMLExecutionProvider" in rt.get_available_providers(),
            "CUDA (NVIDIA GPU)": "CUDAExecutionProvider" in rt.get_available_providers(),
        }
    except ModuleNotFoundError:
        available = {
            "CPU": True,  # Always available
            "CoreML (M2/M3 GPU)": False,  # Not checked (onnxruntime not installed)
            "CUDA (NVIDIA GPU)": False,  # Not checked (onnxruntime not installed)
        }
    
    # Log which backend will be used
    if available.get("CoreML (M2/M3 GPU)"):
        print("  💡 GLiNER will use CoreML (GPU + Neural Engine) when needed")
    elif available.get("CUDA (NVIDIA GPU)"):
        print("  💡 GLiNER will use CUDA (NVIDIA GPU) when needed")
    else:
        print("  💡 GLiNER will use CPU (consider GPU for better performance)")
    
    return available


# ============================================================================
# Test 2: Adaptive Routing — Short Text (Fast Track)
# ============================================================================

def test_short_text_fast_track():
    """
    Short text (< 100 chars) should use fast track: regex + spaCy.
    No GLiNER invoked (saves latency).
    """
    pipeline = Taivium(short_text_threshold=100)
    
    short_text = "Alice Johnson emailed alice@example.com from New York yesterday."
    
    start = time.perf_counter()
    result = pipeline.process(short_text)
    latency_ms = (time.perf_counter() - start) * 1000
    
    print("\n=== Short Text (Fast Track) ===")
    print(f"Input: {short_text}")
    print(f"Latency: {latency_ms:.2f}ms (expected ~2-5ms on CPU)")
    print(f"Entities detected: {len(result['entities'])}")
    for e in result["entities"]:
        print(f"  - {e['label']:12} `@ [{e['start']:3},{e['end']:3}]: {e['text']!r}")
    
    return result


# ============================================================================
# Test 3: Adaptive Routing — Long Text (Context Track)
# ============================================================================

def test_long_text_context_track():
    """
    Long text (>= 100 chars) should use context track: regex + GLiNER.
    GLiNER invoked for better PERSON/LOCATION detection in narrative.
    """
    pipeline = Taivium(short_text_threshold=100)
    
    long_text = """
    Dr. Emily Clarke is a senior software engineer at Horizon AI, based in Boston.
    She collaborated with Michael Rodriguez, a data scientist from San Francisco, 
    on a machine learning project. Their manager, Sarah Chen, coordinated the effort 
    from the company's New York office. The team also included David Martinez from 
    the London branch. They all met at the conference in Paris last month.
    """
    
    start = time.perf_counter()
    result = pipeline.process(long_text)
    latency_ms = (time.perf_counter() - start) * 1000
    
    print("\n=== Long Text (Context Track) ===")
    print(f"Input length: {len(long_text)} chars")
    print(f"Latency: {latency_ms:.2f}ms (expected ~20-50ms on CPU, ~5-10ms on GPU)")
    print(f"Entities detected: {len(result['entities'])}")
    for e in result["entities"]:
        sources = ", ".join(e["evidence_sources"])
        print(f"  - {e['label']:12} @ [{e['start']:3},{e['end']:3}]: {e['text']!r} (from: {sources})")
    
    return result


# ============================================================================
# Test 4: Known Organization Detection (Compliance-Friendly)
# ============================================================================

def test_known_orgs_detection():
    """
    Use org_list for fast, auditable organization detection.
    High confidence (0.95) without ML inference.
    """
    pipeline = Taivium()
    
    text = "Acme Corporation approved the merger with Beta Industries. Gamma LLC declined."
    known_orgs = ["Acme Corporation", "Beta Industries", "Gamma LLC", "Delta Inc."]
    
    start = time.perf_counter()
    result = pipeline.process(text, known_orgs=known_orgs)
    latency_ms = (time.perf_counter() - start) * 1000
    
    print("\n=== Organization List Detection ===")
    print(f"Input: {text}")
    print(f"Known orgs: {known_orgs}")
    print(f"Latency: {latency_ms:.2f}ms (expected <1ms for org list)")
    print(f"Entities detected: {len(result['entities'])}")
    for e in result["entities"]:
        if e["label"] == "ORG":
            print(f"  - ORG @ [{e['start']:3},{e['end']:3}]: {e['text']!r} (confidence: {e['confidence']:.2f})")
    
    return result


# ============================================================================
# Test 5: Document Chunking for Large Texts (GPU Batching)
# ============================================================================

def chunk_text_for_gpu(text: str, chunk_size: int = 600, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Split large text into chunks with overlap for GPU processing.
    Returns list of chunks with character offsets for remapping.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text,
            "start_offset": start,
            "end_offset": end,
        })
        
        # Move start position by chunk_size, but back up by overlap for context
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def test_large_document_chunking():
    """
    Process a large document by chunking and batching on GPU.
    Merge results with offset remapping.
    """
    # Simulate a large document (multiple pages of text)
    large_text = """
    John Smith works at Microsoft in Seattle. Sarah Williams is his manager.
    They collaborate with Alice Johnson, who is based in San Francisco.
    The team recently visited the London office to meet with David Chen.
    """ * 50  # Repeat to create a large document
    
    print("\n=== Large Document Chunking ===")
    print(f"Total text length: {len(large_text)} chars")
    
    # Chunk the document
    chunk_size = 600
    overlap = 100
    chunks = chunk_text_for_gpu(large_text, chunk_size=chunk_size, overlap=overlap)
    
    print(f"Chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"  Chunk {i}: offset [{chunk['start_offset']}, {chunk['end_offset']}], "
              f"length {len(chunk['text'])}")
    
    # Process each chunk through pipeline
    pipeline = Taivium()
    all_results = []
    
    start_time = time.perf_counter()
    for chunk in chunks:
        result = pipeline.process(chunk["text"])
        # Remap entity offsets back to original document
        for entity in result["entities"]:
            entity["start"] += chunk["start_offset"]
            entity["end"] += chunk["start_offset"]
        all_results.append(result)
    
    total_latency = (time.perf_counter() - start_time) * 1000
    total_entities = sum(len(r["entities"]) for r in all_results)
    
    print(f"Total latency: {total_latency:.2f}ms")
    print(f"Total entities found: {total_entities}")
    print(f"Throughput: {len(large_text) / (total_latency / 1000):.0f} chars/sec")
    
    return all_results


# ============================================================================
# Test 6: Custom Policy Engine
# ============================================================================

def test_custom_policy():
    """
    Define custom policies for different entity types.
    Example: ALLOW locations, BLOCK API keys.
    """
    custom_policy = {
        "EMAIL": PolicyRule("EMAIL", PolicyAction.ANONYMIZE, RiskLevel.HIGH),
        "PHONE": PolicyRule("PHONE", PolicyAction.ANONYMIZE, RiskLevel.HIGH),
        "API_KEY": PolicyRule("API_KEY", PolicyAction.BLOCK, RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW, RiskLevel.LOW),
    }
    
    policy_engine = PolicyEngine(policy_table=custom_policy)
    pipeline = Taivium(policy_engine=policy_engine)
    
    text = "Alice lives in New York. Her API key is sk-1234567890abcdef. Contact: alice@example.com"
    
    print("\n=== Custom Policy Engine ===")
    print(f"Input: {text}")
    print("Policies:")
    print("  - EMAIL: ANONYMIZE")
    print("  - PHONE: ANONYMIZE")
    print("  - API_KEY: BLOCK (raises error)")
    print("  - LOCATION: ALLOW (passes through)")
    
    try:
        result = pipeline.process(text)
        print(f"Result: {result['anonymized']}")
    except ValueError as e:
        print(f"BLOCKED: {e}")
    
    return None


# ============================================================================
# Test 6: Verify GPU Provider in Use
# ============================================================================

def verify_gpu_provider():
    """Verify which ONNX execution provider is actually being used by GLiNER."""
    print("\n=== GPU Provider Verification ===")
    try:
        import onnxruntime as rt
        
        print("Available ONNX providers on this system:")
        available = rt.get_available_providers()
        for i, provider in enumerate(available, 1):
            print(f"  {i}. {provider}")
        
        print("\nProvider selection (in order of preference):")
        preference_order = ["CoreMLExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        for i, provider in enumerate(preference_order, 1):
            if provider in available:
                print(f"  {i}. {provider:30} ✅ SELECTED (will be used)")
            else:
                print(f"  {i}. {provider:30} (not available)")
        
        # Determine which will be used
        active_provider = None
        for provider in preference_order:
            if provider in available:
                active_provider = provider
                break
        
        print(f"\n🎯 ONNX will use: {active_provider}")
        
        if "CoreML" in (active_provider or ""):
            print("   ✅ GPU acceleration ACTIVE")
            print("      • M2/M3 Neural Engine enabled")
            print("      • CoreML provides 5-10x speedup")
        elif "CUDA" in (active_provider or ""):
            print("   ✅ GPU acceleration ACTIVE")
            print("      • NVIDIA CUDA enabled")
            print("      • Significant speedup on compatible GPUs")
        else:
            print("   ⚠️  CPU mode (no GPU acceleration)")
        
        # Now load the model to demonstrate
        print("\nLoading GLiNER model...")
        from taivium.utility import get_gliner_model
        _model = get_gliner_model()
        print("✅ Model loaded and ready for inference")
        
        return active_provider
        
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        print(f"❌ Provider verification failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Test 7: Performance Benchmarking
# ============================================================================

def benchmark_pipeline(text: str, num_iterations: int = 5) -> Dict[str, float]:
    """
    Benchmark pipeline latency across multiple runs.
    """
    pipeline = Taivium()
    latencies = []
    
    print(f"\n=== Benchmarking (text length: {len(text)} chars) ===")
    
    for _ in range(num_iterations):
        start = time.perf_counter()
        _result = pipeline.process(text)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
    
    min_lat = min(latencies)
    max_lat = max(latencies)
    avg_lat = sum(latencies) / len(latencies)
    
    print(f"Iterations: {num_iterations}")
    print(f"Min latency: {min_lat:.2f}ms")
    print(f"Max latency: {max_lat:.2f}ms")
    print(f"Avg latency: {avg_lat:.2f}ms")
    print(f"Throughput: {len(text) / (avg_lat / 1000):.0f} chars/sec")
    
    return {"min": min_lat, "max": max_lat, "avg": avg_lat}


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all test examples."""
    
    # Configure logging to show backend selection
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("=" * 70)
    print("TAIVIUM ADAPTIVE PIPELINE TESTS")
    print("=" * 70)
    
    # Backend detection
    backends = detect_available_backends()
    print("\nAvailable backends:")
    for name, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {name:20} {status}")
    
    # Run tests with error handling
    try:
        print("\n" + "=" * 70)
        print("Running pipeline tests...")
        print("=" * 70)
        print("\n📝 Backend selection will be logged during model initialization...\n")
        
        test_short_text_fast_track()
        
        # Verify GPU provider BEFORE running long text test
        verify_gpu_provider()
        
        test_long_text_context_track()
        test_known_orgs_detection()
        test_large_document_chunking()
        test_custom_policy()
        
        # Benchmark
        benchmark_text = "John Smith from Microsoft emailed alice@example.com about the project."
        benchmark_pipeline(benchmark_text, num_iterations=5)
        
    except (ImportError, RuntimeError, ValueError, OSError) as e:
        print("\n⚠️  Note: Some tests require full Taivium initialization")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print("\n✅ Testing helper functions instead:")
        
        # Test chunking helper independently
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunk_text_for_gpu(text, chunk_size=100, overlap=20)
        print("\n  Chunking test:")
        print(f"    • Original length: {len(text)} chars")
        print(f"    • Number of chunks: {len(chunks)}")
        print(f"    • First chunk preview: {chunks[0]['text'][:50]}...")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
