"""
Tests for examples/example_adaptive_pipeline.py

Tests helper functions and pipeline examples, with mocking for GPU provider checks.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import sys
from pathlib import Path

# Ensure examples module is importable
examples_dir = Path(__file__).parent.parent / "examples"
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from example_adaptive_pipeline import (
    detect_available_backends,
    test_short_text_fast_track,
    test_long_text_context_track,
    test_known_orgs_detection,
    chunk_text_for_gpu,
    test_large_document_chunking,
    test_custom_policy,
    verify_gpu_provider,
    benchmark_pipeline,
)


# ============================================================================
# Test: Backend Detection
# ============================================================================

class TestBackendDetection:
    """Test suite for detect_available_backends()."""

    def test_detect_backends_with_onnxruntime(self, capsys):
        """Test backend detection when onnxruntime is available."""
        mock_rt = MagicMock()
        mock_rt.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        with patch.dict("sys.modules", {"onnxruntime": mock_rt}):
            backends = detect_available_backends()
        
        assert backends is not None
        assert isinstance(backends, dict)
        assert "CPU" in backends
        assert "CoreML (M2/M3 GPU)" in backends or "CUDA (NVIDIA GPU)" in backends
        
        captured = capsys.readouterr()
        assert "💡 GLiNER will use" in captured.out

    def test_detect_backends_without_onnxruntime(self, capsys):
        """Test backend detection when onnxruntime is not available."""
        with patch("builtins.__import__", side_effect=ModuleNotFoundError("No module named 'onnxruntime'")):
            backends = detect_available_backends()
            
            # Should still return dict with CPU=True
            assert backends is not None
            assert backends.get("CPU") is True


# ============================================================================
# Test: Short Text Fast Track
# ============================================================================

class TestShortTextFastTrack:
    """Test suite for test_short_text_fast_track()."""

    def test_short_text_processes_successfully(self, capsys):
        """Test that short text is processed and returns entities."""
        # Suppress output during test
        with capsys.disabled():
            result = test_short_text_fast_track()
        
        assert result is not None
        assert "entities" in result
        assert isinstance(result["entities"], list)

    def test_short_text_result_structure(self):
        """Test that short text result has expected structure."""
        # Suppress output during test
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = test_short_text_fast_track()
        finally:
            sys.stdout = old_stdout
        
        for entity in result.get("entities", []):
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity
            assert "text" in entity
            assert isinstance(entity["start"], int)
            assert isinstance(entity["end"], int)


# ============================================================================
# Test: Long Text Context Track
# ============================================================================

class TestLongTextContextTrack:
    """Test suite for test_long_text_context_track()."""

    def test_long_text_processes_successfully(self, capsys):
        """Test that long text is processed and returns entities."""
        result = test_long_text_context_track()
        
        assert result is not None
        assert "entities" in result
        assert isinstance(result["entities"], list)
        
        captured = capsys.readouterr()
        assert "=== Long Text (Context Track) ===" in captured.out
        assert "Latency:" in captured.out

    def test_long_text_finds_entities(self):
        """Test that long text detection finds expected entity types."""
        result = test_long_text_context_track()
        
        labels = {e["label"] for e in result.get("entities", [])}
        # Expect PERSON, LOCATION, ORG entities in the test text
        assert len(labels) > 0


# ============================================================================
# Test: Organization Detection
# ============================================================================

class TestOrgDetection:
    """Test suite for test_known_orgs_detection()."""

    def test_org_detection_processes_successfully(self, capsys):
        """Test that org list detection works."""
        result = test_known_orgs_detection()
        
        assert result is not None
        assert "entities" in result
        
        captured = capsys.readouterr()
        assert "=== Organization List Detection ===" in captured.out

    def test_org_detection_finds_known_orgs(self):
        """Test that known orgs are detected."""
        result = test_known_orgs_detection()
        
        org_entities = [e for e in result.get("entities", []) if e["label"] == "ORG"]
        assert len(org_entities) > 0
        
        # Check that detected org names match known orgs
        detected_texts = {e["text"] for e in org_entities}
        known_orgs_set = {"Acme Corporation", "Beta Industries", "Gamma LLC"}
        assert detected_texts & known_orgs_set  # Should have overlap


# ============================================================================
# Test: Chunking Helper
# ============================================================================

class TestChunkingHelper:
    """Test suite for chunk_text_for_gpu()."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = chunk_text_for_gpu(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        assert isinstance(chunks, list)

    def test_chunk_structure(self):
        """Test that chunks have required structure."""
        text = "Sample text " * 50
        chunks = chunk_text_for_gpu(text, chunk_size=100, overlap=20)
        
        for chunk in chunks:
            assert "text" in chunk
            assert "start_offset" in chunk
            assert "end_offset" in chunk
            assert isinstance(chunk["start_offset"], int)
            assert isinstance(chunk["end_offset"], int)
            assert chunk["start_offset"] < chunk["end_offset"]

    def test_chunk_offsets_valid(self):
        """Test that chunk offsets correctly reference original text."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
        chunks = chunk_text_for_gpu(text, chunk_size=50, overlap=10)
        
        for chunk in chunks:
            start = chunk["start_offset"]
            end = chunk["end_offset"]
            assert text[start:end] == chunk["text"]

    def test_chunk_with_overlap(self):
        """Test that overlapping chunks have correct overlap."""
        text = "Word " * 100
        chunk_size = 50
        overlap = 10
        chunks = chunk_text_for_gpu(text, chunk_size=chunk_size, overlap=overlap)
        
        # Check that consecutive chunks overlap
        for i in range(len(chunks) - 1):
            curr_end = chunks[i]["end_offset"]
            next_start = chunks[i + 1]["start_offset"]
            # Next chunk should start before current chunk ends (overlap)
            assert next_start < curr_end

    def test_chunk_single_chunk(self):
        """Test chunking when text is smaller than chunk size."""
        text = "Short text"
        chunks = chunk_text_for_gpu(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["start_offset"] == 0
        assert chunks[0]["end_offset"] == len(text)

    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        text = ""
        chunks = chunk_text_for_gpu(text, chunk_size=100, overlap=10)
        
        assert len(chunks) == 0


# ============================================================================
# Test: Large Document Chunking
# ============================================================================

class TestLargeDocumentChunking:
    """Test suite for test_large_document_chunking()."""

    def test_large_document_chunking_processes(self, capsys):
        """Test that large document chunking completes."""
        result = test_large_document_chunking()
        
        assert result is not None
        assert isinstance(result, list)
        
        captured = capsys.readouterr()
        assert "=== Large Document Chunking ===" in captured.out
        assert "Total latency:" in captured.out

    def test_large_document_entity_counts(self):
        """Test that entities are detected in large documents."""
        result = test_large_document_chunking()
        
        # Should have multiple results (one per chunk)
        assert len(result) > 0
        
        # Each result should have entities key
        for chunk_result in result:
            assert "entities" in chunk_result


# ============================================================================
# Test: Custom Policy Engine
# ============================================================================

class TestCustomPolicy:
    """Test suite for test_custom_policy()."""

    def test_custom_policy_with_blocking(self, capsys):
        """Test custom policy that blocks API keys."""
        test_custom_policy()
        
        captured = capsys.readouterr()
        assert "=== Custom Policy Engine ===" in captured.out
        # Should either block or process the text
        assert "BLOCKED:" in captured.out or "Result:" in captured.out


# ============================================================================
# Test: GPU Provider Verification
# ============================================================================

class TestGPUProviderVerification:
    """Test suite for verify_gpu_provider()."""

    def test_verify_gpu_provider_with_mock(self, capsys):
        """Test GPU provider verification with mocked onnxruntime."""
        mock_rt = MagicMock()
        mock_rt.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        
        with patch.dict("sys.modules", {"onnxruntime": mock_rt}):
            result = verify_gpu_provider()
        
        # Should return a provider or None
        assert result is None or isinstance(result, str)


# ============================================================================
# Test: Benchmarking
# ============================================================================

class TestBenchmarking:
    """Test suite for benchmark_pipeline()."""

    def test_benchmark_runs_successfully(self, capsys):
        """Test that benchmarking completes without error."""
        text = "John Smith from Microsoft emailed alice@example.com about the project."
        result = benchmark_pipeline(text, num_iterations=2)
        
        assert result is not None
        assert isinstance(result, dict)
        
        captured = capsys.readouterr()
        assert "=== Benchmarking" in captured.out

    def test_benchmark_returns_metrics(self):
        """Test that benchmark returns expected metrics."""
        text = "Test text with entity alice@example.com here."
        result = benchmark_pipeline(text, num_iterations=2)
        
        assert "min" in result
        assert "max" in result
        assert "avg" in result
        assert result["min"] > 0
        assert result["max"] > 0
        assert result["avg"] > 0
        assert result["min"] <= result["avg"] <= result["max"]

    def test_benchmark_latency_reasonable(self):
        """Test that benchmark latencies are reasonable (< 60s per iteration on short text)."""
        text = "Quick test text."
        result = benchmark_pipeline(text, num_iterations=1)
        
        # Should be fast for short text
        assert result["avg"] < 60000  # 60 seconds in ms


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for multiple functions together."""

    def test_chunking_and_processing_pipeline(self):
        """Test chunking followed by processing."""
        text = "John works at Acme. Mary works at Beta. " * 30
        chunks = chunk_text_for_gpu(text, chunk_size=200, overlap=50)
        
        from taivium import Taivium
        pipeline = Taivium()
        
        all_entities = []
        for chunk in chunks:
            result = pipeline.process(chunk["text"])
            # Remap offsets
            for entity in result["entities"]:
                entity["start"] += chunk["start_offset"]
                entity["end"] += chunk["start_offset"]
            all_entities.extend(result["entities"])
        
        assert len(all_entities) > 0

    def test_all_examples_runnable(self):
        """Test that all example functions can be called without crashing."""
        # These should not raise exceptions
        try:
            detect_available_backends()
            test_short_text_fast_track()
            chunk_text_for_gpu("test", 10, 2)
            benchmark_pipeline("test", 1)
        except Exception as e:
            pytest.fail(f"Example function raised unexpected exception: {e}")
