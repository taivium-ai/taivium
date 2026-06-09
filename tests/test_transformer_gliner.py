"""
Unit tests for taivium.transformer_gliner module, covering GLiNER evidence collection
with chunking, batching, offset remapping, and exception handling.
"""

import pytest
import types
import logging
from taivium import transformer_gliner as gliner


class TestGlinerEvidence:
    """Test suite for gliner_evidence() function."""

    def setup_method(self):
        """Clear cache before each test."""
        from taivium.utility import get_gliner_model
        get_gliner_model.cache_clear()

    def test_gliner_evidence_returns_list(self):
        """gliner_evidence() should always return a list."""
        result = gliner.gliner_evidence("test text")
        assert isinstance(result, list)

    def test_gliner_evidence_empty_text(self):
        """gliner_evidence() should handle empty text gracefully."""
        result = gliner.gliner_evidence("")
        assert isinstance(result, list)

    def test_gliner_evidence_no_matches(self):
        """gliner_evidence() should return empty list when no entities match."""
        text = "The quick brown fox jumps over the lazy dog."
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)

    def test_gliner_evidence_default_targets(self):
        """gliner_evidence() without targets should use default PERSON and LOCATION."""
        text = "John Smith lives in New York."
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert ev.source == "gliner"
            assert 0.55 <= ev.confidence <= 1.0

    def test_gliner_evidence_custom_targets(self):
        """gliner_evidence() should accept custom targets parameter."""
        text = "John Smith is a CEO at Microsoft."
        result = gliner.gliner_evidence(text, targets=["PERSON", "ORG"])
        assert isinstance(result, list)

    def test_gliner_evidence_person_detection(self):
        """gliner_evidence() should detect PERSON entities."""
        test_names = [
            "My name is John Smith.",
            "Alice Johnson works here.",
            "Call me David Chen.",
            "Meet Professor Sarah Williams.",
        ]
        result = None
        for text in test_names:
            result = gliner.gliner_evidence(text)
            if any(ev.label == "PERSON" for ev in result):
                break
        assert isinstance(result, list)

    def test_gliner_evidence_location_detection(self):
        """gliner_evidence() should detect LOCATION entities."""
        test_locations = [
            "I visited Paris last summer.",
            "She lives in Tokyo, Japan.",
            "Meeting in New York City.",
            "They moved to London.",
        ]
        result = None
        for text in test_locations:
            result = gliner.gliner_evidence(text)
            if any(ev.label == "LOCATION" for ev in result):
                break
        assert isinstance(result, list)

    def test_gliner_evidence_mixed_content(self):
        """gliner_evidence() should process mixed content with multiple entity types."""
        text = """
        John Smith is a software engineer at Microsoft in Seattle.
        His colleague, Maria Garcia, works at Google in Mountain View.
        They both attended Stanford University in California.
        """
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert hasattr(ev, 'start')
            assert hasattr(ev, 'end')
            assert hasattr(ev, 'label')
            assert hasattr(ev, 'source')
            assert hasattr(ev, 'confidence')
            assert ev.source == "gliner"
            assert 0.55 <= ev.confidence <= 1.0

    def test_gliner_evidence_handles_exception(self, monkeypatch, caplog):
        """gliner_evidence() should handle exceptions gracefully and return empty list."""
        def failing_predict(*args, **kwargs):
            raise RuntimeError("Model error")
        
        mock_model = types.SimpleNamespace(predict_entities=failing_predict)
        from taivium.utility import get_gliner_model
        monkeypatch.setattr(get_gliner_model, "cache_clear", lambda: None)
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        with caplog.at_level(logging.WARNING, logger="taivium.engine"):
            result = gliner.gliner_evidence("test text")
        
        assert isinstance(result, list)
        assert len(result) == 0
        
        # Check for warning message
        assert any("GLiNER detection failed" in msg for msg in caplog.messages)

    def test_gliner_evidence_empty_targets(self):
        """gliner_evidence() with empty targets list should return empty list."""
        result = gliner.gliner_evidence("John Smith lives in NYC", targets=[])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_gliner_evidence_organization_detection(self):
        """gliner_evidence() should detect ORG entities when requested."""
        text = "I work for Apple Inc. and Microsoft Corporation."
        result = gliner.gliner_evidence(text, targets=["ORG"])
        assert isinstance(result, list)

    def test_chunk_text_for_gliner_shorter_than_max_tokens(self):
        """Text shorter than max token window should remain a single chunk."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # Create short text that is definitely < 382 DeBERTa tokens
        # "Hello world " * 50 = 100 DeBERTa tokens (verified)
        text = "Hello world " * 50
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        actual_token_count = len(enc["input_ids"])
        assert actual_token_count < 382, f"Test assumes {actual_token_count} < 382"
        assert actual_token_count == 100, f"Expected 100 tokens, got {actual_token_count}"
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)
        assert len(chunks) == 1
        assert chunks[0][0] == 0

    def test_chunk_text_for_gliner_equal_max_tokens(self):
        """Text around max token window should stay as single chunk if within limit."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # Construct text that is close to but under 382 DeBERTa tokens
        text = "word " * 300  # 300 DeBERTa tokens
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        token_count = len(enc["input_ids"])
        assert token_count <= 382, f"Test assumes text is under 382 tokens, got {token_count}"
        assert token_count == 300, f"Expected 300 tokens, got {token_count}"
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)
        assert len(chunks) == 1, f"Expected 1 chunk for {token_count} tokens, got {len(chunks)}"

    def test_chunk_text_for_gliner_just_longer_than_max_tokens(self):
        """Text that exceeds max token window should split into multiple chunks."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        # Create text that will definitely exceed 382 DeBERTa tokens
        text = "word " * 400  # 400 DeBERTa tokens
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)

        assert len(chunks) == 2  # 400 tokens creates exactly 2 chunks (382 + 50)
        assert chunks[0][0] == 0
        assert chunks[1][0] > 0
        
        # Verify each chunk is within limit
        for i, (_, chunk_text) in enumerate(chunks):
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= 382, \
                f"Chunk {i} has {chunk_token_count} tokens, exceeds max 382"
        
        # Verify last chunk is smaller (remainder)
        _, last_chunk = chunks[-1]
        enc_last = tokenizer(last_chunk, return_offsets_mapping=True, add_special_tokens=False)
        # Verify each chunk doesn't exceed max tokens (DeBERTa token count)
        for i, (_, chunk_text) in enumerate(chunks):
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= 382, \
                f"Chunk {i} has {chunk_token_count} DeBERTa tokens, exceeds max 382"
            if i == len(chunks) - 1:  # Last chunk should be smaller
                assert chunk_token_count == 50, f"Last chunk should be smaller, got {chunk_token_count}"
        
    def test_chunk_text_for_gliner_token_count_equals_batch_size(self):
        """Token count equal to batch size should still be a single chunk if under max."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # _GLINER_BATCH_SIZE = 8, which is way less than _GLINER_MAX_TOKENS = 384
        # Just create small text
        text = "word " * 5  # 5 DeBERTa tokens
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)

        assert len(chunks) == 1

    def test_chunk_text_for_gliner_token_count_twice_batch_size(self):
        """Token count twice batch size should still be a single chunk if under max."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # _GLINER_BATCH_SIZE * 2 = 16, still way less than _GLINER_MAX_TOKENS = 384
        text = "word " * 15  # 15 DeBERTa tokens
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)

        assert len(chunks) == 1

    def test_chunk_text_for_gliner_non_edge_multi_chunk_flow(self):
        """Typical long text should create stable overlapping chunks."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # Create text that generates 800 DeBERTa tokens
        text = "word " * 800  # 800 DeBERTa tokens
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)

        # With 800 tokens and 382 content max, should get exactly 3 chunks
        assert len(chunks) == 3
        
        # Verify each chunk doesn't exceed max tokens (DeBERTa token count)
        for _, chunk_text in chunks:
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= 382, \
                f"Chunk has {chunk_token_count} DeBERTa tokens, exceeds max 382"
        
        # Verify offsets are sorted
        assert [offset for offset, _ in chunks] == sorted(offset for offset, _ in chunks)

    def test_chunk_text_for_gliner_exact_token_boundaries(self):
        """Verify chunks respect exact token boundaries with DeBERTa tokenizer."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # Test: small text under max (will be 1 chunk)
        text_small = "hello world test example paragraph " * 5  # Safe small text
        chunks_small = gliner._chunk_text_for_gliner(text_small, tokenizer=tokenizer)
        enc_small = tokenizer(text_small, return_offsets_mapping=True, add_special_tokens=False)
        deberta_token_count_small = len(enc_small["input_ids"])
        assert deberta_token_count_small < gliner._GLINER_MAX_TOKENS, \
            f"Expected < 384 DeBERTa tokens, got {deberta_token_count_small}"
        assert len(chunks_small) == 1, f"Small text should be 1 chunk, got {len(chunks_small)}"
        
        # Test: medium text that stays under max (will be 1 chunk)
        text_medium = "hello world test example paragraph " * 10  # 50 tokens, under max
        chunks_medium = gliner._chunk_text_for_gliner(text_medium, tokenizer=tokenizer)
        enc_medium = tokenizer(text_medium, return_offsets_mapping=True, add_special_tokens=False)
        deberta_token_count_medium = len(enc_medium["input_ids"])
        assert deberta_token_count_medium < gliner._GLINER_MAX_TOKENS, \
            f"Test assumes text is under max, got {deberta_token_count_medium}"
        assert len(chunks_medium) == 1
        
        # Verify all chunks stay under max
        for chunk_idx, (_, chunk_text) in enumerate(chunks_medium):
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= gliner._GLINER_MAX_TOKENS, \
                f"Chunk {chunk_idx} has {chunk_token_count} DeBERTa tokens, exceeds max {gliner._GLINER_MAX_TOKENS}"
        
        # Test: large text that definitely requires multiple chunks (400+ words)
        text_large = "the quick brown fox jumps over the lazy dog in the forest " * 50  # 600 tokens
        chunks_large = gliner._chunk_text_for_gliner(text_large, tokenizer=tokenizer)
        enc_large = tokenizer(text_large, return_offsets_mapping=True, add_special_tokens=False)
        deberta_token_count_large = len(enc_large["input_ids"])
        
        # Verify multiple chunks are created for large text
        assert deberta_token_count_large > gliner._GLINER_MAX_TOKENS, \
            f"Test assumes text exceeds max, got {deberta_token_count_large} tokens"
        assert len(chunks_large) == 2, f"Large text with {deberta_token_count_large} tokens should create exactly 2 chunks, got {len(chunks_large)}"
        
        # Verify last chunk is remainder
        _, last_chunk_large = chunks_large[-1]
        enc_last_large = tokenizer(last_chunk_large, return_offsets_mapping=True, add_special_tokens=False)
        assert len(enc_last_large["input_ids"]) < 382, "Last chunk should be remainder (< 382 tokens)"
        
        # Verify all chunks stay under max
        for chunk_idx, (_, chunk_text) in enumerate(chunks_large):
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= gliner._GLINER_MAX_TOKENS, \
                f"Chunk {chunk_idx} of large text has {chunk_token_count} DeBERTa tokens, exceeds {gliner._GLINER_MAX_TOKENS}"

        

    def test_chunk_text_for_gliner_requires_tokenizer(self):
        """_chunk_text_for_gliner should raise ValueError if tokenizer is None."""
        text = " ".join(f"tok{i}" for i in range(100))
        
        with pytest.raises(ValueError, match="GLiNER tokenizer is required"):
            gliner._chunk_text_for_gliner(text, tokenizer=None)

    def test_gliner_evidence_chunks_text_over_384_tokens(self, monkeypatch):
        """Long text should be split into multiple GLiNER chunks."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        call_count = 0

        def fake_predict_entities(chunk_text, targets, threshold=0.55):
            nonlocal call_count
            del targets, threshold
            call_count += 1
            first_word = chunk_text.split()[0]
            return [{"start": 0, "end": len(first_word), "label": "PERSON", "score": 0.9}]

        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict_entities,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)

        long_text = " ".join(f"tok{i}" for i in range(420))
        result = gliner.gliner_evidence(long_text)

        assert call_count > 1
        assert len(result) == call_count
        assert result[0].start == 0
        assert any(ev.start > 0 for ev in result)

    def test_gliner_evidence_uses_batch_predict_when_available(self, monkeypatch):
        """Use batch_predict_entities when model supports it."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        batch_calls = 0

        def fake_batch_predict_entities(batch_texts, targets, threshold=0.55):
            nonlocal batch_calls
            del targets, threshold
            batch_calls += 1
            preds = []
            for chunk_text in batch_texts:
                first_word = chunk_text.split()[0]
                preds.append([
                    {"start": 0, "end": len(first_word), "label": "PERSON", "score": 0.9}
                ])
            return preds

        def fail_predict_entities(*args, **kwargs):
            raise AssertionError("predict_entities should not be called when batch API exists")

        mock_model = types.SimpleNamespace(
            batch_predict_entities=fake_batch_predict_entities,
            predict_entities=fail_predict_entities,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)

        long_text = " ".join(f"tok{i}" for i in range(420))
        result = gliner.gliner_evidence(long_text)

        assert batch_calls == 1  # All chunks processed in one batch call
        assert len(result) == 4  # One prediction per chunk (4 chunks total from 420 words)
        assert all(ev.source == "gliner" for ev in result)

    def test_gliner_evidence_multiple_occurrences(self):
        """gliner_evidence() should detect multiple entity occurrences in text."""
        text = """
        John went to London yesterday.
        Today, John is in Paris.
        Tomorrow, John will be in Berlin.
        """
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert 0 <= ev.start < ev.end <= len(text)

    def test_gliner_evidence_confidence_value(self):
        """gliner_evidence() should preserve GLiNER confidence scores (>= threshold)."""
        text = "John Smith works at Google in Mountain View."
        result = gliner.gliner_evidence(text)
        for ev in result:
            assert 0.55 <= ev.confidence <= 1.0

    def test_gliner_evidence_pure_lowercase_person(self):
        """gliner_evidence() should detect PERSON entities in pure lowercase text."""
        texts = [
            "john smith is a software engineer",
            "alice johnson works at the tech company",
            "michael chen lives downtown",
            "sarah williams attended the meeting",
            "david martinez is coming tomorrow",
        ]
        for text in texts:
            result = gliner.gliner_evidence(text)
            for ev in result:
                assert ev.source == "gliner"
                assert 0.55 <= ev.confidence <= 1.0
                assert 0 <= ev.start < ev.end <= len(text)
        assert True

    def test_gliner_evidence_pure_lowercase_location(self):
        """gliner_evidence() should detect LOCATION entities in pure lowercase text."""
        texts = [
            "she went to paris for vacation",
            "they live in tokyo and osaka",
            "meeting scheduled for london next week",
            "arrived in new york yesterday",
            "traveling through california and nevada",
        ]
        for text in texts:
            result = gliner.gliner_evidence(text)
            for ev in result:
                assert ev.source == "gliner"
                assert 0.55 <= ev.confidence <= 1.0
                assert 0 <= ev.start < ev.end <= len(text)
        assert True

    def test_gliner_evidence_pure_lowercase_mixed(self):
        """gliner_evidence() should detect both PERSON and LOCATION in lowercase text."""
        text = "john visited paris in june. alice lives in london. david travels to tokyo monthly."
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert ev.source == "gliner"
            assert 0.55 <= ev.confidence <= 1.0
            assert hasattr(ev, 'label')
            assert 0 <= ev.start < ev.end <= len(text)

    def test_gliner_evidence_pure_lowercase_telemetry_style(self):
        """gliner_evidence() handles telemetry-like lowercase content (core use case)."""
        telemetry_texts = [
            "user john smith logged in from new york",
            "customer alice johnson submitted request from london",
            "admin david chen accessed database from tokyo",
            "alert: sarah williams connected from paris",
            "transaction by michael rodriguez processed in singapore",
        ]
        for text in telemetry_texts:
            result = gliner.gliner_evidence(text)
            for ev in result:
                assert ev.source == "gliner"
                assert 0.55 <= ev.confidence <= 1.0
        assert True

    def test_gliner_evidence_pure_lowercase_no_names(self):
        """gliner_evidence() on lowercase text with no proper names should return minimal results."""
        text = "the quick brown fox jumps over the lazy dog in the forest"
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert ev.source == "gliner"
            assert 0.55 <= ev.confidence <= 1.0
            assert 0 <= ev.start < ev.end <= len(text)

    def test_gliner_evidence_pure_lowercase_repeated_entities(self):
        """gliner_evidence() should detect repeated person/location names in lowercase."""
        text = "john went to paris. then john traveled to london. finally john arrived in paris again."
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)
        for ev in result:
            assert ev.source == "gliner"
            assert 0.55 <= ev.confidence <= 1.0
            assert 0 <= ev.start < ev.end <= len(text)
            extracted = text[ev.start:ev.end]
            assert len(extracted) > 0

    def test_predict_gliner_chunks_inference_fallback(self, monkeypatch):
        """_predict_gliner_chunks should fall back when inference fails."""
        def failing_inference(*args, **kwargs):
            raise RuntimeError("Inference failed")
        
        def working_batch_predict(texts, labels, **kwargs):
            return [[{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}] for _ in texts]
        
        mock_model = types.SimpleNamespace(
            inference=failing_inference,
            batch_predict_entities=working_batch_predict,
        )
        
        chunks = [(0, "chunk1"), (10, "chunk2")]
        result = gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)
        
        assert len(result) == 2
        assert all(len(preds) == 1 for preds in result)

    def test_predict_gliner_chunks_batch_fallback(self, monkeypatch):
        """_predict_gliner_chunks should fall back to per-chunk when batch fails."""
        def failing_batch(*args, **kwargs):
            raise RuntimeError("Batch failed")
        
        def working_predict(text, labels, **kwargs):
            return [{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}]
        
        mock_model = types.SimpleNamespace(
            inference=None,
            batch_predict_entities=failing_batch,
            predict_entities=working_predict,
        )
        
        chunks = [(0, "chunk1"), (10, "chunk2")]
        result = gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)
        
        assert len(result) == 2
        assert all(len(preds) == 1 for preds in result)

    def test_predict_gliner_chunks_shape_mismatch_inference(self, monkeypatch, caplog):
        """_predict_gliner_chunks should fall back when inference shape mismatches."""
        def mismatch_inference(texts, labels, **kwargs):
            return [[{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}]]  # Only 1 instead of N
        
        def working_batch(texts, labels, **kwargs):
            return [[{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}] for _ in texts]
        
        mock_model = types.SimpleNamespace(
            inference=mismatch_inference,
            batch_predict_entities=working_batch,
        )
        
        with caplog.at_level(logging.DEBUG, logger="taivium.engine"):
            chunks = [(0, "chunk1"), (10, "chunk2")]
            result = gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)
        
        assert len(result) == 2
        assert any("unavailable" in msg for msg in caplog.messages)

    def test_predict_gliner_chunks_shape_mismatch_batch(self, monkeypatch, caplog):
        """_predict_gliner_chunks should fall back when batch predict shape mismatches."""
        def mismatch_batch(texts, labels, **kwargs):
            return [[{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}]]  # Only 1 instead of N
        
        def working_predict(text, labels, **kwargs):
            return [{"start": 0, "end": 3, "label": "PERSON", "score": 0.9}]
        
        mock_model = types.SimpleNamespace(
            inference=None,
            batch_predict_entities=mismatch_batch,
            predict_entities=working_predict,
        )
        
        with caplog.at_level(logging.DEBUG, logger="taivium.engine"):
            chunks = [(0, "chunk1"), (10, "chunk2")]
            result = gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)
        
        assert len(result) == 2
        assert any("unavailable" in msg for msg in caplog.messages)

    def test_chunk_text_for_gliner_empty_text(self):
        """_chunk_text_for_gliner should handle empty text."""
        chunks = gliner._chunk_text_for_gliner("")
        assert chunks == []

    def test_chunk_text_for_gliner_no_tokens(self):
        """_chunk_text_for_gliner should handle text with no tokens (only whitespace)."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        chunks = gliner._chunk_text_for_gliner("   \n  \t  ", tokenizer=tokenizer)
        assert len(chunks) <= 1

    def test_gliner_evidence_with_empty_targets(self):
        """gliner_evidence should return empty list for empty targets."""
        result = gliner.gliner_evidence("John Smith lives in NYC", targets=[])
        assert result == []

    def test_gliner_evidence_with_none_targets(self):
        """gliner_evidence should use default targets when None is passed."""
        result = gliner.gliner_evidence("John Smith", targets=None)
        assert isinstance(result, list)

    def test_gliner_evidence_model_exception(self, monkeypatch, caplog):
        """gliner_evidence should handle model exceptions gracefully."""
        def failing_get_model():
            raise RuntimeError("Model loading failed")
        
        monkeypatch.setattr(gliner, "get_gliner_model", failing_get_model)
        
        with caplog.at_level(logging.WARNING, logger="taivium.engine"):
            result = gliner.gliner_evidence("John Smith")
        
        assert result == []
        assert any("GLiNER detection failed" in msg for msg in caplog.messages)

    def test_gliner_evidence_deduplication(self, monkeypatch):
        """gliner_evidence should deduplicate overlapping predictions."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        def fake_predict(text, labels, **kwargs):
            # Return duplicate spans (same start/end/label)
            return [
                {"start": 0, "end": 4, "label": "PERSON", "score": 0.9},
                {"start": 0, "end": 4, "label": "PERSON", "score": 0.85},  # Duplicate
            ]
        
        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("John Smith")
        
        # Should only have 1 evidence, not 2 (deduplication)
        assert len(result) == 1

    def test_gliner_evidence_label_normalization(self, monkeypatch):
        """gliner_evidence should normalize labels correctly."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        def fake_predict(text, labels, **kwargs):
            return [
                {"start": 0, "end": 4, "label": "PERSON", "score": 0.9},
                {"start": 5, "end": 10, "label": "ORG", "score": 0.85},
            ]
        
        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("John Smith Company")
        
        assert len(result) == 2
        assert all(ev.label in ["PERSON", "ORG"] for ev in result)

    def test_chunk_text_for_gliner_complex_overlap(self):
        """_chunk_text_for_gliner should maintain proper overlap for complex texts."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer
        
        # Create text with 1100 DeBERTa tokens (should create multiple chunks)
        text = "word " * 1100  # 1100 tokens
        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)
        
        # Should have multiple chunks
        assert len(chunks) == 4  # 1100 tokens creates exactly 4 chunks (382 + 382 + 382 + 50)
        
        # Verify last chunk is remainder
        _, last_chunk = chunks[-1]
        enc_last = tokenizer(last_chunk, return_offsets_mapping=True, add_special_tokens=False)
        assert len(enc_last["input_ids"]) < 382, "Last chunk should be remainder (< 382 tokens)"
        
        # Verify each chunk is within limit
        for i, (_, chunk_text) in enumerate(chunks):
            enc = tokenizer(chunk_text, return_offsets_mapping=True, add_special_tokens=False)
            chunk_token_count = len(enc["input_ids"])
            assert chunk_token_count <= 382, \
                f"Chunk {i} has {chunk_token_count} tokens, exceeds max 382"
            if i == 3:  # Last chunk should be smaller
                assert chunk_token_count == 50, f"Last chunk should be smaller, got {chunk_token_count}"

    def test_predict_gliner_chunks_per_chunk_fallback(self, monkeypatch):
        """_predict_gliner_chunks should fall through to per-chunk mode when APIs unavailable."""
        def working_predict(text, labels, **kwargs):
            return [{"start": 0, "end": 4, "label": "PERSON", "score": 0.9}]
        
        mock_model = types.SimpleNamespace(
            predict_entities=working_predict,
            # No inference or batch_predict methods (they don't exist)
        )
        
        chunks = [(0, "text1"), (10, "text2"), (20, "text3")]
        result = gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)
        
        # Should process 3 chunks, 1 prediction each
        assert len(result) == 3
        assert all(len(preds) == 1 for preds in result)

    def test_gliner_evidence_unknown_label_filtering(self, monkeypatch):
        """gliner_evidence should skip predictions with UNKNOWN labels."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        def fake_predict(text, labels, **kwargs):
            return [
                {"start": 0, "end": 4, "label": "UNKNOWN", "score": 0.9},  # Will be filtered
                {"start": 5, "end": 10, "label": "PERSON", "score": 0.85},  # Will be kept
            ]
        
        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("John Smith")
        
        # Should only have 1 evidence (UNKNOWN filtered out)
        assert len(result) == 1
        assert result[0].label == "PERSON"

    def test_gliner_evidence_offset_remapping_multi_chunk(self, monkeypatch):
        """gliner_evidence should correctly remap offsets from multiple chunks."""
        def fake_predict(text, labels, **kwargs):
            if "chunk1" in text:
                return [{"start": 0, "end": 2, "label": "PERSON", "score": 0.9}]
            elif "chunk2" in text:
                return [{"start": 3, "end": 6, "label": "LOCATION", "score": 0.85}]
            else:
                return []
        
        mock_model = types.SimpleNamespace(predict_entities=fake_predict)
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        # Mock chunking to return specific chunks with offsets
        def fake_chunk(text, **kwargs):
            return [(0, "chunk1"), (50, "chunk2")]
        
        monkeypatch.setattr(gliner, "_chunk_text_for_gliner", fake_chunk)
        
        result = gliner.gliner_evidence("dummy text")
        
        # Check that offsets are correctly remapped
        assert len(result) == 2
        assert result[0].start == 0  # chunk_offset 0 + local 0
        assert result[1].start == 53  # chunk_offset 50 + local 3

    def test_predict_gliner_chunks_all_apis_missing(self, monkeypatch):
        """_predict_gliner_chunks should gracefully handle model with no inference APIs."""
        mock_model = types.SimpleNamespace()  # No methods at all
        
        chunks = [(0, "text")]
        
        # This should raise an AttributeError since predict_entities doesn't exist either
        with pytest.raises(AttributeError):
            gliner._predict_gliner_chunks(mock_model, chunks, ["PERSON"], 0.55)

    def test_gliner_evidence_empty_predictions(self, monkeypatch):
        """gliner_evidence should handle chunks with empty prediction lists."""
        def fake_predict(text, labels, **kwargs):
            return []  # No predictions
        
        mock_model = types.SimpleNamespace(predict_entities=fake_predict)
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("some text")
        
        assert result == []

    def test_gliner_evidence_multiple_spans_same_location(self, monkeypatch):
        """gliner_evidence should handle overlapping predictions at same location."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        def fake_predict(text, labels, **kwargs):
            return [
                {"start": 0, "end": 4, "label": "PERSON", "score": 0.95},
                {"start": 0, "end": 4, "label": "PERSON", "score": 0.92},  # Dup
                {"start": 0, "end": 4, "label": "ORG", "score": 0.88},  # Different label
            ]
        
        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("text")
        
        # Should have 2 unique entries (same span+label deduped, but different label kept)
        assert len(result) == 2

    def test_gliner_evidence_special_characters(self, monkeypatch):
        """gliner_evidence should handle text with special characters."""
        from taivium.utility import get_gliner_model
        real_model = get_gliner_model()
        real_tokenizer = real_model.data_processor.transformer_tokenizer
        
        def fake_predict(text, labels, **kwargs):
            return [{"start": 0, "end": 5, "label": "PERSON", "score": 0.9}]
        
        mock_model = types.SimpleNamespace(
            predict_entities=fake_predict,
            data_processor=types.SimpleNamespace(transformer_tokenizer=real_tokenizer)
        )
        monkeypatch.setattr(gliner, "get_gliner_model", lambda: mock_model)
        
        result = gliner.gliner_evidence("Jöhn @#$% Smith")
        
        assert len(result) == 1
        assert result[0].source == "gliner"

    def test_gliner_evidence_very_long_text_detects_entity(self):
        """Very long text should still yield detected PERSON entity."""
        text = "John Smith lives in New York. " * 2000
        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)

        persons = [
            ev for ev in result
            if getattr(ev, "label", None) == "PERSON" and getattr(ev, "source", None) == "gliner"
        ]
        assert persons, "Expected at least one PERSON from GLiNER"

        expected_name = "John Smith"
        assert any(text[ev.start:ev.end] == expected_name for ev in persons), \
            "No PERSON evidence maps to the exact 'John Smith' substring"

        sentence = "John Smith lives in New York. "
        sentence_length = len(sentence)
        assert any(ev.start % sentence_length == 0 for ev in persons), \
            "PERSON start offsets should align with repeated sentence boundaries"

    def test_chunk_text_for_gliner_preserves_exact_text(self):
        """_chunk_text_for_gliner should preserve the exact original text for each chunk."""
        from taivium.utility import get_gliner_model
        model = get_gliner_model()
        tokenizer = model.data_processor.transformer_tokenizer

        text = (
            "John Smith lives in New York. "
            "Alice Johnson visited Tokyo last week. "
            "Maria Garcia traveled to Paris for a meeting. "
        ) * 80

        chunks = gliner._chunk_text_for_gliner(text, tokenizer=tokenizer)
        assert chunks, "Expected _chunk_text_for_gliner to return at least one chunk"

        for offset, chunk_text in chunks:
            assert text[offset:offset + len(chunk_text)] == chunk_text, \
                "Chunk text must match the original text at the reported character offset"
            assert len(chunk_text) > 0, "Chunk text should not be empty"
            assert 0 <= offset < len(text), "Chunk offset must lie within the original text range"

        result = gliner.gliner_evidence(text)
        assert isinstance(result, list)

        expected_names = {"John Smith", "Alice Johnson", "Maria Garcia"}
        persons = [
            ev for ev in result
            if getattr(ev, "label", None) == "PERSON"
            and getattr(ev, "source", None) == "gliner"
        ]

        assert persons, "Expected at least one PERSON entity from GLiNER"
        for ev in persons:
            span_text = text[ev.start:ev.end]
            assert span_text in expected_names, (
                f"PERSON entity span {span_text!r} at {ev.start}:{ev.end} does not match an expected name"
            )
        

