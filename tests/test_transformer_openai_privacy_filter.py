"""Unit tests for taivium.backend.transformer_openai_privacy_filter."""

import logging
import sys
import types

from taivium.backend import transformer_openai_privacy_filter as openai_filter


class TestOpenaiPrivacyFilterEvidence:
    """Test suite for openai_privacy_filter_evidence()."""

    def setup_method(self):
        """Clear cached pipeline between tests for isolation."""
        openai_filter._get_openai_ner_pipeline.cache_clear()

    def test_returns_empty_list_when_pipeline_unavailable(self, monkeypatch):
        """Should degrade gracefully when pipeline cannot be loaded."""
        monkeypatch.setattr(openai_filter, "_get_openai_ner_pipeline", lambda: None)

        result = openai_filter.openai_privacy_filter_evidence("Alice at Acme")

        assert isinstance(result, list)
        assert result == []

    def test_returns_empty_list_when_pipeline_raises(self, monkeypatch, caplog):
        """Should handle backend runtime errors without crashing."""

        def failing_pipeline(_text):
            raise RuntimeError("pipeline failure")

        monkeypatch.setattr(openai_filter, "_get_openai_ner_pipeline", lambda: failing_pipeline)

        with caplog.at_level(logging.ERROR, logger="taivium.backend.transformer_openai_privacy_filter"):
            result = openai_filter.openai_privacy_filter_evidence("Alice at Acme")

        assert result == []
        assert any("OpenAI privacy filter evidence extraction failed" in msg for msg in caplog.messages)

    def test_maps_labels_and_builds_evidence(self, monkeypatch):
        """Should map backend labels to canonical labels and emit Evidence objects."""
        text = "Alice works at Acme in Paris."

        def fake_pipeline(_text):
            return [
                {"entity_group": "private_person", "start": 0, "end": 5, "score": 0.93},
                {"entity_group": "ORG", "start": 15, "end": 19, "score": 0.88},
                {"entity_group": "LOC", "start": 23, "end": 28, "score": 0.77},
                {"entity_group": "MISC", "start": 0, "end": 5, "score": 0.10},
            ]

        monkeypatch.setattr(openai_filter, "_get_openai_ner_pipeline", lambda: fake_pipeline)

        result = openai_filter.openai_privacy_filter_evidence(text)

        assert len(result) == 3
        assert [ev.label for ev in result] == ["PERSON", "ORG", "LOCATION"]
        assert all(ev.source == "openai_privacy_filter" for ev in result)
        assert [ev.start for ev in result] == [0, 15, 23]
        assert [ev.end for ev in result] == [5, 19, 28]

    def test_skips_invalid_span_predictions(self, monkeypatch):
        """Should filter out out-of-range or malformed spans."""
        text = "Alice"

        def fake_pipeline(_text):
            return [
                {"entity_group": "PER", "start": -1, "end": 2, "score": 0.9},
                {"entity_group": "PER", "start": 0, "end": 0, "score": 0.9},
                {"entity_group": "PER", "start": 0, "end": 10, "score": 0.9},
                {"entity_group": "PER", "start": 0, "end": 5, "score": 0.9},
            ]

        monkeypatch.setattr(openai_filter, "_get_openai_ner_pipeline", lambda: fake_pipeline)

        result = openai_filter.openai_privacy_filter_evidence(text)

        assert len(result) == 1
        assert result[0].label == "PERSON"
        assert result[0].start == 0
        assert result[0].end == 5

    def test_unknown_label_is_skipped(self, monkeypatch):
        """Unknown/unmapped labels should be ignored."""

        def fake_pipeline(_text):
            return [
                {"entity_group": "UNMAPPED_LABEL", "start": 0, "end": 5, "score": 0.9},
            ]

        monkeypatch.setattr(openai_filter, "_get_openai_ner_pipeline", lambda: fake_pipeline)

        result = openai_filter.openai_privacy_filter_evidence("Alice")

        assert result == []

    def test_get_openai_ner_pipeline_success(self, monkeypatch):
        """Loader should return initialized pipeline when dependencies are available."""
        openai_filter._get_openai_ner_pipeline.cache_clear()

        hf_module = types.ModuleType("huggingface_hub")
        hf_module.snapshot_download = lambda _repo: "/tmp/fake-model"

        openmed_module = types.ModuleType("openmed")
        mlx_module = types.ModuleType("openmed.mlx")
        inference_module = types.ModuleType("openmed.mlx.inference")

        class FakePipeline:
            def __init__(self, model_path):
                self.model_path = model_path

        inference_module.PrivacyFilterMLXPipeline = FakePipeline

        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
        monkeypatch.setitem(sys.modules, "openmed", openmed_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx.inference", inference_module)

        pipeline = openai_filter._get_openai_ner_pipeline()

        assert pipeline is not None
        assert isinstance(pipeline, FakePipeline)
        assert pipeline.model_path == "/tmp/fake-model"

    def test_get_openai_ner_pipeline_expected_load_failure(self, monkeypatch, caplog):
        """Known loader/import failures should return None and warn."""
        openai_filter._get_openai_ner_pipeline.cache_clear()

        hf_module = types.ModuleType("huggingface_hub")

        def raise_oserror(_repo):
            raise OSError("download failed")

        hf_module.snapshot_download = raise_oserror

        openmed_module = types.ModuleType("openmed")
        mlx_module = types.ModuleType("openmed.mlx")
        inference_module = types.ModuleType("openmed.mlx.inference")
        inference_module.PrivacyFilterMLXPipeline = lambda _path: object()

        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
        monkeypatch.setitem(sys.modules, "openmed", openmed_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx.inference", inference_module)

        with caplog.at_level(logging.WARNING, logger="taivium.backend.transformer_openai_privacy_filter"):
            pipeline = openai_filter._get_openai_ner_pipeline()

        assert pipeline is None
        assert any("Transformer NER pipeline unavailable" in msg for msg in caplog.messages)

    def test_get_openai_ner_pipeline_unexpected_failure(self, monkeypatch, caplog):
        """Unexpected failures should hit the broad-except path and return None."""
        openai_filter._get_openai_ner_pipeline.cache_clear()

        hf_module = types.ModuleType("huggingface_hub")

        def raise_unexpected(_repo):
            raise AssertionError("unexpected")

        hf_module.snapshot_download = raise_unexpected

        openmed_module = types.ModuleType("openmed")
        mlx_module = types.ModuleType("openmed.mlx")
        inference_module = types.ModuleType("openmed.mlx.inference")
        inference_module.PrivacyFilterMLXPipeline = lambda _path: object()

        monkeypatch.setitem(sys.modules, "huggingface_hub", hf_module)
        monkeypatch.setitem(sys.modules, "openmed", openmed_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx", mlx_module)
        monkeypatch.setitem(sys.modules, "openmed.mlx.inference", inference_module)

        with caplog.at_level(logging.ERROR, logger="taivium.backend.transformer_openai_privacy_filter"):
            pipeline = openai_filter._get_openai_ner_pipeline()

        assert pipeline is None
        assert any("Unexpected error in transformer NER pipeline" in msg for msg in caplog.messages)
