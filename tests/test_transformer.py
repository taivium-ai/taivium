"""Tests for the transformer-based NER evidence collector (transformer.py)."""
import sys
import pytest

import taivium.transformer as tr
from taivium.engine import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pipeline(predictions: list):
    """Returns a callable mock NER pipeline that echoes *predictions*."""
    def _call(_text: str):
        return predictions
    return _call


def _clear_cache():
    """Clears the _get_ner_pipeline lru_cache between tests."""
    tr._get_ner_pipeline.cache_clear()


# ---------------------------------------------------------------------------
# _get_ner_pipeline
# ---------------------------------------------------------------------------

class TestGetNerPipeline:
    def test_returns_none_and_warns_when_transformers_missing(self, monkeypatch):
        """When 'transformers' is not importable, returns None with a RuntimeWarning."""
        _clear_cache()
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.warns(RuntimeWarning, match="Transformer NER pipeline unavailable"):
            result = tr._get_ner_pipeline()
        assert result is None
        _clear_cache()

    def test_returns_none_and_warns_on_os_error(self, monkeypatch):
        """OSError during model loading (e.g. missing weights) also returns None."""
        _clear_cache()

        class _FakeTransformers:
            @staticmethod
            def pipeline(*args, **kwargs):
                raise OSError("model not found")

        monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers)
        with pytest.warns(RuntimeWarning):
            result = tr._get_ner_pipeline()
        assert result is None
        _clear_cache()


# ---------------------------------------------------------------------------
# transformer_evidence — pipeline unavailable / error paths
# ---------------------------------------------------------------------------

class TestTransformerEvidenceNoPipeline:
    def test_returns_empty_when_pipeline_is_none(self, monkeypatch):
        """Returns [] when the NER pipeline could not be loaded."""
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: None)
        assert tr.transformer_evidence("Alice works at Acme Corp.") == []

    def test_returns_empty_on_pipeline_runtime_error(self, monkeypatch):
        """A RuntimeError raised by the pipeline is swallowed; returns []."""
        def _bad(_text):
            raise RuntimeError("CUDA out of memory")
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _bad)
        assert tr.transformer_evidence("Alice works at Acme Corp.") == []

    def test_returns_empty_on_pipeline_value_error(self, monkeypatch):
        """A ValueError raised by the pipeline is swallowed; returns []."""
        def _bad(_text):
            raise ValueError("unexpected input")
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _bad)
        assert tr.transformer_evidence("test") == []


# ---------------------------------------------------------------------------
# transformer_evidence — label mapping
# ---------------------------------------------------------------------------

class TestLabelMapping:
    def test_per_maps_to_person(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": 0, "end": 5, "score": 0.95}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice is here.")
        assert len(result) == 1
        assert result[0].label == "PERSON"

    def test_org_stays_org(self, monkeypatch):
        preds = [{"entity_group": "ORG", "start": 10, "end": 19, "score": 0.90}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice at Acme Corp.")
        assert result[0].label == "ORG"

    def test_loc_maps_to_location(self, monkeypatch):
        preds = [{"entity_group": "LOC", "start": 9, "end": 15, "score": 0.88}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Based in London.")
        assert result[0].label == "LOCATION"

    def test_misc_is_skipped(self, monkeypatch):
        """MISC is too ambiguous — produces no Evidence."""
        preds = [{"entity_group": "MISC", "start": 0, "end": 5, "score": 0.80}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence("Oscar goes there.") == []

    def test_unknown_entity_group_is_skipped(self, monkeypatch):
        """Entity groups outside the label map produce no Evidence."""
        preds = [{"entity_group": "DATE", "start": 0, "end": 8, "score": 0.75}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence("Tomorrow is uncertain.") == []

    def test_missing_entity_group_key_is_skipped(self, monkeypatch):
        """Predictions missing 'entity_group' key map to UNKNOWN and are skipped."""
        preds = [{"start": 0, "end": 5, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence("Alice is here.") == []


# ---------------------------------------------------------------------------
# transformer_evidence — span validation
# ---------------------------------------------------------------------------

class TestSpanValidation:
    def test_start_equals_end_is_filtered(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": 3, "end": 3, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence("Hi Alice.") == []

    def test_negative_start_is_filtered(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": -1, "end": 5, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence("Alice is here.") == []

    def test_end_beyond_text_length_is_filtered(self, monkeypatch):
        text = "Alice"
        preds = [{"entity_group": "PER", "start": 0, "end": len(text) + 1, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        assert tr.transformer_evidence(text) == []

    def test_valid_span_at_text_boundaries(self, monkeypatch):
        """Span covering the entire text is valid."""
        text = "Alice"
        preds = [{"entity_group": "PER", "start": 0, "end": len(text), "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence(text)
        assert len(result) == 1
        assert result[0].start == 0
        assert result[0].end == len(text)


# ---------------------------------------------------------------------------
# transformer_evidence — Evidence field correctness
# ---------------------------------------------------------------------------

class TestEvidenceFields:
    def test_source_is_transformer(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": 0, "end": 5, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice is here.")
        assert all(e.source == "transformer" for e in result)

    def test_confidence_is_preserved(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": 0, "end": 5, "score": 0.9321}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice is here.")
        assert result[0].confidence == pytest.approx(0.9321)

    def test_start_end_are_preserved(self, monkeypatch):
        preds = [{"entity_group": "ORG", "start": 10, "end": 19, "score": 0.85}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice at Acme Corp.")
        assert result[0].start == 10
        assert result[0].end == 19

    def test_returns_evidence_instances(self, monkeypatch):
        preds = [{"entity_group": "PER", "start": 0, "end": 5, "score": 0.9}]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence("Alice is here.")
        assert all(isinstance(e, Evidence) for e in result)


# ---------------------------------------------------------------------------
# transformer_evidence — multiple entities
# ---------------------------------------------------------------------------

class TestMultipleEntities:
    def test_multiple_predictions_all_converted(self, monkeypatch):
        text = "Alice works at Acme Corp in London."
        preds = [
            {"entity_group": "PER", "start": 0, "end": 5, "score": 0.95},
            {"entity_group": "ORG", "start": 15, "end": 24, "score": 0.92},
            {"entity_group": "LOC", "start": 28, "end": 34, "score": 0.88},
        ]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence(text)
        assert len(result) == 3
        assert {e.label for e in result} == {"PERSON", "ORG", "LOCATION"}

    def test_mixed_valid_invalid_predictions(self, monkeypatch):
        """Valid predictions pass through while invalid ones are silently dropped."""
        text = "Alice at Acme."
        preds = [
            {"entity_group": "PER", "start": 0, "end": 5, "score": 0.95},   # valid
            {"entity_group": "MISC", "start": 9, "end": 13, "score": 0.70}, # skipped
            {"entity_group": "ORG", "start": 9, "end": 13, "score": 0.85},  # valid
        ]
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline(preds))
        result = tr.transformer_evidence(text)
        assert len(result) == 2
        assert {e.label for e in result} == {"PERSON", "ORG"}

    def test_empty_predictions_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: _pipeline([]))
        assert tr.transformer_evidence("No entities here.") == []
