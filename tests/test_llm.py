"""Tests for the LLM-assisted NER evidence collector (llm.py)."""
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

import taivium.llm as llm_mod
from taivium.engine import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_llama_instance(content: str) -> MagicMock:
    """Returns a mock Llama instance that returns the given content."""
    llama = MagicMock()
    response = {"choices": [{"text": content}]}
    llama.return_value = response
    return llama


def _get_mock_llama_module(content: str) -> MagicMock:
    """Returns a mock llama_cpp module with a Llama class."""
    mock_module = MagicMock()
    mock_llama_class = MagicMock(return_value=_mock_llama_instance(content))
    mock_module.Llama = mock_llama_class
    return mock_module


# ---------------------------------------------------------------------------
# API key guard
# ---------------------------------------------------------------------------

class TestApiKeyGuard:
    def test_returns_empty_and_warns_when_key_missing(self, monkeypatch):
        """Without LLM_MODEL_PATH the function emits a warning and returns []."""
        monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
        # Reset the warning flag so the warning is always emitted for this test
        llm_mod._WarnState.warned_no_api_key = False
        with pytest.warns(RuntimeWarning, match="LLM_MODEL_PATH"):
            result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_proceeds_when_key_is_set(self, monkeypatch):
        """When the model path is set the function attempts inference (mocked here)."""
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        mock_module = _get_mock_llama_module(json.dumps([]))
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("No entities here.")
        assert result == []


# ---------------------------------------------------------------------------
# Basic entity extraction
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_person_and_email_extracted(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([
            {"text": "Alice", "type": "PERSON"},
            {"text": "alice@acme.com", "type": "EMAIL"},
        ])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice sent from alice@acme.com today.")
        labels = {e.label for e in result}
        assert "PERSON" in labels
        assert "EMAIL" in labels

    def test_org_extracted(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Acme Corp", "type": "ORG"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("She joined Acme Corp last year.")
        assert any(e.label == "ORG" for e in result)

    def test_location_mapped(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "London", "type": "LOCATION"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("She lives in London.")
        assert result[0].label == "LOCATION"


# ---------------------------------------------------------------------------
# Character offset recovery
# ---------------------------------------------------------------------------

class TestOffsetRecovery:
    def test_offsets_match_text_span(self, monkeypatch):
        text = "Alice works at Acme Corp."
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence(text)
        assert len(result) == 1
        assert text[result[0].start:result[0].end] == "Alice"

    def test_all_occurrences_of_entity_found(self, monkeypatch):
        """Each occurrence of the entity surface form gets its own Evidence record."""
        text = "Alice met Alice at the Alice conference."
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence(text)
        assert len(result) == 3
        for ev in result:
            assert text[ev.start:ev.end] == "Alice"

    def test_entity_not_present_in_text_produces_no_evidence(self, monkeypatch):
        """If the LLM hallucinates an entity that isn't in the text, nothing is added."""
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Bob", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("There is no one here.")
        assert result == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_surface_label_pair_added_once(self, monkeypatch):
        """Duplicate (surface, label) pairs from the LLM are deduplicated."""
        text = "Alice is Alice."
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        # LLM returns the same entity twice — should be treated as one unique surface
        payload = json.dumps([
            {"text": "Alice", "type": "PERSON"},
            {"text": "Alice", "type": "PERSON"},
        ])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence(text)
        # Two occurrences in text → 2 Evidence records (from the one unique surface)
        assert len(result) == 2

    def test_same_surface_different_labels_both_kept(self, monkeypatch):
        """Same text with two different labels generates evidence for both."""
        text = "Python is great and Python Inc. is a company."
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([
            {"text": "Python", "type": "ORG"},
            {"text": "Python", "type": "PERSON"},
        ])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence(text)
        labels = {e.label for e in result}
        assert "ORG" in labels
        assert "PERSON" in labels


# ---------------------------------------------------------------------------
# Label filtering
# ---------------------------------------------------------------------------

class TestLabelFiltering:
    def test_unknown_type_is_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "tomorrow", "type": "DATE"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("See you tomorrow.")
        assert result == []

    def test_empty_type_is_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": ""}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert result == []

    def test_empty_text_is_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert result == []


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:
    def test_json_in_backtick_fence_parsed(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        raw = "```\n[{\"text\": \"Alice\", \"type\": \"PERSON\"}]\n```"
        mock_module = _get_mock_llama_module(raw)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert any(e.label == "PERSON" for e in result)

    def test_json_in_json_fence_parsed(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        raw = "```json\n[{\"text\": \"Alice\", \"type\": \"PERSON\"}]\n```"
        mock_module = _get_mock_llama_module(raw)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert any(e.label == "PERSON" for e in result)


# ---------------------------------------------------------------------------
# Error / degradation paths
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_api_exception_is_raised(self, monkeypatch):
        """Any exception during inference is logged and re-raised."""
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")

        def _explode(**kw):
            raise RuntimeError("model load error")

        mock_module = MagicMock()
        mock_module.Llama = _explode
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        with pytest.raises(RuntimeError, match="model load error"):
            llm_mod.llm_evidence("Alice works at Acme.")

    def test_invalid_json_raises_error(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        mock_module = _get_mock_llama_module("not valid json {{{")
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        with pytest.raises(Exception):
            llm_mod.llm_evidence("Alice works at Acme.")

    def test_non_list_json_returns_empty(self, monkeypatch):
        """Non-list JSON is still acceptable (returns empty evidence)."""
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        mock_module = _get_mock_llama_module('{"text": "Alice"}')
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_non_dict_items_in_list_skipped(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps(["not a dict", None, 42])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_none_response_content_raises_error(self, monkeypatch):
        """If the model returns None content, raises an error."""
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        mock_module = _get_mock_llama_module(None)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        with pytest.raises(Exception):
            llm_mod.llm_evidence("Alice works at Acme.")


# ---------------------------------------------------------------------------
# Evidence field correctness
# ---------------------------------------------------------------------------

class TestEvidenceFields:
    def test_source_is_llm(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(e.source == "llm" for e in result)

    def test_confidence_is_0_85(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(e.confidence == pytest.approx(0.85) for e in result)

    def test_returns_evidence_instances(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        mock_module = _get_mock_llama_module(payload)
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(isinstance(e, Evidence) for e in result)


# ---------------------------------------------------------------------------
# Model env var
# ---------------------------------------------------------------------------

class TestModelEnvVar:
    def test_default_gpu_layers_is_zero(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        monkeypatch.delenv("LLM_N_GPU_LAYERS", raising=False)
        mock_llama_class = MagicMock(return_value=_mock_llama_instance("[]"))
        mock_module = MagicMock()
        mock_module.Llama = mock_llama_class
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        llm_mod.llm_evidence("Alice is here.")
        _, call_kwargs = mock_llama_class.call_args
        assert call_kwargs["n_gpu_layers"] == 0

    def test_custom_gpu_layers_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
        monkeypatch.setenv("LLM_N_GPU_LAYERS", "35")
        mock_llama_class = MagicMock(return_value=_mock_llama_instance("[]"))
        mock_module = MagicMock()
        mock_module.Llama = mock_llama_class
        monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
        llm_mod.llm_evidence("Alice is here.")
        _, call_kwargs = mock_llama_class.call_args
        assert call_kwargs["n_gpu_layers"] == 35


def test_llm_warns_on_missing_key(monkeypatch):
    monkeypatch.delenv("LLM_MODEL_PATH", raising=False)
    llm_mod._WarnState.warned_no_api_key = False
    with pytest.warns(RuntimeWarning):
        llm_mod.llm_evidence("test")


def test_llm_handles_inference_error(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", "/path/to/model.gguf")
    class FakeLlama:
        def __init__(self, **kwargs):
            raise RuntimeError("model not found")
    mock_module = MagicMock()
    mock_module.Llama = FakeLlama
    monkeypatch.setitem(sys.modules, "llama_cpp", mock_module)
    with pytest.raises(RuntimeError, match="model not found"):
        llm_mod.llm_evidence("test")


def test_llm_warn_state_is_warned_and_reset():
    """_WarnState.is_warned() and reset_warning() work correctly (lines 35, 40)."""
    from taivium import llm as llm_module
    
    llm_module._WarnState.reset_warning()
    assert not llm_module._WarnState.is_warned()
    
    # Manually set warned state
    llm_module._WarnState.warned_no_api_key = True
    assert llm_module._WarnState.is_warned()
    
    # Reset
    llm_module._WarnState.reset_warning()
    assert not llm_module._WarnState.is_warned()