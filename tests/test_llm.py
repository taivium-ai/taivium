"""Tests for the LLM-assisted NER evidence collector (llm.py)."""
import json
from unittest.mock import MagicMock

import pytest

import taivium.llm as llm_mod
from taivium.engine import Evidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_response(content: str) -> MagicMock:
    """Builds a minimal mock that looks like an openai ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_client(content: str) -> MagicMock:
    """Returns a mock OpenAI client whose completions.create returns *content*."""
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(content)
    return client


# ---------------------------------------------------------------------------
# API key guard
# ---------------------------------------------------------------------------

class TestApiKeyGuard:
    def test_returns_empty_and_warns_when_key_missing(self, monkeypatch):
        """Without OPENAI_API_KEY the function emits a warning and returns []."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Reset the warning flag so the warning is always emitted for this test
        llm_mod._WarnState.warned_no_api_key = False
        with pytest.warns(RuntimeWarning, match="OPENAI_API_KEY"):
            result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_proceeds_when_key_is_set(self, monkeypatch):
        """When the key is set the function attempts the API call (mocked here)."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(
            "openai.OpenAI",
            lambda **kw: _mock_client(json.dumps([])),
        )
        result = llm_mod.llm_evidence("No entities here.")
        assert result == []


# ---------------------------------------------------------------------------
# Basic entity extraction
# ---------------------------------------------------------------------------

class TestBasicExtraction:
    def test_person_and_email_extracted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([
            {"text": "Alice", "type": "PERSON"},
            {"text": "alice@acme.com", "type": "EMAIL"},
        ])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice sent from alice@acme.com today.")
        labels = {e.label for e in result}
        assert "PERSON" in labels
        assert "EMAIL" in labels

    def test_org_extracted(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Acme Corp", "type": "ORG"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("She joined Acme Corp last year.")
        assert any(e.label == "ORG" for e in result)

    def test_location_mapped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "London", "type": "LOCATION"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("She lives in London.")
        assert result[0].label == "LOCATION"


# ---------------------------------------------------------------------------
# Character offset recovery
# ---------------------------------------------------------------------------

class TestOffsetRecovery:
    def test_offsets_match_text_span(self, monkeypatch):
        text = "Alice works at Acme Corp."
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence(text)
        assert len(result) == 1
        assert text[result[0].start:result[0].end] == "Alice"

    def test_all_occurrences_of_entity_found(self, monkeypatch):
        """Each occurrence of the entity surface form gets its own Evidence record."""
        text = "Alice met Alice at the Alice conference."
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence(text)
        assert len(result) == 3
        for ev in result:
            assert text[ev.start:ev.end] == "Alice"

    def test_entity_not_present_in_text_produces_no_evidence(self, monkeypatch):
        """If the LLM hallucinates an entity that isn't in the text, nothing is added."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Bob", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("There is no one here.")
        assert result == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_surface_label_pair_added_once(self, monkeypatch):
        """Duplicate (surface, label) pairs from the LLM are deduplicated."""
        text = "Alice is Alice."
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # LLM returns the same entity twice — should be treated as one unique surface
        payload = json.dumps([
            {"text": "Alice", "type": "PERSON"},
            {"text": "Alice", "type": "PERSON"},
        ])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence(text)
        # Two occurrences in text → 2 Evidence records (from the one unique surface)
        assert len(result) == 2

    def test_same_surface_different_labels_both_kept(self, monkeypatch):
        """Same text with two different labels generates evidence for both."""
        text = "Python is great and Python Inc. is a company."
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([
            {"text": "Python", "type": "ORG"},
            {"text": "Python", "type": "PERSON"},
        ])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence(text)
        labels = {e.label for e in result}
        assert "ORG" in labels
        assert "PERSON" in labels


# ---------------------------------------------------------------------------
# Label filtering
# ---------------------------------------------------------------------------

class TestLabelFiltering:
    def test_unknown_type_is_skipped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "tomorrow", "type": "DATE"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("See you tomorrow.")
        assert result == []

    def test_empty_type_is_skipped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": ""}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice is here.")
        assert result == []

    def test_empty_text_is_skipped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice is here.")
        assert result == []


# ---------------------------------------------------------------------------
# Markdown fence stripping
# ---------------------------------------------------------------------------

class TestMarkdownFenceStripping:
    def test_json_in_backtick_fence_parsed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        raw = "```\n[{\"text\": \"Alice\", \"type\": \"PERSON\"}]\n```"
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(raw))
        result = llm_mod.llm_evidence("Alice is here.")
        assert any(e.label == "PERSON" for e in result)

    def test_json_in_json_fence_parsed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        raw = "```json\n[{\"text\": \"Alice\", \"type\": \"PERSON\"}]\n```"
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(raw))
        result = llm_mod.llm_evidence("Alice is here.")
        assert any(e.label == "PERSON" for e in result)


# ---------------------------------------------------------------------------
# Error / degradation paths
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_api_exception_returns_empty(self, monkeypatch):
        """Any exception during the API call is swallowed; returns []."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        def _explode(**kw):
            raise RuntimeError("network error")

        monkeypatch.setattr("openai.OpenAI", _explode)
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_invalid_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client("not valid json {{{"))
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_non_list_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client('{"text": "Alice"}'))
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_non_dict_items_in_list_skipped(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps(["not a dict", None, 42])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []

    def test_none_response_content_returns_empty(self, monkeypatch):
        """If the model returns None content, returns []."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(None))
        result = llm_mod.llm_evidence("Alice works at Acme.")
        assert result == []


# ---------------------------------------------------------------------------
# Evidence field correctness
# ---------------------------------------------------------------------------

class TestEvidenceFields:
    def test_source_is_llm(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(e.source == "llm" for e in result)

    def test_confidence_is_0_85(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(e.confidence == pytest.approx(0.85) for e in result)

    def test_returns_evidence_instances(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        payload = json.dumps([{"text": "Alice", "type": "PERSON"}])
        monkeypatch.setattr("openai.OpenAI", lambda **kw: _mock_client(payload))
        result = llm_mod.llm_evidence("Alice is here.")
        assert all(isinstance(e, Evidence) for e in result)


# ---------------------------------------------------------------------------
# Model env var
# ---------------------------------------------------------------------------

class TestModelEnvVar:
    def test_default_model_is_gpt_4o_mini(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("PRIVACYZE_LLM_MODEL", raising=False)
        client = _mock_client("[]")
        monkeypatch.setattr("openai.OpenAI", lambda **kw: client)
        llm_mod.llm_evidence("Alice is here.")
        _, call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_custom_model_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("PRIVACYZE_LLM_MODEL", "gpt-4o")
        client = _mock_client("[]")
        monkeypatch.setattr("openai.OpenAI", lambda **kw: client)
        llm_mod.llm_evidence("Alice is here.")
        _, call_kwargs = client.chat.completions.create.call_args
        assert call_kwargs["model"] == "gpt-4o"
