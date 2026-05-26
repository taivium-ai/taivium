"""
Tests that validate the examples shown in examples/example_client.py.

All tests are fully offline — no OpenAI API key or network access is required.
A fake PrivacyClient is injected via the ``client=`` parameter exposed by the
example function, following the same pattern as test_client.py.

Any breaking change to the example — renamed import, changed constructor
signature, altered return shape, or removed method — is caught at the import
or call site.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from example_client import (  # pylint: disable=import-error
    EXAMPLE_MESSAGES,
    run_basic_client_example,
)
from taivium.client import (  # pyright: ignore[reportPrivateImportUsage]
    PrivacyClient,
    _PrivacyChat,
)
from taivium.engine import Taivium

# pylint: disable=protected-access,too-few-public-methods,duplicate-code


# ---------------------------------------------------------------------------
# Fake OpenAI infrastructure (no network, no openai package required)
# ---------------------------------------------------------------------------

def _fake_response(content: str) -> Any:
    """Build a minimal ChatCompletion-shaped object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeCompletions:
    def __init__(self, response_content: str = "Noted.") -> None:
        self.calls: List[Dict[str, Any]] = []
        self._response_content = response_content

    def create(self, *, messages, **kwargs) -> Any:
        """Record call and return preset fake response."""
        self.calls.append({"messages": messages, **kwargs})
        return _fake_response(self._response_content)


class _FakeChat:
    def __init__(self, response_content: str = "Noted.") -> None:
        self.completions = _FakeCompletions(response_content)


def _make_example_client(
    response_content: str = "I'll follow up with them.",
) -> tuple[PrivacyClient, _FakeCompletions]:
    """Return a PrivacyClient wired to a fake OpenAI backend."""
    fake_chat = _FakeChat(response_content)
    client = object.__new__(PrivacyClient)
    client._pipeline = Taivium()
    client.chat = _PrivacyChat(fake_chat, client._pipeline)
    return client, fake_chat.completions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunBasicClientExample:
    """Mirrors run_basic_client_example in examples/example_client.py."""

    def test_pii_is_anonymized_before_llm(self):
        """PII in EXAMPLE_MESSAGES must not reach the fake LLM."""
        client, fake = _make_example_client()
        run_basic_client_example(client=client, verbose=False)

        sent_content = fake.calls[0]["messages"][0]["content"]
        assert "John Doe" not in sent_content
        assert "123-456-7890" not in sent_content

    def test_returns_response_and_session_mapping_keys(self):
        """Return dict must contain exactly 'response' and 'session_mapping'."""
        client, _ = _make_example_client()
        result = run_basic_client_example(client=client, verbose=False)

        assert set(result.keys()) == {"response", "session_mapping"}

    def test_session_mapping_populated_before_reset(self):
        """session_mapping snapshot in the return value is non-empty."""
        client, _ = _make_example_client()
        result = run_basic_client_example(client=client, verbose=False)

        assert len(result["session_mapping"]) > 0

    def test_session_mapping_entries_have_required_keys(self):
        """Each entry in session_mapping has at least 'text' and 'label'."""
        client, _ = _make_example_client()
        result = run_basic_client_example(client=client, verbose=False)

        for token, entry in result["session_mapping"].items():
            assert "text" in entry, f"Entry for {token!r} missing 'text'"
            assert "label" in entry, f"Entry for {token!r} missing 'label'"

    def test_session_reset_after_call(self):
        """reset_session() is called inside the function — client mapping is
        empty after the call returns."""
        client, _ = _make_example_client()
        run_basic_client_example(client=client, verbose=False)

        assert len(client.session_mapping) == 0

    def test_response_is_chatcompletion_shaped(self):
        """The 'response' value has .choices[0].message.content."""
        client, _ = _make_example_client(response_content="Will do.")
        result = run_basic_client_example(client=client, verbose=False)

        assert hasattr(result["response"], "choices")
        assert hasattr(result["response"].choices[0], "message")
        assert isinstance(result["response"].choices[0].message.content, str)

    def test_custom_messages_override_defaults(self):
        """Custom messages replace EXAMPLE_MESSAGES when provided."""
        custom = [{"role": "user", "content": "Alice Smith called."}]
        client, fake = _make_example_client()
        run_basic_client_example(client=client, messages=custom, verbose=False)

        sent_content = fake.calls[0]["messages"][0]["content"]
        assert "Alice Smith" not in sent_content  # anonymized

    def test_default_messages_are_example_messages(self):
        """When no messages are provided the function uses EXAMPLE_MESSAGES."""
        client, _ = _make_example_client()
        run_basic_client_example(client=client, verbose=False)

        # The raw content before anonymization contained the EXAMPLE_MESSAGES text
        # We can verify the session_mapping contains entities from that text
        result = run_basic_client_example(
            client=_make_example_client()[0], verbose=False
        )
        labels = {v["label"] for v in result["session_mapping"].values()}
        # EXAMPLE_MESSAGES contains a phone number and a person name
        assert "PHONE" in labels or "PERSON" in labels

    def test_no_api_key_raises_environment_error(self, monkeypatch):
        """EnvironmentError is raised when client=None and OPENAI_KEY is unset."""
        monkeypatch.delenv("OPENAI_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_KEY"):
            run_basic_client_example(client=None, verbose=False)

    def test_deid_response_reverses_tokens_in_llm_reply(self):
        """deid_response=True causes entity tokens in the LLM reply to be
        replaced with the original values before the response is returned.
        """
        # First pass: process the text to discover the token ID for "John Doe"
        pipeline = Taivium()
        processed = pipeline.process(EXAMPLE_MESSAGES[0]["content"])
        person_token = next(
            (eid for eid, v in processed["mapping"].items() if v["label"] == "PERSON"),
            None,
        )
        if person_token is None:
            pytest.skip("No PERSON entity detected in EXAMPLE_MESSAGES")

        # Set up the fake LLM to echo the token back
        client, _ = _make_example_client(
            response_content=f"I will call {person_token} back."
        )
        result = run_basic_client_example(client=client, verbose=False)

        # deid_response=True should have replaced the token with "John Doe"
        assert "John Doe" in result["response"].choices[0].message.content


# ---------------------------------------------------------------------------
# EXAMPLE_MESSAGES constant
# ---------------------------------------------------------------------------

def test_example_messages_constant():
    """EXAMPLE_MESSAGES is a non-empty list of valid OpenAI message dicts."""
    assert isinstance(EXAMPLE_MESSAGES, list)
    assert len(EXAMPLE_MESSAGES) > 0
    for msg in EXAMPLE_MESSAGES:
        assert "role" in msg
        assert "content" in msg
        assert isinstance(msg["content"], str)
