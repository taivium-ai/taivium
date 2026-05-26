"""Tests for PrivacyClient — OpenAI-compatible drop-in SDK wrapper.

All tests are fully offline: the openai package and network are never used.
The inner OpenAI client is replaced with a minimal fake that records the
messages it receives and returns a configurable fake response.
"""
# pylint: disable=protected-access,duplicate-code
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from tarvium.client import PrivacyClient, _patch_response_content
from tarvium.engine import (
    PolicyAction,
    PolicyEngine,
    PolicyRule,
    Tarvium,
    RiskLevel,
    reverse_transform,
)
from tarvium.session_store import InMemorySessionStore


# ---------------------------------------------------------------------------
# Fake OpenAI infrastructure (no network, no openai package required)
# ---------------------------------------------------------------------------

def _fake_response(content: str) -> Any:
    """Builds a minimal ChatCompletion-shaped object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class _FakeCompletions:  # pylint: disable=too-few-public-methods
    """Records every ``create()`` call and returns a preset response."""

    def __init__(self, response_content: str = "Thank you.") -> None:
        self.calls: List[Dict] = []
        self._response_content = response_content

    def create(self, *, messages, **kwargs) -> Any:
        """Record call and return configured fake response."""
        self.calls.append({"messages": messages, **kwargs})
        return _fake_response(self._response_content)


class _FakeChat:  # pylint: disable=too-few-public-methods
    def __init__(self, response_content: str = "Thank you.") -> None:
        self.completions = _FakeCompletions(response_content)


class _FakeOpenAI:  # pylint: disable=too-few-public-methods
    def __init__(self, response_content: str = "Thank you.", **_kwargs) -> None:
        self.chat = _FakeChat(response_content)


def _make_client(
    response_content: str = "Thank you.",
    policy_engine: PolicyEngine | None = None,
    session_store: InMemorySessionStore | None = None,
) -> tuple[PrivacyClient, _FakeCompletions]:
    """Returns a PrivacyClient wired to a fake OpenAI backend."""
    fake_openai = _FakeOpenAI(response_content)
    with patch("tarvium.client.PrivacyClient.__init__", wraps=lambda self, **kw: None):
        pass  # not used — we build manually below

    # Build directly without going through openai import
    client = object.__new__(PrivacyClient)
    client._pipeline = Tarvium(policy_engine=policy_engine, session_store=session_store)
    from tarvium.client import _PrivacyChat  # pylint: disable=import-outside-toplevel
    client.chat = _PrivacyChat(fake_openai.chat, client._pipeline)
    return client, fake_openai.chat.completions


# ---------------------------------------------------------------------------
# 1. Message anonymisation
# ---------------------------------------------------------------------------

def test_pii_is_anonymised_before_reaching_llm():
    """Email and name in the message must be replaced with tokens before the
    fake LLM receives them."""
    client, fake = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com for details."}],
    )

    sent_content = fake.calls[0]["messages"][0]["content"]
    assert "alice@acme.com" not in sent_content, "Raw email must not reach the LLM"
    assert "EMAIL_" in sent_content, "Email should be replaced with an EMAIL_ token"


def test_non_string_content_forwarded_unchanged():
    """Messages with None content (e.g. tool-call assistant turns) must be
    forwarded as-is without error."""
    client, fake = _make_client()
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "user", "content": "Reply to alice@acme.com"},
    ]
    client.chat.completions.create(model="gpt-4o", messages=messages)

    assert fake.calls[0]["messages"][0]["content"] is None
    assert "alice@acme.com" not in fake.calls[0]["messages"][1]["content"]


def test_message_role_and_extra_keys_preserved():
    """Anonymisation must only mutate ``content``; all other message keys must
    be forwarded unchanged."""
    client, fake = _make_client()
    msg = {"role": "user", "content": "Email bob@example.com", "name": "tester"}
    client.chat.completions.create(model="gpt-4o", messages=[msg])

    sent = fake.calls[0]["messages"][0]
    assert sent["role"] == "user"
    assert sent["name"] == "tester"
    assert "bob@example.com" not in sent["content"]


def test_kwargs_forwarded_to_inner_client():
    """Extra kwargs (model, temperature, …) must reach the inner LLM call."""
    client, fake = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=100,
    )
    assert fake.calls[0]["model"] == "gpt-4o"
    assert fake.calls[0]["temperature"] == 0.2
    assert fake.calls[0]["max_tokens"] == 100


# ---------------------------------------------------------------------------
# 2. Session mapping accumulation
# ---------------------------------------------------------------------------

def test_session_mapping_populated_after_create():
    """session_mapping must contain an entry for every anonymised entity."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    assert len(client.session_mapping) > 0
    original_texts = {v["text"] for v in client.session_mapping.values()}
    assert "alice@acme.com" in original_texts


def test_session_mapping_accumulates_across_calls():
    """Entities from multiple calls must all appear in session_mapping."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "bob@example.com"}],
    )
    texts = {v["text"] for v in client.session_mapping.values()}
    assert "alice@acme.com" in texts
    assert "bob@example.com" in texts


def test_same_entity_maps_to_same_token_across_calls():
    """The same entity appearing in two separate calls must produce the same token."""
    client, fake = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com"}],
    )
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Again alice@acme.com"}],
    )
    token_1 = next(
        eid for eid, m in client.session_mapping.items() if m["text"] == "alice@acme.com"
    )
    # both calls must have emitted the same token
    assert token_1 in fake.calls[0]["messages"][0]["content"]
    assert token_1 in fake.calls[1]["messages"][0]["content"]


def test_session_mapping_is_a_copy():
    """Mutating the returned session_mapping must not affect the client's internal state."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    snapshot = client.session_mapping
    snapshot.clear()
    assert len(client.session_mapping) > 0, "Internal mapping must not be mutated"


def test_reset_session_clears_mapping():
    """reset_session() must empty session_mapping."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    assert len(client.session_mapping) > 0
    client.reset_session()
    assert client.session_mapping == {}


# ---------------------------------------------------------------------------
# 3. deid_response — reverse mapping
# ---------------------------------------------------------------------------

def test_deid_response_restores_entity_in_reply():
    """When deid_response=True the entity token in the LLM reply must be
    replaced with the original value before the response is returned."""
    client, _ = _make_client()

    # First call: get the token emitted for alice@acme.com
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com"}],
    )
    token = next(
        eid for eid, m in client.session_mapping.items() if m["text"] == "alice@acme.com"
    )

    # Second call: LLM "echoes" the token back; deid_response should restore it
    client2, _ = _make_client(response_content=f"You asked about {token}.")
    # Seed client2's session store with the same mapping
    client2._pipeline.session_store.set_many(client.session_mapping)

    response = client2.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Who did I ask about?"}],
        deid_response=True,
    )
    assert "alice@acme.com" in response.choices[0].message.content
    assert token not in response.choices[0].message.content


def test_deid_response_false_leaves_tokens_in_reply():
    """When deid_response=False (default) the response content must not be
    modified — tokens remain as-is."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    token = next(
        eid for eid, m in client.session_mapping.items() if m["text"] == "alice@acme.com"
    )

    client2, _ = _make_client(response_content=f"Token: {token}")
    client2._pipeline.session_store.set_many(client.session_mapping)

    response = client2.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        deid_response=False,
    )
    assert token in response.choices[0].message.content


# ---------------------------------------------------------------------------
# 4. Policy engine integration
# ---------------------------------------------------------------------------

def test_block_policy_raises_before_llm_is_called():
    """A BLOCK-policy entity must raise ValueError and must not reach the LLM."""
    policy = PolicyEngine(policy_table={
        "EMAIL": PolicyRule("EMAIL", PolicyAction.BLOCK, RiskLevel.CRITICAL),
    })
    client, fake = _make_client(policy_engine=policy)

    with pytest.raises(ValueError, match="Blocked"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Email alice@acme.com"}],
        )
    assert len(fake.calls) == 0, "LLM must not be called when an entity is blocked"


def test_allow_policy_passes_entity_through():
    """An ALLOW-policy entity must appear unchanged in the message sent to the LLM."""
    policy = PolicyEngine(policy_table={
        "EMAIL": PolicyRule("EMAIL", PolicyAction.ALLOW, RiskLevel.LOW),
    })
    client, fake = _make_client(policy_engine=policy)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Email alice@acme.com"}],
    )
    sent = fake.calls[0]["messages"][0]["content"]
    assert "alice@acme.com" in sent, "ALLOW policy must leave entity unchanged"


# ---------------------------------------------------------------------------
# 5. reverse_transform (standalone utility)
# ---------------------------------------------------------------------------

def test_reverse_transform_replaces_tokens():
    """Token placeholders in text are replaced with their original values."""
    mapping = {
        "PERSON_abc": {"text": "Alice Johnson"},
        "EMAIL_xyz":  {"text": "alice@acme.com"},
    }
    result = reverse_transform("Hello PERSON_abc, your email is EMAIL_xyz.", mapping)
    assert result == "Hello Alice Johnson, your email is alice@acme.com."


def test_reverse_transform_no_match_returns_unchanged():
    """Text with no matching tokens is returned unchanged."""
    mapping = {"PERSON_abc": {"text": "Alice"}}
    text = "No entities here."
    assert reverse_transform(text, mapping) == text


def test_reverse_transform_longest_token_first():
    """If one token is a prefix of another, the longer one must be replaced first
    to avoid partial substitution corruption."""
    mapping = {
        "PERSON_ab":    {"text": "Bob"},
        "PERSON_abcdef": {"text": "Alice"},
    }
    text = "Hello PERSON_abcdef and PERSON_ab."
    result = reverse_transform(text, mapping)
    assert result == "Hello Alice and Bob."


# ---------------------------------------------------------------------------
# 6. _patch_response_content
# ---------------------------------------------------------------------------

def test_patch_response_content_mutates_in_place():
    """Token in response content is replaced with the original value."""
    mapping = {"EMAIL_tok": {"text": "alice@acme.com"}}
    response = _fake_response("Contact EMAIL_tok for info.")
    _patch_response_content(response, mapping)
    assert response.choices[0].message.content == "Contact alice@acme.com for info."


def test_patch_response_content_ignores_none_content():
    """Responses with None message content must not raise."""
    response = _fake_response(None)
    response.choices[0].message.content = None
    _patch_response_content(response, {"X": {"text": "y"}})  # must not raise


def test_patch_response_content_ignores_malformed_response():
    """Non-standard response shapes (no .choices) must be silently ignored."""
    _patch_response_content(SimpleNamespace(), {"X": {"text": "y"}})  # must not raise


# ---------------------------------------------------------------------------
# 7. PolicyEngine — extended integration via PrivacyClient
# ---------------------------------------------------------------------------

def test_policy_engine_default_action_anonymize_via_client():
    """default_action=ANONYMIZE: unlisted entity labels are anonymised."""
    policy = PolicyEngine(default_action=PolicyAction.ANONYMIZE)
    client, fake = _make_client(policy_engine=policy)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com"}],
    )
    sent = fake.calls[0]["messages"][0]["content"]
    assert "alice@acme.com" not in sent
    assert "EMAIL_" in sent


def test_policy_engine_default_action_allow_via_client():
    """default_action=ALLOW with empty policy_table: entities pass through unchanged."""
    policy = PolicyEngine(policy_table={}, default_action=PolicyAction.ALLOW)
    client, fake = _make_client(policy_engine=policy)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com"}],
    )
    sent = fake.calls[0]["messages"][0]["content"]
    assert "alice@acme.com" in sent


def test_policy_engine_explicit_rule_overrides_default_allow():
    """An explicit BLOCK rule fires even when default_action=ALLOW."""
    policy = PolicyEngine(
        policy_table={"EMAIL": PolicyRule("EMAIL", PolicyAction.BLOCK, RiskLevel.CRITICAL)},
        default_action=PolicyAction.ALLOW,
    )
    client, fake = _make_client(policy_engine=policy)
    with pytest.raises(ValueError, match="Blocked"):
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Email alice@acme.com"}],
        )
    assert len(fake.calls) == 0


def test_policy_engine_explicit_anonymize_rule_replaces_token():
    """An explicit ANONYMIZE rule for a label replaces the entity with a token."""
    policy = PolicyEngine(policy_table={
        "EMAIL": PolicyRule("EMAIL", PolicyAction.ANONYMIZE, RiskLevel.MEDIUM),
    })
    client, fake = _make_client(policy_engine=policy)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Send to alice@acme.com"}],
    )
    sent = fake.calls[0]["messages"][0]["content"]
    assert "alice@acme.com" not in sent
    assert "EMAIL_" in sent


def test_policy_engine_decision_reason_stored_in_mapping():
    """The policy decision reason is recorded in session_mapping metadata."""
    policy = PolicyEngine(policy_table={
        "EMAIL": PolicyRule("EMAIL", PolicyAction.ANONYMIZE, RiskLevel.HIGH),
    })
    client, _ = _make_client(policy_engine=policy)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    reasons = {v.get("reason") for v in client.session_mapping.values()}
    assert any(r is not None for r in reasons), "reason should be present in metadata"


# ---------------------------------------------------------------------------
# 8. SessionStore integration via PrivacyClient
# ---------------------------------------------------------------------------

def test_client_uses_inmemory_store_by_default():
    """The default pipeline uses InMemorySessionStore."""
    client, _ = _make_client()
    assert isinstance(client._pipeline.session_store, InMemorySessionStore)


def test_client_custom_store_receives_entries():
    """A custom InMemorySessionStore injected at construction is populated by chat calls."""
    custom_store = InMemorySessionStore()
    client, _ = _make_client(session_store=custom_store)
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Contact alice@acme.com"}],
    )
    texts = {v["text"] for v in custom_store.get_all().values()}
    assert "alice@acme.com" in texts


def test_session_mapping_matches_store_get_all():
    """session_mapping always reflects the store's current state."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    assert client.session_mapping == client._pipeline.session_store.get_all()


def test_reset_session_empties_store():
    """reset_session() clears the underlying store — get_all() returns {} afterwards."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    assert len(client._pipeline.session_store.get_all()) > 0
    client.reset_session()
    assert client._pipeline.session_store.get_all() == {}


def test_two_clients_sharing_store_see_each_others_entities():
    """Two clients backed by the same store accumulate a unified entity mapping."""
    shared_store = InMemorySessionStore()
    client_a, _ = _make_client(session_store=shared_store)
    client_b, _ = _make_client(session_store=shared_store)

    client_a.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    client_b.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "bob@example.com"}],
    )
    texts = {v["text"] for v in shared_store.get_all().values()}
    assert "alice@acme.com" in texts
    assert "bob@example.com" in texts


def test_store_entry_metadata_includes_label():
    """Each store entry carries the entity label (e.g. EMAIL, PERSON)."""
    client, _ = _make_client()
    client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "alice@acme.com"}],
    )
    labels = {v.get("label") for v in client._pipeline.session_store.get_all().values()}
    assert "EMAIL" in labels
