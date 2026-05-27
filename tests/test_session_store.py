"""Tests for session identity store backends.

Covers:
- InMemorySessionStore: get/set/set_many/get_all/clear
- RedisSessionStore: same interface via fakeredis (no real Redis required)
- Taivium.session_store integration: mapping is persisted per call
- Enum/tuple serialization round-trip for Redis store
- Cross-call identity persistence: same entity_id across separate process() calls
- reset_session (clear) removes all entries
"""

from __future__ import annotations
import sys
from typing import Any, Dict

import pytest

from taivium.engine import Taivium
from taivium.session_store import InMemorySessionStore, RedisSessionStore, _serialize_metadata
import taivium.session_store as store_mod


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------

class TestInMemorySessionStore:
    """Tests for the in-process InMemorySessionStore backend."""
    def test_get_returns_none_when_empty(self) -> None:
        """get() on an empty store returns None."""
        store = InMemorySessionStore()
        assert store.get("PERSON_abc123") is None

    def test_set_and_get_roundtrip(self) -> None:
        """set() persists metadata; get() retrieves it."""
        store = InMemorySessionStore()
        meta: Dict[str, Any] = {"text": "Alice", "label": "PERSON"}
        store.set("PERSON_abc123", meta)
        assert store.get("PERSON_abc123") == meta

    def test_set_many_stores_all_entries(self) -> None:
        """set_many() stores multiple entries atomically."""
        store = InMemorySessionStore()
        mapping = {
            "PERSON_aaa": {"text": "Alice", "label": "PERSON"},
            "EMAIL_bbb": {"text": "alice@acme.com", "label": "EMAIL"},
        }
        store.set_many(mapping)
        assert store.get("PERSON_aaa") == mapping["PERSON_aaa"]
        assert store.get("EMAIL_bbb") == mapping["EMAIL_bbb"]

    def test_get_all_returns_all_entries(self) -> None:
        """get_all() returns every stored entry."""
        store = InMemorySessionStore()
        store.set("PERSON_aaa", {"text": "Alice"})
        store.set("ORG_bbb", {"text": "Acme"})
        result = store.get_all()
        assert set(result.keys()) == {"PERSON_aaa", "ORG_bbb"}

    def test_get_all_returns_shallow_copy(self) -> None:
        """Mutating the get_all() result does not affect the store."""
        store = InMemorySessionStore()
        store.set("PERSON_aaa", {"text": "Alice"})
        snapshot = store.get_all()
        snapshot["NEW_KEY"] = {"text": "injected"}
        assert store.get("NEW_KEY") is None

    def test_clear_removes_all_entries(self) -> None:
        """clear() empties the store completely."""
        store = InMemorySessionStore()
        store.set("PERSON_aaa", {"text": "Alice"})
        store.clear()
        assert not store.get_all()

    def test_overwrite_existing_key(self) -> None:
        """set() overwrites a previously stored entry for the same key."""
        store = InMemorySessionStore()
        store.set("PERSON_aaa", {"text": "Alice"})
        store.set("PERSON_aaa", {"text": "Bob"})
        assert store.get("PERSON_aaa") == {"text": "Bob"}


# ---------------------------------------------------------------------------
# _serialize_metadata
# ---------------------------------------------------------------------------

class TestSerializeMetadata:
    """Tests for the _serialize_metadata helper."""
    def test_enum_is_serialized_to_value(self) -> None:
        """Enum values are stored as their .value strings."""
        from taivium.engine import PolicyAction, RiskLevel  # pylint: disable=import-outside-toplevel
        meta = {"action": PolicyAction.ANONYMIZE, "risk": RiskLevel.HIGH}
        result = _serialize_metadata(meta)
        assert result["action"] == "anonymize"
        assert result["risk"] == "high"

    def test_tuple_is_serialized_to_list(self) -> None:
        """Tuple values are stored as lists."""
        meta = {"evidence_sources": ("spacy", "regex")}
        result = _serialize_metadata(meta)
        assert result["evidence_sources"] == ["spacy", "regex"]

    def test_plain_types_pass_through(self) -> None:
        """Strings, ints, and floats are unchanged."""
        meta = {"text": "Alice", "confidence": 0.9, "start": 0}
        assert _serialize_metadata(meta) == meta


# ---------------------------------------------------------------------------
# RedisSessionStore (via fakeredis)
# ---------------------------------------------------------------------------

fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")


class TestRedisSessionStore:
    """Tests for the Redis-backed session store (via fakeredis, no real Redis needed)."""
    @pytest.fixture()
    def redis_store(
            self, monkeypatch: pytest.MonkeyPatch) -> RedisSessionStore:  # type: ignore[return]
        """Returns a RedisSessionStore backed by a fakeredis server."""
        fake_server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)

        import redis  # pylint: disable=import-outside-toplevel  # type: ignore[import]
        monkeypatch.setattr(redis, "from_url", lambda *_a, **_kw: fake_client)

        return RedisSessionStore(session_id="test-session-001")

    def test_get_returns_none_when_empty(self, redis_store: RedisSessionStore) -> None:
        """get() returns None for an unknown key in an empty Redis store."""
        assert redis_store.get("PERSON_abc") is None

    def test_set_and_get_roundtrip(self, redis_store: RedisSessionStore) -> None:
        """set() + get() round-trips plain-string metadata."""
        redis_store.set("PERSON_abc", {"text": "Alice", "label": "PERSON"})
        result = redis_store.get("PERSON_abc")
        assert result is not None
        assert result["text"] == "Alice"
        assert result["label"] == "PERSON"

    def test_enum_values_round_trip(self, redis_store: RedisSessionStore) -> None:
        """Enum fields are serialized to their .value on write and returned as strings."""
        from taivium.engine import PolicyAction, RiskLevel  # pylint: disable=import-outside-toplevel
        redis_store.set("EMAIL_abc", {
            "text": "alice@acme.com",
            "action": PolicyAction.ANONYMIZE,
            "risk": RiskLevel.HIGH,
        })
        result = redis_store.get("EMAIL_abc")
        assert result["action"] == "anonymize"
        assert result["risk"] == "high"

    def test_tuple_values_round_trip(self, redis_store: RedisSessionStore) -> None:
        """Tuple fields are serialized as lists and returned as lists."""
        redis_store.set("PERSON_abc", {"evidence_sources": ("spacy", "regex")})
        result = redis_store.get("PERSON_abc")
        assert result["evidence_sources"] == ["spacy", "regex"]

    def test_set_many_and_get_all(self, redis_store: RedisSessionStore) -> None:
        """set_many() stores multiple entries; get_all() retrieves all."""
        mapping = {
            "PERSON_aaa": {"text": "Alice", "label": "PERSON"},
            "EMAIL_bbb": {"text": "alice@acme.com", "label": "EMAIL"},
        }
        redis_store.set_many(mapping)
        result = redis_store.get_all()
        assert set(result.keys()) == {"PERSON_aaa", "EMAIL_bbb"}
        assert result["PERSON_aaa"]["text"] == "Alice"

    def test_clear_removes_all_entries(self, redis_store: RedisSessionStore) -> None:
        """clear() deletes all session keys from Redis."""
        redis_store.set("PERSON_aaa", {"text": "Alice"})
        redis_store.clear()
        assert redis_store.get_all() == {}

    def test_session_namespacing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two stores with different session_ids do not share entries."""
        fake_server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)

        import redis  # pylint: disable=import-outside-toplevel  # type: ignore[import]
        monkeypatch.setattr(redis, "from_url", lambda *_a, **_kw: fake_client)

        store_a = RedisSessionStore(session_id="session-A")
        store_b = RedisSessionStore(session_id="session-B")

        store_a.set("PERSON_aaa", {"text": "Alice"})
        assert store_b.get("PERSON_aaa") is None
        assert store_a.get("PERSON_aaa") is not None


# ---------------------------------------------------------------------------
# Taivium + session store integration
# ---------------------------------------------------------------------------

class TestPipelineSessionStoreIntegration:
    """Integration tests: Taivium with session store backends."""
    def test_process_saves_mapping_to_inmemory_store(self) -> None:
        """process() persists the call mapping to the session store."""
        store = InMemorySessionStore()
        pipeline = Taivium(session_store=store)
        result = pipeline.process("alice@acme.com sent an email.")
        # The mapping returned by process() must match what's in the store.
        for eid, meta in result["mapping"].items():
            stored = store.get(eid)
            assert stored is not None
            assert stored["text"] == meta["text"]

    def test_session_store_accumulates_across_calls(self) -> None:
        """Separate process() calls accumulate in the same store."""
        store = InMemorySessionStore()
        pipeline = Taivium(session_store=store)
        pipeline.process("alice@acme.com sent an email.")
        pipeline.process("bob@acme.com replied.")
        all_entries = store.get_all()
        texts = {v["text"] for v in all_entries.values()}
        assert "alice@acme.com" in texts
        assert "bob@acme.com" in texts

    def test_default_pipeline_uses_inmemory_store(self) -> None:
        """Taivium() with no store arg uses InMemorySessionStore."""
        pipeline = Taivium()
        assert isinstance(pipeline.session_store, InMemorySessionStore)

    def test_process_with_redis_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """process() works correctly with a Redis-backed store."""
        fake_server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)

        import redis  # pylint: disable=import-outside-toplevel  # type: ignore[import]
        monkeypatch.setattr(redis, "from_url", lambda *_a, **_kw: fake_client)

        store = RedisSessionStore(session_id="pipeline-test")
        pipeline = Taivium(session_store=store)
        result = pipeline.process("alice@acme.com sent an email.")

        for eid in result["mapping"]:
            assert store.get(eid) is not None

    def test_cross_call_identity_is_stable(self) -> None:
        """The same entity text maps to the same ID in both calls."""
        store = InMemorySessionStore()
        pipeline = Taivium(session_store=store)
        r1 = pipeline.process("alice@acme.com sent an email.")
        r2 = pipeline.process("Reply from alice@acme.com.")

        ids_1 = {v["text"]: k for k, v in r1["mapping"].items()}
        ids_2 = {v["text"]: k for k, v in r2["mapping"].items()}

        if "alice@acme.com" in ids_1 and "alice@acme.com" in ids_2:
            assert ids_1["alice@acme.com"] == ids_2["alice@acme.com"], (
                "Same entity must map to same ID across calls"
            )

def test_redis_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    with pytest.raises(ImportError):
        store_mod.RedisSessionStore(session_id="abc")

# --- session_store.py: set_many Redis pipeline ---
    class FakeClient:
        def pipeline(self):
            class Pipe:
                def __init__(self): self.calls = []
                def set(self, k, v, ex=None): self.calls.append((k, v, ex))
                def execute(self): return self.calls
            return Pipe()
    s = store_mod.RedisSessionStore.__new__(store_mod.RedisSessionStore)
    s._client = FakeClient()
    s._prefix = "test:"
    s._ttl = 123
    s.set_many({"id": {"meta": 1}})  # Should not raise