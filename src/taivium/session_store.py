"""
Session Identity Store
----------------------------------------------------------------------
Pluggable backends for persisting entity-ID → original metadata
mapping across pipeline calls, processes, or nodes.

Default:   InMemorySessionStore  — in-process, cleared on restart.
Optional:  RedisSessionStore     — Redis-backed, survives restarts,
           shared across processes, with configurable TTL.

Usage::

    from taivium.session_store import RedisSessionStore
    from taivium import Taivium

    store = RedisSessionStore(session_id="user-abc123")
    pipeline = Taivium(session_store=store)

    result = pipeline.process("Alice at alice@acme.com needs help.")
    # mapping is persisted in Redis under key taivium:default:session:user-abc123:*
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Protocol, cast


class SessionStore(Protocol):
    """Protocol defining the interface for pluggable session stores.

    Both ``InMemorySessionStore`` and ``RedisSessionStore`` satisfy this
    interface.  Implement it to provide custom persistence backends.
    """

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Returns the stored metadata for *entity_id*, or ``None``."""

    def set(self, entity_id: str, metadata: Dict[str, Any]) -> None:
        """Stores *metadata* under *entity_id*."""

    def set_many(self, mapping: Dict[str, Dict[str, Any]]) -> None:
        """Bulk-stores all entries in *mapping*."""

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Returns all stored entries as ``{entity_id: metadata}``."""
        raise NotImplementedError()

    def clear(self) -> None:
        """Removes all entries."""


def _resolve_default_session_ttl(_logger: logging.Logger) -> int:
    """Resolve default session TTL from SESSION_TTL_SECONDS with validation."""
    ttl_str = os.getenv("SESSION_TTL_SECONDS", "86400")
    try:
        ttl = int(ttl_str)
        if ttl <= 0:
            raise ValueError("TTL must be > 0")
        return ttl
    except ValueError:
        _logger.warning("Invalid SESSION_TTL_SECONDS: %s, using default 86400", ttl_str)
        return 86400


def _resolve_tenant_session_ttl(tenant_id: str, default_ttl: int, _logger: logging.Logger) -> int:
    """Resolve per-tenant TTL override from TENANT_SESSION_TTL_SECONDS policy.

    Environment format:
        TENANT_SESSION_TTL_SECONDS='{"tenant-a": 1800, "tenant-b": 7200}'
    """
    policy_str = os.getenv("TENANT_SESSION_TTL_SECONDS", "").strip()
    if not policy_str:
        return default_ttl

    try:
        policy = json.loads(policy_str)
    except json.JSONDecodeError as exc:
        _logger.warning(
            "Invalid TENANT_SESSION_TTL_SECONDS JSON: %s; using SESSION_TTL_SECONDS",
            exc,
        )
        return default_ttl

    if not isinstance(policy, dict):
        _logger.warning(
            "TENANT_SESSION_TTL_SECONDS must decode to a dict; got %s; using SESSION_TTL_SECONDS",
            type(policy).__name__,
        )
        return default_ttl

    tenant_ttl_raw = policy.get(tenant_id,None)
    if tenant_ttl_raw is None:
        return default_ttl

    try:
        tenant_ttl = int(tenant_ttl_raw)
        if tenant_ttl <= 0:
            raise ValueError("TTL must be > 0")
        return tenant_ttl
    except (TypeError, ValueError):
        _logger.warning(
            "Invalid tenant TTL override in TENANT_SESSION_TTL_SECONDS for tenant %s: %s; "
            "falling back to SESSION_TTL_SECONDS",
            tenant_id,
            tenant_ttl_raw,
        )
        return default_ttl


def _build_tenant_session_store(
    tenant_id: Optional[str],
    _logger: logging.Logger,
) -> SessionStore:
    """Create a tenant-aware session store, or in-memory fallback."""
    if not tenant_id:
        return InMemorySessionStore()

    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        _logger.debug(
            "tenant_id provided but REDIS_URL not configured; "
            "using in-memory session store"
        )
        return InMemorySessionStore()

    default_ttl = _resolve_default_session_ttl(_logger)
    ttl = _resolve_tenant_session_ttl(str(tenant_id), default_ttl, _logger)
    try:
        session_store = RedisSessionStore(
            session_id="tenant-session",  # Placeholder; unused in per-tenant mode
            redis_url=redis_url,
            ttl=ttl,
            tenant_id=tenant_id,
        )
        _logger.info(
            "Created per-tenant RedisSessionStore for tenant: %s (ttl=%ss)",
            tenant_id,
            ttl,
        )
        return session_store
    except (OSError, IOError) as exc:
        _logger.warning(
            "Failed to create per-tenant RedisSessionStore for tenant %s: %s; "
            "falling back to in-memory",
            tenant_id,
            exc,
        )
        return InMemorySessionStore()


def _serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Converts non-JSON-native values (Enum, tuple) to JSON-safe types.

    Enums are stored as their ``.value`` strings.
    Tuples are stored as lists.
    All other values are passed through unchanged.
    """
    result: Dict[str, Any] = {}
    for k, v in metadata.items():
        if hasattr(v, "value"):      # Enum subclasses
            result[k] = v.value
        elif isinstance(v, tuple):
            result[k] = list(v)  # type: ignore[arg-type]
        else:
            result[k] = v
    return result


class InMemorySessionStore:
    """Default in-process session store.

    Retains the accumulated entity-ID → metadata mapping for as long as the
    store object is alive (i.e. the lifetime of ``Taivium`` or
    ``PrivacyClient``).  No persistence across process restarts.

    This is the backward-compatible default used when no Redis URL is provided.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core interface (duck-typed; also matches RedisSessionStore)
    # ------------------------------------------------------------------

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Returns the stored metadata for *entity_id*, or ``None``."""
        return self._data.get(entity_id)

    def set(self, entity_id: str, metadata: Dict[str, Any]) -> None:
        """Stores *metadata* under *entity_id*."""
        self._data[entity_id] = metadata

    def set_many(self, mapping: Dict[str, Dict[str, Any]]) -> None:
        """Bulk-stores all entries in *mapping*."""
        self._data.update(mapping)

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Returns a shallow copy of all stored entries."""
        return dict(self._data)

    def clear(self) -> None:
        """Removes all entries (e.g. between independent conversations)."""
        self._data.clear()


class RedisSessionStore:
    """Redis-backed session store for cross-call and cross-process persistence.

    Every entry is stored as a JSON string under the key::

        taivium:<tenant_id>:session:<session_id>:<entity_id>

    Entries expire after *ttl* seconds (default 24 h) to prevent unbounded
    growth.  Pass ``ttl=None`` to disable expiry.

    Args:
        session_id: Unique identifier for the session (e.g. user ID, request
            correlation ID).  Used to namespace keys in Redis so that
            different sessions never collide.
        tenant_id: Optional tenant identifier for multi-tenant isolation.
            Defaults to ``"default"`` when not provided.
        redis_url: Redis connection URL (default ``"redis://localhost:6379"``).
            Supports ``redis://``, ``rediss://`` (TLS), and
            ``unix://`` socket URLs supported by ``redis-py``.
        ttl: Key expiry in seconds.  Defaults to 86 400 (24 h).  Set to
            ``None`` to keep keys indefinitely.

    Raises:
        ImportError: If the ``redis`` package is not installed.

    Usage::

        store = RedisSessionStore(
            session_id="user-abc123",
            redis_url="redis://localhost:6379",
            ttl=3600,
        )
        pipeline = Taivium(session_store=store)
    """

    def __init__(
        self,
        session_id: str,
        tenant_id: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        ttl: Optional[int] = 86400,
    ) -> None:
        try:
            import redis  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "The 'redis' package is required to use RedisSessionStore. "
                "Install it with:  pip install redis"
            ) from exc

        self._client = redis.from_url(redis_url, decode_responses=True)
        resolved_tenant_id = (tenant_id or "default").strip() or "default"
        self.tenant_id = resolved_tenant_id
        self.session_id = session_id
        self._prefix = f"taivium:{resolved_tenant_id}:session:{session_id}:"
        self._ttl = ttl

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Returns the stored metadata for *entity_id*, or ``None``."""
        raw = self._client.get(self._prefix + entity_id)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[return-value]

    def set(self, entity_id: str, metadata: Dict[str, Any]) -> None:
        """Serializes and stores *metadata* under *entity_id* in Redis."""
        safe = _serialize_metadata(metadata)
        self._client.set(
            self._prefix + entity_id,
            json.dumps(safe),
            ex=self._ttl,
        )

    def set_many(self, mapping: Dict[str, Dict[str, Any]]) -> None:
        """Bulk-stores all entries using a Redis pipeline for efficiency."""
        if not mapping:
            return
        pipe = self._client.pipeline()  # type: ignore[attr-defined]
        for entity_id, metadata in mapping.items():
            safe = _serialize_metadata(metadata)
            pipe.set(
                self._prefix + entity_id,
                json.dumps(safe),
                ex=self._ttl,
            )
        pipe.execute()

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Returns all entries for this session as a plain dict.

        Uses ``SCAN`` iteration to avoid blocking Redis on large key sets.
        """
        pattern = self._prefix + "*"
        prefix_len = len(self._prefix)
        keys = list(self._client.scan_iter(pattern))
        if not keys:
            return {}
        values = cast(List[Optional[str]], self._client.mget(keys))  # type: ignore[arg-type]
        result: Dict[str, Dict[str, Any]] = {}
        for key, raw in zip(keys, values):
            if raw is not None:
                result[key[prefix_len:]] = json.loads(raw)  # type: ignore[arg-type]
        return result

    def clear(self) -> None:
        """Deletes all keys for this session from Redis."""
        keys = list(self._client.scan_iter(self._prefix + "*"))
        if keys:
            self._client.delete(*keys)
