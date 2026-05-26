"""
PrivacyClient — OpenAI-compatible drop-in SDK wrapper
----------------------------------------------------------------------
De-identifies text before it reaches any OpenAI-compatible LLM endpoint
and optionally re-identifies entity tokens in the response.

Typical usage::

    from taivium import PrivacyClient

    client = PrivacyClient(api_key="sk-...")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Alice at alice@acme.com needs help."}],
    )
    # Message was anonymised before leaving this process.
    # Original entity values are preserved in client.session_mapping.

Pass ``deid_response=True`` to ``create()`` to reverse-map entity tokens
in the LLM's reply back to the original values before returning.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .engine import PolicyAction, PolicyEngine, Taivium, reverse_transform
from .session_store import InMemorySessionStore, RedisSessionStore

# pylint: disable=too-few-public-methods
# ---------------------------------------------------------------------------
# Internal: thin wrappers that mirror the openai client's attribute hierarchy
# ---------------------------------------------------------------------------


class _PrivacyCompletions:
    """Wraps ``openai.resources.chat.completions.Completions``."""

    def __init__(
        self,
        inner_completions: Any,
        pipeline: Taivium,
    ) -> None:
        self._inner = inner_completions
        self._pipeline = pipeline

    def create(
        self,
        *,
        messages: List[Dict[str, Any]],
        deid_response: bool = False,
        **kwargs: Any,
    ) -> Any:
        """De-identifies every user/system message then calls the underlying LLM.

        Args:
            messages: OpenAI-format message list
                (``[{"role": "user", "content": "..."}]``).
            deid_response: When ``True``, entity tokens in the LLM's reply are
                replaced with their original values before the response is returned.
                Defaults to ``False`` (anonymised response is returned as-is).
            **kwargs: All remaining arguments are forwarded verbatim to the
                underlying ``openai`` completions endpoint (``model``, ``temperature``,
                ``stream``, etc.).

        Returns:
            The ``openai.types.chat.ChatCompletion`` object returned by the LLM,
            with message content optionally re-identified.

        Raises:
            ValueError: If a BLOCK-policy entity is detected in any message.
        """
        safe_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if content and isinstance(content, str):
                result = self._pipeline.process(content)
                safe_messages.append({**msg, "content": result["anonymized"]})
            else:
                # Non-text content (None, list of parts) is forwarded
                # unchanged.
                safe_messages.append(msg)

        response = self._inner.create(messages=safe_messages, **kwargs)

        if deid_response:
            _patch_response_content(response, self._pipeline.session_store.get_all())

        return response


class _PrivacyChat:
    """Mirrors ``openai.resources.chat.Chat`` with a ``completions`` attribute."""

    def __init__(
        self,
        inner_chat: Any,
        pipeline: Taivium,
    ) -> None:
        self.completions = _PrivacyCompletions(
            inner_chat.completions, pipeline
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PrivacyClient:
    """OpenAI-compatible drop-in client that de-identifies text before it
    reaches the LLM.

    ``PrivacyClient`` accepts every keyword argument that ``openai.OpenAI``
    accepts (``api_key``, ``base_url``, ``organization``, ``timeout``, etc.) and
    forwards them unchanged.  The only additions are:

    * ``policy_engine`` — optional :class:`PolicyEngine` instance to override
      the default de-identification policy.
    * ``default_action`` — :class:`~taivium.engine.PolicyAction`
      applied to entity labels not covered by the policy table. Defaults to
      ``PolicyAction.ANONYMIZE`` (strict). Pass ``PolicyAction.ALLOW`` for
      permissive mode. Ignored if ``policy_engine`` is provided.
    * ``session_id`` — optional string to namespace session identity memory.
      When combined with ``redis_url``, mappings are persisted in Redis so
      the same entity always receives the same token across process restarts
      and API calls.
    * ``redis_url`` — Redis connection URL (e.g. ``"redis://localhost:6379"``)
      to enable cross-call, cross-process session persistence.  Requires the
      ``redis`` package (``pip install redis``).  Ignored if ``session_id`` is
      not provided.
    * ``redis_ttl`` — TTL in seconds for Redis keys (default 86 400 / 24 h).

    The ``chat.completions.create()`` interface is identical to the upstream
    ``openai`` library's.  No other code changes are required::

        # Before
        import openai
        client = openai.OpenAI(api_key="sk-...")

        # After
        from taivium import PrivacyClient
        client = PrivacyClient(api_key="sk-...")

    Attributes:
        chat: Drop-in replacement for ``openai.OpenAI().chat``.
        session_mapping: Read-only snapshot of every entity anonymised so far
            in this session, keyed by the token ID emitted to the LLM.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        policy_engine: Optional[PolicyEngine] = None,
        default_action: Optional[PolicyAction] = None,
        session_id: Optional[str] = None,
        redis_url: str = "redis://localhost:6379",
        redis_ttl: Optional[int] = 86400,
        **openai_kwargs: Any,
    ) -> None:
        try:
            # pylint: disable=too-few-public-methods,unused-argument,import-outside-toplevel
            import openai  # deferred — openai is an optional dependency
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required to use PrivacyClient. "
                "Install it with:  pip install openai"
            ) from exc

        self._inner = openai.OpenAI(**openai_kwargs)

        # Build session store: Redis-backed when session_id + redis_url given,
        # otherwise plain in-memory (backward-compatible default).
        if session_id is not None:
            store: Any = RedisSessionStore(
                session_id=session_id,
                redis_url=redis_url,
                ttl=redis_ttl,
            )
        else:
            store = InMemorySessionStore()

        if policy_engine is None and default_action is not None:
            policy_engine = PolicyEngine(default_action=default_action)
        self._pipeline = Taivium(policy_engine=policy_engine, session_store=store)
        self.chat = _PrivacyChat(self._inner.chat, self._pipeline)

    @property
    def session_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Accumulated entity mapping for this session (shallow copy).

        When backed by ``RedisSessionStore``, this reflects all entries
        persisted in Redis for the current ``session_id`` — including those
        written by previous process runs.
        """
        return self._pipeline.session_store.get_all()

    def reset_session(self) -> None:
        """Clears the accumulated session entity mapping.

        Call this between independent conversations to prevent entity tokens
        from leaking across unrelated sessions.  When backed by
        ``RedisSessionStore``, this also deletes all Redis keys for this
        session.
        """
        self._pipeline.session_store.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_response_content(response: Any, mapping: Dict[str, Dict[str, Any]]) -> None:
    """Mutates the response's message content in-place to reverse anonymization.

    Silently does nothing for response shapes that do not match the standard
    ``ChatCompletion`` structure (e.g. streaming chunks, future API versions).
    """
    try:
        for choice in response.choices:
            msg = choice.message
            if msg and isinstance(msg.content, str):
                msg.content = reverse_transform(msg.content, mapping)
    except AttributeError:
        pass
