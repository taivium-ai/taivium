"""
Taivium — Privacy-preserving de-identification SDK
------------------------------------------------------
Drop-in OpenAI-compatible client that de-identifies sensitive data
before it reaches any LLM endpoint.

Quickstart::

    from taivium import PrivacyClient

    client = PrivacyClient(api_key="sk-...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Alice at alice@acme.com needs help."}],
    )

For custom policies::

    from taivium import PrivacyClient, PolicyEngine, PolicyRule, PolicyAction, RiskLevel

    policy = PolicyEngine(policy_table={
        "PERSON":   PolicyRule("PERSON",   PolicyAction.ANONYMIZE, RiskLevel.HIGH),
        "EMAIL":    PolicyRule("EMAIL",    PolicyAction.BLOCK,     RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW,     RiskLevel.LOW),
    })
    client = PrivacyClient(api_key="sk-...", policy_engine=policy)
"""

from importlib.metadata import version, PackageNotFoundError

from .client import PrivacyClient
from .engine import (
    Entity,
    Evidence,
    PolicyAction,
    PolicyDecision,
    PolicyContext,
    PolicyDecisionReason,
    PolicyEngine,
    PolicyRule,
    Taivium,
    RiskLevel,
    find_recurrences,
    recurrence_evidence,
    reverse_transform,
)
from .session_store import InMemorySessionStore, RedisSessionStore, SessionStore


try:
    __version__ = version("taivium")
except PackageNotFoundError:
    # Fallback for local/dev usage (package not installed)
    __version__ = "0.0.0"


__all__ = [
    # High-level SDK entry point
    "PrivacyClient",
    # Core pipeline (usable standalone, without the OpenAI wrapper)
    "Taivium",
    # Policy primitives
    "PolicyEngine",
    "PolicyRule",
    "PolicyAction",
    "PolicyDecision",
    "PolicyContext",
    "PolicyDecisionReason",
    "RiskLevel",
    # Data types
    "Evidence",
    "Entity",
    # Utilities
    "find_recurrences",
    "recurrence_evidence",
    "reverse_transform",
    # Session stores
    "InMemorySessionStore",
    "RedisSessionStore",
]