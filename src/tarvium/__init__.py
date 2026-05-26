"""
Tarvium — Privacy-preserving de-identification SDK
------------------------------------------------------
Drop-in OpenAI-compatible client that de-identifies sensitive data
before it reaches any LLM endpoint.

Quickstart::

    from tarvium import PrivacyClient

    client = PrivacyClient(api_key="sk-...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Alice at alice@acme.com needs help."}],
    )

For custom policies::

    from tarvium import PrivacyClient, PolicyEngine, PolicyRule, PolicyAction, RiskLevel

    policy = PolicyEngine(policy_table={
        "PERSON":   PolicyRule("PERSON",   PolicyAction.ANONYMIZE, RiskLevel.HIGH),
        "EMAIL":    PolicyRule("EMAIL",    PolicyAction.BLOCK,     RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW,     RiskLevel.LOW),
    })
    client = PrivacyClient(api_key="sk-...", policy_engine=policy)
"""

from .client import PrivacyClient
from .engine import (Entity, Evidence, PolicyAction, PolicyDecision,
                               PolicyContext, PolicyDecisionReason, PolicyEngine, PolicyRule,
                               Tarvium, RiskLevel, find_recurrences, recurrence_evidence,
                               reverse_transform)
from .session_store import InMemorySessionStore, RedisSessionStore, SessionStore

__all__ = [
    # High-level SDK entry point
    "PrivacyClient",
    # Core pipeline (usable standalone, without the OpenAI wrapper)
    "Tarvium",
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
