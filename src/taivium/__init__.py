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
import os
from .client import PrivacyClient
from .engine import (Entity, Evidence, PolicyAction, PolicyDecision,
                               PolicyContext, PolicyDecisionReason, PolicyEngine, PolicyRule,
                               Taivium, RiskLevel, find_recurrences, recurrence_evidence,
                               reverse_transform)
from .session_store import InMemorySessionStore, RedisSessionStore, SessionStore

# Dynamically set __version__ from package metadata or pyproject.toml if available
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("taivium")
except (ImportError, AttributeError, ModuleNotFoundError):
    try:
        from pkg_resources import get_distribution, DistributionNotFound
        try:
            __version__ = get_distribution("taivium").version
        except DistributionNotFound:
            pass
    except (ImportError, ModuleNotFoundError):
        try:
            import tomllib  # Python 3.11+
            pyproject_path = os.path.join(os.path.dirname(__file__), "..", "..", "pyproject.toml")
            with open(os.path.abspath(pyproject_path), "rb") as f:
                pyproject = tomllib.load(f)
                __version__ = pyproject["project"]["version"]
        except (ImportError, ModuleNotFoundError):
            try:
                import toml  # fallback for older Python
                pyproject_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", "pyproject.toml")
                with open(os.path.abspath(pyproject_path), "r", encoding="utf-8") as f:
                    pyproject = toml.load(f)
                    __version__ = pyproject["project"]["version"]
            except Exception as exc:
                raise ImportError(
                    "Could not determine taivium version from package metadata or pyproject.toml."
                    ) from exc

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
