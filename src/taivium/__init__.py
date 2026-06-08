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
import subprocess
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

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
from .audit_logger import log_audit_event
from .session_store import InMemorySessionStore, RedisSessionStore, SessionStore


def _get_git_commit() -> str:
    env_commit = os.getenv("TAIVIUM_GIT_COMMIT") or os.getenv("GIT_COMMIT")
    if env_commit:
        return env_commit.strip()

    try:
        repo_root = Path(__file__).resolve().parent.parent
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass

    return ""

try:
    __version__ = version("taivium")
except PackageNotFoundError:
    # Fallback for local/dev usage (package not installed)
    __version__ = "0.0.0"

__commit__ = _get_git_commit()

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
    # Audit logging (enterprise override applies automatically if taivium-enterprise is installed)
    "log_audit_event",
]
