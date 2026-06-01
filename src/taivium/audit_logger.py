"""Audit logging for security and compliance event tracking."""

import json
import sys
from datetime import datetime, timezone
from typing import Iterable


# pylint: disable=too-many-arguments,too-many-positional-arguments
def log_audit_event(
    operation: str,
    session_id: str,
    entity_count: int,
    entity_types: Iterable[str],
    duration_ms: float,
    status: str,
) -> None:
    """Log an audit event with operation details and metrics.
    
    Args:
        operation: The operation being performed.
        session_id: The session identifier.
        entity_count: Number of entities affected.
        entity_types: Types of entities involved.
        duration_ms: Duration in milliseconds.
        status: Operation status.
    """
    event: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "session_id": session_id,
        "entity_count": entity_count,
        "entity_types": list(entity_types),
        "duration_ms": round(duration_ms, 3),
        "status": status,
    }
    try:
        print(json.dumps(event), file=sys.stdout, flush=True)
    except (TypeError, ValueError, OSError):
        # Audit logging must not break request handling.
        pass


# Allow enterprise pack to override log_audit_event with an enriched implementation.
# If taivium_enterprise is installed, its log_audit_event replaces this one transparently.
try:
    import importlib as _importlib
    _enterprise = _importlib.import_module("taivium_enterprise.audit")
    log_audit_event = _enterprise.log_audit_event  # noqa: F811
    del _importlib, _enterprise
except ImportError:
    pass
