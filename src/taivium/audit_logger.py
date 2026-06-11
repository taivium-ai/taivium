"""Audit logging for security and compliance event tracking."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Iterable

logger = logging.getLogger("taivium.audit")

# pylint: disable=too-many-arguments
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

    Notes:
        Emits one JSON event through the ``taivium.audit`` logger at INFO level.
        Set ``TAIVIUM_AUDIT_STDOUT=0`` to suppress emission from this function.
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

    # Set TAIVIUM_AUDIT_STDOUT=0 to suppress terminal audit JSON output.
    audit_stdout_enabled = os.getenv("TAIVIUM_AUDIT_STDOUT", "1").strip().lower()
    if audit_stdout_enabled in {"0", "false", "off", "no"}:
        return

    try:
        logger.info(json.dumps(event))
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
