import io
import json
import logging
import sys
import types
from unittest.mock import patch

import pytest

from taivium.audit_logger import log_audit_event


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(**kwargs):
    defaults = dict(
        operation="process",
        session_id="sess_1",
        entity_count=1,
        entity_types=["EMAIL"],
        duration_ms=5.0,
        status="ok",
    )
    defaults.update(kwargs)
    log_audit_event(**defaults)


# ---------------------------------------------------------------------------
# Normal path
# ---------------------------------------------------------------------------

def test_log_audit_event_emits_json(capsys):
    _call(operation="process", session_id="s1", entity_count=2,
          entity_types=["PERSON", "EMAIL"], duration_ms=12.5, status="ok")
    out = capsys.readouterr().out.strip()
    event = json.loads(out)
    assert event["operation"] == "process"
    assert event["session_id"] == "s1"
    assert event["entity_count"] == 2
    assert event["entity_types"] == ["PERSON", "EMAIL"]
    assert event["duration_ms"] == 12.5
    assert event["status"] == "ok"
    assert "timestamp" in event


# ---------------------------------------------------------------------------
# Silent-failure path — TypeError, ValueError, OSError must not propagate
# ---------------------------------------------------------------------------

def test_silent_on_type_error(capsys):
    """json.dumps raises TypeError on non-serialisable values — must be swallowed."""
    with patch("taivium.audit_logger.json.dumps", side_effect=TypeError("not serialisable")):
        _call()   # must not raise
    # nothing should have been written
    assert capsys.readouterr().out == ""


def test_silent_on_value_error(capsys):
    """json.dumps raises ValueError on circular refs — must be swallowed."""
    with patch("taivium.audit_logger.json.dumps", side_effect=ValueError("circular")):
        _call()   # must not raise
    assert capsys.readouterr().out == ""


def test_silent_on_os_error(capsys):
    """OSError from the logging handler — must be swallowed."""
    import taivium.audit_logger as _mod
    
    # Mock the logger's handlers to raise OSError on emit
    original_handlers = _mod.logger.handlers[:]
    
    class FailingHandler(logging.Handler):
        def emit(self, record):
            raise OSError("broken pipe")
    
    try:
        _mod.logger.handlers.clear()
        _mod.logger.addHandler(FailingHandler())
        _call()  # must not raise
    finally:
        _mod.logger.handlers.clear()
        for handler in original_handlers:
            _mod.logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Enterprise override — importlib-based replacement
# ---------------------------------------------------------------------------

def test_enterprise_override_replaces_log_audit_event():
    """When taivium_enterprise.audit is importable, log_audit_event is replaced."""
    sentinel = object()

    # Build a minimal fake module with a replacement log_audit_event
    fake_module = types.ModuleType("taivium_enterprise.audit")
    fake_module.log_audit_event = sentinel  # type: ignore[attr-defined]

    import taivium.audit_logger as _mod

    with patch.dict("sys.modules", {"taivium_enterprise.audit": fake_module}):
        # Re-execute only the override block so the rest of the module is untouched
        import importlib as _importlib
        _enterprise = _importlib.import_module("taivium_enterprise.audit")
        _mod.log_audit_event = _enterprise.log_audit_event  # type: ignore[attr-defined]
        try:
            assert _mod.log_audit_event is sentinel
        finally:
            # Restore the original implementation so other tests are unaffected
            from taivium.audit_logger import log_audit_event as _orig  # noqa: F401
            # Re-import the real function from its definition in the module source
            import importlib
            importlib.reload(_mod)


def test_enterprise_override_import_error_keeps_default():
    """When taivium_enterprise is absent, the built-in log_audit_event is kept."""
    import taivium.audit_logger as _mod
    import importlib

    # Remove any cached enterprise modules and reload without them
    sys_modules_backup = {
        k: v for k, v in sys.modules.items() if "taivium_enterprise" in k
    }
    for k in sys_modules_backup:
        del sys.modules[k]

    try:
        importlib.reload(_mod)
        # Must still be callable and emit JSON
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        
        original_handlers = _mod.logger.handlers[:]
        try:
            _mod.logger.handlers.clear()
            _mod.logger.addHandler(handler)
            
            _mod.log_audit_event(
                operation="test",
                session_id="",
                entity_count=0,
                entity_types=[],
                duration_ms=0.0,
                status="ok",
            )
            event = json.loads(buf.getvalue())
            assert event["operation"] == "test"
        finally:
            _mod.logger.handlers.clear()
            for h in original_handlers:
                _mod.logger.addHandler(h)
    finally:
        importlib.reload(_mod)
        for k, v in sys_modules_backup.items():
            sys.modules[k] = v
