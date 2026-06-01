import io
import json
import sys
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
    """print() raises OSError when stdout is a broken pipe — must be swallowed."""
    broken = io.TextIOWrapper(io.RawIOBase())  # write() always raises OSError
    with patch("taivium.audit_logger.sys.stdout", broken):
        _call()   # must not raise
