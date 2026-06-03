"""Pytest configuration and shared fixtures for the Taivium test suite."""
import logging
import sys
import os
from pathlib import Path


class _CapsysCompatibleHandler(logging.Handler):
    """A logging handler that writes to sys.stdout at write time, not initialization time.
    
    This allows pytest's capsys fixture to properly capture the output,
    since capsys replaces sys.stdout after imports.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
        except Exception:
            self.handleError(record)


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets variables that are not already in the environment."""
    if not path.is_file():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(Path(__file__).parent.parent / ".env")

# Configure the taivium.audit logger to output JSON to stdout
_audit_logger = logging.getLogger("taivium.audit")
_audit_logger.handlers.clear()
_audit_logger.propagate = False
_audit_logger.setLevel(logging.INFO)
_handler = _CapsysCompatibleHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(message)s"))
_audit_logger.addHandler(_handler)
