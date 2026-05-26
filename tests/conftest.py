"""Pytest configuration and shared fixtures for the Tarvium test suite."""
import os
from pathlib import Path


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
