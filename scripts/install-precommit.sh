#!/usr/bin/env bash
set -euo pipefail

# Install pre-commit into the active Python environment and register the git hook.
# Prefer a virtualenv if available; fall back to user-level install.
if [ -n "${VIRTUAL_ENV:-}" ]; then
  python -m pip install --upgrade pre-commit
else
  python -m pip install --user --upgrade pre-commit
fi

pre-commit install
echo "Pre-commit installed. To test: pre-commit run --all-files"
