# Development Guide

## Project Structure

```
Taivium/
├── src/
│   └── taivium/
│       ├── __init__.py           # Public SDK exports (PrivacyClient, Taivium, …)
│       ├── client.py             # PrivacyClient — OpenAI-compatible drop-in wrapper
│       ├── engine.py             # Core privacy engine (detection, policy, anonymization)
│       └── session_store.py      # Pluggable session identity stores (InMemory / Redis)
├── tests/
│   ├── conftest.py               # Auto-loads .env before tests run
│   ├── test_client.py            # Tests for PrivacyClient (fully offline, no openai needed)
│   ├── test_privacy_pipeline.py  # Unit tests for pipeline components and fallback logic
│   ├── test_session_store.py     # Tests for InMemorySessionStore and RedisSessionStore
│   └── test_requirements.py      # Functional requirements test suite
├── docs/
│   ├── desing_document.md        # Software design document
│   ├── requirements.md           # Functional requirements
│   ├── openapi_3.0_spec.md       # OpenAPI 3.0 spec for proxy server
│   ├── design_document_MAS.md    # Multi-agent system design
│   ├── privata_product_suite_design.md
│   ├── product_def.md
│   ├── issue.md
│   └── market/
│       ├── market_observation.md
│       └── 90 days plan.md
├── examples/
│   ├── example_privacy_pipeline.py
│   └── example_client.py
├── .env                          # Local environment variables (OPENAI_KEY, PYTHONPATH)
├── DEVELOPMENT.md
├── requirements.txt              # Runtime dependencies
└── requirements-test.txt         # Test dependencies
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-test.txt
python -m spacy download en_core_web_sm
```

## Running Tests

```bash
PYTHONPATH=src pytest tests/
```

> `PYTHONPATH=src` is required so that `import taivium` resolves to `src/taivium/`.
> This value is also set in the `.env` file, which `tests/conftest.py` loads automatically.

## Core Pipeline Flow

`Taivium.process()` follows this sequence:

1. Collect detector evidence (`spacy_evidence`, `regex_evidence` [calibrated confidence], `transformer_evidence`, `llm_evidence`)
2. Canonicalize spans (`canonicalize_spans`) — sweep-line overlap-cluster grouping produces one canonical entity per non-overlapping cluster via weighted label vote and longest-span selection
2b. Find semantic recurrences (`find_recurrences`) — add repeated surface-form mentions of canonical entities missed by NER for recurrence-eligible entities only (token-boundary safe, non-overlapping; avoids ambiguous short PERSON/LOCATION/acronym cloning)
3. Resolve deterministic IDs (`IdentityEngine.resolve`)
4. Persist mapping to `session_store` (`InMemorySessionStore` or `RedisSessionStore`)
5. Evaluate policy (`PolicyEngine.evaluate`)
6. Transform text (`transform`)


`Taivium.process()` follows this sequence:

1. Collect detector evidence (`spacy_evidence`, `regex_evidence` [calibrated confidence], `transformer_evidence`, `llm_evidence`)
2. Canonicalize spans (`canonicalize_spans`) — sweep-line overlap-cluster grouping produces one canonical entity per non-overlapping cluster via weighted label vote and longest-span selection
2b. Find semantic recurrences (`find_recurrences`) — add repeated surface-form mentions of canonical entities missed by NER for recurrence-eligible entities only (token-boundary safe, non-overlapping; avoids ambiguous short PERSON/LOCATION/acronym cloning)
3. Resolve deterministic IDs (`IdentityEngine.resolve`) with privacy-preserving options:
  - `id_salt`: Optional salt to scope entity IDs to a tenant, session, or namespace. Prevents cross-tenant/session linkage of identical entities.
  - `id_hash_len`: Number of hex digits to use from the hash (default 12; can be increased for more collision resistance).
  - **Privacy note:** If no salt is provided, IDs are globally stable (legacy behavior), which may allow cross-document or cross-tenant linkage. For privacy-preserving deployments, always set a unique salt per tenant or session.
4. Persist mapping to `session_store` (`InMemorySessionStore` or `RedisSessionStore`)
5. Evaluate policy (`PolicyEngine.evaluate`)
6. Transform text (`transform`)
## Session Identity Store

The pipeline's `session_store` persists the `entity_id → metadata` mapping across calls:

```python
# Default (in-memory, within-process only)
pipeline = Taivium()

# Redis-backed (cross-call, cross-process)
from taivium.session_store import RedisSessionStore
store = RedisSessionStore(session_id="user-abc123", redis_url="redis://localhost:6379")
pipeline = Taivium(session_store=store)

# Via PrivacyClient
from taivium import PrivacyClient
client = PrivacyClient(
    api_key="sk-...",
    session_id="user-abc123",
    redis_url="redis://localhost:6379",
    redis_ttl=3600,  # seconds; default 86400
)
```

The pipeline's `session_store` persists the `entity_id → metadata` mapping across calls. Entity IDs can be scoped for privacy:

```python
# Default (global, legacy-stable IDs; not privacy-preserving)
pipeline = Taivium()

# Tenant-scoped IDs (prevents cross-tenant linkage)
pipeline = Taivium(id_salt="tenant_1234")

# Session-scoped IDs (prevents cross-session linkage)
pipeline = Taivium(id_salt="session_5678")

# Custom hash length (longer IDs)
pipeline = Taivium(id_hash_len=24)

# Both salt and custom hash length
pipeline = Taivium(id_salt="tenant_1234", id_hash_len=24)

# Redis-backed (cross-call, cross-process)
from taivium.session_store import RedisSessionStore
store = RedisSessionStore(session_id="user-abc123", redis_url="redis://localhost:6379")
pipeline = Taivium(session_store=store)

# Via PrivacyClient
from taivium import PrivacyClient
client = PrivacyClient(
  api_key="sk-...",
  session_id="user-abc123",
  redis_url="redis://localhost:6379",
  redis_ttl=3600,  # seconds; default 86400
)
```

**Privacy best practice:** Always set a unique salt per tenant or session in regulated or multi-tenant environments to prevent cross-tenant or cross-session linkage of anonymized IDs.

## Dependencies

### Runtime

| Package | Purpose |
|---------|----------|
| `spacy` | NER detection |
| `en_core_web_sm` | spaCy English NER model |
| `transformers` + `torch` | Transformer-based NER (optional, only needed for `use_transformer=True`) |
| `openai` | LLM client (optional, only needed for `PrivacyClient` in client.py or `use_llm=True`) |
| `redis` | Redis client (optional, only needed for `RedisSessionStore`) |
| `openai` | LLM client (optional, only needed for `PrivacyClient` in client.py) |
| `redis` | Redis client (optional, only needed for `RedisSessionStore`) |

### Test

| Package | Purpose |
|---------|----------|
| `pytest` | Test runner |
| `pytest-watch` | File-watch test runner (`ptw`) |
| `pylint` | Static analysis |
| `fakeredis` | Redis test double (no real Redis server needed) |

#### Linting

`pylint` is used for static code analysis and linting. To check code style and catch common errors, run:

```bash
pylint src/ tests/
```

### Standard Library (no install needed)

`re`, `hashlib`, `dataclasses`, `enum`, `collections`, `time`, `functools`, `warnings`

## VS Code Workspace Configuration

Project-specific VS Code settings and tasks are stored in the `.vscode/` directory at the project root:

- `.vscode/settings.json`: Recommended editor and extension settings for all contributors (e.g., Python interpreter, formatting, linting, etc.).
- `.vscode/tasks.json`: Shared automation tasks (e.g., test runner, build commands) for consistent development workflows.

These files help ensure a consistent development environment across the team.

### Example: Current VS Code Settings

```jsonc
// .vscode/settings.json
{
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.envFile": "${workspaceFolder}/.env",
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

```jsonc
// .env
PYTHONPATH=src
OPENAI_KEY=your_openai_api_key_here
```

```jsonc
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "ptw",
      "type": "shell",
      "command": "${command:python.interpreterPath}",
      "args": ["-m", "pytest_watch", "--ext=.py"],
      "isBackground": true,
      "runOptions": {
        "runOn": "folderOpen"
      }
    },
    {
      "label": "pylint",
      "type": "shell",
      "command": "${command:python.interpreterPath}",
      "args": ["-m", "pylint", "./src"],
      "runOptions": {
        "runOn": "folderOpen"
      }
    }
  ]
}
```