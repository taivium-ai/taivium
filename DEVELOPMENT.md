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
# Optional: override pinned GLiNER model snapshot revision
# export TAIVIUM_GLINER_REVISION=8142fb00740ccea973e64b1272949ff48653df5e
```

## Running Tests

```bash
PYTHONPATH=src pytest tests/
```

> `PYTHONPATH=src` is required so that `import taivium` resolves to `src/taivium/`.
> This value is also set in the `.env` file, which `tests/conftest.py` loads automatically.

## Core Pipeline Flow

`Taivium.process()` follows this sequence:

1. Collect detector evidence via adaptive routing:
  - Fast track (`len(text) < 100`): `regex_evidence` (includes structured field detection) + `spacy_evidence`
  - Context track (`len(text) >= 100`): `regex_evidence` (includes structured field detection) + selected context backend
    (`gliner_evidence` or OpenAI privacy filter)
    - `gliner_evidence` chunks inputs over 384 tokens with overlap and processes
      chunks in batches before offset remapping/deduplication
  - Optional layers: `org_list_evidence`, `transformer_evidence`, `llm_evidence`
2. Canonicalize spans (`canonicalize_spans`) — weighted interval scheduling over exact span+label candidates selects the best global non-overlapping set; includes a guard that penalizes PERSON/ORG/LOCATION fragment spans inside regex-validated EMAIL spans
2b. Find semantic recurrences (`find_recurrences`) — add repeated surface-form mentions of canonical entities missed by NER for recurrence-eligible entities only (token-boundary safe, non-overlapping; avoids ambiguous short PERSON/LOCATION/acronym cloning)
3. Resolve deterministic IDs (`IdentityEngine.resolve`) with privacy-preserving options:
  - `id_salt`: Optional salt to scope entity IDs to a tenant, session, or namespace
  - `id_hash_len`: Number of hex digits to use from the hash (default 12)
4. Persist mapping to `session_store` (`InMemorySessionStore` or `RedisSessionStore`)
5. Evaluate policy (`PolicyEngine.evaluate`)
6. Transform text (`transform`)

## Detector Features

### Structured Field Detection

The `regex_evidence()` function includes generalized field detection that extracts sensitive values from structured formats (JSON, YAML, Markdown, XML). This increases recall for values that might be missed by NER patterns alone.

**Key Features:**

- **Value-only labeling**: Only the VALUE is labeled, not the key (e.g., `"email": "john@example.com"` → only the email value is labeled)
- **50+ field key mappings**: Maps field names like `email`, `phone`, `username`, `organization`, `api_key`, etc. to canonical entity labels
- **Format support**: JSON, YAML, Markdown, and XML field structures
- **Validation**: Deduplicates against existing spans, validates USERNAME fields against emails
- **Label-specific confidence**: Adjusts confidence by label type (API_KEY: 0.93, EMAIL: 0.90, PHONE: 0.80, etc.)

**Example:**

```python
from taivium import Taivium

text = '''
{
  "email": "alice@example.com",
  "phone": "+1-555-1234",
  "username": "alice_smith"
}
'''

pipeline = Taivium()
result = pipeline.process(text)
# All three values are detected with appropriate labels and high confidence
```

### Organization List (Compliance-Friendly Detection)

The organization list layer provides fast, deterministic, fully-auditable detection of known organizations. Pass a list of known organizations to enable this compliance-friendly Tier 1 detection:

```python
from taivium import Taivium

# Tier 1: Curated organization list (fast, auditable, confidence=0.95)
# Falls back to Tier 2 (GLiNER) for unknown organizations
pipeline = Taivium()
result = pipeline.process(
    text="Acme Corporation approved the request.",
    known_orgs=["Acme Corporation", "Beta Industries", "Gamma LLC"]
)

# Organizations in known_orgs are detected via exact-match lookup
# with case-insensitive matching and Unicode support.
# Results include source="org_list" for audit trails.
for entity_id, entity_meta in result["mapping"].items():
    if entity_meta["source"] == "org_list":
        print(f"Known org detected: {entity_meta['text']} → {entity_id}")
```

**Why use org_list?**

- **Compliance**: Satisfies GDPR Article 32 (Privacy by Design) via deterministic detection rules, not probabilistic ML
- **Performance**: <1ms per text vs 5-10ms for GLiNER
- **Auditability**: Explicit organization list, exact-match logic, deterministic results
- **Precision**: 0.95 confidence from curated list; no false positives

**Architecture:**

- **Layer 0 (org_list)**: Known organizations, exact-match, confidence=0.95, <1ms
- **Layer 1 (GLiNER fallback)**: Unknown organizations, ML-based, confidence=0.55, 5-10ms
- **Layer 2 (optional recurrence)**: Repeated mentions of detected organizations

### spaCy NER Configuration

The spaCy detector defaults to `en_core_web_sm` and is configurable via `spacy_model_name`.
Adaptive routing uses a default `short_text_threshold=100` characters and is configurable
via both `Taivium(short_text_threshold=...)` and `module_engine_process(..., options={"short_text_threshold": ...})`:

```python
from taivium.engine import Taivium, module_engine_process

# Default model
pipeline = Taivium()

# Configure a different installed spaCy model
pipeline = Taivium(spacy_model_name="en_core_web_lg")

# Configure long-text context backend (short-text route remains spaCy)
pipeline = Taivium(context_ner_backend="gliner")
pipeline = Taivium(context_ner_backend="openai-privacy-filter")

# Also configurable in module_engine_process options
result = module_engine_process(
  "Alice Johnson from Acme Corp",
  options={
    "spacy_model_name": "en_core_web_lg",
    "context_ner_backend": "gliner",
  },
)
```

## Session Identity Store

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