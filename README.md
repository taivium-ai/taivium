# Taivium

> **Deterministic identity + privacy layer for LLM applications**  
> Sensitive data in → safe prompts out → coherent responses back

[**View on GitHub**](https://github.com/taivium-ai/taivium)

Taivium sits between your application and any LLM provider, automatically preventing sensitive data leakage — **without breaking reasoning, context, or output quality**.

**No prompt engineering. No degraded responses. No data exposure.**

---

[![CI](https://github.com/taivium-ai/taivium/actions/workflows/ci.yml/badge.svg)](https://github.com/taivium-ai/taivium/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/taivium-ai/taivium/branch/main/graph/badge.svg)](https://codecov.io/gh/taivium-ai/taivium)
[![PyPI version](https://img.shields.io/pypi/v/taivium?cacheSeconds=0)](https://pypi.org/project/taivium/)
[![Downloads](https://img.shields.io/pypi/dm/taivium)](https://pypi.org/project/taivium/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

Every LLM call is a potential data leak.

- Support copilots handle customer PII (names, emails, accounts)
- Developers paste secrets (API keys, credentials)
- Healthcare & legal workflows send sensitive records
- RAG pipelines embed confidential company data

**Naive redaction breaks AI quality.**

```
"Alice emailed Bob" → "[NAME] emailed [NAME]"
```

→ Identity is lost → reasoning collapses.

---

## The Solution

**Taivium preserves identity *and* privacy.**

```
"Alice emailed Bob" → "PERSON_a1b2 emailed PERSON_c3d4"
```

- Identity remains distinguishable  
- Context stays intact  
- Zero sensitive data reaches the LLM  

---

## How It Works

```
Your App ──► Taivium ──► LLM (OpenAI / Claude / local)
                │
    ┌──────────────────────────────┐
    │ Detection Layer              │  spaCy · regex · transformer · LLM
    │ Span Canonicalization        │  one entity per span
    │ Identity Engine              │  deterministic pseudonyms
    │ Policy Engine                │  ALLOW · ANONYMIZE · BLOCK
    │ Session Store                │  memory or Redis
    └──────────────────────────────┘
                │
        Optional response restoration
```

**Key idea:**  
Each real-world entity gets a **stable pseudonymous ID** across the session.

---

## Why Taivium Is Different

| Capability           | Typical Tools        | **Taivium**                  |
|----------------------|---------------------|------------------------------|
| PII detection        | ✔                   | ✔                            |
| Anonymization        | Masking             | **Semantic preservation**    |
| Identity tracking    | ✗                   | **Deterministic + persistent** |
| Coreference          | Weak                | **Cross-session consistent** |
| Utility preservation | ✗                   | **Primary objective**        |

---

## Determinism

Taivium is **fully deterministic by default**:

- Same input → same output  
- Reproducible + auditable  
- No randomness  

Optional layers:  
- Transformer → deterministic  
- LLM → **non-deterministic (opt-in)**  

---

## Installation

```bash
pip install taivium
```

With Redis:

```bash
pip install taivium[redis]
```

### Model Dependency

Taivium uses spaCy for entity detection. You must manually install the English model:

```bash
pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
```

The default model is `en_core_web_sm`, and you can configure a different spaCy model at runtime:

```python
from taivium.engine import Taivium

# Default
pipeline = Taivium()

# Configured model
pipeline = Taivium(spacy_model_name="en_core_web_lg")
```

With Transformers (deep learning NER):

```bash
pip install taivium[transformers]
```

With multiple options (e.g., Redis and Transformers and dev tools):

```bash
pip install taivium[redis,transformers,dev]
```

> **Note:**
> Optional dependencies let you install only what you need. For example, if you want transformer-based NER, use the `transformers` extra. If you want Redis-backed session storage, use the `redis` extra. You can combine extras as needed.

---

## Quick Start

### 1. Drop-in OpenAI Client

```python
from taivium.client import PrivacyClient
import os

client = PrivacyClient(api_key=os.environ["OPENAI_API_KEY"])

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "John Doe's email is john@acme.com. Draft a follow-up."
    }],
    deid_response=True,
)

print(response.choices[0].message.content)
```

✔ LLM never sees real data  
✔ Output still uses real names  

---

### 2. Sanitize Text Directly

```python
from taivium.engine import Taivium

pipeline = Taivium()

result = pipeline.process(
    "Alice Johnson from Acme Corp emailed alice@acme.com"
)

print(result["anonymized"])
```

Example output:

```
PERSON_xxxx from ORG_xxxx emailed EMAIL_xxxx
```

---

### 3. Custom Policies

```python
from taivium.engine import (
    Taivium, PolicyEngine, PolicyRule, PolicyAction, RiskLevel
)

pipeline = Taivium(
    policy_engine=PolicyEngine(policy_table={
        "API_KEY": PolicyRule("API_KEY", PolicyAction.BLOCK, RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW, RiskLevel.LOW),
    })
)
```

---

### 4. Redis Persistence

```python
from taivium.engine import Taivium
from taivium.session_store import RedisSessionStore

store = RedisSessionStore(
    session_id="user-123",
    tenant_id="tenant-acme",  # optional (defaults to "default")
    redis_url="redis://localhost:6379",
)

pipeline = Taivium(session_store=store)
```

Redis keys are namespaced as:

```text
taivium:<tenant_id>:session:<session_id>:<entity_id>
```

**Multi-tenant Isolation:**  
When using `tenant_id`, it becomes the **primary isolation boundary**. The session ID component is fixed per-tenant; different requests with the same `tenant_id` share the same Redis namespace (intended for distributed session persistence across multiple requests). Different tenants get completely isolated namespaces regardless of session ID. This ensures full data isolation in multi-tenant gRPC deployments.

---

### 5. Optional Detection Layers

```python
pipeline = Taivium(
    use_transformer=True,
    use_llm=True
)
```

- Transformer → higher recall  
- LLM → broader detection  
- Both optional  

### 6. Configure spaCy Model

```python
from taivium.engine import Taivium, module_engine_process

# Class API: defaults to en_core_web_sm
pipeline = Taivium(spacy_model_name="en_core_web_sm")

# Switch to another installed spaCy model
pipeline = Taivium(spacy_model_name="en_core_web_lg")

# module_engine_process options
result = module_engine_process(
    "Alice emailed alice@acme.com",
    options={"spacy_model_name": "en_core_web_lg"},
)
```

---

## Environment Configuration

Taivium respects environment variables for logging and audit control. **Your application is responsible** for loading `.env` and configuring logging — Taivium itself does not call `logging.basicConfig()` (library best practice).

### Controlling Logger Output

**In your application code** (before importing Taivium):

```python
import logging

# Suppress engine logs
logging.getLogger("taivium.engine").setLevel(logging.WARNING)

# Suppress audit logs
logging.getLogger("taivium.audit").setLevel(logging.WARNING)

# Or configure all loggers at once
logging.basicConfig(level=logging.WARNING)

from taivium.engine import Taivium
pipeline = Taivium()
```

### Audit Output Control

Disable audit JSON emission to stdout:

```python
import os
os.environ["TAIVIUM_AUDIT_STDOUT"] = "0"

from taivium.engine import Taivium
pipeline = Taivium()
```

Or via shell:

```bash
export TAIVIUM_AUDIT_STDOUT=0
python your_app.py
```

Allowed values: `0`, `false`, `off`, `no`.  
Default: `1` (enabled).

### Example: Loading `.env` in Your App

If you want to use `.env` files for configuration:

```python
import os
import logging
from pathlib import Path

# Load .env (your app's responsibility)
def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

load_dotenv(Path(__file__).parent / ".env")

# Your app configures logging
level = getattr(logging, os.getenv("LOG_LEVEL", "WARNING"))
logging.basicConfig(level=level)

# Now use Taivium
from taivium.engine import Taivium
pipeline = Taivium()
```

---

## What Gets Detected

| Entity Type | Examples |
|-------------|----------|
| PERSON      | Alice Johnson |
| ORG         | Acme Corp |
| EMAIL       | alice@acme.com |
| PHONE       | +1 415-555-1234 |
| API_KEY     | sk-xxxx |
| LOCATION    | New York |

---

## Why Not Simple Redaction?

| | Redaction | Taivium |
|--|----------|--------|
| Identity preserved | ✗ | ✓ |
| Same entity consistency | ✗ | ✓ |
| LLM reasoning intact | ✗ | ✓ |
| Response restoration | ✗ | ✓ |
| Policy control | ✗ | ✓ |

---

## Features

- Multi-layer detection (spaCy + regex + transformer + LLM)
- Deterministic pseudonymous identity
- Policy engine (ALLOW / ANONYMIZE / BLOCK)
- Response restoration
- Redis session persistence
- Audit trail per request
- OpenAI-compatible client

---

## Example Output

```python
result = pipeline.process("Alice emailed alice@acme.com")
```

```json
{
    "original": "Alice emailed alice@acme.com",
    "anonymized": "PERSON_xxxx emailed EMAIL_xxxx",
    "store_type": "InMemorySessionStore",
    "mapping = {
        "PERSON_7e4a1c2b3d4e5f6a7b8c9d0e": "Alice",
        "EMAIL_1a2b3c4d5e6f7a8b9c0d1e2f": "alice@acme.com"
    }": {},
    "entities": [
        {
            "text": "Alice",
            "label": "PERSON",
            "start": 0,
            "end": 5,
            "source": "spacy",
            "evidence_sources": ["spacy", "regex"],
            "confidence": 0.92
        },
        {
            "text": "alice@acme.com",
            "label": "EMAIL",
            "start": 14,
            "end": 28,
            "source": "regex",
            "evidence_sources": ["regex"],
            "confidence": 0.95
        }
    ]
}
```

---

## Architecture

```
engine.py
├── collect_evidence()
├── canonicalize_spans()
├── IdentityEngine.resolve()
├── PolicyEngine.evaluate()
├── transform()
└── session_store
```

---

## Documentation

See [`docs/`](docs/) for full design details.

---

## Examples

See [`examples/`](examples/):

- privacy pipeline  
- OpenAI client  
- custom detectors  

---

## Contributing

1. Fork the repo  
2. Create a branch  
3. Add tests  
4. Open PR  

---

## License

MIT — see [LICENSE](LICENSE)

---

## Support

Open an issue for bugs, questions, or feature requests.