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
    redis_url="redis://localhost:6379",
)

pipeline = Taivium(session_store=store)
```

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
  "mapping": {...},
  "entities": [...]
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