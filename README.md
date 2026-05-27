# Taivium

> **Privacy firewall for AI applications.**
> Sensitive data in — safe prompts out — coherent responses back.

Stop PII, secrets, and customer data from leaking into LLMs.  
Taivium sits between your app and any AI provider, automatically protecting every request — with zero prompt engineering and zero LLM quality loss.

[![CI](https://github.com/taivium-ai/taivium/actions/workflows/ci.yml/badge.svg)](https://github.com/taivium-ai/taivium/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/taivium-ai/taivium/branch/main/graph/badge.svg)](https://codecov.io/gh/taivium-ai/taivium)
[![PyPI version](https://img.shields.io/pypi/v/taivium)](https://pypi.org/project/taivium/)
[![PyPI downloads](https://img.shields.io/pypi/dm/taivium)](https://pypi.org/project/taivium/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

Every AI call is a potential data leak.

- Support copilots receive customer names, account numbers, and emails
- Developer tools get pasted with API keys and credentials
- Healthcare and legal workflows feed patient and case data directly to hosted LLMs
- RAG pipelines embed confidential business context into external vector stores

Standard redaction tools make it worse: replacing `Alice` and `Bob` with `[NAME]` and `[NAME]` destroys identity — your LLM can no longer reason about who is who.

**Taivium solves both problems at once.**

---

## How It Works

```
Your App ──► Taivium ──► LLM (OpenAI / Claude / local)
                │
    ┌───────────┴────────────┐
    │  Detection Layer       │  spaCy · regex · transformer · LLM
    │  Span Canonicalization │  one authoritative entity per span
    │  Identity Engine       │  PERSON_a1b2c3 — consistent forever
    │  Policy Engine         │  ALLOW · ANONYMIZE · BLOCK
    │  Session Store         │  in-memory or Redis-backed
    └────────────────────────┘
                │
    Restore original entities in LLM response (optional)
```

Unlike dumb masking, Taivium assigns **consistent pseudonymous tokens** across the entire session. `Alice Johnson` always becomes the same `PERSON_a1b2c3` — so your LLM can still reason coherently about identity across multi-turn conversations, RAG pipelines, and agent workflows.

**Full AI quality. Zero data exposure.**

---

## Installation

```bash
pip install taivium
```

Redis support (optional, for cross-session persistence):

```bash
pip install taivium[redis]
```

---

## Quick Start

### Drop-in OpenAI client

Two lines change. Everything else stays the same.

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
    deid_response=True,  # restore original names in the response
)

print(response.choices[0].message.content)
# → Response refers to "John Doe" naturally. The LLM never saw the real name.
```

### Sanitize text directly

```python
from taivium.engine import Taivium

pipeline = Taivium()
result = pipeline.process(
    "Alice Johnson from Acme Corp emailed alice@acme.com. "
    "Her API key is sk-1234567890abcdef."
)

print(result["anonymized"])
# → "PERSON_c54953ca from ORG_8d92f143 emailed EMAIL_3fedc406.
#    Her API key is APIKEY_1a2b3c4d."

print(result["store_type"])
# → "InMemorySessionStore"
```

The same entity always maps to the same token — coherent AI reasoning, guaranteed.

### Custom policies

Block certain entity types outright. Allow others through unchanged.

```python
from taivium.engine import (
    Taivium, PolicyEngine, PolicyRule, PolicyAction, RiskLevel
)

pipeline = Taivium(
    policy_engine=PolicyEngine(policy_table={
        "API_KEY":  PolicyRule("API_KEY",  PolicyAction.BLOCK,     RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW,     RiskLevel.LOW),
    })
)

try:
    result = pipeline.process("My key is sk-secret123 and I'm in New York.")
except ValueError as e:
    print(e)  # Blocked sensitive entity: sk-secret123 (API_KEY)
```

### Redis-backed session persistence

Keep the same pseudonymous tokens across restarts, workers, and long-running sessions.

```python
from taivium.engine import Taivium
from taivium.session_store import RedisSessionStore

store = RedisSessionStore(
    session_id="user-abc123",
    redis_url="redis://localhost:6379",
    ttl=3600,
)
pipeline = Taivium(session_store=store)

r1 = pipeline.process("Alice Johnson sent a report.")
r2 = pipeline.process("Alice Johnson followed up.")

# r1 and r2 use the same PERSON_* token for Alice — always.
print(r1["store_type"])  # → "RedisSessionStore"
```

Start Redis in seconds:

```bash
docker run -d -p 6379:6379 redis:7-alpine
```

### Transformer + LLM evidence (optional)

Two extra detection layers can be enabled on top of the default spaCy + regex
detectors. Both are **opt-in** and degrade gracefully when unavailable.

```python
from taivium.engine import Taivium

# Transformer layer — requires: pip install transformers torch
# Uses dslim/bert-base-NER (BERT fine-tuned on CoNLL-2003).
pipeline = Taivium(use_transformer=True)

result = pipeline.process(
    "Dr. Emily Clarke joined Horizon AI in Boston. "
    "Reach her at emily.clarke@horizonai.io or call +1-617-555-0199."
)

for entity in result["entities"]:
    print(
        f"[{entity['label']:10}] {entity['text']!r:30}"
        f" sources={entity['evidence_sources']}  conf={entity['confidence']:.2f}"
    )
# [PERSON    ] 'Emily Clarke'                 sources=('spacy', 'transformer')  conf=0.87
# [ORG       ] 'Horizon AI'                   sources=('transformer',)           conf=1.00
# [LOCATION  ] 'Boston'                       sources=('spacy', 'transformer')   conf=0.87
# [EMAIL     ] 'emily.clarke@horizonai.io'    sources=('regex',)                 conf=0.90
# [PHONE     ] '+1-617-555-0199'              sources=('regex',)                 conf=0.80
```

Enable the LLM layer (OpenAI) alongside or independently:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

# Both layers active — broadest detection coverage
pipeline = Taivium(use_transformer=True, use_llm=True)
result = pipeline.process("...")
```

When multiple detectors agree on a span, `evidence_sources` lists all of them
and `confidence` is their mean score (see [Confidence Scoring](#confidence-scoring)).
If a layer is unavailable (missing package or missing API key) it is silently
skipped — the other detectors still run.

### Replace a detector with your own

Pass `transformer_fn` or `llm_fn` to swap the built-in implementation with any
callable that accepts a `str` and returns `List[Evidence]`. The `use_transformer`
/ `use_llm` flag must still be `True` for the layer to run.

```python
from taivium.engine import Evidence, Taivium

def my_ner(text: str) -> list[Evidence]:
    """Drop-in replacement — use any model, API, or rule set."""
    # example: tag a known codename
    idx = text.find("Agent X")
    if idx == -1:
        return []
    return [Evidence(start=idx, end=idx + 7, label="PERSON",
                     source="transformer", confidence=0.95)]

pipeline = Taivium(use_transformer=True, transformer_fn=my_ner)
result = pipeline.process("Agent X joined the briefing.")
# → "PERSON_... joined the briefing."
```

The same pattern applies to `use_llm=True, llm_fn=my_llm_fn` — useful for
routing LLM calls to a private endpoint or a different model provider.

---

## What Gets Detected

| Entity Type | Examples |
|---|---|
| `PERSON` | Alice Johnson, Bob Smith |
| `ORG` | Acme Corp, Beta LLC |
| `EMAIL` | alice@acme.com |
| `PHONE` | +1 415-555-1234, +44 20 7946 0958 |
| `API_KEY` | sk-1234567890abcdef, ZXCVBNMASDF |
| `LOCATION` | San Francisco, New York, Sahara Desert |

Detection combines **spaCy NER**, **regex patterns**, and optionally a
**transformer NER** model and an **LLM evidence layer** — all fused by a span
canonicalization engine into one authoritative entity per span.

---

## Confidence Scoring

Every detected entity carries a `confidence` value in `[0, 1]`. It is computed
in two stages:

**1. Per-detector raw score** — each detector assigns a fixed or model-derived
score to every span it emits:

| Detector | Score | Source |
|---|---|---|
| `spacy` | `0.75` | Fixed (spaCy `en_core_web_sm` has no per-entity probability) |
| `regex` | `0.90` (email) · `0.80` (phone) · `0.95` (API key) | Fixed per pattern type |
| `transformer` | Model probability | Mean softmax score across the entity's tokens (`pred["score"]` from HuggingFace pipeline with `aggregation_strategy="simple"`) |
| `llm` | `0.85` | Fixed (OpenAI does not return token-level log-probs by default) |

**2. Final entity confidence** — when multiple detectors agree on the same
span and label, the `confidence` stored on the `Entity` is the **mean** of
their individual scores:

```
confidence = mean(score_detector_1, score_detector_2, ...)
```

Example — "Emily Clarke" detected by both spaCy (`0.75`) and the transformer
(`1.00`):

```
confidence = (0.75 + 1.00) / 2 = 0.875  →  rounded to 0.87 in display
```

> **Span selection vs. reported confidence** — when detector spans *overlap*,
> the optimizer picks the winning span using a richer internal score (source
> reliability weights + span-length bonus). That score is only used for
> selection; the `confidence` field on the final entity is always the simple
> mean of the contributing detectors.

---

## Why Not Simple Redaction?

| | Simple redaction | Taivium |
|---|---|---|
| Input | `John emailed Mary` | `John emailed Mary` |
| To LLM | `[NAME] emailed [NAME]` | `PERSON_a1b2 emailed PERSON_c3d4` |
| LLM can distinguish identities? | ✗ | ✓ |
| Same entity = same token across calls? | ✗ | ✓ |
| Response restoration | ✗ | ✓ |
| Policy engine (block / allow / anonymize) | ✗ | ✓ |
| Audit trail per request | ✗ | ✓ |

---

## Features

| Capability | Detail |
|---|---|
| Multi-layer detection | spaCy + regex + transformer + LLM evidence (pluggable) |
| Span canonicalization | Weighted voting resolves overlapping detector evidence — see [Confidence Scoring](#confidence-scoring) |
| Consistent identity | Same entity text → same token, always |
| Policy engine | Per-label ALLOW / ANONYMIZE / BLOCK rules |
| Response restoration | `reverse_transform()` rebuilds original names in LLM output |
| Session persistence | In-memory (default) or Redis-backed |
| Audit trail | Full entity mapping + risk + policy decision per call |
| Latency tracking | Per-call processing time logged in `pipeline.latency_history` |
| OpenAI drop-in | `PrivacyClient` wraps any OpenAI-compatible endpoint |
| `store_type` in result | Every `process()` result reports which store backend is active |

---

## The `process()` Result

```python
result = pipeline.process(text)
# {
#   "original":   "Alice Johnson emailed alice@acme.com",
#   "anonymized": "PERSON_c549 emailed EMAIL_3fed",
#   "store_type": "InMemorySessionStore",       # or RedisSessionStore
#   "mapping": {
#     "PERSON_c549": {
#       "text": "Alice Johnson", "label": "PERSON",
#       "action": <PolicyAction.ANONYMIZE>, "risk": <RiskLevel.HIGH>, ...
#     },
#     ...
#   },
#   "entities": [{"text": "Alice Johnson", "label": "PERSON", "id": "PERSON_c549", ...}]
# }
```

---

## Architecture

Taivium is a single Python package. All detection, identity resolution, policy enforcement, and transformation run inside `Taivium` — no external engine required.

Everything in the pipeline runs in-process:

```
engine.py
├── collect_evidence()        spaCy + regex + transformer (stub) + LLM (stub)
├── canonicalize_spans()      weighted voting → one entity per span
├── IdentityEngine.resolve()  deterministic pseudonymous IDs
├── PolicyEngine.evaluate()   ALLOW / ANONYMIZE / BLOCK per label
├── transform()               span replacement → safe text
└── session_store             InMemorySessionStore or RedisSessionStore
```

---

## Documentation

See design document in [docs/](docs/) for detailed architecture, design decisions, and implementation details.

---

## Examples

See the [`examples/`](examples/) folder for runnable end-to-end scripts:

| File | Demonstrates |
|---|---|
| [`example_privacy_pipeline.py`](examples/example_privacy_pipeline.py) | Default pipeline · custom policies · Redis session · transformer + LLM layers · custom detector injection |
| [`example_client.py`](examples/example_client.py) | Drop-in `PrivacyClient` (OpenAI-compatible) with de-identification |

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Add tests — all changes require test coverage
4. Submit a pull request with a clear description

---

## License

MIT — see [LICENSE](LICENSE).

---

## Support

Open an issue for bugs, questions, or feature requests.
