# Tarvium — Software Design Document

## 1. Introduction

### 1.1 Purpose

Tarvium is a middleware system that protects sensitive data (PII, secrets, and internal business entities) before it reaches Large Language Models (LLMs), while preserving semantic meaning and downstream model utility.

It provides:

* SDKs (Python + Node.js)
* Optional OpenAI-compatible proxy
* Core Privacy Engine for detection, anonymization, and policy enforcement

### 1.2 Design Goals

* Deterministic privacy transformation
* High utility preservation for LLM outputs
* Low-latency middleware (<200ms overhead target)
* OpenAI-compatible drop-in integration
* Strong auditability and observability  
    _(Current implementation: `PrivacyPipeline.session_store` accumulates the full entity audit trail across calls. `PrivacyPipeline` records per-call processing latency in `latency_history` (capped at the last 1 000 entries). Redis-backed session persistence is supported via `RedisSessionStore`. Structured logging and external observability integrations are planned but not yet implemented.)_
* Consistent entity anonymization across sessions

### 1.3 Key Principle

> A privacy firewall between your application data and any LLM — sensitive data in, safe prompts out, coherent responses back.

---

## 2. System Architecture

### 2.1 High-Level Architecture

Client App
→ SDK (Python / Node.js) OR Proxy Server
→ Privacy Engine

* Detection Layer (PII / secrets / entities)
* Policy Engine (block / redact / anonymize / route)
* Semantic Anonymization Engine
* Entity Mapping Store
* Output Leakage Detector
* Observability Layer
  → LLM Router
* OpenAI / Anthropic APIs
* Local Models
* Fallback / Block responses

### End-to-End Architecture (MVP)
[1] **Detection Layer**
    - spaCy evidence
    - Regex evidence
    - Transformer evidence (placeholder)
    - LLM evidence (placeholder)
   ↓
[2] **Evidence Layer**
    - `Evidence(start, end, label, source, confidence)`
    - merged by `collect_evidence()`
   ↓
[3] **Span Canonicalizer** (`canonicalize_spans`)
    - sweep-line overlap-cluster grouping (any overlapping spans join one cluster)
    - weighted label vote per cluster (`confidence + SOURCE_WEIGHT[source]`)
    - longest span selected as canonical span within each cluster
    - one `Entity(source="canonical")` per cluster; retains `evidence_sources` lineage and canonical vote `confidence`; clusters are non-overlapping by construction
   ↓
[4] **Identity Engine (CORE)**
    - deterministic mapping
    - PERSON_abcdef123456, ORG_7b3c9e012345, etc.
    - same (text, label) → same ID
   ↓
[5] **Policy Engine**
    - ALLOW / ANONYMIZE / BLOCK
   ↓
[6] **Anonymization Engine**
    - span replacement
    - safe text rebuild
   ↓
[7] **(Optional) Reverse Mapping Layer**
    - restore original entities

---

### 2.2 Trust Zones

#### Raw Data Zone (Untrusted)

* Original user input
* Contains sensitive data
* Must never leave system untransformed

#### Transformed Data Zone (LLM-Safe)

* Anonymized representation
* Safe for external LLM processing
* Deterministic entity mappings

#### Reconstructed Output Zone (Optional)

* De-anonymized outputs
* Strict access control required

---

## 3. Core Components

### 3.1 SDK Layer (`PrivacyClient`)

`PrivacyClient` is an OpenAI-compatible drop-in client implemented in `src/tarvium/client.py`. It de-identifies every outgoing message through `Tarvium` before the request leaves the process, and optionally reverses entity tokens in the LLM response.

#### Responsibilities

* Intercept `chat.completions.create()` calls
* Run each string `content` field through `Tarvium.process()`
* Persist the entity mapping via the pipeline's `session_store` (in-memory or Redis)
* Optionally reverse-map tokens in the response (`deid_response=True`)
* Forward all other kwargs unchanged to the underlying `openai.OpenAI` client

#### Public API

```python
from tarvium import PrivacyClient

# Default: in-memory session store
client = PrivacyClient(api_key="sk-...")

# Redis-backed: mappings survive process restarts
client = PrivacyClient(
    api_key="sk-...",
    session_id="user-abc123",
    redis_url="redis://localhost:6379",
    redis_ttl=3600,   # seconds; default 86400 (24 h)
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Alice at alice@acme.com needs help."}],
    deid_response=True,   # optional: reverse-map tokens in the reply
)

print(client.session_mapping)  # full entity audit trail
client.reset_session()         # clear between independent conversations
```

#### Attribute hierarchy (mirrors `openai.OpenAI`)

```
PrivacyClient
└── chat  (_PrivacyChat)
    └── completions  (_PrivacyCompletions)
        └── create(messages, deid_response=False, **kwargs)
```

`openai` is a deferred, optional import — the core pipeline works without it.

#### `reverse_transform(text, mapping)`

Module-level utility in `engine.py` that replaces every entity ID token in `text` with its original value from `mapping`. Tokens are substituted longest-first to prevent prefix-collision corruption. Used internally by `PrivacyClient` when `deid_response=True`, and available as a standalone export.

---

### 3.2 Privacy Engine

#### 3.2.1 Detection Layer
##### Recurrence Evidence (Semantic Recurrence Layer)

**Summary:** The recurrence evidence layer is a deterministic, privacy-preserving mechanism that increases recall for repeated sensitive entities by finding additional surface-form mentions of canonical entities that may have been missed by NER detectors. It is strictly gated for privacy: only eligible canonical entities are considered, and all recurrence logic is deterministic, non-overlapping, and idempotent. No new labels, widened spans, or invented entities are ever introduced.

**Privacy guarantees:**
- Only canonical entities that pass strict eligibility gating are considered for recurrence (default: `EMAIL`, `PHONE`, `API_KEY`; `PERSON` and `ORG` are gated by heuristics to prevent ambiguous or unsafe cloning).
- Recurrence evidence is generated by scanning the input text for exact, boundary-validated substring matches of canonical entity surfaces.
- All recurrence spans are checked to ensure they do not overlap with canonical entities, and no new labels or wider spans are ever created.
- Every recurrence entity is marked with `source="recurrence"` and inherits provenance and confidence from its canonical source, ensuring full auditability.
- The process is deterministic and idempotent: running recurrence detection multiple times produces the same result, and recurrence is never applied recursively.

**Privacy rationale:** Recurrence evidence increases recall for repeated sensitive entities (e.g., multiple mentions of the same email or name) while maintaining strict safety and auditability guarantees. This ensures that privacy boundaries are never violated, and all recurrence logic is fully explainable and testable.

**How it works:**
- After canonicalization, the pipeline applies the recurrence layer to recover repeated surface-form mentions of eligible canonical entities.
- For each eligible canonical entity, the text is scanned for exact substring matches, with strict boundary checks (word/non-word/whitespace) to avoid overmatching or partial matches.
- New evidence records (`Evidence`) with `source="recurrence"` are emitted for each safe, non-overlapping recurrence found.
- Recurrence evidence is merged with detector evidence and re-canonicalized, so all spans are globally optimized together.

**Detects:**
- PII (names, emails, phones, addresses)
- Credentials (API keys, tokens)
- Financial identifiers
- Internal business entities

**Methods:**
- Regex
- spaCy NER
- Transformer-based NER
- Optional LLM tagging

##### Hybrid Detection Architecture

The detector stage emits evidence from all enabled detector layers, then resolves that noisy evidence to canonical entities, with a semantic recurrence layer to recover missed repeated entities:

```
Input Text
    |
    v
[Layer 1] spaCy evidence
    |
    v
[Layer 2] regex evidence
    |
    v
[Layer 3] transformer evidence (opt-in: use_transformer=True)
    |
    v
[Layer 4] LLM evidence (opt-in: use_llm=True)
    |
    v
collect_evidence()  -> raw evidence list
    |
    v
canonicalize_spans() -> canonical entity set
    |
    v
find_recurrences() -> recurrence entities (optional, non-overlapping)
```

* Each layer runs independently; failures in one layer do not block others
* Evidence is merged first, then canonicalized
* Canonicalization uses a sweep-line overlap-cluster algorithm: overlapping evidence spans are grouped into connected clusters, then each cluster is resolved to one canonical entity via weighted label voting
* After canonicalization, the semantic recurrence layer finds repeated surface-form mentions only for recurrence-eligible canonical entities (default: `EMAIL`, `PHONE`, `API_KEY`; gated heuristics for `PERSON` and `ORG`) and adds them as new, non-overlapping entities with source="recurrence", inheriting canonical `evidence_sources` and `confidence`. Matching uses exact substring scanning plus manual boundary validation (Unicode-aware character classes), not regex `\\b` heuristics.
* Clusters and recurrences are non-overlapping by construction; `transform()` applies a defensive overlap check as a safety net

---

#### 3.2.1.1 Canonical Span Resolution

Evidence from multiple detectors can conflict. `canonicalize_spans(text, evidence)` resolves this into a single canonical entity set using a **weighted interval scheduling (DP) algorithm**:

1. **Filter** — discard any evidence with invalid span boundaries or unknown labels.
2. **Group** — merge evidence records by exact `(start, end, label)` equivalence into `SpanCandidate` objects; different span boundaries remain distinct competing hypotheses.
3. **Score** — score each candidate deterministically: sum of detector confidences, source reliability weights (`SOURCE_WEIGHT`), a capped linear span-length bonus (`min(0.15 * span_len, 2.0)`), and a label prior. Higher score = preferred candidate.
4. **Schedule** — solve weighted interval scheduling via DP to select the globally optimal, strictly non-overlapping candidate set. Tie-breaking is deterministic using `(score, coverage, -count, -start, -end, label)` keys so equal-score ties never depend on iteration order.
5. **Emit** — produce one `Entity(source="canonical")` per selected candidate with retained `evidence_sources` (sorted union of contributing detector sources) and averaged `confidence`; the selected set is non-overlapping by construction.


#### 3.2.1.2 Canonicalization and Span Integrity Contract

The canonicalization and transformation pipeline must satisfy the following hard invariants:

1. **Deterministic**
    For the same input evidence set, canonicalization must produce the same output entities and ordering.
2. **Non-overlapping**
    Canonical entities must never overlap in character spans. This is enforced by `assert_non_overlapping` after canonicalization, recurrence, and before transformation.
2b. **Monotonic ordering**
    Entity sets must be sorted in ascending `start` order and each entity must satisfy `start < end`.
2c. **Text-span integrity**
    Every entity must satisfy `entity.text == text[entity.start:entity.end]`.
2d. **Canonical immutability**
    Canonical entities must never mutate after creation (no widening, shrinking, label mutation, or text/source drift).
3. **Stable under detector variance**
    Small detector-level variation should not cause unstable identity outcomes when evidence still supports the same semantic entity.
4. **Span-safe**
    Every emitted span must be valid (`0 <= start < end <= len(text)`) and safe for downstream replacement.
5. **Recurrence-safe**
    Recurrence expansion must not introduce overlaps with canonical spans and must preserve inherited label/provenance.
6. **Idempotent canonicalization**
    `canonicalize_spans(canonicalize_spans(E)) == canonicalize_spans(E)` for any evidence set `E`. Canonicalization is stable and repeatable.
7. **Idempotent recurrence**
    `find_recurrences(text, canonical + recurrences)` must produce no new entities. Recurrence expansion is stable and does not recursively amplify.
8. **Transform/reverse_transform roundtrip**
    `reverse_transform(transform(text, ...), mapping) == text` for all valid entity sets and mappings. This roundtrip is tested aggressively and catches overlap corruption, offset drift, and replacement ordering bugs.
9. **Auditable**
    Canonical output must retain decision lineage (`source`, `evidence_sources`, `confidence`) for explainability and enterprise review.

These invariants are enforced by explicit assertions in code and validated by comprehensive tests. Any violation is a critical bug.

**Test Coverage Mapping:**

- Non-overlapping spans: `test_assert_non_overlapping_detects_overlap`, `test_assert_non_overlapping_passes_for_non_overlap`, enforced after canonicalization, recurrence, and before transform.
- Monotonic ordering and valid spans: `test_assert_non_overlapping_detects_unsorted_entities`, `test_assert_non_overlapping_detects_invalid_span`
- Text-span integrity: `test_assert_text_span_integrity_detects_mismatch`, `test_transform_raises_on_text_span_mismatch`
- Canonical immutability: `test_assert_canonical_immutability_passes_when_unchanged`, `test_assert_canonical_immutability_detects_label_mutation`, `test_assert_canonical_immutability_detects_span_mutation`
- Canonicalization idempotence: `test_canonicalize_idempotent`
- Recurrence idempotence: `test_find_recurrences_idempotent`
- Transform/reverse_transform roundtrip: `test_transform_reverse_transform_roundtrip`

This mapping ensures traceability and auditability for all span integrity and idempotence guarantees. Any future changes must update both the invariants and their corresponding tests.


##### Recurrence Layer Invariants (Explicit)

The recurrence layer (`find_recurrences`) enforces the following invariants to guarantee privacy and safety:

1. **No invention**: Recurrences may ONLY replicate existing canonical entities. No new labels, widened spans, or invented entities are allowed.
2. **Label, identity, and boundaries preserved**: Recurrences must exactly match the canonical entity's label and surface text (span boundaries).
3. **No invented labels or widened spans**: Recurrences cannot introduce new labels or expand the span beyond the canonical entity.
4. **No overlap with canonical spans**: Recurrences must not overlap any canonical entity span.
5. **No recursive recurrence chains**: Recurrences are only generated from canonical entities, never from other recurrences.
6. **Source is always "recurrence"**: All recurrence entities have `source="recurrence"` for auditability.
7. **Recurrence label and text exactly match canonical**: Ensures deterministic, safe anonymization.

These invariants are enforced by explicit assertions in code and validated by comprehensive tests.

**Test Coverage Mapping:**

Each recurrence invariant is validated by one or more tests in `tests/test_privacy_pipeline.py`:

- Only canonical entities are replicated (no invention):
    - `test_find_recurrences_catches_second_mention`, `test_find_recurrences_multiple_occurrences`
- Recurrence eligibility gate blocks ambiguous lexical cloning:
    - `test_find_recurrences_skips_single_token_person`
- Recurrence eligibility gate preserves safe deterministic classes:
    - `test_find_recurrences_allows_email_recurrence`
- Label, identity, and boundaries are preserved:
    - `test_find_recurrences_inherits_provenance`, `test_find_recurrences_catches_second_mention`
- No invented labels or widened spans:
    - `test_find_recurrences_respects_word_boundaries`, `test_find_recurrences_case_sensitive`
- Manual boundary guards are Unicode-safe and regex-boundary independent:
    - `test_find_recurrences_manual_boundary_apostrophe_org`, `test_find_recurrences_manual_boundary_hyphen_org`, `test_find_recurrences_manual_boundary_cjk_org`
- No overlap with canonical spans:
    - `test_find_recurrences_no_duplicates_of_canonical`
- No recursive recurrence chains:
    - `test_find_recurrences_idempotent`
- Source is always "recurrence":
    - `test_find_recurrences_inherits_provenance`, `test_find_recurrences_catches_second_mention`
- Recurrence label and text exactly match canonical:
    - `test_find_recurrences_inherits_provenance`, `test_find_recurrences_catches_second_mention`

This mapping ensures traceability and auditability for all recurrence safety guarantees. Any future changes to recurrence logic must update both the invariants and their corresponding tests.


#### 3.2.2 Policy Engine

The policy engine currently determines actions from entity labels. It is implemented via `PolicyEngine`, `PolicyRule`, `PolicyAction`, and `RiskLevel`.

Current decision surface:

* label-only (`entity.label`) for deterministic MVP behavior

Forward-compatible direction (already wired into the API shape):

* `(label, context, confidence, source)` via optional `PolicyContext`
* `PolicyEngine.evaluate(entity, context=None)` accepts optional context while preserving label-only behavior by default
* Subclasses can override `evaluate(...)` to implement contextual rules (for example: public executive names in press releases versus internal emails in stack traces)

##### `PolicyAction` (enum)

| Value | Description |
|-------|-------------|
| `ALLOW` | Entity is passed through untransformed |
| `ANONYMIZE` | Entity is replaced with a deterministic hash-based token |
| `BLOCK` | Processing is halted and a `ValueError` is raised |

> Note: `redact` and `route` are planned actions (see requirements) but not yet implemented.

##### `RiskLevel` (enum)

| Value | Description |
|-------|-------------|
| `LOW` | Minimal sensitivity (e.g. location) |
| `MEDIUM` | Moderate sensitivity (e.g. person name, org) |
| `HIGH` | High sensitivity (e.g. email, phone) |
| `CRITICAL` | Secrets / credentials (e.g. API keys) |
| `UNKNOWN` | Unlabeled or undefined entity type |

##### `PolicyRule` (dataclass)

```python
@dataclass
class PolicyRule:
    label: str           # entity label (e.g. "PERSON")
    action: PolicyAction # action to take
    risk: RiskLevel      # risk classification
```

##### `DEFAULT_POLICY`

| Label | Action | Risk |
|-------|--------|------|
| `PERSON` | `ANONYMIZE` | `MEDIUM` |
| `ORG` | `ANONYMIZE` | `MEDIUM` |
| `LOCATION` | `ANONYMIZE` | `LOW` |
| `EMAIL` | `ANONYMIZE` | `HIGH` |
| `PHONE` | `ANONYMIZE` | `HIGH` |
| `API_KEY` | `ANONYMIZE` | `CRITICAL` |

##### Unknown Label Default

Entity labels not present in the policy table receive a default rule. The default action is controlled by the `default_action` parameter of `PolicyEngine.__init__`:

* `default_action=PolicyAction.ANONYMIZE` (default) → `ANONYMIZE` with `RiskLevel.UNKNOWN`
* `default_action=PolicyAction.ALLOW` → `ALLOW` with `RiskLevel.UNKNOWN`

##### `PolicyDecisionReason` (enum)

Records why a particular policy rule was applied:

| Value | Description |
|-------|-------------|
| `EXPLICIT` | Entity label matched an explicit entry in the policy table |
| `FALLBACK` | Entity label was not in the policy table; default action applied |

##### `PolicyDecision` (frozen dataclass)

The object returned by `PolicyEngine.evaluate()`:

```python
@dataclass(frozen=True)
class PolicyDecision:
    label: str                    # entity label
    action: PolicyAction          # action to take
    risk: RiskLevel               # risk classification
    reason: PolicyDecisionReason  # why this rule was selected
```

`PolicyDecision` is immutable and is stored alongside the entity and its ID in `Tarvium.process()` as a `(Entity, str, PolicyDecision)` triple — ensuring the action, risk, and reason are always co-located with the entity they describe.

##### Custom Policy

A custom `Dict[str, PolicyRule]` can be passed to `PolicyEngine(policy_table=...)` and injected into `PrivacyPipeline(policy_engine=...)` to override default behaviour per entity label.

---

### 3.2.4 Entity Mapping Store (`IdentityEngine`)


Maps detected entities to deterministic, collision-resistant IDs.

### 3.2.4.1 Privacy-Preserving ID Scoping and Hash Length

By default, entity IDs are globally stable: the same entity text and label always produce the same anonymized token (e.g., `PERSON_abcdef123456`).
This enables consistent anonymization across documents and sessions, but also allows cross-document and cross-tenant linkage of identical entities.

**Tarvium now supports privacy-preserving deployments via two new options:**

- `id_salt`: An optional salt (string) that scopes entity IDs to a tenant, session, or namespace. When set, identical entities in different tenants/sessions will receive different anonymized IDs, preventing cross-tenant or cross-session linkage.
- `id_hash_len`: The number of hex digits to use from the hash (default 12 for legacy compatibility). Longer hashes increase collision resistance; shorter hashes may be used for more compact tokens.

**Usage Examples:**

```python
# Default (global, legacy-stable IDs; not privacy-preserving)
engine = Tarvium()

# Tenant-scoped IDs (prevents cross-tenant linkage)
engine = Tarvium(id_salt="tenant_1234")

# Session-scoped IDs (prevents cross-session linkage)
engine = Tarvium(id_salt="session_5678")

# Custom hash length (longer IDs)
engine = Tarvium(id_hash_len=24)

# Both salt and custom hash length
engine = Tarvium(id_salt="tenant_1234", id_hash_len=24)
```

**Privacy Implications:**

- Setting a unique salt per tenant or session ensures that identical entities across customers or sessions cannot be correlated by their anonymized IDs.
- This is strongly recommended for privacy-preserving deployments, especially in multi-tenant or regulated environments.
- The default (no salt) is provided for backward compatibility and for use cases where global linkage is desired.

**Best practice:** Always set a unique salt per tenant or session in regulated or multi-tenant environments to prevent cross-tenant or cross-session linkage of anonymized IDs.

See the `Tarvium` class docstring in `src/tarvium/engine.py` for more details and usage patterns.

**ID generation key:** `(label, normalize_identity_text(text))` — purely semantic. The same entity text and label always produce the same token, regardless of position.

`normalize_identity_text()` applies:

* Unicode normalization (NFKC)
* Case folding
* Internal whitespace collapsing
* Edge punctuation stripping

**Positional deduplication:** `resolve()` does **not** deduplicate by position. It returns one `(Entity, id)` tuple per input entity, preserving all positional occurrences so that `transform()` can replace every span in the document. Entities sharing the same `(text, label)` receive the same deterministic ID.

Stores anonymization state:

* entity hash (SHA-256, first 12 hex chars, deterministic per entity type and value)
* original value
* token
* type

#### Session Identity Store

`PrivacyPipeline` holds a pluggable `session_store` that accumulates the entity-ID → metadata mapping across pipeline calls. Two backends are provided:

| Backend | Class | Persistence | Use case |
|---------|-------|-------------|----------|
| In-memory (default) | `InMemorySessionStore` | Within-process lifetime only | Development, single-process apps |
| Redis | `RedisSessionStore` | Cross-call, cross-process, configurable TTL | Production, multi-instance, long-lived sessions |

`RedisSessionStore` namespaces keys as `tarvium:session:<session_id>:<entity_id>` and stores metadata as JSON. Enum and tuple values are serialized to JSON-safe types on write.

```python
from tarvium.session_store import RedisSessionStore
from tarvium import Tarvium

store = RedisSessionStore(
    session_id="user-abc123",
    redis_url="redis://localhost:6379",
    ttl=3600,
)
pipeline = Tarvium(session_store=store)
```

#### Hashing Standard

Entity hashes are generated using SHA-256 on the string `"<LABEL>:<normalized_entity_text>"`, where `<normalized_entity_text>` is produced by `normalize_identity_text()`. The digest is truncated to the first 12 hexadecimal characters. This ensures deterministic, collision-resistant mapping for anonymization tokens (e.g., `PERSON_abcdef123456`) while collapsing semantic surface variants like case, repeated spaces, punctuation edges, and equivalent Unicode forms.

---

### 3.2.5 LLM Router

Routing logic:

* Low risk → external LLM
* Medium risk → anonymized LLM
* High risk → local model
* Critical → blocked

---

### 3.2.6 Output Leakage Detector

Prevents sensitive regeneration:

* regex scanning
* reverse mapping validation
* policy re-check

---

### 3.2.7 Observability Layer

Logs:

* transformations
* mappings
* policy decisions
* routing decisions
* latency metrics

**Current implementation:** `PrivacyPipeline` records the wall-clock processing time for every `process()` call in `self.latency_history` (a `List[float]` of millisecond values, capped at 1 000 entries). No external log sink or metrics export is implemented yet.

---

## 3.3 Known Limitations

### spaCy Single-Name Detection and Recurrence Layer

spaCy's `en_core_web_sm` model frequently fails to tag single-token names (e.g. `"Alice"`, `"Bob"`) as `PERSON` entities when they appear without context clues such as a title or surname. This means first-name-only inputs may not be anonymized by the spaCy layer. The semantic recurrence layer (`find_recurrences`) addresses this by searching for repeated surface-form mentions of canonical entities and adding them as new, non-overlapping entities. This improves recall for repeated names (e.g. "Alice met Alice.") but does not help if the name is missed entirely by all detectors.

**Workaround:** Include surrounding context (e.g. `"Alice Johnson"` rather than `"Alice"`) or supply a custom policy rule that forces `ANONYMIZE` for any `PERSON`-labelled entity to ensure all detected names are handled.

### Privacy Risk of Global IDs

If no salt is provided, entity IDs are globally stable and can be linked across documents, tenants, or sessions. This may be a privacy risk in regulated or multi-tenant environments. **Always set a unique salt per tenant or session for privacy-preserving deployments.**

### Transformer and LLM Detection Layers

Both layers are fully implemented and opt-in. `transformer_evidence()` uses `dslim/bert-base-NER` via HuggingFace `transformers`; `llm_evidence()` uses `gpt-4o-mini` via the OpenAI API. Enable them with `use_transformer=True` and `use_llm=True` on `Tarvium`. All layers run additively — each adds to the evidence pool; `canonicalize_spans()` resolves conflicts.

### Regex Confidence Calibration

Regex detectors now use calibrated confidence values (email: 0.90, phone: 0.80, API key: 0.95) instead of absolute 0.99/1.0. This reflects the real-world risk of overmatching and false positives in logs, code, and noisy text.

### spaCy Model Load Time

The spaCy model is lazy-loaded on the first call to `PrivacyPipeline.process()`. The first call incurs a one-time startup cost (typically ~300–400 ms) while the model is loaded into memory. All subsequent calls run in ~10–20 ms. The model is loaded with unused pipeline components disabled (`tagger`, `parser`, `lemmatizer`, `attribute_ruler`) to minimise inference latency.
The measurement is done on a typical development machine (Macbook Pro M2; actual load times may vary based on hardware and environment).

### Latency History Cap

`PrivacyPipeline.latency_history` retains the most recent 1 000 entries. Older entries are discarded automatically.

---

## 4. Proxy Server

### Endpoint

POST /v1/chat/completions

### Pipeline

Request → Privacy Engine → LLM Router → Post-processing → Response

---

## 5. Data Flow

Input Flow:
User Input → `PrivacyClient` → `PrivacyPipeline.process()` → anonymized payload → LLM

Output Flow (optional re-identification):
LLM Output → `reverse_transform(response, session_mapping)` → original entity values restored → caller

---

## 6. Evaluation System

### 6.1 Dimensions

* Latency (p50/p95/p99)
* Detection Quality (precision/recall/F1)
* Consistency (mapping stability)
* Utility Preservation (embedding similarity)

---

### 6.2 Test Harness

Dataset → Privacy Pipeline → LLM → Scoring Engine → Benchmark Aggregation

Datasets:

* synthetic PII
* CoNLL / WikiAnn / Enron
* injected adversarial samples

---

## 7. Failure Modes

* Under-detection → conservative mode
* Over-redaction → role-preserving anonymization
* Entity collision → type-aware hashing
* Prompt injection → post-transform validation
* Output leakage → re-scanning layer

---

## 8. Performance Requirements

* SDK overhead <200ms
* deterministic transformations
* scalable stateless design (except mapping store)

---

## 9. Security

* no raw PII persistence by default
* encrypted mapping store optional
* strict access control for reconstruction mode
* audit logs for all transformations

---

## 10. Scope

* SDKs (Python + Node.js)
* regex + spaCy detection
* redaction
* basic logging
* semantic anonymization
* policy engine
* leakage detection

---

## 11. Out of Scope (MVP)

* SOC2 automation
* dashboards
* multi-region infra
* RBAC
* advanced agent orchestration

---

## 12. Success Criteria

* <10 min integration
* consistent anonymization
* OpenAI-compatible API
* minimal LLM utility loss
* measurable privacy improvement

---

## 13. Key Insight

A deterministic semantic privacy transformation engine that enforces privacy boundaries while preserving LLM utility.
