""" Example usage of Taivium """

from pprint import pprint
from typing import Any, Dict, List

from taivium.engine import (
    Evidence,
    PolicyAction,
    PolicyEngine,
    PolicyRule,
    Taivium,
    RiskLevel,
)
from taivium.session_store import RedisSessionStore

# pylint: disable=invalid-name

EXAMPLE_TEXT = """
Alice Johnson from Acme Corp emailed alice@acme.com.
Her API key is sk-1234567890abcdef.
She lives in San Francisco.
Bob Smith called +1 415-555-1234 and sent his API_KEY: ZXCVBNMASDF.
Carol from Beta LLC visited New York and used sk-abcdef1234567890.
Contact: carol@beta.com or +44 20 7946 0958.
The expedition crossed the Sahara Desert and camped by the Pacific Ocean.
They met at Central Park before heading to the Amazon River.
"""


# ------------------------------------------------------------
# Section 1: Default pipeline
# Demonstrates running the privacy pipeline with default settings.
# All supported entity types (PERSON, ORG, EMAIL, PHONE, API_KEY, LOCATION)
# are anonymized using pseudonymous placeholder IDs.
# ------------------------------------------------------------

def run_section1_default_pipeline(
    text: str = EXAMPLE_TEXT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the default Taivium and return the process() result.

    Args:
        text: Input text to process. Defaults to the shared example text.
        verbose: When True, pretty-prints the result to stdout.

    Returns:
        dict with keys: ``original``, ``anonymized``, ``entities``, ``mapping``.

    Expected output (abridged)::

        {'anonymized': '\\nPERSON_1 from ORG_1 emailed EMAIL_1.\\n'
                       'Her API key is APIKEY_1.\\n'
                       'She lives in LOCATION_1.\\n...',
         'entities': [{'id': 'PERSON_1', 'label': 'PERSON',
                       'text': 'Alice Johnson', ...}, ...],
         'mapping':   {'PERSON_1': {'action': <PolicyAction.ANONYMIZE: 'anonymize'>,
                                    'text': 'Alice Johnson', ...}, ...},
         'original':  '\\nAlice Johnson from Acme Corp ...'}
    """
    pipeline = Taivium()
    result = pipeline.process(text)
    if verbose:
        pprint(result)
    return result


# ------------------------------------------------------------
# Section 2: Custom policy engine
# Demonstrates overriding the default policy.
# API_KEY entities are BLOCKED (raises ValueError on detection),
# LOCATION entities are ALLOWED (passed through unchanged).
# All other entity types use the default policy.
# ------------------------------------------------------------

def run_section2_custom_policy(
    text: str = EXAMPLE_TEXT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a custom-policy pipeline and return the process() result.

    The policy blocks API_KEY entities (raises :exc:`ValueError`) and allows
    LOCATION entities through unchanged.  All other labels fall back to
    ``PolicyAction.ANONYMIZE``.

    Args:
        text: Input text to process. Defaults to :data:`EXAMPLE_TEXT`, which
              contains API keys — so the default call will raise.
        verbose: When True, pretty-prints the result to stdout.

    Raises:
        ValueError: when a BLOCK-policy entity is detected in *text*.
            Message format: ``Blocked sensitive entity: <value> (<LABEL>)``

    Returns:
        dict with keys: ``original``, ``anonymized``, ``entities``, ``mapping``.
        Only reached when *text* contains no API keys.
    """
    custom_policy = {
        "API_KEY": PolicyRule("API_KEY", PolicyAction.BLOCK, RiskLevel.CRITICAL),
        "LOCATION": PolicyRule("LOCATION", PolicyAction.ALLOW, RiskLevel.LOW),
    }
    pipeline = Taivium(policy_engine=PolicyEngine(policy_table=custom_policy))
    result = pipeline.process(text)
    if verbose:
        pprint(result)
    return result


# ------------------------------------------------------------
# Section 3: Redis-backed session store
# Demonstrates cross-call identity persistence using Redis.
# The same entity (e.g. "Alice") will map to the same placeholder ID
# across multiple pipeline.process() calls within the same session.
# Requires a running Redis instance at the configured URL.
# ------------------------------------------------------------

def run_section3_redis_session(
    text: str = EXAMPLE_TEXT,
    session_id: str = "user-abc123",
    redis_url: str = "redis://localhost:6379",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a Redis-backed pipeline and return the process() result.

    Args:
        text: Input text to process. Defaults to the shared example text.
        session_id: Namespace for the Redis session identity store.
        redis_url: Connection URL for the Redis instance.
        verbose: When True, pretty-prints the result to stdout.

    Raises:
        Exception: if Redis is unavailable at *redis_url*.

    Returns:
        dict with keys: ``original``, ``anonymized``, ``entities``, ``mapping``.
        Entity IDs are consistent across calls that share the same *session_id*.

    Expected output (abridged — entity IDs consistent with prior calls)::

        {'anonymized': '\\nPERSON_1 from ORG_1 emailed EMAIL_1.\\n...',
         'mapping':    {'PERSON_1': {'text': 'Alice Johnson', ...}, ...},
         ...}

    Note: PERSON_1, ORG_1, etc. are consistent across calls because the
    identity graph is persisted in Redis under *session_id*.
    """
    store = RedisSessionStore(
        session_id=session_id,
        redis_url=redis_url,
        ttl=3600,
    )
    pipeline = Taivium(session_store=store)
    result = pipeline.process(text)
    if verbose:
        pprint(result)
    return result


# ------------------------------------------------------------
# Section 4: Transformer + LLM evidence layers
# Demonstrates enabling the optional BERT NER transformer and/or the
# OpenAI LLM detector on top of the default spaCy + regex detectors.
#
# use_transformer=True  — requires: pip install transformers torch
# use_llm=True          — requires: OPENAI_API_KEY env var to be set
#
# Both flags are False by default; enable only what is available in
# your environment.  The pipeline degrades gracefully when a layer is
# unavailable (missing packages or missing API key).
# ------------------------------------------------------------

EXAMPLE_TEXT_NAMES = (
    "Dr. Emily Clarke joined Horizon AI in Boston. "
    "Reach her at emily.clarke@horizonai.io or call +1-617-555-0199."
)


def run_section4_transformer_and_llm(
    text: str = EXAMPLE_TEXT_NAMES,
    use_transformer: bool = True,
    use_llm: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline with transformer and/or LLM evidence layers enabled.

    Enables the optional BERT NER transformer (``dslim/bert-base-NER``) and/or
    the OpenAI LLM detector on top of the default spaCy + regex detectors.
    Both layers are opt-in and degrade gracefully when unavailable.

    Args:
        text: Input text to process.
        use_transformer: Enable the HuggingFace BERT NER layer.
            Requires ``transformers`` and ``torch`` to be installed.
        use_llm: Enable the OpenAI LLM evidence layer.
            Requires the ``OPENAI_API_KEY`` environment variable to be set.
        verbose: When True, pretty-prints the result and per-entity
            evidence sources to stdout.

    Returns:
        dict with keys: ``original``, ``anonymized``, ``entities``, ``mapping``.

    Expected output (transformer only, abridged)::

        {'anonymized': 'Dr. PERSON_1 joined ORG_1 in LOCATION_1. '
                       'Reach her at EMAIL_1 or call PHONE_1.',
         'entities': [
             {'id': 'PERSON_1', 'label': 'PERSON',
              'evidence_sources': ('spacy', 'transformer'), ...},
             ...
         ], ...}

    Note:
        When both layers are enabled, ``evidence_sources`` on each entity
        lists every detector that agreed on that span, e.g.
        ``('regex', 'spacy', 'transformer')``.  The reported ``confidence``
        is the mean across all contributing detectors.
    """
    import os  # pylint: disable=import-outside-toplevel

    pipeline = Taivium(use_transformer=use_transformer, use_llm=use_llm)
    result = pipeline.process(text)

    if verbose:
        active = []
        if use_transformer:
            active.append("transformer")
        if use_llm:
            if os.getenv("OPENAI_API_KEY"):
                active.append("llm")
            else:
                active.append("llm (skipped — OPENAI_API_KEY not set)")
        print(f"Active extra layers: {active or ['none']}")
        print(f"Anonymized: {result['anonymized']}")
        print("Entities:")
        for entity in result["entities"]:
            sources = entity.get("evidence_sources", ())
            conf = entity.get("confidence", 0.0)
            print(
                f"  [{entity['label']:10}] {entity['text']!r:30} "
                f"sources={sources}  conf={conf:.2f}"
            )

    return result


# ------------------------------------------------------------
# Section 5: Custom detector injection
# Demonstrates replacing or extending the built-in transformer
# and LLM detectors with any callable that accepts a str and
# returns List[Evidence].  Useful for:
#   - swapping in a different NER model (e.g. flair, stanza)
#   - routing LLM calls to a private/on-prem endpoint
#   - writing deterministic detectors for testing
# ------------------------------------------------------------

def _custom_transformer_detector(text: str) -> List[Evidence]:
    """Example custom transformer detector.

    This stub tags every occurrence of a known codename as PERSON.
    In practice, replace this body with any NER model of your choice.
    """
    codenames = ["Agent X", "Agent Y"]
    evidence: List[Evidence] = []
    for name in codenames:
        start = 0
        while True:
            idx = text.find(name, start)
            if idx == -1:
                break
            evidence.append(Evidence(
                start=idx,
                end=idx + len(name),
                label="PERSON",
                source="transformer",
                confidence=0.95,
            ))
            start = idx + len(name)
    return evidence


def _custom_llm_detector(text: str) -> List[Evidence]:
    """Example custom LLM detector.

    This stub returns a hardcoded entity.  In practice, replace this
    body with a call to any LLM or external API, returning
    ``List[Evidence]`` in the same shape.
    """
    org_name = "Horizon AI"
    idx = text.find(org_name)
    if idx == -1:
        return []
    return [Evidence(
        start=idx,
        end=idx + len(org_name),
        label="ORG",
        source="llm",
        confidence=0.92,
    )]


def run_section5_custom_detectors(
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the pipeline with custom transformer and LLM detector callables.

    Demonstrates passing ``transformer_fn`` and ``llm_fn`` to :class:`Taivium`
    to replace the built-in detectors with any callable that accepts a ``str``
    and returns ``List[Evidence]``.  Useful for swapping in different NER models,
    routing LLM calls to a private endpoint, or writing deterministic stubs for
    testing.

    Args:
        verbose: When True, prints entities with evidence sources.

    Returns:
        dict with keys: ``original``, ``anonymized``, ``entities``, ``mapping``.

    Expected output (abridged)::

        [PERSON    ] 'Agent X'     sources=('transformer',)        conf=0.95
        [PERSON    ] 'Agent Y'     sources=('transformer',)        conf=0.95
        [ORG       ] 'Horizon AI'  sources=('llm', 'spacy')        conf=0.84
    """
    text = "Agent X and Agent Y joined Horizon AI for the briefing."

    pipeline = Taivium(
        use_transformer=True,
        transformer_fn=_custom_transformer_detector,
        use_llm=True,
        llm_fn=_custom_llm_detector,
    )
    result = pipeline.process(text)

    if verbose:
        print(f"Anonymized: {result['anonymized']}")
        print("Entities:")
        for entity in result["entities"]:
            sources = entity.get("evidence_sources", ())
            conf = entity.get("confidence", 0.0)
            print(
                f"  [{entity['label']:10}] {entity['text']!r:15}"
                f" sources={sources}  conf={conf:.2f}"
            )

    return result


if __name__ == "__main__":
    # Section 1
    print("=== Section 1: Default pipeline ===")
    run_section1_default_pipeline()

    # Section 2
    print("\n=== Section 2: Custom policy engine ===")
    try:
        run_section2_custom_policy()
    except ValueError as e:
        print(f"Policy violation: {e}")
        # Expected: Policy violation: Blocked sensitive entity: sk-1234567890abcdef (API_KEY)

    # Section 3
    print("\n=== Section 3: Redis-backed session store ===")
    try:
        run_section3_redis_session()
    except Exception as e:  # pylint: disable=broad-except
        print(f"Redis session store unavailable: {e}")
        print("Start Redis with: docker run -p 6379:6379 redis")

    # Section 4
    print("\n=== Section 4: Transformer + LLM evidence layers ===")
    run_section4_transformer_and_llm()

    # Section 5
    print("\n=== Section 5: Custom detector injection ===")
    run_section5_custom_detectors()

