"""
Tests that validate the examples shown in examples/example_privacy_pipeline.py.

Each test class calls the corresponding example function directly.  Any
breaking change — renamed class, changed constructor signature, altered return
shape, or removed method — is caught at the import or call site rather than
through a separate canary.

Section 3 tests substitute ``InMemorySessionStore`` for ``RedisSessionStore``
so the suite runs without a live Redis instance.
"""
from unittest import mock

import pytest

from example_privacy_pipeline import (  # pylint: disable=import-error
    EXAMPLE_TEXT,
    EXAMPLE_TEXT_NAMES,
    run_section1_default_pipeline,
    run_section2_custom_policy,
    run_section3_redis_session,
    run_section4_transformer_and_llm,
    run_section5_custom_detectors,
)
from tarvium.engine import (
    PolicyAction,
    PolicyEngine,
    PolicyRule,
    Tarvium,
    RiskLevel,
)
from tarvium.session_store import InMemorySessionStore


# ---------------------------------------------------------------------------
# Module-scoped fixture — run the NLP pipeline once for all Section 1 tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def default_result():
    """Result of Section 1 example — computed once per test module."""
    return run_section1_default_pipeline(verbose=False)


# ---------------------------------------------------------------------------
# Section 1: Default pipeline
# ---------------------------------------------------------------------------

class TestSection1DefaultPipeline:
    """Mirrors Section 1 of example_privacy_pipeline.py."""

    # pylint: disable=redefined-outer-name  # standard pytest fixture injection pattern

    def test_result_keys(self, default_result):
        """process() returns the documented top-level keys."""
        assert set(default_result.keys()) == {"original", "anonymized", "entities", "mapping", "store_type"}

    def test_original_preserved(self, default_result):
        """'original' equals the input text verbatim."""
        assert default_result["original"] == EXAMPLE_TEXT

    def test_persons_anonymized(self, default_result):
        """Full-name PERSON entities are replaced with placeholder tokens in 'anonymized'.

        Note: single-word first names (e.g. 'Carol') may not be detected by
        the underlying spaCy model; only multi-token names are asserted here.
        """
        anonymized = default_result["anonymized"]
        assert "Alice Johnson" not in anonymized
        assert "Bob Smith" not in anonymized

    def test_emails_anonymized(self, default_result):
        """EMAIL entities do not appear in the anonymized output."""
        assert "alice@acme.com" not in default_result["anonymized"]
        assert "carol@beta.com" not in default_result["anonymized"]

    def test_api_keys_anonymized(self, default_result):
        """API_KEY entities do not appear in the anonymized output."""
        anonymized = default_result["anonymized"]
        assert "sk-1234567890abcdef" not in anonymized
        assert "ZXCVBNMASDF" not in anonymized
        assert "sk-abcdef1234567890" not in anonymized

    def test_entities_list_non_empty(self, default_result):
        """'entities' is a non-empty list with the required keys per entry."""
        assert len(default_result["entities"]) > 0
        required_keys = {
            "id", "text", "label", "start", "end", "source", "confidence", "evidence_sources"
        }
        for entity in default_result["entities"]:
            assert required_keys.issubset(entity.keys()), f"Entity missing keys: {entity}"

    def test_mapping_structure(self, default_result):
        """'mapping' entries each contain the documented fields."""
        assert len(default_result["mapping"]) > 0
        required_keys = {"text", "label", "action", "risk", "reason", "source", "confidence"}
        for eid, entry in default_result["mapping"].items():
            assert required_keys.issubset(entry.keys()), (
                f"Mapping entry {eid!r} missing keys: {entry}"
            )

    def test_mapping_action_is_anonymize(self, default_result):
        """Default policy sets action=ANONYMIZE for all detected entities."""
        for eid, entry in default_result["mapping"].items():
            assert entry["action"] == PolicyAction.ANONYMIZE, (
                f"Expected ANONYMIZE for {eid!r}, got {entry['action']}"
            )

    def test_deterministic_ids(self):
        """The same text produces the same placeholder IDs on repeated calls."""
        r1 = run_section1_default_pipeline(verbose=False)
        r2 = run_section1_default_pipeline(verbose=False)
        assert r1["anonymized"] == r2["anonymized"]
        assert set(r1["mapping"].keys()) == set(r2["mapping"].keys())


# ---------------------------------------------------------------------------
# Section 2: Custom policy engine
# ---------------------------------------------------------------------------

class TestSection2CustomPolicyEngine:
    """Mirrors Section 2 of example_privacy_pipeline.py."""

    def test_api_key_raises_value_error(self):
        """BLOCK policy on API_KEY raises ValueError when an API key is detected."""
        with pytest.raises(ValueError, match="Blocked sensitive entity"):
            run_section2_custom_policy(verbose=False)

    def test_location_passed_through(self):
        """ALLOW policy on LOCATION leaves location text in the anonymized output.

        Uses the exact same function as the example but with text that contains
        no API keys, so the BLOCK never fires.
        """
        text = "Alice Johnson lives in San Francisco and visited New York."
        result = run_section2_custom_policy(text=text, verbose=False)
        # Locations should appear verbatim (ALLOW); persons fall back to ANONYMIZE
        assert "San Francisco" in result["anonymized"] or "New York" in result["anonymized"]
        assert "Alice Johnson" not in result["anonymized"]

    def test_block_error_message_includes_entity_text(self):
        """The ValueError message identifies the blocked entity."""
        with pytest.raises(ValueError) as exc_info:
            run_section2_custom_policy(verbose=False)
        assert "API_KEY" in str(exc_info.value)

    def test_custom_policy_overrides_default(self):
        """Explicitly provided policy rules override the built-in defaults."""
        custom_policy = {
            "EMAIL": PolicyRule("EMAIL", PolicyAction.ALLOW, RiskLevel.LOW),
        }
        pipeline = Tarvium(policy_engine=PolicyEngine(policy_table=custom_policy))
        text = "Contact alice@example.com for details."
        result = pipeline.process(text)
        assert "alice@example.com" in result["anonymized"]

    def test_policy_table_replaces_default_policy_entirely(self):
        """policy_table= replaces DEFAULT_POLICY in full — labels not in the table
        fall back to default_action (ANONYMIZE by default), not the built-in defaults.
        """
        # Only EMAIL is in the table; PERSON falls back to ANONYMIZE
        custom_policy = {
            "EMAIL": PolicyRule("EMAIL", PolicyAction.ALLOW, RiskLevel.LOW),
        }
        pipeline = Tarvium(policy_engine=PolicyEngine(policy_table=custom_policy))
        text = "Alice Johnson emailed alice@example.com."
        result = pipeline.process(text)
        # EMAIL is explicitly ALLOW — appears verbatim
        assert "alice@example.com" in result["anonymized"]
        # PERSON is not in the custom table — falls back to ANONYMIZE
        assert "Alice Johnson" not in result["anonymized"]


# ---------------------------------------------------------------------------
# Section 3: Redis-backed session store (tested with InMemorySessionStore)
# ---------------------------------------------------------------------------

class TestSection3SessionPersistence:
    """Mirrors Section 3 of example_privacy_pipeline.py.

    Redis is not required in CI.  ``run_section3_redis_session`` is called
    directly with ``RedisSessionStore`` patched to ``InMemorySessionStore``
    so the real function code runs without a live Redis instance.
    Additional persistence tests use InMemorySessionStore directly.
    """

    def test_same_entity_gets_same_id_across_calls(self):
        """The same entity text maps to the same placeholder ID across repeated process() calls."""
        store = InMemorySessionStore()
        pipeline = Tarvium(session_store=store)

        text1 = "Alice Johnson sent an email."
        text2 = "Alice Johnson called back."

        r1 = pipeline.process(text1)
        r2 = pipeline.process(text2)

        # Find the placeholder assigned to "Alice Johnson" in each call
        ids_1 = {v["text"]: k for k, v in r1["mapping"].items()}
        ids_2 = {v["text"]: k for k, v in r2["mapping"].items()}

        assert "Alice Johnson" in ids_1, "Alice not detected in first call"
        assert "Alice Johnson" in ids_2, "Alice not detected in second call"
        assert ids_1["Alice Johnson"] == ids_2["Alice Johnson"], (
            "Same entity received different IDs across calls"
        )

    def test_session_store_accumulates_mappings(self):
        """Entities from multiple process() calls all appear in session_store.get_all()."""
        store = InMemorySessionStore()
        pipeline = Tarvium(session_store=store)

        pipeline.process("Alice Johnson works here.")
        pipeline.process("Contact alice@example.com.")

        all_entries = store.get_all()
        labels = {v["label"] for v in all_entries.values()}
        assert "PERSON" in labels
        assert "EMAIL" in labels

    def test_pipeline_accepts_session_store_kwarg(self):
        """Tarvium accepts a session_store keyword argument without error."""
        store = InMemorySessionStore()
        pipeline = Tarvium(session_store=store)
        result = pipeline.process("Bob Smith is here.")
        assert result["anonymized"] is not None

    def test_run_section3_returns_expected_keys(self):
        """run_section3_redis_session returns a dict with the four documented keys."""
        with mock.patch(
            "example_privacy_pipeline.RedisSessionStore",
            return_value=InMemorySessionStore(),
        ):
            result = run_section3_redis_session(verbose=False)
        assert set(result.keys()) == {"original", "anonymized", "entities", "mapping", "store_type"}

    def test_run_section3_anonymizes_persons(self):
        """run_section3_redis_session anonymizes PERSON entities in the output."""
        with mock.patch(
            "example_privacy_pipeline.RedisSessionStore",
            return_value=InMemorySessionStore(),
        ):
            result = run_section3_redis_session(verbose=False)
        assert "Alice Johnson" not in result["anonymized"]

    def test_run_section3_consistent_ids_across_calls(self):
        """Two calls sharing the same (mocked) store produce the same placeholder IDs."""
        store = InMemorySessionStore()
        with mock.patch(
            "example_privacy_pipeline.RedisSessionStore", return_value=store
        ):
            r1 = run_section3_redis_session(verbose=False)
            r2 = run_section3_redis_session(verbose=False)
        assert r1["anonymized"] == r2["anonymized"]


# ---------------------------------------------------------------------------
# Section 4: Transformer + LLM evidence layers
# ---------------------------------------------------------------------------

class TestSection4TransformerAndLlm:
    """Mirrors Section 4 of example_privacy_pipeline.py.

    Transformer and LLM pipelines are fully mocked so this suite runs without
    ``transformers``/``torch`` installed and without an ``OPENAI_API_KEY``.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mock_transformer(monkeypatch, predictions: list):
        """Patch _get_ner_pipeline to return a mock callable."""
        import tarvium.transformer as tr  # pylint: disable=import-outside-toplevel
        tr._get_ner_pipeline.cache_clear()
        monkeypatch.setattr(tr, "_get_ner_pipeline", lambda: (lambda text, **kw: predictions))

    @staticmethod
    def _mock_llm(monkeypatch, entities: list):
        """Patch openai.OpenAI to return a mock client emitting *entities*."""
        import json  # pylint: disable=import-outside-toplevel

        class _Choice:
            class _Message:
                content = json.dumps(entities)
            message = _Message()

        class _Completion:
            choices = [_Choice()]

        class _Chat:
            class _Completions:
                @staticmethod
                def create(**_kw):
                    return _Completion()
            completions = _Completions()

        class _Client:
            chat = _Chat()

        monkeypatch.setattr("openai.OpenAI", lambda **kw: _Client())

    # ------------------------------------------------------------------
    # Return-shape tests (no extra layers)
    # ------------------------------------------------------------------

    def test_returns_expected_keys(self):
        """run_section4_transformer_and_llm returns the standard result dict."""
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=False, verbose=False
        )
        assert set(result.keys()) == {"original", "anonymized", "entities", "mapping", "store_type"}

    def test_persons_anonymized_without_extra_layers(self):
        """spaCy alone anonymizes the PERSON entity in EXAMPLE_TEXT_NAMES."""
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=False, verbose=False
        )
        assert "Emily Clarke" not in result["anonymized"]

    def test_emails_anonymized_without_extra_layers(self):
        """Regex alone catches the email in EXAMPLE_TEXT_NAMES."""
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=False, verbose=False
        )
        assert "emily.clarke@horizonai.io" not in result["anonymized"]

    # ------------------------------------------------------------------
    # Transformer layer
    # ------------------------------------------------------------------

    def test_transformer_evidence_adds_source(self, monkeypatch):
        """With use_transformer=True the transformer source appears in evidence_sources."""
        self._mock_transformer(monkeypatch, [
            {"entity_group": "PER", "score": 0.99, "start": 4, "end": 16, "word": "Emily Clarke"},
        ])
        result = run_section4_transformer_and_llm(
            use_transformer=True, use_llm=False, verbose=False
        )
        person_entities = [
            e for e in result["entities"] if e["label"] == "PERSON"
        ]
        assert any("transformer" in e["evidence_sources"] for e in person_entities)

    def test_transformer_blends_confidence(self, monkeypatch):
        """Confidence is the mean of spaCy (0.75) and transformer scores."""
        self._mock_transformer(monkeypatch, [
            {"entity_group": "PER", "score": 1.0, "start": 4, "end": 16, "word": "Emily Clarke"},
        ])
        result = run_section4_transformer_and_llm(
            use_transformer=True, use_llm=False, verbose=False
        )
        person = next(
            (e for e in result["entities"]
             if e["label"] == "PERSON" and "transformer" in e["evidence_sources"]),
            None,
        )
        assert person is not None
        # mean(0.75, 1.0) = 0.875
        assert abs(person["confidence"] - 0.875) < 0.01

    def test_transformer_disabled_has_no_transformer_source(self, monkeypatch):
        """With use_transformer=False, no entity has transformer in evidence_sources."""
        self._mock_transformer(monkeypatch, [
            {"entity_group": "PER", "score": 0.99, "start": 4, "end": 16, "word": "Emily Clarke"},
        ])
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=False, verbose=False
        )
        for entity in result["entities"]:
            assert "transformer" not in entity["evidence_sources"]

    # ------------------------------------------------------------------
    # LLM layer
    # ------------------------------------------------------------------

    def test_llm_evidence_adds_source(self, monkeypatch):
        """With use_llm=True the llm source appears in evidence_sources."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        self._mock_llm(monkeypatch, [
            {"text": "Emily Clarke", "type": "PERSON"},
        ])
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=True, verbose=False
        )
        person_entities = [e for e in result["entities"] if e["label"] == "PERSON"]
        assert any("llm" in e["evidence_sources"] for e in person_entities)

    def test_llm_skipped_without_api_key(self, monkeypatch):
        """LLM layer is silently skipped when OPENAI_API_KEY is unset."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = run_section4_transformer_and_llm(
            use_transformer=False, use_llm=True, verbose=False
        )
        for entity in result["entities"]:
            assert "llm" not in entity["evidence_sources"]


# ---------------------------------------------------------------------------
# Section 5: Custom detector injection
# ---------------------------------------------------------------------------

class TestSection5CustomDetectors:
    """Mirrors Section 5 of example_privacy_pipeline.py."""

    def test_returns_expected_keys(self):
        """run_section5_custom_detectors returns the standard result dict."""
        from example_privacy_pipeline import run_section5_custom_detectors  # pylint: disable=import-error,import-outside-toplevel
        result = run_section5_custom_detectors(verbose=False)
        assert set(result.keys()) == {"original", "anonymized", "entities", "mapping", "store_type"}

    def test_custom_transformer_fn_called(self):
        """transformer_fn callable is invoked and its evidence appears in results."""
        calls = []

        def _det(text: str):
            calls.append(text)
            from tarvium.engine import Evidence  # pylint: disable=import-outside-toplevel
            return [Evidence(start=0, end=7, label="PERSON", source="transformer", confidence=0.99)]

        pipeline = Tarvium(use_transformer=True, transformer_fn=_det)
        result = pipeline.process("Agent X joined Horizon AI.")
        assert len(calls) == 1
        person = next((e for e in result["entities"] if e["label"] == "PERSON"), None)
        assert person is not None
        assert "transformer" in person["evidence_sources"]

    def test_custom_llm_fn_called(self):
        """llm_fn callable is invoked and its evidence appears in results."""
        calls = []

        def _det(text: str):
            calls.append(text)
            from tarvium.engine import Evidence  # pylint: disable=import-outside-toplevel
            idx = text.find("Horizon AI")
            if idx == -1:
                return []
            return [Evidence(start=idx, end=idx + 10, label="ORG", source="llm", confidence=0.92)]

        pipeline = Tarvium(use_llm=True, llm_fn=_det)
        result = pipeline.process("Agent X joined Horizon AI.")
        assert len(calls) == 1
        org = next((e for e in result["entities"] if "llm" in e["evidence_sources"]), None)
        assert org is not None

    def test_transformer_fn_overrides_builtin(self):
        """Providing transformer_fn does not call the built-in transformer_evidence."""
        import tarvium.transformer as tr  # pylint: disable=import-outside-toplevel
        tr._get_ner_pipeline.cache_clear()

        builtin_calls = []
        original = tr.transformer_evidence

        def _spy(text):
            builtin_calls.append(text)
            return original(text)

        custom_calls = []

        def _custom(text):
            custom_calls.append(text)
            return []

        # Temporarily patch the built-in to spy on it
        tr.transformer_evidence = _spy
        try:
            pipeline = Tarvium(use_transformer=True, transformer_fn=_custom)
            pipeline.process("Alice works at Acme.")
        finally:
            tr.transformer_evidence = original

        assert len(custom_calls) == 1, "Custom fn should be called"
        assert len(builtin_calls) == 0, "Built-in should NOT be called when transformer_fn is set"

    def test_llm_fn_overrides_builtin(self, monkeypatch):
        """Providing llm_fn does not call the built-in llm_evidence."""
        import tarvium.llm as llm_mod  # pylint: disable=import-outside-toplevel

        builtin_calls = []
        monkeypatch.setattr(
            llm_mod, "llm_evidence",
            lambda text: builtin_calls.append(text) or [],
        )

        custom_calls = []
        pipeline = Tarvium(use_llm=True, llm_fn=lambda text: custom_calls.append(text) or [])
        pipeline.process("Alice works at Acme.")

        assert len(custom_calls) == 1
        assert len(builtin_calls) == 0

    def test_transformer_fn_without_use_transformer_flag(self):
        """transformer_fn is NOT called when use_transformer=False (default)."""
        called = []
        pipeline = Tarvium(transformer_fn=lambda text: called.append(text) or [])
        pipeline.process("Test text.")
        assert len(called) == 0, "fn should be ignored when use_transformer=False"

    def test_llm_fn_without_use_llm_flag(self):
        """llm_fn is NOT called when use_llm=False (default)."""
        called = []
        pipeline = Tarvium(llm_fn=lambda text: called.append(text) or [])
        pipeline.process("Test text.")
        assert len(called) == 0, "fn should be ignored when use_llm=False"

    def test_custom_fn_confidence_blended(self):
        """Confidence is the mean of spaCy and custom detector scores."""
        from tarvium.engine import Evidence  # pylint: disable=import-outside-toplevel

        # spaCy will detect 'Alice Johnson' at 0.75; custom adds 1.0 on same span.
        def _det(text: str):
            idx = text.find("Alice Johnson")
            if idx == -1:
                return []
            return [Evidence(start=idx, end=idx + 13, label="PERSON",
                             source="transformer", confidence=1.0)]

        pipeline = Tarvium(use_transformer=True, transformer_fn=_det)
        result = pipeline.process("Alice Johnson works here.")
        person = next(
            (e for e in result["entities"]
             if e["label"] == "PERSON" and "transformer" in e["evidence_sources"]),
            None,
        )
        assert person is not None
        # mean(0.75, 1.0) = 0.875
        assert abs(person["confidence"] - 0.875) < 0.01
