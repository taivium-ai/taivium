"""Tests for the evidence-first Taivium flow."""

import unicodedata

import pytest

from taivium.engine import (
    Evidence,
    Entity,
    IdentityEngine,
    Taivium,
    assert_text_span_integrity,
    canonicalize_spans,
    find_recurrences,
    reverse_transform,
    transform,
    assert_non_overlapping,
)
from taivium.session_store import InMemorySessionStore


def test_transform_raises_on_overlapping_spans() -> None:
    """transform() must fail hard on overlapping resolved spans."""
    overlapping = [
        (Entity("Alice Johnson", "PERSON", 0, 13, "canonical"), "PERSON_aaa"),
        (Entity("Johnson", "PERSON", 6, 13, "canonical"), "PERSON_bbb"),
    ]
    with pytest.raises(ValueError, match="Overlapping spans"):
        transform("Alice Johnson works here.", overlapping)


def test_transform_no_raise_on_adjacent_spans() -> None:
    """Adjacent spans are valid and should be transformed."""
    adjacent = [
        (Entity("Alice", "PERSON", 0, 5, "canonical"), "PERSON_aaa"),
        (Entity("Johnson", "PERSON", 5, 12, "canonical"), "PERSON_bbb"),
    ]
    transformed = transform("AliceJohnson works here.", adjacent)
    assert "PERSON_aaa" in transformed
    assert "PERSON_bbb" in transformed


def test_collect_evidence_calls_all_detectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evidence collector must invoke all detector layers."""
    calls = {"spacy": 0, "regex": 0, "transformer": 0, "llm": 0}

    import taivium.engine as pp  # pylint: disable=import-outside-toplevel

    def _spacy(_: str):
        calls["spacy"] += 1
        return [Evidence(0, 5, "PERSON", "spacy", 0.7)]

    def _regex(_: str):
        calls["regex"] += 1
        return [Evidence(6, 10, "ORG", "regex", 0.99)]

    def _transformer(_: str):
        calls["transformer"] += 1
        return []

    def _llm(_: str):
        calls["llm"] += 1
        return []

    monkeypatch.setattr(pp, "spacy_evidence", _spacy)
    monkeypatch.setattr(pp, "regex_evidence", _regex)
    monkeypatch.setattr(pp, "transformer_evidence", _transformer)
    monkeypatch.setattr(pp, "llm_evidence", _llm)

    evidence = pp.collect_evidence("Alice Acme", use_transformer=True, use_llm=True)

    assert len(evidence) == 2
    assert calls == {"spacy": 1, "regex": 1, "transformer": 1, "llm": 1}


def test_canonicalize_spans_weighted_vote() -> None:
    """For the same span, canonicalization must choose the highest weighted label."""
    text = "Alice"
    evidence = [
        Evidence(0, 5, "PERSON", "spacy", 0.80),
        Evidence(0, 5, "ORG", "regex", 0.95),
    ]
    entities = canonicalize_spans(text, evidence)

    assert len(entities) == 1
    assert entities[0].label == "ORG"
    assert entities[0].source == "canonical"
    assert entities[0].evidence_sources == ("regex",)
    assert entities[0].confidence > 0


def test_canonicalize_spans_explicit_tie_breaker_uses_structural_fields() -> None:
    """Exact score/cov/count ties should resolve deterministically by structural fields."""
    text = "Alice"
    evidence = [
        # Same span, same source, same confidence => same score/coverage/count.
        Evidence(0, 5, "ORG", "spacy", 0.80),
        Evidence(0, 5, "PERSON", "spacy", 0.80),
    ]

    entities = canonicalize_spans(text, evidence)

    assert len(entities) == 1
    # With include_key(..., candidate.label) tie-break, PERSON deterministically wins.
    assert entities[0].label == "PERSON"


def test_canonicalize_spans_groups_by_exact_span() -> None:
    """Identical spans from multiple detectors collapse into one canonical entity."""
    text = "Alice met Bob"
    evidence = [
        Evidence(0, 5, "PERSON", "spacy", 0.7),
        Evidence(0, 5, "PERSON", "regex", 0.9),
        Evidence(10, 13, "PERSON", "spacy", 0.7),
    ]
    entities = canonicalize_spans(text, evidence)

    spans = {(e.start, e.end) for e in entities}
    assert spans == {(0, 5), (10, 13)}
    assert len(entities) == 2


def test_canonicalize_spans_overlap_cluster() -> None:
    """Overlapping non-identical spans may resolve to precise non-overlapping entities.

    spaCy   [0:10] PERSON  — full name "John Smith"
    regex   [0:4]  PERSON  — "John"
    transformer [5:10] PERSON — "Smith"

    With bounded span-length bonus, the optimizer should not force a long
    weak span if stronger precise spans explain the text better.
    """
    text = "John Smith met Alice."
    evidence = [
        Evidence(0, 10, "PERSON", "spacy", 0.75),
        Evidence(0, 4, "PERSON", "regex", 0.99),
        Evidence(5, 10, "PERSON", "transformer", 0.80),
    ]
    entities = canonicalize_spans(text, evidence)

    spans = {(e.start, e.end, e.label) for e in entities}
    assert spans == {(0, 4, "PERSON"), (5, 10, "PERSON")}


def test_canonicalize_spans_longest_span_wins() -> None:
    """Longest span should not win solely due to length when shorter spans are stronger."""
    text = "John Smith"
    evidence = [
        Evidence(0, 4, "PERSON", "regex", 0.99),   # short span
        Evidence(0, 10, "PERSON", "spacy", 0.75),  # longest span
        Evidence(5, 10, "PERSON", "regex", 0.99),  # short span
    ]
    entities = canonicalize_spans(text, evidence)

    spans = {(e.start, e.end) for e in entities}
    assert spans == {(0, 4), (5, 10)}


def test_canonicalize_spans_avoids_transitive_overlap_collapse() -> None:
    """Optimizer should avoid forced single-span collapse across transitive overlaps."""
    text = "New York Times Square"
    evidence = [
        Evidence(0, 8, "LOCATION", "regex", 0.95),
        Evidence(0, 14, "ORG", "spacy", 0.30),
        Evidence(9, 21, "LOCATION", "regex", 0.95),
    ]

    entities = canonicalize_spans(text, evidence)

    spans = {(e.start, e.end, e.label) for e in entities}
    assert spans == {(0, 8, "LOCATION"), (9, 21, "LOCATION")}


def test_canonicalize_spans_does_not_overweight_long_noisy_span() -> None:
    """Long low-confidence spans must not dominate shorter high-confidence spans."""
    text = "John Smith from Acme Corporation"
    evidence = [
        # Precise, high-confidence person span
        Evidence(0, 10, "PERSON", "regex", 0.99),
        # Longer but weaker span that should not win by length alone
        Evidence(0, len(text), "PERSON", "spacy", 0.20),
    ]

    entities = canonicalize_spans(text, evidence)

    assert len(entities) == 1
    assert entities[0].start == 0
    assert entities[0].end == 10
    assert entities[0].text == "John Smith"


def test_identityengine_resolve_preserves_all_positions() -> None:
    """resolve() returns one output per input entity, while IDs remain deterministic."""
    entities = [
        Entity("Alice", "PERSON", 0, 5, "canonical"),
        Entity("Alice", "PERSON", 40, 45, "canonical"),
        Entity("alice@example.com", "EMAIL", 10, 26, "canonical"),
    ]
    engine = IdentityEngine()
    resolved = engine.resolve(entities)

    assert len(resolved) == 3
    alice_ids = [eid for entity, eid in resolved if entity.text == "Alice"]
    assert len(set(alice_ids)) == 1


def test_identityengine_normalizes_case_whitespace_and_punctuation() -> None:
    """Semantic surface variants should hash to the same deterministic ID."""
    engine = IdentityEngine()
    label = "PERSON"
    variants = [
        "John Smith",
        "john smith",
        "John  Smith",
        "JOHN SMITH",
        "John Smith.",
        "  John Smith  ",
    ]

    ids = {engine.generate_id(text, label) for text in variants}
    assert len(ids) == 1


def test_identityengine_normalizes_unicode_forms() -> None:
    """Unicode equivalent forms should produce identical IDs."""
    engine = IdentityEngine()
    label = "PERSON"
    nfc = "José"
    nfd = unicodedata.normalize("NFD", nfc)

    assert engine.generate_id(nfc, label) == engine.generate_id(nfd, label)


def test_pipeline_process_uses_canonical_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end process should consume canonical entities and emit deterministic mapping."""
    import taivium.engine as pp  # pylint: disable=import-outside-toplevel

    def _collect(_: str, **kwargs):
        return [
            Evidence(0, 5, "PERSON", "spacy", 0.8),
            Evidence(0, 5, "PERSON", "regex", 0.99),
        ]

    monkeypatch.setattr(pp, "collect_evidence", _collect)
    pipeline = Taivium()
    output = pipeline.process("Alice")

    assert output["entities"][0]["label"] == "PERSON"
    assert output["entities"][0]["id"].startswith("PERSON_")
    assert output["entities"][0]["source"] == "canonical"
    assert output["entities"][0]["evidence_sources"] == ("regex", "spacy")
    assert output["entities"][0]["confidence"] > 0
    mapping_meta = next(iter(output["mapping"].values()))
    assert mapping_meta["evidence_sources"] == ("regex", "spacy")
    assert mapping_meta["confidence"] > 0
    assert "Alice" not in output["anonymized"]


def test_find_recurrences_inherits_provenance() -> None:
    """Recurrences should preserve canonical provenance lineage and confidence."""
    text = "Alice Johnson met Alice Johnson."
    canonical = [
        Entity(
            "Alice Johnson",
            "PERSON",
            0,
            13,
            "canonical",
            evidence_sources=("regex", "spacy"),
            confidence=0.87,
        )
    ]

    extras = find_recurrences(text, canonical)
    assert len(extras) == 1
    assert extras[0].source == "recurrence"
    assert extras[0].evidence_sources == ("regex", "spacy")
    assert extras[0].confidence == pytest.approx(0.87)


def test_reverse_transform_replaces_tokens() -> None:
    """Tokens in text are replaced with their original values."""
    mapping = {
        "PERSON_abc": {"text": "Alice Johnson"},
        "EMAIL_xyz": {"text": "alice@acme.com"},
    }
    transformed = reverse_transform("Hello PERSON_abc, your email is EMAIL_xyz.", mapping)
    assert transformed == "Hello Alice Johnson, your email is alice@acme.com."


def test_reverse_transform_no_match_returns_unchanged() -> None:
    """Text with no matching tokens is returned unchanged."""
    mapping = {"PERSON_abc": {"text": "Alice"}}
    text = "No entities here."
    assert reverse_transform(text, mapping) == text


def test_reverse_transform_longest_token_first() -> None:
    """Longer IDs should be replaced first to avoid prefix collisions."""
    mapping = {
        "PERSON_ab": {"text": "Bob"},
        "PERSON_abcdef": {"text": "Alice"},
    }
    text = "Hello PERSON_abcdef and PERSON_ab."
    transformed = reverse_transform(text, mapping)
    assert transformed == "Hello Alice and Bob."



# -----------------------------------------------------------------------
# find_recurrences
# -----------------------------------------------------------------------
#
# Recurrence Invariants Validated by These Tests:
# - Only canonical entities are replicated (no invention):
#     test_find_recurrences_catches_second_mention, test_find_recurrences_multiple_occurrences
# - Label, identity, and boundaries are preserved:
#     test_find_recurrences_inherits_provenance, test_find_recurrences_catches_second_mention
# - No invented labels or widened spans:
#     test_find_recurrences_respects_word_boundaries, test_find_recurrences_case_sensitive
# - No overlap with canonical spans:
#     test_find_recurrences_no_duplicates_of_canonical
# - No recursive recurrence chains (recurrences only from canonical):
#     All recurrence tests (recurrences only generated from canonical, never from recurrences)
# - Source is always "recurrence":
#     test_find_recurrences_inherits_provenance, test_find_recurrences_catches_second_mention
# - Recurrence label and text exactly match the canonical entity:
#     test_find_recurrences_inherits_provenance, test_find_recurrences_catches_second_mention
# - Eligibility gate prevents ambiguous lexical cloning:
#     test_find_recurrences_skips_single_token_person, test_find_recurrences_allows_email_recurrence
#

def test_find_recurrences_catches_second_mention() -> None:
    """Second occurrence of a canonical name must be found as a recurrence."""
    text = "Alice Johnson met Alice Johnson."
    canonical = [Entity("Alice Johnson", "PERSON", 0, 13, "canonical")]
    extras = find_recurrences(text, canonical)

    assert len(extras) == 1
    assert extras[0].start == 18
    assert extras[0].end == 31
    assert extras[0].label == "PERSON"
    assert extras[0].source == "recurrence"


def test_find_recurrences_no_duplicates_of_canonical() -> None:
    """Already-canonical spans must not be re-emitted as recurrences."""
    text = "Alice Johnson met Bob Stone."
    canonical = [
        Entity("Alice Johnson", "PERSON", 0, 13, "canonical"),
        Entity("Bob Stone", "PERSON", 18, 27, "canonical"),
    ]
    extras = find_recurrences(text, canonical)
    assert not extras


def test_find_recurrences_respects_word_boundaries() -> None:
    """Substring matches must not fire; only whole-token occurrences count."""
    text = "Alice Johnsonson is not Alice Johnson."
    # "Alice Johnson" standalone is at [24:37]; canonical points there.
    # The recurrence search must NOT re-fire inside "Alice Johnsonson" [0:14].
    canonical = [Entity("Alice Johnson", "PERSON", 24, 37, "canonical")]
    extras = find_recurrences(text, canonical)
    assert not extras


def test_find_recurrences_case_sensitive() -> None:
    """Recurrence search is case-sensitive for eligible multi-token PERSON spans."""
    text = "Alice Johnson said alice johnson was tired."
    canonical = [Entity("Alice Johnson", "PERSON", 0, 13, "canonical")]
    extras = find_recurrences(text, canonical)
    assert not extras


def test_find_recurrences_multiple_occurrences() -> None:
    """Three total mentions, one canonical — two recurrences returned."""
    text = "Bob Stone called Bob Stone, but Bob Stone did not answer."
    #       Bob Stone[0:9]  Bob Stone[17:26]  Bob Stone[32:41]
    canonical = [Entity("Bob Stone", "PERSON", 0, 9, "canonical")]
    extras = find_recurrences(text, canonical)

    assert len(extras) == 2
    starts = {e.start for e in extras}
    assert starts == {17, 32}


def test_pipeline_recurrence_replaces_all_mentions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: both occurrences of an eligible person name are anonymized."""
    import taivium.engine as pp  # pylint: disable=import-outside-toplevel

    def _collect(_: str, **kwargs) -> list:
        # spaCy only detects the first full name
        return [Evidence(0, 13, "PERSON", "spacy", 0.75)]

    monkeypatch.setattr(pp, "collect_evidence", _collect)
    pipeline = Taivium()
    output = pipeline.process("Alice Johnson met Alice Johnson.")

    assert "Alice Johnson" not in output["anonymized"]
    # Both spans should map to the same deterministic ID
    assert output["anonymized"].count("PERSON_") == 2


def test_find_recurrences_skips_single_token_person() -> None:
    """Single-token PERSON entities are ineligible to prevent lexical ambiguity."""
    text = "May met May."
    canonical = [Entity("May", "PERSON", 0, 3, "canonical")]
    extras = find_recurrences(text, canonical)
    assert not extras


def test_find_recurrences_allows_email_recurrence() -> None:
    """EMAIL is recurrence-eligible and should still produce safe recurrences."""
    text = "alice@example.com wrote to alice@example.com."
    canonical = [Entity("alice@example.com", "EMAIL", 0, 17, "canonical")]
    extras = find_recurrences(text, canonical)

    assert len(extras) == 1
    assert extras[0].start == 27
    assert extras[0].end == 44
    assert extras[0].label == "EMAIL"


def test_find_recurrences_manual_boundary_apostrophe_org() -> None:
    """Apostrophe-containing names should match exact token boundaries only."""
    text = "X O'Connor Y O'ConnorZ"
    canonical = [Entity("O'Connor", "ORG", 2, 10, "canonical")]

    extras = find_recurrences(text, canonical)

    assert not extras


def test_find_recurrences_manual_boundary_hyphen_org() -> None:
    """Hyphenated names should not match as a prefix inside longer tokens."""
    text = "A Jean-Luc B Jean-Lucx"
    canonical = [Entity("Jean-Luc", "ORG", 2, 10, "canonical")]

    extras = find_recurrences(text, canonical)

    assert not extras


def test_find_recurrences_manual_boundary_cjk_org() -> None:
    """Unicode CJK word-like tokens should respect manual non-word boundaries."""
    text = "A 東京 B 東京X"
    canonical = [Entity("東京", "ORG", 2, 4, "canonical")]

    extras = find_recurrences(text, canonical)

    assert not extras

def test_assert_non_overlapping_detects_overlap():
    """Hard span-integrity guard: overlapping entities must fail immediately."""
    ents = [
        Entity("Alice", "PERSON", 0, 5, "canonical"),
        Entity("Bob", "PERSON", 4, 8, "canonical"),
    ]
    with pytest.raises(AssertionError, match="Overlapping entities"):
        assert_non_overlapping(ents)

def test_assert_non_overlapping_passes_for_non_overlap():
    """Adjacent, non-overlapping spans are valid and should pass the invariant."""
    ents = [
        Entity("Alice", "PERSON", 0, 5, "canonical"),
        Entity("Bob", "PERSON", 5, 8, "canonical"),
    ]
    assert_non_overlapping(ents)  # Should not raise


def test_assert_non_overlapping_detects_unsorted_entities():
    """Entity lists must be sorted ascending by start offset."""
    ents = [
        Entity("Bob", "PERSON", 5, 8, "canonical"),
        Entity("Alice", "PERSON", 0, 5, "canonical"),
    ]
    with pytest.raises(AssertionError, match="not sorted"):
        assert_non_overlapping(ents)


def test_assert_non_overlapping_detects_invalid_span():
    """Each entity must satisfy start < end."""
    ents = [Entity("Bad", "PERSON", 3, 3, "canonical")]
    with pytest.raises(AssertionError, match="Invalid entity span"):
        assert_non_overlapping(ents)


def test_assert_text_span_integrity_detects_mismatch():
    """Entity text must always match text[start:end]."""
    text = "Alice met Bob."
    ents = [Entity("Alice", "PERSON", 0, 5, "canonical")]
    assert_text_span_integrity(text, ents)

    bad = [Entity("Alicia", "PERSON", 0, 5, "canonical")]
    with pytest.raises(AssertionError, match="Text-span mismatch"):
        assert_text_span_integrity(text, bad)


def test_transform_raises_on_text_span_mismatch() -> None:
    """transform() must reject resolved entities with mismatched text/span."""
    bad = [(Entity("Alicia", "PERSON", 0, 5, "canonical"), "PERSON_aaa")]
    with pytest.raises(ValueError, match="transform"):
        transform("Alice works here.", bad)

def test_canonicalize_idempotent():
    """Canonicalization idempotence: canonicalize(canonicalize(E)) == canonicalize(E)."""
    text = "Alice met Bob."
    evidence = [
        Evidence(0, 5, "PERSON", "spacy", 0.8),
        Evidence(9, 12, "PERSON", "spacy", 0.8),
    ]
    c1 = canonicalize_spans(text, evidence)
    # Feed canonical output back as evidence to prove the truth layer is stable.
    c2 = canonicalize_spans(
        text,
        [Evidence(e.start, e.end, e.label, e.source, e.confidence) for e in c1],
    )
    # Compare structural identity of spans and labels across both passes.
    assert [e.start for e in c1] == [e.start for e in c2]
    assert [e.end for e in c1] == [e.end for e in c2]
    assert [e.label for e in c1] == [e.label for e in c2]

def test_find_recurrences_idempotent():
    """Recurrence expansion idempotence: rerunning with prior recurrences adds nothing new."""
    text = "Alice met Alice."
    canonical = [Entity("Alice", "PERSON", 0, 5, "canonical")]
    recurrences = find_recurrences(text, canonical)
    # If this ever returns extra entities, recurrence is recursively amplifying.
    recurrences2 = find_recurrences(text, canonical + recurrences)
    assert not recurrences2

def test_transform_reverse_transform_roundtrip():
    """Critical roundtrip invariant: reverse_transform(transform(text)) == text."""
    text = "Alice met Bob."
    evidence = [
        Evidence(0, 5, "PERSON", "spacy", 0.8),
        Evidence(9, 12, "PERSON", "spacy", 0.8),
    ]
    canonical = canonicalize_spans(text, evidence)
    resolved = IdentityEngine().resolve(canonical)
    transformed = transform(text, resolved)
    mapping = {eid: {"text": ent.text, "label": ent.label} for ent, eid in resolved}
    roundtrip = reverse_transform(transformed, mapping)
    assert roundtrip == text


def test_process_result_contains_store_type_key() -> None:
    """process() result must include a 'store_type' key."""
    pipeline = Taivium()
    result = pipeline.process("Alice Johnson works here.")
    assert "store_type" in result


def test_process_store_type_default_is_inmemory() -> None:
    """Default pipeline uses InMemorySessionStore, reflected in store_type."""
    pipeline = Taivium()
    result = pipeline.process("Alice Johnson works here.")
    assert result["store_type"] == "InMemorySessionStore"


def test_process_store_type_reflects_custom_store() -> None:
    """When a custom session store is injected, store_type matches its class name."""
    store = InMemorySessionStore()
    pipeline = Taivium(session_store=store)
    result = pipeline.process("Bob Smith is here.")
    assert result["store_type"] == "InMemorySessionStore"
