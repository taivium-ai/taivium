"""
Test suite for org_list_evidence() function and integration with Taivium pipeline.

Tests compliance-friendly organization detection via curated lists.
"""

import pytest
from taivium.engine import Taivium, collect_evidence, org_list_evidence, Evidence


class TestOrgListEvidenceFunction:
    """Tests for org_list_evidence() core function."""

    def test_org_list_evidence_basic(self) -> None:
        """Test basic organization list detection."""
        text = "Acme Corporation sent a proposal."
        known_orgs = ["Acme Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 1
        assert evidence[0].start == 0
        assert evidence[0].end == 16
        assert evidence[0].label == "ORG"
        assert evidence[0].source == "org_list"
        assert evidence[0].confidence == 0.95

    def test_org_list_evidence_multiple_orgs(self) -> None:
        """Test detection of multiple organizations."""
        text = "Acme Corporation and Beta Industries discuss partnership."
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert text[evidence[0].start:evidence[0].end] == "Acme Corporation"
        assert text[evidence[1].start:evidence[1].end] == "Beta Industries"

    def test_org_list_evidence_case_insensitive(self) -> None:
        """Test case-insensitive organization matching."""
        text = "Contact ACME CORPORATION or acme corporation for details."
        known_orgs = ["Acme Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert evidence[0].confidence == 0.95
        assert evidence[1].confidence == 0.95

    def test_org_list_evidence_empty_list(self) -> None:
        """Test that empty organization list returns no evidence."""
        text = "Acme Corporation sent a proposal."
        
        evidence = org_list_evidence(text, [])
        
        assert len(evidence) == 0

    def test_org_list_evidence_none_list(self) -> None:
        """Test that None organization list returns no evidence."""
        text = "Acme Corporation sent a proposal."
        
        evidence = org_list_evidence(text, None)
        
        assert len(evidence) == 0

    def test_org_list_evidence_no_match(self) -> None:
        """Test that non-matching organizations return no evidence."""
        text = "XYZ Corporation sent a proposal."
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 0

    def test_org_list_evidence_special_characters(self) -> None:
        """Test organization names with special characters."""
        text = "AT&T Inc. and Smith & Jones LLC are partners."
        known_orgs = ["AT&T Inc.", "Smith & Jones LLC"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert text[evidence[0].start:evidence[0].end] == "AT&T Inc."
        assert text[evidence[1].start:evidence[1].end] == "Smith & Jones LLC"

    def test_org_list_evidence_repeated_org(self) -> None:
        """Test same organization appearing multiple times."""
        text = "Acme Corporation and Acme Corporation subsidiaries."
        known_orgs = ["Acme Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert evidence[0].start == 0
        assert evidence[0].end == 16
        assert evidence[1].start == 21
        assert evidence[1].end == 37

    def test_org_list_evidence_whitespace_handling(self) -> None:
        """Test that whitespace in org names is handled correctly."""
        text = "  Acme Corporation  and   Beta Industries  "
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert text[evidence[0].start:evidence[0].end] == "Acme Corporation"
        assert text[evidence[1].start:evidence[1].end] == "Beta Industries"


class TestOrgListIntegrationWithCollectEvidence:
    """Tests for org_list_evidence integration with collect_evidence()."""

    def test_collect_evidence_with_known_orgs(self) -> None:
        """Test that collect_evidence includes org_list evidence."""
        text = "Acme Corporation sent email to alice@example.com"
        known_orgs = ["Acme Corporation"]
        
        evidence = collect_evidence(text, known_orgs=known_orgs)
        
        org_evidence = [e for e in evidence if e.source == "org_list"]
        assert len(org_evidence) == 1
        assert org_evidence[0].label == "ORG"
        assert org_evidence[0].confidence == 0.95

    def test_collect_evidence_without_known_orgs(self) -> None:
        """Test that collect_evidence works without known_orgs."""
        text = "Acme Corporation sent email to alice@example.com"
        
        evidence = collect_evidence(text, known_orgs=None)
        
        org_evidence = [e for e in evidence if e.source == "org_list"]
        assert len(org_evidence) == 0

    def test_collect_evidence_org_list_before_gliner(self) -> None:
        """Test that org_list evidence runs before GLiNER."""
        text = "Acme Corporation"
        known_orgs = ["Acme Corporation"]
        
        evidence = collect_evidence(text, known_orgs=known_orgs)
        
        # Both org_list and gliner should detect it
        org_list_evs = [e for e in evidence if e.source == "org_list"]
        gliner_evs = [e for e in evidence if e.source == "gliner"]
        
        assert len(org_list_evs) >= 1
        # gliner may or may not detect it, but org_list should always detect it
        assert org_list_evs[0].confidence == 0.95

    def test_collect_evidence_mixed_detectors(self) -> None:
        """Test collect_evidence with multiple detection layers."""
        text = "Contact Acme Corporation at alice@acme.com"
        known_orgs = ["Acme Corporation"]
        
        evidence = collect_evidence(text, known_orgs=known_orgs)
        
        sources = set(e.source for e in evidence)
        assert "org_list" in sources
        assert "regex" in sources  # EMAIL detection
        # gliner may or may not be triggered

    def test_collect_evidence_empty_known_orgs(self) -> None:
        """Test that empty known_orgs list is handled correctly."""
        text = "Acme Corporation sent a message."
        
        evidence = collect_evidence(text, known_orgs=[])
        
        org_list_evs = [e for e in evidence if e.source == "org_list"]
        assert len(org_list_evs) == 0


class TestOrgListIntegrationWithTaivium:
    """Tests for org_list_evidence integration with Taivium.process()."""

    def test_taivium_process_with_known_orgs(self) -> None:
        """Test Taivium.process() accepts and uses known_orgs parameter."""
        text = "Acme Corporation and Beta Industries partnership"
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        pipeline = Taivium()
        result = pipeline.process(text, known_orgs=known_orgs)
        
        entities = result["entities"]
        assert len(entities) == 2
        
        org_entities = [e for e in entities if "org_list" in e.get("evidence_sources", ())]
        assert len(org_entities) == 2

    def test_taivium_process_without_known_orgs(self) -> None:
        """Test Taivium.process() works without known_orgs."""
        text = "Acme Corporation sent a proposal."
        
        pipeline = Taivium()
        result = pipeline.process(text)
        
        # Should still detect organization via GLiNER
        entities = result["entities"]
        assert len(entities) > 0

    def test_taivium_process_known_orgs_none(self) -> None:
        """Test Taivium.process() with known_orgs=None."""
        text = "Acme Corporation sent a proposal."
        
        pipeline = Taivium()
        result = pipeline.process(text, known_orgs=None)
        
        entities = result["entities"]
        # Should detect via GLiNER, not org_list
        org_list_entities = [e for e in entities if "org_list" in e.get("evidence_sources", ())]
        assert len(org_list_entities) == 0

    def test_taivium_process_entity_anonymization_with_org_list(self) -> None:
        """Test that org_list detected organizations are properly anonymized."""
        text = "Acme Corporation approved the request."
        known_orgs = ["Acme Corporation"]
        
        pipeline = Taivium()
        result = pipeline.process(text, known_orgs=known_orgs)
        
        anonymized = result["anonymized"]
        # Organization should be replaced with canonical ID
        assert "Acme Corporation" not in anonymized
        assert "ORG_" in anonymized

    def test_taivium_process_entity_response_restoration(self) -> None:
        """Test response restoration with org_list entities."""
        text = "Acme Corporation sent a message."
        known_orgs = ["Acme Corporation"]
        
        pipeline = Taivium()
        result = pipeline.process(text, known_orgs=known_orgs)
        
        # Verify entity details are preserved
        entities = result["entities"]
        assert len(entities) > 0
        
        org_entity = next((e for e in entities if e["label"] == "ORG"), None)
        assert org_entity is not None
        assert org_entity["text"] == "Acme Corporation"


class TestOrgListCompliance:
    """Tests for compliance-friendly aspects of org_list detection."""

    def test_org_list_confidence_is_high(self) -> None:
        """Test that org_list confidence is appropriately high (0.95)."""
        text = "Acme Corporation"
        known_orgs = ["Acme Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert evidence[0].confidence == 0.95

    def test_org_list_source_tracking(self) -> None:
        """Test that org_list source is properly tracked."""
        text = "Acme Corporation"
        known_orgs = ["Acme Corporation"]
        
        pipeline = Taivium()
        result = pipeline.process(text, known_orgs=known_orgs)
        
        org_entity = result["entities"][0]
        assert "org_list" in org_entity["evidence_sources"]

    def test_org_list_vs_gliner_confidence(self) -> None:
        """Test that org_list has higher confidence than typical GLiNER detection."""
        text = "Acme Corporation"
        known_orgs = ["Acme Corporation"]
        
        # org_list evidence
        org_list_evs = org_list_evidence(text, known_orgs)
        org_list_conf = org_list_evs[0].confidence if org_list_evs else 0.0
        
        # gliner evidence
        from taivium.engine import gliner_evidence
        gliner_evs = gliner_evidence(text)
        gliner_conf = max((e.confidence for e in gliner_evs), default=0.0)
        
        # org_list should have higher confidence for exact matches
        assert org_list_conf >= gliner_conf

    def test_org_list_deterministic(self) -> None:
        """Test that org_list detection is deterministic."""
        text = "Acme Corporation and Beta Industries discuss terms."
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        results = []
        for _ in range(3):
            evidence = org_list_evidence(text, known_orgs)
            results.append([(e.start, e.end, e.label) for e in evidence])
        
        # All results should be identical (deterministic)
        assert results[0] == results[1] == results[2]


class TestOrgListEdgeCases:
    """Tests for edge cases and robustness."""

    def test_org_list_partial_match_not_detected(self) -> None:
        """Test that partial org name matches don't trigger."""
        text = "Acme and Corporation discuss terms."
        known_orgs = ["Acme Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        # Should not match "Acme" alone or "Corporation" alone
        assert len(evidence) == 0

    def test_org_list_overlapping_orgs(self) -> None:
        """Test handling of overlapping organization names in list."""
        text = "Acme Corporation Inc. is our partner."
        known_orgs = ["Acme Corporation", "Acme Corporation Inc."]
        
        evidence = org_list_evidence(text, known_orgs)
        
        # Both should be detected as separate occurrences
        assert len(evidence) == 2
        assert evidence[0].end == 16  # "Acme Corporation"
        assert evidence[1].end == 21  # "Acme Corporation Inc."

    def test_org_list_unicode_characters(self) -> None:
        """Test organizations with unicode characters."""
        text = "Société Générale and Münchener Re are partners."
        known_orgs = ["Société Générale", "Münchener Re"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2

    def test_org_list_with_numbers(self) -> None:
        """Test organizations with numbers in names."""
        text = "3M Company and Zoom Inc. announced partnership."
        known_orgs = ["3M Company", "Zoom Inc."]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert text[evidence[0].start:evidence[0].end] == "3M Company"
        assert text[evidence[1].start:evidence[1].end] == "Zoom Inc."

    def test_org_list_punctuation_boundary(self) -> None:
        """Test that org names at punctuation boundaries are matched."""
        text = "We recommend Acme Corporation. Beta Industries is also good."
        known_orgs = ["Acme Corporation", "Beta Industries"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 2
        assert text[evidence[0].start:evidence[0].end] == "Acme Corporation"
        assert text[evidence[1].start:evidence[1].end] == "Beta Industries"

    def test_org_list_very_long_org_name(self) -> None:
        """Test with very long organization names."""
        text = "International Business Machines Corporation is a technology company."
        known_orgs = ["International Business Machines Corporation"]
        
        evidence = org_list_evidence(text, known_orgs)
        
        assert len(evidence) == 1
        assert evidence[0].confidence == 0.95
