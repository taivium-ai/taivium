import taivium.engine as eng


def _spans_for_label(evidence, label):
    return [(item.start, item.end, item.label) for item in evidence if item.label == label]


def test_regex_evidence_detects_ip_date_and_socialnumber():
    text = "Server 192.168.1.5 logged in on 2026-06-04 with SSN 123-45-6789"

    evidence = eng.regex_evidence(text)

    ip_spans = _spans_for_label(evidence, "IP")
    date_spans = _spans_for_label(evidence, "DATE")
    ssn_spans = _spans_for_label(evidence, "SOCIALNUMBER")

    assert ip_spans, "Expected IP evidence"
    assert date_spans, "Expected DATE evidence"
    assert ssn_spans, "Expected SOCIALNUMBER evidence"


def test_regex_evidence_does_not_match_invalid_ip_octets():
    text = "Invalid ip 999.10.10.10 should not be matched"

    evidence = eng.regex_evidence(text)
    ip_spans = _spans_for_label(evidence, "IP")

    assert ip_spans == []


def test_regex_evidence_supports_multiple_date_formats():
    text = "Dates: 2026-06-04 and 06/04/2026 are both valid"

    evidence = eng.regex_evidence(text)
    date_texts = [text[item.start:item.end] for item in evidence if item.label == "DATE"]

    assert "2026-06-04" in date_texts
    assert "06/04/2026" in date_texts


def test_regex_evidence_requires_dashed_socialnumber_format():
    text = "Dashed SSN 123-45-6789 and plain 123456789"

    evidence = eng.regex_evidence(text)
    ssn_texts = [text[item.start:item.end] for item in evidence if item.label == "SOCIALNUMBER"]

    assert "123-45-6789" in ssn_texts
    assert "123456789" not in ssn_texts
