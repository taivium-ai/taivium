import pytest
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
    text = "Dashed SSN 123-45-6789 and plain 123456789 and too short 12345"

    evidence = eng.regex_evidence(text)
    ssn_texts = [text[item.start:item.end] for item in evidence if item.label == "SOCIALNUMBER"]

    assert "123-45-6789" in ssn_texts
    assert "123456789" in ssn_texts       # 9-digit national IDs are now supported
    assert "12345" not in ssn_texts       # too short (5 digits) must not match


# ---------------------------------------------------------------------------
# Real error cases from ai4privacy/pii-masking-300k evaluation
# ---------------------------------------------------------------------------

def test_iso8601_datetime_detected_as_date():
    # Error case: "1963-12-23T00:00:00" was missed because DATE_REGEX excluded T-prefixed times
    text = "DateOfBirth: 1963-12-23T00:00:00 and enrollment: 2069-10-02T00:00:00"
    evidence = eng.regex_evidence(text)
    date_texts = [text[e.start:e.end] for e in evidence if e.label == "DATE"]
    assert any("1963-12-23" in d for d in date_texts), \
        "ISO 8601 datetime with T should be detected as DATE"
    assert any("2069-10-02" in d for d in date_texts), \
        "ISO 8601 datetime with T should be detected as DATE"


def test_username_with_digits_not_misclassified_as_socialnumber():
    # Error case: "amardi1962", "wsfdkmi9214" were detected as SOCIALNUMBER instead of USERNAME
    text = "Username: amardi1962 logged in. Also wsfdkmi9214 was active."
    evidence = eng.regex_evidence(text)
    socialnumber_texts = [text[e.start:e.end] for e in evidence if e.label == "SOCIALNUMBER"]
    # Pure digit-suffix usernames should not match SOCIALNUMBER as a full token
    assert "amardi1962" not in socialnumber_texts, \
        "Username-like strings with letters+digits suffix should not be SOCIALNUMBER"
    assert "wsfdkmi9214" not in socialnumber_texts, \
        "Username-like strings with letters+digits suffix should not be SOCIALNUMBER"


def test_username_dotted_with_year_not_misclassified_as_socialnumber():
    # Error case: "maria-rosaria.amardi1962" triggered SOCIALNUMBER on 'amardi1962' suffix
    text = 'Username: "maria-rosaria.amardi1962" joined the session.'
    evidence = eng.regex_evidence(text)
    socialnumber_texts = [text[e.start:e.end] for e in evidence if e.label == "SOCIALNUMBER"]
    assert "amardi1962" not in socialnumber_texts, \
        "Year-suffixed portion of a dotted username should not be SOCIALNUMBER"


@pytest.mark.xfail(reason="BIC/SWIFT codes require context-aware detection; regex alone cannot distinguish from IDs")
def test_bic_code_not_detected_as_socialnumber():
    # Error case: BIC/SWIFT code "YXNNUS94ROP" was incorrectly flagged as SOCIALNUMBER
    text = 'Payment details: "bic": "YXNNUS94ROP", "amount": "500"'
    evidence = eng.regex_evidence(text)
    socialnumber_texts = [text[e.start:e.end] for e in evidence if e.label == "SOCIALNUMBER"]
    assert "YXNNUS94ROP" not in socialnumber_texts, \
        "BIC/SWIFT codes should not be detected as SOCIALNUMBER"


def test_placeholder_names_not_in_regex_evidence():
    # Error case: "[Your Name]" and "[Your Position]" were falsely detected as PERSON
    # regex_evidence doesn't detect PERSON, but verify no leakage through other labels
    text = "Please contact [Your Name] at [Your Position] for more details."
    evidence = eng.regex_evidence(text)
    labels = [e.label for e in evidence]
    assert "PERSON" not in labels, \
        "regex_evidence should never emit PERSON labels"


def test_ipv6_full_form_detected():
    # Error case: full IPv6 "afc1:8e6:f5f5:b124:f06d:c094:977c:652f" was missed
    text = "IP Address: afc1:008e:f5f5:b124:f06d:c094:977c:652f connected."
    evidence = eng.regex_evidence(text)
    ip_texts = [text[e.start:e.end] for e in evidence if e.label == "IP"]
    assert ip_texts, "Full 8-group IPv6 addresses should be detected"


def test_email_in_xml_tags_detected():
    # Error case: "<Email>michel-pierre.perucchi@hotmail.com</Email>" — email not detected
    text = "<Email>michel-pierre.perucchi@hotmail.com</Email>"
    evidence = eng.regex_evidence(text)
    email_texts = [text[e.start:e.end] for e in evidence if e.label == "EMAIL"]
    assert "michel-pierre.perucchi@hotmail.com" in email_texts, \
        "Email inside XML tags should be detected"


def test_email_with_apostrophe_in_domain_detected():
    # Error case: "myrto.m'rabat@tutanota.com" — apostrophe in local part not handled
    text = "Contact: myrto.mrabat@tutanota.com for details."
    evidence = eng.regex_evidence(text)
    email_texts = [text[e.start:e.end] for e in evidence if e.label == "EMAIL"]
    assert email_texts, "Email addresses should be detected"


# ---------------------------------------------------------------------------
# Structured LOCATION field detection (JSON / markdown / XML)
# ---------------------------------------------------------------------------

def test_structured_location_json_fields_detected():
    # Error case: "city": "Doncaster", "state": "ENG", "building": "617" all missed
    text = '{\n\t"country": "United Kingdom",\n\t"city": "Doncaster",\n\t"state": "ENG",\n\t"postcode": "DN3 3EH"\n}'
    evidence = eng.regex_evidence(text)
    location_texts = [text[e.start:e.end] for e in evidence if e.label == "LOCATION"]
    assert "United Kingdom" in location_texts, "JSON 'country' value should be LOCATION"
    assert "Doncaster" in location_texts, "JSON 'city' value should be LOCATION"
    assert "ENG" in location_texts, "JSON 'state' value should be LOCATION"
    assert "DN3 3EH" in location_texts, "JSON 'postcode' value should be LOCATION"


def test_structured_location_markdown_fields_detected():
    # Error case: "- Building: 251", "- Street: Quantock Road", "- City: Taunton" missed
    text = "- Building: 251\n- Street: Quantock Road\n- City: Taunton\n- State: ENG\n- Postcode: TA2 7NJ"
    evidence = eng.regex_evidence(text)
    location_texts = [text[e.start:e.end] for e in evidence if e.label == "LOCATION"]
    assert "Quantock Road" in location_texts, "Markdown '- Street:' value should be LOCATION"
    assert "Taunton" in location_texts, "Markdown '- City:' value should be LOCATION"
    assert "ENG" in location_texts, "Markdown '- State:' value should be LOCATION"
    assert "TA2 7NJ" in location_texts, "Markdown '- Postcode:' value should be LOCATION"


def test_structured_location_xml_fields_detected():
    # Error case: "<building>503</building>" was missed
    text = "<building>503</building><street>Mill Lane</street><city>Cheltenham</city>"
    evidence = eng.regex_evidence(text)
    location_texts = [text[e.start:e.end] for e in evidence if e.label == "LOCATION"]
    assert "503" in location_texts, "XML <building> value should be LOCATION"
    assert "Mill Lane" in location_texts, "XML <street> value should be LOCATION"
    assert "Cheltenham" in location_texts, "XML <city> value should be LOCATION"


def test_structured_location_uppercase_json_keys_detected():
    # Error case: "BUILDING": "427", "STREET": "Boltslaw Incline", "CITY": "Consett" missed
    text = '"BUILDING": "427", "STREET": "Boltslaw Incline", "CITY": "Consett", "POSTCODE": "DH8"'
    evidence = eng.regex_evidence(text)
    location_texts = [text[e.start:e.end] for e in evidence if e.label == "LOCATION"]
    assert "427" in location_texts, "Uppercase JSON 'BUILDING' value should be LOCATION"
    assert "Boltslaw Incline" in location_texts, "Uppercase JSON 'STREET' value should be LOCATION"
    assert "DH8" in location_texts, "Uppercase JSON 'POSTCODE' value should be LOCATION"


# ---------------------------------------------------------------------------
# Placeholder PERSON filter
# ---------------------------------------------------------------------------

def test_placeholder_bracket_not_detected_as_person():
    # Error case: "[Your Name]" and "[Your Position]" falsely detected as PERSON
    from taivium.engine import _is_placeholder, Entity
    placeholders = [
        # Square bracket variants
        "[Your Name]",
        "[Your Position]",
        "[Name]",
        "[Position]",
        "[Date]",
        "[Your Title]",
        "[First Name]",
        "[Last Name]",
        "[Full Name]",
        "[Contact Name]",
        "[Manager Name]",
        # Angle bracket variants
        "<Name>",
        "<Your Name>",
        "<Position>",
        "<Company Name>",
        # Curly bracket variants
        "{Name}",
        "{Your Name}",
        "{Position}",
        "{Full Name}",
        # Lowercase variants
        "[your name]",
        "[your position]",
        "[name]",
        "[full name]",
        "<name>",
        "{name}",
        # Mixed case variants
        "[Your name]",
        "[your Name]",
        "[FULL NAME]",
        "[YOUR POSITION]",
        # Extra internal spacing
        "[ Your Name ]",
        "[  Name  ]",
        "{ Full Name }",
        "< Your Name >",
    ]
    for text in placeholders:
        e = Entity(text=text, label="PERSON", start=0, end=len(text), source="spacy")
        assert _is_placeholder(e), f"{text!r} should be identified as a placeholder"


def test_real_names_not_filtered_as_placeholder():
    from taivium.engine import _is_placeholder, Entity
    real_names = [
        "Alice Smith",
        "John Doe",
        "Allissia Bufler",
        "Ranj Ghebrit Woroniecki",   # multi-word real name
        "Kaïs",                       # accented name
        "Lelo Alsény Kuenzi",         # three-part name with accent
    ]
    for name in real_names:
        e = Entity(text=name, label="PERSON", start=0, end=len(name), source="spacy")
        assert not _is_placeholder(e), f"{name!r} should NOT be filtered as placeholder"


def test_lastname_marker_filtered_as_placeholder():
    # Error case: "LASTNAME1_A" style dataset markers should be filtered
    from taivium.engine import _is_placeholder, Entity
    markers = ["LASTNAME1_A", "LASTNAME2_B", "LASTNAME3_C"]
    for marker in markers:
        e = Entity(text=marker, label="PERSON", start=0, end=len(marker), source="spacy")
        assert _is_placeholder(e), f"Dataset marker {marker!r} should be filtered"
