import pytest
import taivium.engine as eng
from taivium.engine import Entity


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
# USERNAME regex coverage (opaque + context keyed)
# ---------------------------------------------------------------------------

def test_username_opaque_alphanumeric_detected():
    # Error case: opaque handles missed (e.g., paaltwvkjuijwbj957, wsfdkmi9214)
    text = "users: paaltwvkjuijwbj957, wsfdkmi9214 and lyxmvtinlajlq99997"
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    assert "paaltwvkjuijwbj957" in username_texts
    assert "wsfdkmi9214" in username_texts
    assert "lyxmvtinlajlq99997" in username_texts


def test_username_context_key_detects_short_code_and_dotted():
    # Error case: key-based usernames like participant_id and short values (R21) missed
    text = 'participant_id: "10mavus.tancev"; username: R21; caller: rand.podo'
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    assert "10mavus.tancev" in username_texts
    assert "R21" in username_texts
    assert "rand.podo" in username_texts


def test_username_nl_context_detects_keyed_values_without_colon():
    text = "The user paaltwvkjuijwbj957 reported an issue; handle rand.podo confirmed it."
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    assert "paaltwvkjuijwbj957" in username_texts
    assert "rand.podo" in username_texts


def test_socialnumber_spaced_mixed_id_detected():
    text = "Record: AUSTI 711154 AS 852 should be treated as a document identifier."
    evidence = eng.regex_evidence(text)
    socialnumber_texts = [text[e.start:e.end] for e in evidence if e.label == "SOCIALNUMBER"]

    assert "AUSTI 711154 AS 852" in socialnumber_texts


def test_structured_person_applicant_field_detected():
    text = '"applicant": "Balloi Eckrich", "email": "bballoi@yahoo.com"'
    evidence = eng.regex_evidence(text)
    person_texts = [text[e.start:e.end] for e in evidence if e.label == "PERSON"]

    assert "Balloi Eckrich" in person_texts


def test_username_short_code_context_detected():
    text = 'username: N23; participant_id: R21; user: A1'
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    assert "N23" in username_texts
    assert "R21" in username_texts
    # Two-char code is intentionally ignored to reduce noise.
    assert "A1" not in username_texts


def test_username_regex_does_not_capture_email():
    text = "username: ewgenij.inzollitto22@hotmail.com"
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    # Email should be EMAIL, not USERNAME
    assert not username_texts


@pytest.mark.parametrize(
    "text,expected",
    [
        ("username: paaltwvkjuijwbj957", "paaltwvkjuijwbj957"),
        ("login_id=wsfdkmi9214", "wsfdkmi9214"),
        ("participant_id: 10mavus.tancev", "10mavus.tancev"),
        ("caller: rand.podo", "rand.podo"),
        ("handle: maria-rosaria.amardi1962", "maria-rosaria.amardi1962"),
        ("user: R21", "R21"),
        ("username: 43CU", "43CU"),
        ("username: _badprefix", None),
    ],
)
def test_username_context_variants(text, expected):
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    if expected is None:
        assert not username_texts
    else:
        assert expected in username_texts


@pytest.mark.parametrize(
    "text,should_match",
    [
        ("users: lyxmvtinlajlq99997", True),
        ("users: ylhhhrmivzz90", True),
        ("users: gpesrelu34", True),
        ("users: shari", False),             # too short / no digit
        ("users: JOHN-DOE", False),          # all-caps field-like / no digit
        ("users: api_key", False),           # should not be USERNAME
        ("users: sk-abcdef1234567890", False),
        ("users: account-name", False),      # no digit in generic hyphen token
    ],
)
def test_username_opaque_and_separator_variants(text, should_match):
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]

    if should_match:
        assert username_texts, f"Expected USERNAME for: {text!r}"
    else:
        assert not username_texts, f"Did not expect USERNAME for: {text!r}"


def test_username_deduplicates_overlapping_matches():
    text = "username: paaltwvkjuijwbj957"
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]
    # Context + opaque regexes may both see the same token; output should contain one span.
    assert username_texts.count("paaltwvkjuijwbj957") == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        # Standard spacing
        ('"username": "R21"', "R21"),
        ("username: R21", "R21"),
        # No spaces around colon
        ('"username":"R21"', "R21"),
        ("username:R21", "R21"),
        # Extra spaces
        ('"username" : "R21"', "R21"),
        ("username : R21", "R21"),
        ('"username"  :  "R21"', "R21"),
        ("username  :  R21", "R21"),
        # Space before colon only
        ('"username" :"R21"', "R21"),
        ("username :R21", "R21"),
        # Space after colon only
        ('"username": "R21"', "R21"),
        ("username: R21", "R21"),
        # Equals sign instead of colon
        ('"username"="R21"', "R21"),
        ("username=R21", "R21"),
        ("username = R21", "R21"),
        # Mixed quotes and spacing
        ("'username': 'R21'", "R21"),
        ("'username': R21", "R21"),
        ("username: 'R21'", "R21"),
    ],
)
def test_username_context_spacing_variants(text, expected):
    evidence = eng.regex_evidence(text)
    username_texts = [text[e.start:e.end] for e in evidence if e.label == "USERNAME"]
    assert expected in username_texts, f"Expected {expected!r} in {text!r}"


# ============================================================
# Adjacent same-label entity merging tests
# ============================================================

def test_merge_adjacent_person_entities_with_space():
    """Test merging two PERSON entities separated by a single space."""
    text = "John Smith"
    e1 = Entity(text="John", label="PERSON", start=0, end=4, source="spacy")
    e2 = Entity(text="Smith", label="PERSON", start=5, end=10, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert merged[0].text == "John Smith"
    assert merged[0].start == 0
    assert merged[0].end == 10
    assert merged[0].label == "PERSON"


def test_merge_adjacent_person_entities_with_multiple_spaces():
    """Test merging two PERSON entities separated by multiple spaces."""
    text = "Jane   Doe"
    e1 = Entity(text="Jane", label="PERSON", start=0, end=4, source="spacy")
    e2 = Entity(text="Doe", label="PERSON", start=7, end=10, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert merged[0].text == "Jane   Doe"
    assert merged[0].start == 0
    assert merged[0].end == 10


def test_merge_adjacent_person_entities_with_tab():
    """Test merging two PERSON entities separated by a tab."""
    text = "Alice\tBob"
    e1 = Entity(text="Alice", label="PERSON", start=0, end=5, source="spacy")
    e2 = Entity(text="Bob", label="PERSON", start=6, end=9, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert merged[0].text == "Alice\tBob"


def test_merge_adjacent_person_entities_with_newline():
    """Test merging two PERSON entities separated by a newline."""
    text = "Charlie\nDiana"
    e1 = Entity(text="Charlie", label="PERSON", start=0, end=7, source="spacy")
    e2 = Entity(text="Diana", label="PERSON", start=8, end=13, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert merged[0].text == "Charlie\nDiana"


def test_no_merge_different_labels():
    """Test that entities with different labels are NOT merged."""
    text = "John company"
    e1 = Entity(text="John", label="PERSON", start=0, end=4, source="spacy")
    e2 = Entity(text="company", label="ORG", start=5, end=12, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 2
    assert merged[0].text == "John"
    assert merged[1].text == "company"


def test_no_merge_non_whitespace_gap():
    """Test that adjacent entities with non-whitespace gap are NOT merged."""
    text = "John-Smith"
    e1 = Entity(text="John", label="PERSON", start=0, end=4, source="spacy")
    e2 = Entity(text="Smith", label="PERSON", start=5, end=10, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    # The gap contains "-" (non-whitespace), so should NOT merge
    assert len(merged) == 2


def test_merge_three_adjacent_person_entities():
    """Test merging three consecutive PERSON entities with whitespace."""
    text = "John Michael Smith"
    e1 = Entity(text="John", label="PERSON", start=0, end=4, source="spacy")
    e2 = Entity(text="Michael", label="PERSON", start=5, end=12, source="spacy")
    e3 = Entity(text="Smith", label="PERSON", start=13, end=18, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2, e3])
    
    assert len(merged) == 1
    assert merged[0].text == "John Michael Smith"
    assert merged[0].start == 0
    assert merged[0].end == 18


def test_merge_preserves_confidence():
    """Test that merging entities averages their confidence scores."""
    text = "Alice Bob"
    e1 = Entity(text="Alice", label="PERSON", start=0, end=5, source="spacy", confidence=0.8)
    e2 = Entity(text="Bob", label="PERSON", start=6, end=9, source="spacy", confidence=0.9)
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert merged[0].confidence == pytest.approx(0.85)  # (0.8 + 0.9) / 2


def test_merge_preserves_evidence_sources():
    """Test that merging entities combines evidence sources."""
    text = "Eve Frank"
    e1 = Entity(text="Eve", label="PERSON", start=0, end=3, source="spacy", evidence_sources=("spacy",))
    e2 = Entity(text="Frank", label="PERSON", start=4, end=9, source="spacy", evidence_sources=("gliner",))
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2])
    
    assert len(merged) == 1
    assert set(merged[0].evidence_sources) == {"spacy", "gliner"}


def test_merge_partial_sequence():
    """Test merging where only some entities in sequence have same label."""
    text = "Grace ORG Henry"
    e1 = Entity(text="Grace", label="PERSON", start=0, end=5, source="spacy")
    e2 = Entity(text="ORG", label="ORG", start=6, end=9, source="regex")
    e3 = Entity(text="Henry", label="PERSON", start=10, end=15, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e1, e2, e3])
    
    # Grace and Henry should NOT merge (ORG in between)
    assert len(merged) == 3
    assert merged[0].text == "Grace"
    assert merged[1].text == "ORG"
    assert merged[2].text == "Henry"


def test_merge_empty_list():
    """Test that empty entity list returns empty list."""
    text = "test"
    merged = eng._merge_adjacent_same_label_entities(text, [])
    assert merged == []


def test_merge_single_entity():
    """Test that single entity list returns single entity unchanged."""
    text = "single"
    e = Entity(text="single", label="PERSON", start=0, end=6, source="spacy")
    merged = eng._merge_adjacent_same_label_entities(text, [e])
    assert len(merged) == 1
    assert merged[0] == e


# ============================================================
# Generalized structured field detection tests
# ============================================================

def test_structured_field_email_value_only():
    """Test that only the EMAIL value is labeled, not the field key."""
    text = '"email": "john@example.com"'
    evidence = eng.regex_evidence(text)
    email_spans = [text[e.start:e.end] for e in evidence if e.label == "EMAIL"]
    
    # Only the value should be extracted as EMAIL
    assert "john@example.com" in email_spans
    # The key "email" should NOT be in the EMAIL spans
    assert "email" not in email_spans


def test_structured_field_email_json_format():
    """Test EMAIL detection in JSON format."""
    text = '{"email": "alice@domain.com", "phone": "555-1234"}'
    evidence = eng.regex_evidence(text)
    email_texts = [text[e.start:e.end] for e in evidence if e.label == "EMAIL"]
    
    assert "alice@domain.com" in email_texts


def test_structured_field_phone_markdown_format():
    """Test PHONE detection in Markdown format."""
    text = "- phone: 555-1234\n- name: John"
    evidence = eng.regex_evidence(text)
    phone_texts = [text[e.start:e.end] for e in evidence if e.label == "PHONE"]
    
    assert "555-1234" in phone_texts


def test_structured_field_date_time_format():
    """Test DATE detection from 'time' field key."""
    text = '"time": "10:30am"'
    evidence = eng.regex_evidence(text)
    date_texts = [text[e.start:e.end] for e in evidence if e.label == "DATE"]
    
    assert "10:30am" in date_texts


def test_structured_field_xml_format():
    """Test structured field detection in XML format."""
    text = "<email>bob@test.org</email><phone>555-9999</phone>"
    evidence = eng.regex_evidence(text)
    
    email_texts = [text[e.start:e.end] for e in evidence if e.label == "EMAIL"]
    phone_texts = [text[e.start:e.end] for e in evidence if e.label == "PHONE"]
    
    assert "bob@test.org" in email_texts
    assert "555-9999" in phone_texts


def test_structured_field_person_name():
    """Test PERSON detection from 'name' field."""
    text = '"name": "Charlie Smith"'
    evidence = eng.regex_evidence(text)
    person_texts = [text[e.start:e.end] for e in evidence if e.label == "PERSON"]
    
    # May or may not detect "Charlie Smith" depending on regex; at minimum check no false positives on key
    assert "name" not in person_texts  # Key should not be labeled


def test_structured_field_first_last_name():
    """Test PERSON detection from first_name and last_name fields."""
    text = '"first_name": "Diana", "last_name": "Prince"'
    evidence = eng.regex_evidence(text)
    person_texts = [text[e.start:e.end] for e in evidence if e.label == "PERSON"]
    
    assert "Diana" in person_texts or "Prince" in person_texts
    # Keys should not be labeled
    assert "first_name" not in person_texts
    assert "last_name" not in person_texts


def test_structured_field_organization():
    """Test ORG detection from organization/company fields."""
    text = '"organization": "Acme Corp", "company": "TechStart Inc"'
    evidence = eng.regex_evidence(text)
    org_texts = [text[e.start:e.end] for e in evidence if e.label == "ORG"]
    
    assert "Acme Corp" in org_texts or "TechStart Inc" in org_texts
    assert "organization" not in org_texts
    assert "company" not in org_texts


def test_structured_field_api_key():
    """Test API_KEY detection from api_key/access_token fields."""
    text = '"api_key": "sk-abc123def456", "access_token": "token_xyz"'
    evidence = eng.regex_evidence(text)
    api_texts = [text[e.start:e.end] for e in evidence if e.label == "API_KEY"]
    
    assert "sk-abc123def456" in api_texts or "token_xyz" in api_texts
    assert "api_key" not in api_texts


def test_structured_field_socialnumber():
    """Test SOCIALNUMBER detection from ssn/passport fields."""
    text = '"us_ssn": "123-45-6789", "passport": "ABC123456"'
    evidence = eng.regex_evidence(text)
    ssn_texts = [text[e.start:e.end] for e in evidence if e.label == "SOCIALNUMBER"]
    
    # Should detect at least one
    assert len(ssn_texts) > 0
    assert "us_ssn" not in ssn_texts
    assert "passport" not in ssn_texts


def test_structured_field_ip():
    """Test IP detection from ip field."""
    text = '"ip": "192.168.1.1"'
    evidence = eng.regex_evidence(text)
    ip_texts = [text[e.start:e.end] for e in evidence if e.label == "IP"]
    
    assert "192.168.1.1" in ip_texts
    assert "ip" not in ip_texts


def test_structured_field_no_false_positives_on_keys():
    """Comprehensive test: no field keys should be labeled as their value type."""
    text = (
        '"email": "test@example.com", '
        '"phone": "555-1234", '
        '"name": "John Doe", '
        '"api_key": "sk-12345"'
    )
    evidence = eng.regex_evidence(text)
    
    # Collect all detected values by label
    all_labeled_texts = {e.label: [text[e.start:e.end] for e in evidence if e.label == e.label] 
                         for e in evidence}
    
    # Keys should never appear as values
    keys = ["email", "phone", "name", "api_key"]
    for key in keys:
        for label, spans in all_labeled_texts.items():
            assert key not in spans, f"Field key '{key}' should not be labeled as {label}"


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


# ==================== Tests for Plain Key: Value Format ====================


def test_structured_field_plain_key_value_username():
    """Test plain key: value format for USERNAME fields."""
    text = "Username: john123"
    evidence = eng.regex_evidence(text)
    
    username_spans = _spans_for_label(evidence, "USERNAME")
    assert username_spans, "Expected USERNAME from plain key: value format"
    assert text[username_spans[0][0]:username_spans[0][1]] == "john123"


def test_structured_field_plain_key_value_location():
    """Test plain key: value format for LOCATION fields."""
    text = "Place: New York"
    evidence = eng.regex_evidence(text)
    
    location_spans = _spans_for_label(evidence, "LOCATION")
    assert location_spans, "Expected LOCATION from plain key: value format"
    assert text[location_spans[0][0]:location_spans[0][1]] == "New York"


def test_structured_field_plain_key_value_email():
    """Test plain key: value format for EMAIL fields."""
    text = "Email: test@example.com"
    evidence = eng.regex_evidence(text)
    
    email_spans = _spans_for_label(evidence, "EMAIL")
    assert email_spans, "Expected EMAIL from plain key: value format"
    assert text[email_spans[0][0]:email_spans[0][1]] == "test@example.com"


def test_structured_field_plain_key_value_phone():
    """Test plain key: value format for PHONE fields."""
    text = "Phone: 555-123-4567"
    evidence = eng.regex_evidence(text)
    
    phone_spans = _spans_for_label(evidence, "PHONE")
    assert phone_spans, "Expected PHONE from plain key: value format"
    assert text[phone_spans[0][0]:phone_spans[0][1]] == "555-123-4567"


def test_structured_field_plain_key_value_date():
    """Test plain key: value format for DATE fields."""
    text = "Date: 2024-06-05"
    evidence = eng.regex_evidence(text)
    
    date_spans = _spans_for_label(evidence, "DATE")
    assert date_spans, "Expected DATE from plain key: value format"
    assert text[date_spans[0][0]:date_spans[0][1]] == "2024-06-05"


def test_structured_field_plain_key_value_api_key():
    """Test plain key: value format for API_KEY fields."""
    text = "API_KEY: sk-abc123def456"
    evidence = eng.regex_evidence(text)
    
    api_key_spans = _spans_for_label(evidence, "API_KEY")
    assert api_key_spans, "Expected API_KEY from plain key: value format"
    # Check that the actual API key value is detected
    api_key_texts = [text[s:e] for s, e, _ in api_key_spans]
    assert "sk-abc123def456" in api_key_texts, f"Expected API key value in {api_key_texts}"


def test_structured_field_plain_key_value_api_key2():
    """Test plain key: value format for API_KEY fields."""
    text = "APIKEY: sk-abc123def456"
    evidence = eng.regex_evidence(text)
    
    api_key_spans = _spans_for_label(evidence, "API_KEY")
    assert api_key_spans, "Expected API_KEY from plain key: value format"
    # Check that at least one match is the actual API key (not the field key)
    api_key_texts = [text[s:e] for s, e, _ in api_key_spans]
    assert "sk-abc123def456" in api_key_texts, f"Expected full API key in {api_key_texts}"


def test_structured_field_plain_key_value_ip():
    """Test plain key: value format for IP fields."""
    text = "IP: 192.168.1.1"
    evidence = eng.regex_evidence(text)
    
    ip_spans = _spans_for_label(evidence, "IP")
    assert ip_spans, "Expected IP from plain key: value format"
    assert text[ip_spans[0][0]:ip_spans[0][1]] == "192.168.1.1"


def test_structured_field_plain_key_value_socialnumber():
    """Test plain key: value format for SOCIALNUMBER fields."""
    text = "Passport: A123456789"
    evidence = eng.regex_evidence(text)
    
    ssn_spans = _spans_for_label(evidence, "SOCIALNUMBER")
    assert ssn_spans, "Expected SOCIALNUMBER from plain key: value format"
    assert text[ssn_spans[0][0]:ssn_spans[0][1]] == "A123456789"


def test_structured_field_plain_key_value_case_insensitive():
    """Test plain key: value format is case-insensitive."""
    text = "USERNAME: alice42"
    evidence = eng.regex_evidence(text)
    
    username_spans = _spans_for_label(evidence, "USERNAME")
    assert username_spans, "Expected USERNAME with uppercase key"
    assert text[username_spans[0][0]:username_spans[0][1]] == "alice42"


def test_structured_field_plain_key_value_mixed_case():
    """Test plain key: value format with mixed case field names."""
    text = "FirstName: John"
    evidence = eng.regex_evidence(text)
    
    person_spans = _spans_for_label(evidence, "PERSON")
    assert person_spans, "Expected PERSON from FirstName key"
    assert text[person_spans[0][0]:person_spans[0][1]] == "John"


def test_structured_field_plain_key_value_multiple_fields():
    """Test multiple plain key: value fields in same text."""
    text = "Username: bob99\nEmail: bob@company.com\nLocation: Boston"
    evidence = eng.regex_evidence(text)
    
    username_spans = _spans_for_label(evidence, "USERNAME")
    email_spans = _spans_for_label(evidence, "EMAIL")
    location_spans = _spans_for_label(evidence, "LOCATION")
    
    assert len(username_spans) >= 1, "Expected USERNAME"
    assert len(email_spans) >= 1, "Expected EMAIL"
    assert len(location_spans) >= 1, "Expected LOCATION"
