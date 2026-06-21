'''Module for regex-based PII/secret detection patterns and evidence extraction.'''
import re
from typing import List, Optional
from .defs import Entity, Evidence

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+\-\xC0-\xFF]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(
    r"(?<!\d)"
    r"(?!\d{4}[-/]\d{2}[-/]\d{2})"         # Exclude YYYY-MM-DD / YYYY/MM/DD dates
    r"(?!\d{1,3}\.\d{1,3}\.\d{1,3})"          # Exclude IP-like patterns (3 octet groups)
    r"\+?\d[\d \-\.\(\)]{7,}\d"              # Digits, spaces, dashes, dots, parens only
    r"(?!\d)"
)
API_KEY_REGEX = re.compile(
    r"(sk-[a-zA-Z0-9]{10,}|api[_-]?key\s*[:=]\s*[a-zA-Z0-9]+)", re.I)

# IPv4 + IPv6 (full 8-group form)
IP_REGEX = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
    r"|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){7}"  # Full IPv6: 8 colon-separated hex groups
)

# Date formats: numeric, month names, times with AM/PM, o'clock
DATE_REGEX = re.compile(
    r"\b(?:"
    # Numeric dates: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, DD.MM.YYYY
    # Optional ISO 8601 time suffix: T00:00:00 (avoids \b mismatch when T follows digits)
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:T\d{2}:\d{2}:\d{2})?"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    # Month name + day (with optional ordinal)
    #  + optional year: "October 18th, 1980", "June 4", "June/88"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?|[-/]\d{2,4})"
    # Day + optional "of" + month name + optional year: "4th June 2023", "21st of December, 1999"
    r"|\d{1,2}(?:st|nd|rd|th)?(?:\s+of)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:[,\s]+\d{4})?"
    # Times with optional seconds and optional AM/PM: 3:07am, 10:15 PM, 22:41
    # (?<!T) prevents matching times inside ISO 8601 timestamps (e.g. T00:00:00)
    r"|(?<!T)(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[ap]m)?"
    # Hour-only with AM/PM: "1 AM", "8 PM", "3am"
    r"|\d{1,2}\s*[ap]m"
    # o'clock: "5 o'clock", "13 o'clock"
    r"|\d{1,2}\s*o'clock"
    r")\b",
    re.IGNORECASE
)

# Government/document IDs: SSN, passports, national IDs, credit cards, crypto addresses
SOCIALNUMBER_REGEX = re.compile(
    r"(?:"
    # Credit card numbers: 13-19 digits with optional separators (Visa/MC/Amex)
    # Examples: 4532-1234-5678-9010, 4532123456789010, 5234 1234 5678 9010
    r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4,7}\b"
    # SSN with separators: 123-45-6789, 123 45 6789
    r"|\b\d{3}[\-\s]\d{2}[\-\s]\d{4}\b"
    # SSN without separators (exactly 9 digits, not part of longer number)
    r"|\b\d{3}\d{2}\d{4}\b(?!\d)"
    # Bank account numbers and other numeric IDs: 8-17 consecutive digits
    r"|\b\d{8,17}\b"
    # Ethereum addresses: 0x followed by 40 hex characters
    r"|\b0x[a-fA-F0-9]{40}\b"
    # Bitcoin and other crypto addresses: 26-35 alphanumeric chars
    r"|\b[a-km-zA-HJ-NP-Z0-9]{26,35}\b"
    # Passport/ID: letters followed by digits (2+ letters, 4+ digits)
    r"|\b[A-Z]{2,5}\d{4,10}\b"
    # Mixed alphanumeric IDs: 6-12 chars with both letters and digits (passports, IDs)
    # Requires at least one ACTUAL uppercase letter (case-sensitive via (?-i:)) to avoid
    # matching all-lowercase username patterns like 'amardi1962' or 'wsfdkmi9214'
    r"|\b(?=[A-Z\d]*(?-i:[A-Z])[A-Z\d]*\d)[A-Z\d]{6,12}\b"
    # Mixed format: 2-4 letters + 4-8 digits, optionally with separators
    r"|\b[A-Z]{2,4}[\s\-]?\d{4,8}\b"
    # CURP-style with separators: letters + digits + letters + digits
    r"|\b[A-Z]{3,6}\d?[\.\-\s]+\d{4,8}[\.\-\s]+[A-Z0-9]{1,3}[\.\-\s]+\d{2,4}\b"
    # CURP continuous (no separators): e.g. 'CADIJ958032CM645', 'FARLE708293FN375'
    r"|\b[A-Z]{4,6}\d{6,8}[A-Z]{2}\d{3,4}\b"
    # Spaced mixed IDs commonly seen in synthetic privacy datasets:
    # e.g. "AUSTI 711154 AS 852", "ABCD 123456 XY 99"
    r"|\b[A-Z]{3,8}[\s\-]\d{4,8}[\s\-][A-Z]{1,4}[\s\-]\d{2,4}\b"
    # 3-3-4 format: phone-style IDs annotated as national IDs (e.g. '684 916 3578', '873-878-0248')
    r"|\b\d{3}[\s\-\.]\d{3}[\s\-\.]\d{4}\b"
    r")",
    re.IGNORECASE
)

# Structured location fields: detects address components specified via JSON, markdown, XML, or CSV.
# Captures the VALUE from patterns like:
#   JSON:     "city": "Doncaster"   |  "CITY": "Doncaster"
#   Markdown: - City: Doncaster     |  **City:** Doncaster
#   XML:      <city>Doncaster</city>
# The capture group 1 holds the value span; regex_evidence iterates with .start(1)/.end(1).
_LOCATION_FIELD_KEYS = (
    r"country|city|state|street|building|postcode|zipcode|zip_code|"
    r"zip|address|location|district|region|province|county|suburb|locality"
)
STRUCTURED_LOCATION_REGEX = re.compile(
    (r"(?:"
     r"""(?:["']?)(?:""" + _LOCATION_FIELD_KEYS + r""")(?:["']?)"""
     r"""\s*[":]\s*["']([^"'\n,\[\]{}<>]{1,80})["']"""
     r"""|(?:[-*]\s*)?(?:\*{0,2})(?:""" + _LOCATION_FIELD_KEYS + r""")"""
     r"""(?:\*{0,2}):\s*([^\n,\[\]{}<>*|]{1,80})"""
     r"""|<(?:""" + _LOCATION_FIELD_KEYS + r""")>([^<]{1,80})</(?:"""
     + _LOCATION_FIELD_KEYS + r""")>"""
     r"""|(?:""" + _LOCATION_FIELD_KEYS + r"""):\s*([^\n,\[\]{}<>*|]{1,80})"""
     r")"
    ),
    re.IGNORECASE,
)

# Generalized structured field detection mapping: field key names → canonical entity labels.
# When a field key matches a label, the field VALUE is labeled with that entity type.
# Example: "email": "john@example.com" → the VALUE "john@example.com" is labeled EMAIL
_FIELD_KEY_LABEL_MAP = {
    # PERSON fields
    "PERSON": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "FULL_NAME": "PERSON",
    "NAME": "PERSON",
    "GIVENNAME1": "PERSON",
    "GIVENNAME2": "PERSON",
    "LASTNAME1": "PERSON",
    "LASTNAME2": "PERSON",
    "LASTNAME3": "PERSON",
    # ORG fields
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "COMPANY": "ORG",
    # LOCATION fields
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "PLACE": "LOCATION",
    "CITY": "LOCATION",
    "STATE": "LOCATION",
    "COUNTRY": "LOCATION",
    "ADDRESS": "LOCATION",
    "STREET": "LOCATION",
    "BUILDING": "LOCATION",
    "POSTCODE": "LOCATION",
    "SECADDRESS": "LOCATION",
    # EMAIL fields
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    # PHONE fields
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "TEL": "PHONE",
    "MOBILE": "PHONE",
    "MOBILE_PHONE_NUMBER": "PHONE",
    # API_KEY fields
    "API_KEY": "API_KEY",
    "APIKEY": "API_KEY",
    "ACCESS_TOKEN": "API_KEY",
    # DATE/TIME fields
    "DATE": "DATE",
    "TIME": "DATE",
    "BOD": "DATE",
    # IP fields
    "IP": "IP",
    # SOCIALNUMBER fields
    "SOCIALNUMBER": "SOCIALNUMBER",
    "PASSPORT": "SOCIALNUMBER",
    "IDCARD": "SOCIALNUMBER",
    "DRIVERLICENSE": "SOCIALNUMBER",
    "CARDISSUER": "SOCIALNUMBER",
    "PASS": "SOCIALNUMBER",
    "US_SSN": "SOCIALNUMBER",
    "CREDIT_CARD": "SOCIALNUMBER",
    "CRYPTO": "SOCIALNUMBER",
    "CASE_NUMBER": "SOCIALNUMBER",
    # USERNAME fields
    "USERNAME": "USERNAME",
    # Additional field aliases
    "GEOCOORD": "LOCATION",
    "TITLE": "PERSON",
    "PARTICIPANT": "PERSON",
    "APPLICANT": "PERSON",
    "GIVEN_NAME": "PERSON",
    "SURNAME": "PERSON",
    "LASTNAME": "PERSON",
    "MIDDLENAME": "PERSON",
    "FAMILY_NAME": "PERSON",
    "FORENAME": "PERSON",
    "PROVINCE": "LOCATION",
    "REGION": "LOCATION",
    "DISTRICT": "LOCATION",
    "COUNTY": "LOCATION",
}

# Build regex for all field keys except those already in STRUCTURED_LOCATION_REGEX
_LOCATION_FIELD_SET = {
    "COUNTRY", "CITY", "STATE", "STREET", "BUILDING", "POSTCODE",
    "ZIPCODE", "ZIP_CODE", "ZIP", "ADDRESS", "LOCATION", "DISTRICT",
    "REGION", "PROVINCE", "COUNTY", "SUBURB", "LOCALITY"
}
_ALL_FIELD_KEYS_EXCEPT_LOC = "|".join(
    k.lower() for k in _FIELD_KEY_LABEL_MAP
    if k.upper() not in _LOCATION_FIELD_SET
)

STRUCTURED_FIELD_REGEX = re.compile(
    (r"(?:"
     r"""(?:["']?)(?:""" + _ALL_FIELD_KEYS_EXCEPT_LOC + r""")(?:["']?)"""
     r"""\s*[":]\s*["']([^"'\n,\[\]{}<>]{1,80})["']"""
     r"""|(?:[-*]\s*)?(?:\*{0,2})(?:""" + _ALL_FIELD_KEYS_EXCEPT_LOC + r""")"""
     r"""(?:\*{0,2}):\s*([^\n,\[\]{}<>*|]{1,80})"""
     r"""|<(?:""" + _ALL_FIELD_KEYS_EXCEPT_LOC + r""")>([^<]{1,80})</(?:"""
     + _ALL_FIELD_KEYS_EXCEPT_LOC + r""")>"""
     r"""|(?:""" + _ALL_FIELD_KEYS_EXCEPT_LOC + r"""):\s*([^\n,\[\]{}<>*|]{1,80})"""
     r")"
    ),
    re.IGNORECASE,
)

def _extract_field_key_from_match(matched_text: str) -> Optional[str]:
    """Extract the field key name from a structured field regex match.

    Given a matched string like '"email": "john@..."' or '<phone>555-1234</phone>',
    extracts the key ('email' or 'phone'). Returns the key mapped to a canonical label,
    or None if no valid key found.
    """
    # Try JSON/YAML format: "key": or 'key': or key:
    for match in re.finditer(r'(?:["\'"])?([a-z_]+)(?:["\'"])?(?:\s*[:=])',
                             matched_text, re.IGNORECASE):
        potential_key = match.group(1).upper()
        if potential_key in _FIELD_KEY_LABEL_MAP:
            return potential_key

    # Try XML format: <key>
    for match in re.finditer(r'<([a-z_]+)>', matched_text, re.IGNORECASE):
        potential_key = match.group(1).upper()
        if potential_key in _FIELD_KEY_LABEL_MAP:
            return potential_key

    return None

# USERNAME patterns
# 1) Separator format with digit presence (to avoid common hyphenated words):
#    maria-rosaria.amardi1962, user_123
USERNAME_REGEX = re.compile(
    r"\b(?=[a-zA-Z0-9._\-\xC0-\xFF]*\d)"
    r"[a-zA-Z0-9\xC0-\xFF]+[._-][a-zA-Z0-9._\-\xC0-\xFF]{2,}\b"
)

# 2) Opaque alphanumeric handles (lowercase-focused to reduce false positives):
#    paaltwvkjuijwbj957, wsfdkmi9214, lyxmvtinlajlq99997
USERNAME_OPAQUE_REGEX = re.compile(
    r"\b(?=[a-z0-9._-]{8,32}\b)(?=[a-z0-9._-]*[a-z])(?=[a-z0-9._-]*\d)"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*\b"
)

# 3) Context-keyed usernames (captures value after username-ish keys):
#    "username": "R21", participant_id: '10mavus.tancev', caller: ChuWen123
USERNAME_CONTEXT_REGEX = re.compile(
    (r"(?:[\"']?\b(?:username|user|participant_id|caller|login(?:_id)?|handle)"
     r"\b[\"']?\s*[:=]\s*[\"']?)"
     r"([a-zA-Z0-9][a-zA-Z0-9._\-]{2,31})"),
    re.IGNORECASE,
)

# 3b) Keyed short-code usernames (3-6 chars) with letters+digits:
#     username: N23, participant_id=R21
USERNAME_SHORT_CODE_CONTEXT_REGEX = re.compile(
    (r"(?:[\"']?\b(?:username|user|participant_id|caller|login(?:_id)?|handle)"
     r"\b[\"']?\s*[:=]\s*[\"']?)"
     r"([A-Za-z0-9]{3,6})"),
    re.IGNORECASE,
)

# 4) Natural-language keyed usernames without punctuation separators:
#    "user paaltwvkjuijwbj957", "handle rand.podo", "participant 43CU"
USERNAME_NL_CONTEXT_REGEX = re.compile(
    (r"\b(?:username|user|participant|participant_id|caller|login(?:_id)?|handle)\b"
     r"\s+([a-zA-Z0-9][a-zA-Z0-9._\-]{1,31})\b"),
    re.IGNORECASE,
)

def skip_username_candidate(candidate: str) -> bool:
    '''Determines if a USERNAME candidate should be skipped based on heuristics.
    This helps reduce false positives by excluding patterns that are unlikely to be usernames.'''
    cand = candidate.strip()
    low = cand.lower()
    # Do not steal API keys or API-key field names.
    if low.startswith("sk-") or low in {"api_key", "api-key", "apikey"}:
        return True
    # Exclude placeholder tokens and emails.
    if "@" in cand or _PLACEHOLDER_RE.match(cand):
        return True
    # Exclude all-caps field-like tokens (e.g., API_KEY, USER_NAME).
    if cand.upper() == cand and not any(ch.isdigit() for ch in cand):
        return True
    return False

# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def regex_evidence(text: str) -> List[Evidence]:
    """
    Collects high-confidence evidence from regex-based PII/secret patterns.

    Covered patterns: EMAIL, PHONE, API_KEY, IP, DATE, SOCIALNUMBER, LOCATION, USERNAME.
    """
    evidence: List[Evidence] = []

    for m in EMAIL_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "EMAIL", "regex", 0.90))

    for m in PHONE_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "PHONE", "regex", 0.80))

    for m in API_KEY_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "API_KEY", "regex", 0.95))

    for m in IP_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "IP", "regex", 0.88))

    for m in DATE_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "DATE", "regex", 0.75))

    for m in SOCIALNUMBER_REGEX.finditer(text):
        evidence.append(Evidence(m.start(), m.end(), "SOCIALNUMBER", "regex", 0.93))

    _lower_text = text.lower()
    _location_field_signals = (
        "city", "state", "country", "street", "building", "postcode",
        "zipcode", "address", "location",
    )
    if any(sig in _lower_text for sig in _location_field_signals):
        for m in STRUCTURED_LOCATION_REGEX.finditer(text):
            # The value is in one of four capture groups (JSON, markdown, XML, or plain key:value)
            grp = next((i for i in (1, 2, 3, 4) if m.group(i) is not None), None)
            if grp is not None:
                value = m.group(grp).strip()
                if value:
                    vs = m.start(grp) + m.group(grp).index(value.lstrip())
                    ve = vs + len(value)
                    evidence.append(Evidence(vs, ve, "LOCATION", "regex", 0.72))

    # USERNAME detection (medium confidence): context-keyed values + opaque handles.
    # For very long texts, skip username regex scanning for latency safety.
    _username_context_scan = len(text) <= 1200
    _username_full_scan = len(text) <= 800
    _username_seen_spans: set[tuple[int, int]] = set()

    def _is_email_local_part(end_idx: int) -> bool:
        # Matches like "username: localpart@example.com" should remain EMAIL only.
        return end_idx < len(text) and text[end_idx] == "@"

    if _username_context_scan:
        for m in USERNAME_CONTEXT_REGEX.finditer(text):
            candidate = m.group(1)
            if skip_username_candidate(candidate) or _is_email_local_part(m.end(1)):
                continue
            span = (m.start(1), m.end(1))
            if span not in _username_seen_spans:
                _username_seen_spans.add(span)
                evidence.append(Evidence(span[0], span[1], "USERNAME", "regex", 0.78))

        for m in USERNAME_SHORT_CODE_CONTEXT_REGEX.finditer(text):
            candidate = m.group(1)
            # Require mixed alnum to avoid common short words/tokens.
            if not (any(ch.isalpha() for ch in candidate) and any(ch.isdigit() for ch in candidate)):
                continue
            if skip_username_candidate(candidate) or _is_email_local_part(m.end(1)):
                continue
            span = (m.start(1), m.end(1))
            if span not in _username_seen_spans:
                _username_seen_spans.add(span)
                evidence.append(Evidence(span[0], span[1], "USERNAME", "regex", 0.76))

        for m in USERNAME_NL_CONTEXT_REGEX.finditer(text):
            candidate = m.group(1)
            if skip_username_candidate(candidate) or _is_email_local_part(m.end(1)):
                continue
            span = (m.start(1), m.end(1))
            if span not in _username_seen_spans:
                _username_seen_spans.add(span)
                evidence.append(Evidence(span[0], span[1], "USERNAME", "regex", 0.76))

    if _username_full_scan:
        for m in USERNAME_REGEX.finditer(text):
            candidate = m.group(0)
            if skip_username_candidate(candidate) or _is_email_local_part(m.end()):
                continue
            span = (m.start(), m.end())
            if span not in _username_seen_spans:
                _username_seen_spans.add(span)
                evidence.append(Evidence(span[0], span[1], "USERNAME", "regex", 0.70))

        for m in USERNAME_OPAQUE_REGEX.finditer(text):
            candidate = m.group(0)
            if skip_username_candidate(candidate) or _is_email_local_part(m.end()):
                continue
            span = (m.start(), m.end())
            if span not in _username_seen_spans:
                _username_seen_spans.add(span)
                evidence.append(Evidence(span[0], span[1], "USERNAME", "regex", 0.68))

    # Generalized structured field detection for all entity labels.
    # Maps field keys to canonical entity labels. Only the VALUE is labeled.
    # Example: "email": "john@example.com" → "john@example.com" labeled as EMAIL
    # Skip if the value already matches a higher-confidence pattern.
    # Also skip if we've already detected this exact span with the same label.
    existing_spans = {(e.start, e.end, e.label) for e in evidence}

    def _should_skip_field_value(mapped_label: str, value: str) -> bool:
        """Check if a field value should be skipped due to validation rules."""
        if mapped_label == "USERNAME":
            # Skip if value is likely an email address
            if EMAIL_REGEX.search(value):
                return True
            # Skip if username starts with underscore
            if value.startswith("_"):
                return True
        return False

    for m in STRUCTURED_FIELD_REGEX.finditer(text):
        grp = next((i for i in (1, 2, 3, 4) if m.group(i) is not None), None)
        if grp is None:
            continue

        value = m.group(grp).strip()
        field_key = _extract_field_key_from_match(m.group(0))

        if not field_key or field_key not in _FIELD_KEY_LABEL_MAP:
            continue

        mapped_label = _FIELD_KEY_LABEL_MAP[field_key]
        if not value:
            continue

        vs = m.start(grp) + m.group(grp).index(value.lstrip())
        ve = vs + len(value)

        # Skip if we've already detected this exact span with this label
        if (vs, ve, mapped_label) in existing_spans:
            continue

        # Skip if value fails validation checks
        if _should_skip_field_value(mapped_label, value):
            continue

        # Confidence varies by label type
        confidence_map = {
            "API_KEY": 0.93,
            "EMAIL": 0.90,
            "PHONE": 0.80,
            "DATE": 0.75,
            "IP": 0.88,
            "SOCIALNUMBER": 0.85,
        }
        confidence = confidence_map.get(mapped_label, 0.72)
        evidence.append(Evidence(vs, ve, mapped_label, "regex", confidence))
        existing_spans.add((vs, ve, mapped_label))

    return evidence



def org_list_evidence(text: str, known_orgs: Optional[List[str]] = None) -> List[Evidence]:
    """Collect evidence for known organizations via fast exact-match lookup.

    Provides rapid detection for organizations in a pre-compiled list,
    useful for client-specific organization detection with minimal latency.
    Searches are case-insensitive to match common business name variations.

    Args:
        text: Input text to analyze.
        known_orgs: List of organization names to match (case-insensitive).
                   If None or empty, returns empty evidence list.

    Returns:
        List of org_list-origin `Evidence` records with high confidence (0.95).
    """
    evidence: List[Evidence] = []

    if not known_orgs:
        return evidence

    for org_name in known_orgs:
        if not org_name.strip():
            continue
        # Use re.escape to handle special regex chars, re.IGNORECASE for case-insensitive
        pattern = re.escape(org_name.strip())
        for match in re.finditer(pattern, text, re.IGNORECASE):
            evidence.append(Evidence(
                start=match.start(),
                end=match.end(),
                label="ORG",
                source="org_list",
                confidence=0.95,  # Very high confidence for exact matches
            ))

    return evidence

# -----------------------------
# Placeholder filter
# -----------------------------

# Template placeholder patterns that are NOT real PII.
# Examples: [Your Name], [Your Position], [Name], [Position], [Date], etc.
_PLACEHOLDER_RE = re.compile(
    r"^\s*[\[\(<\{]\s*(?:Your\s+)?[A-Za-z][A-Za-z\s]{0,30}\s*[\]\)>\}]\s*$"
    r"|^LASTNAME\d+_[A-Z]$",
    re.IGNORECASE,
)

def _is_placeholder(entity: "Entity") -> bool:
    """Return True if the entity text is a template placeholder, not real PII."""
    return bool(_PLACEHOLDER_RE.match(entity.text.strip()))
