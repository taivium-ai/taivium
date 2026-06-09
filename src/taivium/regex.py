'''
Regex detectors regex definition (PII / secrets)
'''
import re
from typing import Optional
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
     r"([a-zA-Z0-9][a-zA-Z0-9._\-]{1,31})"),
    re.IGNORECASE,
)
