from taivium.engine import find_recurrences, Entity

def test_recurrence_cap(monkeypatch):
    # Create a text with 2000 repeated 'aaaaa@aaaaa.com' substrings
    email = "aaaaa@aaaaa.com"
    text = " ".join([email] * 2000)
    # Canonical entity is 'aaaaa@aaaaa.com' at the start
    canonical = [Entity(text=email, label="EMAIL", start=0, end=len(email), source="canonical")]
    # Should cap at 1000 recurrences
    recurrences = find_recurrences(text, canonical)
    # There are 2000 possible, but cap is 1000
    assert len(recurrences) == 1000
    # All recurrences should be for the email
    for r in recurrences:
        assert r.text == email
        assert r.label == "EMAIL"
        assert r.source == "recurrence"
    # Should not overlap with canonical
    for r in recurrences:
        assert not (r.start == 0 and r.end == len(email))  # skip canonical
    # Should be strictly non-overlapping
    positions = sorted((r.start, r.end) for r in recurrences)
    for i in range(1, len(positions)):
        assert positions[i-1][1] <= positions[i][0]

def test_min_span_length():
    # Should not generate recurrences for spans shorter than 3
    text = "AA AA AA"
    canonical = [Entity(text="AA", label="ORG", start=0, end=2, source="canonical")]
    recurrences = find_recurrences(text, canonical)
    assert recurrences == []
