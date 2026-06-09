'''
Unit tests for taivium.utility module.'''
import pytest
import taivium.utility as utility


# --- spaCy OSError path (get_spacy_model) ---
def test_spacy_model_oserror(monkeypatch):
    def fake_load(*a, **k):
        raise OSError("model not found")
    monkeypatch.setattr(utility.spacy, "load", fake_load)
    with pytest.raises(OSError, match="spaCy model 'en_core_web_sm' not found"):
        utility.get_spacy_model.cache_clear()
        utility.get_spacy_model()
