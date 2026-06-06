'''
Unit tests for taivium.utility module.'''
import pytest
from taivium import engine as eng
from taivium import utility as utility

# --- GLiNER Model Caching Tests ---
class TestGetGlinerModel:
    """Test suite for get_gliner_model() function."""

    def test_get_gliner_model_returns_model_instance(self):
        """get_gliner_model() should return a GLiNER model instance."""
        utility.get_gliner_model.cache_clear()
        model = utility.get_gliner_model()
        assert model is not None
        assert hasattr(model, 'predict_entities')

    def test_get_gliner_model_caches_instance(self):
        """get_gliner_model() should return the same cached instance on repeated calls."""
        utility.get_gliner_model.cache_clear()
        model1 = utility.get_gliner_model()
        model2 = utility.get_gliner_model()
        assert model1 is model2

# --- spaCy OSError path (get_spacy_model) ---
def test_spacy_model_oserror(monkeypatch):
    def fake_load(*a, **k):
        raise OSError("model not found")
    monkeypatch.setattr(utility.spacy, "load", fake_load)
    with pytest.raises(OSError, match="spaCy model 'en_core_web_sm' not found"):
        utility.get_spacy_model.cache_clear()
        utility.get_spacy_model()


# --- _verify_onnx_provider exception handling (line 47-50) ---
def test_verify_onnx_provider_exception_fallback(monkeypatch):
    """_verify_onnx_provider catches exceptions and returns fallback string."""
    class BadModel:
        pass
    
    def bad_get_available_providers(*args):
        raise RuntimeError("ONNX error")
    
    monkeypatch.setattr(utility.rt, "get_available_providers", bad_get_available_providers)
    result = utility._verify_onnx_provider(BadModel())
    assert result == "CPUExecutionProvider (fallback)"
