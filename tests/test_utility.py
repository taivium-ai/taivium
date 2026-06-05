from taivium import engine as eng

# --- GLiNER Model Caching Tests ---
class TestGetGlinerModel:
    """Test suite for get_gliner_model() function."""

    def test_get_gliner_model_returns_model_instance(self):
        """get_gliner_model() should return a GLiNER model instance."""
        eng.get_gliner_model.cache_clear()
        model = eng.get_gliner_model()
        assert model is not None
        assert hasattr(model, 'predict_entities')

    def test_get_gliner_model_caches_instance(self):
        """get_gliner_model() should return the same cached instance on repeated calls."""
        eng.get_gliner_model.cache_clear()
        model1 = eng.get_gliner_model()
        model2 = eng.get_gliner_model()
        assert model1 is model2
