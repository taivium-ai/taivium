'''
Unit tests for taivium.utility module.'''
import pytest
from taivium import utility as utility
from taivium.backend import transformer_gliner as gliner_backend

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

# --- _verify_onnx_provider provider selection paths ---
def test_verify_onnx_provider_coreml(monkeypatch):
    monkeypatch.setattr(
        utility.rt,
        "get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    result = utility._verify_onnx_provider(object())
    assert result == "CoreMLExecutionProvider"


def test_verify_onnx_provider_cuda(monkeypatch):
    monkeypatch.setattr(
        utility.rt,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    result = utility._verify_onnx_provider(object())
    assert result == "CUDAExecutionProvider"


def test_verify_onnx_provider_cpu(monkeypatch):
    monkeypatch.setattr(
        utility.rt,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    result = utility._verify_onnx_provider(object())
    assert result == "CPUExecutionProvider"


# --- get_gliner_model provider selection logic ---
def test_get_gliner_model_coreml_selected(monkeypatch):
    utility.get_gliner_model.cache_clear()
    gliner_backend.get_gliner_model.cache_clear()

    monkeypatch.setattr(
        gliner_backend.rt,
        "get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )

    monkeypatch.setattr(gliner_backend, "snapshot_download", lambda **k: "/tmp/model")

    class DummyModel:
        def predict_entities(self): pass

    monkeypatch.setattr(
        gliner_backend.GLiNER,
        "from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    monkeypatch.setattr(
        gliner_backend, "_verify_onnx_provider", lambda m: "CoreMLExecutionProvider"
    )

    model = utility.get_gliner_model()
    assert hasattr(model, "predict_entities")


def test_get_gliner_model_cuda_selected(monkeypatch):
    utility.get_gliner_model.cache_clear()
    gliner_backend.get_gliner_model.cache_clear()

    monkeypatch.setattr(
        gliner_backend.rt,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider"],
    )

    monkeypatch.setattr(gliner_backend, "snapshot_download", lambda **k: "/tmp/model")

    class DummyModel:
        def predict_entities(self): pass

    monkeypatch.setattr(
        gliner_backend.GLiNER,
        "from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    monkeypatch.setattr(
        gliner_backend, "_verify_onnx_provider", lambda m: "CUDAExecutionProvider"
    )

    model = utility.get_gliner_model()
    assert hasattr(model, "predict_entities")


# --- mismatch warning path ---
def test_get_gliner_model_provider_mismatch(monkeypatch):
    utility.get_gliner_model.cache_clear()
    gliner_backend.get_gliner_model.cache_clear()

    monkeypatch.setattr(
        gliner_backend.rt,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider"],
    )

    monkeypatch.setattr(gliner_backend, "snapshot_download", lambda **k: "/tmp/model")

    class DummyModel:
        def predict_entities(self): pass

    monkeypatch.setattr(
        gliner_backend.GLiNER,
        "from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    # Force mismatch
    monkeypatch.setattr(
        gliner_backend, "_verify_onnx_provider", lambda m: "CPUExecutionProvider"
    )

    model = utility.get_gliner_model()
    assert hasattr(model, "predict_entities")


# --- onnxruntime ImportError fallback ---
def test_get_gliner_model_no_onnxruntime(monkeypatch):
    utility.get_gliner_model.cache_clear()
    gliner_backend.get_gliner_model.cache_clear()

    def raise_import_error():
        raise ImportError()

    monkeypatch.setattr(
        gliner_backend.rt,
        "get_available_providers",
        lambda: raise_import_error(),
    )

    monkeypatch.setattr(gliner_backend, "snapshot_download", lambda **k: "/tmp/model")

    class DummyModel:
        def predict_entities(self): pass

    monkeypatch.setattr(
        gliner_backend.GLiNER,
        "from_pretrained",
        lambda *a, **k: DummyModel(),
    )

    monkeypatch.setattr(
        gliner_backend, "_verify_onnx_provider", lambda m: "CPUExecutionProvider"
    )

    model = utility.get_gliner_model()
    assert hasattr(model, "predict_entities")


# --- ONNX failure fallback to transformer ---
def test_get_gliner_model_fallback_to_transformer(monkeypatch):
    utility.get_gliner_model.cache_clear()
    gliner_backend.get_gliner_model.cache_clear()

    monkeypatch.setattr(
        gliner_backend.rt,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    monkeypatch.setattr(
        gliner_backend, "snapshot_download", lambda **k: "/tmp/model"
    )

    def raise_error(*a, **k):
        raise RuntimeError("ONNX load failed")

    monkeypatch.setattr(gliner_backend.GLiNER, "from_pretrained", raise_error)

    class FallbackModel:
        def predict_entities(self): pass

    # Second call (fallback)
    def fallback_loader(model_name):
        assert model_name == "knowledgator/gliner-pii-small-v1.0"
        return FallbackModel()

    monkeypatch.setattr(
        gliner_backend.GLiNER,
        "from_pretrained",
        lambda *a, **k: raise_error() if "onnx" in str(a) else fallback_loader(a[0]),
    )

    model = utility.get_gliner_model()
    assert hasattr(model, "predict_entities")
