import taivium

def test_version_exists():
    assert hasattr(taivium, "__version__"), "taivium.__version__ should exist"
    assert isinstance(taivium.__version__, str), "taivium.__version__ should be a string"
    assert len(taivium.__version__) > 0, "taivium.__version__ should not be empty"
