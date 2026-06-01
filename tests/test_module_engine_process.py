import json
from taivium.engine import module_engine_process

def test_module_engine_process_basic():
    text = "John Doe's email is john@example.com and his phone is +1234567890."
    result = module_engine_process(text)
    assert isinstance(result, dict)
    assert "entities" in result
    entities = result["entities"]
    assert any(e["label"] == "EMAIL" for e in entities)
    assert any(e["label"] == "PHONE" for e in entities)

def test_module_engine_process_with_options():
    text = "Jane works at Acme Corp. Her email is jane@acme.com."
    options = json.dumps({
        "use_transformer": False,
        "use_llm": False,
        "id_salt": "test_salt",
        "id_hash_len": 12
    })
    result = module_engine_process(text, options)
    assert isinstance(result, dict)
    assert "entities" in result
    entities = result["entities"]
    assert any(e["label"] == "EMAIL" for e in entities)
    assert any(e["label"] == "ORG" for e in entities)

def test_module_engine_process_invalid_options():
    text = "Contact: alice@wonderland.com"
    # Pass invalid JSON
    result = module_engine_process(text, "not a json")
    assert isinstance(result, dict)
    assert "error" in result
    assert "Failed to parse options JSON" in result["error"]

    # Pass JSON that is not a dict
    result = module_engine_process(text, json.dumps([1,2,3]))
    assert isinstance(result, dict)
    assert "error" in result
    assert "Options JSON must decode to a dict" in result["error"]

    # Pass empty string (should fallback to normal processing)
    result = module_engine_process(text, "")
    assert isinstance(result, dict)
    assert "entities" in result
    entities = result["entities"]
    assert any(e["label"] == "EMAIL" for e in entities)


def test_module_engine_process_options_as_dict():
    """options passed as a plain dict (not JSON string) exercises the isinstance(options, dict)
    branch and routes through _options_key with the provided values."""
    text = "Contact bob@example.org for details."
    options = {"use_transformer": False, "use_llm": False, "id_salt": "dict_test", "id_hash_len": 12}
    result = module_engine_process(text, options)
    assert isinstance(result, dict)
    assert "entities" in result
    assert any(e["label"] == "EMAIL" for e in result["entities"])
    # id_salt must be honoured — IDs should differ from the unsalted run
    unsalted = module_engine_process(text)
    salted_ids = {e["id"] for e in result["entities"]}
    unsalted_ids = {e["id"] for e in unsalted["entities"]}
    assert salted_ids != unsalted_ids


def test_module_engine_process_options_wrong_type():
    """options that is neither a dict nor a str triggers the else branch and
    returns an error dict without raising."""
    text = "reach me at carol@example.net"
    for bad_options in (42, 3.14, ["SOC2"], ("a", "b"), True):
        result = module_engine_process(text, bad_options)
        assert isinstance(result, dict), f"expected dict for options={bad_options!r}"
        assert "error" in result, f"expected 'error' key for options={bad_options!r}"
        assert "Options must be a dict or JSON string" in result["error"]
