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
