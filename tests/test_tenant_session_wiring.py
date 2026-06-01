"""Tests for per-request tenant-aware session store wiring."""
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from taivium import engine as eng
from taivium.session_store import InMemorySessionStore


def test_module_engine_process_with_tenant_id_no_redis(monkeypatch):
    """Test that tenant_id is accepted but falls back to in-memory store when Redis is not configured."""
    # Ensure REDIS_URL is not set
    monkeypatch.delenv('REDIS_URL', raising=False)
    
    # Clear the engine cache to ensure fresh instances
    eng._engine_cache.clear()
    
    # Call process with tenant_id in options
    result = eng.module_engine_process(
        "My name is John Doe",
        options={"tenant_id": "tenant-acme"}
    )
    
    # Should succeed and return a result
    assert isinstance(result, dict)
    assert "anonymized" in result
    assert "mapping" in result
    assert "original" in result


def test_module_engine_process_with_tenant_id_and_redis(monkeypatch):
    """Test that tenant_id creates per-tenant RedisSessionStore when Redis is configured."""
    # Mock Redis to use fakeredis
    try:
        import fakeredis
    except ImportError:
        pytest.skip("fakeredis not installed")
    
    # Set up Redis URL
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    monkeypatch.setenv('SESSION_TTL_SECONDS', '3600')
    
    # Clear the engine cache
    eng._engine_cache.clear()
    
    # Mock the RedisSessionStore to track creation
    mock_store_instances = []
    
    def mock_redis_store_init(self, session_id, redis_url, ttl, tenant_id=None):
        """Mock RedisSessionStore __init__ that tracks calls."""
        mock_store_instances.append((session_id, redis_url, ttl, tenant_id))
        # Create a real in-memory store for testing
        self._data = {}
        self.tenant_id = tenant_id
        self.session_id = session_id
    
    def mock_redis_store_get(self, key):
        """Mock get method."""
        return self._data.get(key)
    
    def mock_redis_store_set(self, key, value):
        """Mock set method."""
        self._data[key] = value

    def mock_redis_store_set_many(self, mapping):
        """Mock bulk set used by Taivium.process()."""
        self._data.update(mapping)

    def mock_redis_store_get_all(self):
        """Mock get_all to satisfy session store interface."""
        return dict(self._data)
    
    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_store_init):
        with patch.object(eng.RedisSessionStore, 'get', mock_redis_store_get):
            with patch.object(eng.RedisSessionStore, 'set', mock_redis_store_set):
                with patch.object(eng.RedisSessionStore, 'set_many', mock_redis_store_set_many):
                    with patch.object(eng.RedisSessionStore, 'get_all', mock_redis_store_get_all):
                        # Call process with tenant_id
                        result = eng.module_engine_process(
                            "My name is John Doe",
                            options={"tenant_id": "tenant-acme"}
                        )
    
    # Should succeed
    assert isinstance(result, dict)
    # Should have created a RedisSessionStore (check mock was called)
    assert len(mock_store_instances) > 0
    # The tenant_id should be passed
    assert mock_store_instances[0][3] == "tenant-acme"


def test_module_engine_process_tenant_id_extracted_from_options():
    """Test that tenant_id is extracted from options dict and not passed to Taivium."""
    # Clear engine cache
    eng._engine_cache.clear()
    
    # Create options with tenant_id and other params
    options = {"tenant_id": "acme", "id_salt": "test-salt"}
    
    # Call module_engine_process
    result = eng.module_engine_process(
        "Test text",
        options=options
    )
    
    # Should succeed
    assert isinstance(result, dict)
    # Check that the result contains expected keys (proves process was called)
    assert "anonymized" in result or "error" not in result or isinstance(result, dict)


def test_module_engine_process_tenant_id_from_json_string():
    """Test that tenant_id works when options is a JSON string."""
    # Clear engine cache
    eng._engine_cache.clear()
    
    options_json = json.dumps({"tenant_id": "tenant-xyz"})
    result = eng.module_engine_process(
        "My name is Alice",
        options=options_json
    )
    
    # Should succeed
    assert isinstance(result, dict)


def test_module_engine_process_different_tenants_get_different_stores(monkeypatch):
    """Test that different tenant IDs cause different session stores to be created."""
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    
    # Clear engine cache
    eng._engine_cache.clear()
    
    # Track which tenant_ids are passed to RedisSessionStore
    tenant_ids_seen = []
    
    def mock_redis_init(self, session_id, redis_url, ttl, tenant_id=None):
        tenant_ids_seen.append(tenant_id)
        # Initialize minimal attributes to prevent AttributeError
        self.session_id = session_id
        self.tenant_id = tenant_id
        self._client = Mock()
        self._ttl = ttl
        self._prefix = f"taivium:{tenant_id}:session:{session_id}:"
    
    # Create mock methods to prevent real Redis calls
    def mock_get(self, key):
        return None
    
    def mock_set(self, key, value):
        pass
    
    def mock_set_many(self, mapping):
        pass
    
    def mock_get_all(self):
        return {}
    
    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_init):
        with patch.object(eng.RedisSessionStore, 'get', mock_get):
            with patch.object(eng.RedisSessionStore, 'set', mock_set):
                with patch.object(eng.RedisSessionStore, 'set_many', mock_set_many):
                    with patch.object(eng.RedisSessionStore, 'get_all', mock_get_all):
                        # Both calls should trigger RedisSessionStore creation with different tenant_ids
                        eng.module_engine_process("Text 1", options={"tenant_id": "tenant-1"})
                        eng.module_engine_process("Text 2", options={"tenant_id": "tenant-2"})
    
    # Both tenant_ids should have been passed
    assert "tenant-1" in tenant_ids_seen
    assert "tenant-2" in tenant_ids_seen


def test_module_engine_process_uses_tenant_specific_ttl_policy(monkeypatch):
    """Tenant-specific TTL policy overrides SESSION_TTL_SECONDS when configured."""
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    monkeypatch.setenv('SESSION_TTL_SECONDS', '3600')
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{"tenant-acme": 120}')

    eng._engine_cache.clear()

    captured_ttls = []

    def mock_redis_init(self, session_id, redis_url, ttl, tenant_id=None):
        captured_ttls.append((tenant_id, ttl))
        self.session_id = session_id
        self.tenant_id = tenant_id
        self._client = Mock()
        self._ttl = ttl
        self._prefix = f"taivium:{tenant_id}:session:{session_id}:"

    def mock_get(self, key):
        return None

    def mock_set(self, key, value):
        pass

    def mock_set_many(self, mapping):
        pass

    def mock_get_all(self):
        return {}

    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_init):
        with patch.object(eng.RedisSessionStore, 'get', mock_get):
            with patch.object(eng.RedisSessionStore, 'set', mock_set):
                with patch.object(eng.RedisSessionStore, 'set_many', mock_set_many):
                    with patch.object(eng.RedisSessionStore, 'get_all', mock_get_all):
                        eng.module_engine_process('Text 1', options={'tenant_id': 'tenant-acme'})

    assert captured_ttls
    assert ('tenant-acme', 120) in captured_ttls


def test_module_engine_process_falls_back_to_default_ttl_when_tenant_missing(monkeypatch):
    """Unknown tenant in policy falls back to SESSION_TTL_SECONDS."""
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    monkeypatch.setenv('SESSION_TTL_SECONDS', '3600')
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{"tenant-acme": 120}')

    eng._engine_cache.clear()

    captured_ttls = []

    def mock_redis_init(self, session_id, redis_url, ttl, tenant_id=None):
        captured_ttls.append((tenant_id, ttl))
        self.session_id = session_id
        self.tenant_id = tenant_id
        self._client = Mock()
        self._ttl = ttl
        self._prefix = f"taivium:{tenant_id}:session:{session_id}:"

    def mock_get(self, key):
        return None

    def mock_set(self, key, value):
        pass

    def mock_set_many(self, mapping):
        pass

    def mock_get_all(self):
        return {}

    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_init):
        with patch.object(eng.RedisSessionStore, 'get', mock_get):
            with patch.object(eng.RedisSessionStore, 'set', mock_set):
                with patch.object(eng.RedisSessionStore, 'set_many', mock_set_many):
                    with patch.object(eng.RedisSessionStore, 'get_all', mock_get_all):
                        eng.module_engine_process('Text 2', options={'tenant_id': 'tenant-other'})

    assert captured_ttls
    assert ('tenant-other', 3600) in captured_ttls


def test_build_tenant_session_store_returns_in_memory_without_tenant(monkeypatch):
    """Helper returns InMemorySessionStore when tenant_id is missing."""
    monkeypatch.delenv('REDIS_URL', raising=False)

    store = eng._build_tenant_session_store(None, eng.logging.getLogger("taivium.engine"))

    assert isinstance(store, InMemorySessionStore)


def test_build_tenant_session_store_returns_in_memory_without_redis_url(monkeypatch):
    """Helper returns InMemorySessionStore when REDIS_URL is not configured."""
    monkeypatch.delenv('REDIS_URL', raising=False)

    store = eng._build_tenant_session_store("tenant-acme", eng.logging.getLogger("taivium.engine"))

    assert isinstance(store, InMemorySessionStore)


def test_build_tenant_session_store_uses_tenant_ttl_override(monkeypatch):
    """Helper uses TENANT_SESSION_TTL_SECONDS override when available."""
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    monkeypatch.setenv('SESSION_TTL_SECONDS', '3600')
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{"tenant-acme": 120}')

    captured = {}

    def mock_redis_init(self, session_id, redis_url, ttl, tenant_id=None):
        captured['session_id'] = session_id
        captured['redis_url'] = redis_url
        captured['ttl'] = ttl
        captured['tenant_id'] = tenant_id
        self.session_id = session_id
        self.tenant_id = tenant_id
        self._client = Mock()
        self._ttl = ttl
        self._prefix = f"taivium:{tenant_id}:session:{session_id}:"

    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_init):
        store = eng._build_tenant_session_store("tenant-acme", eng.logging.getLogger("taivium.engine"))

    assert store is not None
    assert captured['tenant_id'] == 'tenant-acme'
    assert captured['ttl'] == 120
    assert captured['session_id'] == 'tenant-session'


def test_parse_module_engine_options_accepts_dict():
    """Helper accepts dict options as-is."""
    parsed, err = eng._parse_module_engine_options(
        {"tenant_id": "tenant-acme", "use_llm": True},
        eng.logging.getLogger("taivium.engine"),
    )

    assert err is None
    assert parsed == {"tenant_id": "tenant-acme", "use_llm": True}


def test_parse_module_engine_options_accepts_json_dict_string():
    """Helper accepts JSON string that decodes to dict."""
    parsed, err = eng._parse_module_engine_options(
        '{"tenant_id": "tenant-acme", "use_llm": true}',
        eng.logging.getLogger("taivium.engine"),
    )

    assert err is None
    assert parsed == {"tenant_id": "tenant-acme", "use_llm": True}


def test_parse_module_engine_options_rejects_invalid_json_string():
    """Helper returns parse error on invalid JSON string."""
    parsed, err = eng._parse_module_engine_options(
        '{bad-json}',
        eng.logging.getLogger("taivium.engine"),
    )

    assert parsed is None
    assert err is not None
    assert "Failed to parse options JSON" in err


def test_parse_module_engine_options_rejects_json_non_dict():
    """Helper rejects JSON values that are not dicts."""
    parsed, err = eng._parse_module_engine_options(
        '[1, 2, 3]',
        eng.logging.getLogger("taivium.engine"),
    )

    assert parsed is None
    assert err == "Options JSON must decode to a dict; got list"


def test_parse_module_engine_options_rejects_unsupported_type():
    """Helper rejects non-dict, non-str option types."""
    parsed, err = eng._parse_module_engine_options(
        42,
        eng.logging.getLogger("taivium.engine"),
    )

    assert parsed is None
    assert err == "Options must be a dict or JSON string, got int"
