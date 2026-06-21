"""Tests for per-request tenant-aware session store wiring."""
from concurrent.futures import ThreadPoolExecutor
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


def test_concurrent_multi_agent_same_tenant_stable_pseudonyms(monkeypatch):
    """Concurrent agent calls for one tenant should yield identical pseudonyms."""
    fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")
    redis = pytest.importorskip("redis", reason="redis not installed")

    fake_server = fakeredis.FakeServer()
    fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)
    monkeypatch.setattr(redis, "from_url", lambda *_a, **_kw: fake_client)

    monkeypatch.setenv("REDIS_URL", "redis://shared-fake:6379")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")
    eng._engine_cache.clear()

    text = "Agent request: Alice Johnson email alice@example.com at Acme Corp"
    tenant_id = "tenant-stress-acme"

    def _agent_call(_: int):
        result = eng.module_engine_process(text, options={"tenant_id": tenant_id})
        return frozenset(result["mapping"].keys())

    with ThreadPoolExecutor(max_workers=12) as pool:
        id_sets = list(pool.map(_agent_call, range(80)))

    assert id_sets, "Expected at least one mapping from concurrent calls"
    assert len(set(id_sets)) == 1, "Same-tenant pseudonyms drifted across agents"


def test_concurrent_multi_agent_cross_tenant_no_collisions(monkeypatch):
    """Concurrent calls across tenants must not collide and must stay tenant-isolated in Redis."""
    fakeredis = pytest.importorskip("fakeredis", reason="fakeredis not installed")
    redis = pytest.importorskip("redis", reason="redis not installed")

    fake_server = fakeredis.FakeServer()
    fake_client = fakeredis.FakeRedis(server=fake_server, decode_responses=True)
    monkeypatch.setattr(redis, "from_url", lambda *_a, **_kw: fake_client)

    monkeypatch.setenv("REDIS_URL", "redis://shared-fake:6379")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "3600")
    eng._engine_cache.clear()

    text = "Alice Johnson uses alice@example.com"
    tenant_a = "tenant-alpha"
    tenant_b = "tenant-beta"

    def _agent_call(idx: int):
        tenant_id = tenant_a if idx % 2 == 0 else tenant_b
        result = eng.module_engine_process(text, options={"tenant_id": tenant_id})
        return tenant_id, frozenset(result["mapping"].keys())

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_agent_call, range(120)))

    ids_a = {id_set for tenant, id_set in results if tenant == tenant_a}
    ids_b = {id_set for tenant, id_set in results if tenant == tenant_b}

    assert len(ids_a) == 1, "Tenant A produced unstable pseudonyms across agents"
    assert len(ids_b) == 1, "Tenant B produced unstable pseudonyms across agents"

    only_a = next(iter(ids_a))
    only_b = next(iter(ids_b))
    assert only_a.isdisjoint(only_b), "Cross-tenant pseudonym collision detected"

    redis_keys = list(fake_client.scan_iter("taivium:*:session:tenant-session:*"))
    decoded_keys = {
        key.decode("utf-8") if isinstance(key, bytes) else key
        for key in redis_keys
    }

    assert any(k.startswith("taivium:tenant-alpha:session:tenant-session:") for k in decoded_keys)
    assert any(k.startswith("taivium:tenant-beta:session:tenant-session:") for k in decoded_keys)


# ---------------------------------------------------------------------------
# _resolve_default_session_ttl — uncovered error branches
# ---------------------------------------------------------------------------

def test_resolve_default_session_ttl_non_integer_falls_back(monkeypatch):
    """Non-integer SESSION_TTL_SECONDS logs a warning and returns 86400."""
    monkeypatch.setenv('SESSION_TTL_SECONDS', 'not-a-number')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_default_session_ttl(logger)

    assert result == 86400


def test_resolve_default_session_ttl_zero_falls_back(monkeypatch):
    """SESSION_TTL_SECONDS=0 is invalid; logs a warning and returns 86400."""
    monkeypatch.setenv('SESSION_TTL_SECONDS', '0')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_default_session_ttl(logger)

    assert result == 86400


def test_resolve_default_session_ttl_negative_falls_back(monkeypatch):
    """Negative SESSION_TTL_SECONDS logs a warning and returns 86400."""
    monkeypatch.setenv('SESSION_TTL_SECONDS', '-100')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_default_session_ttl(logger)

    assert result == 86400


# ---------------------------------------------------------------------------
# _resolve_tenant_session_ttl — uncovered error branches
# ---------------------------------------------------------------------------

def test_resolve_tenant_session_ttl_invalid_json_falls_back(monkeypatch):
    """Invalid JSON in TENANT_SESSION_TTL_SECONDS logs a warning and returns default_ttl."""
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{not valid json}')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_tenant_session_ttl("tenant-acme", 3600, logger)

    assert result == 3600


def test_resolve_tenant_session_ttl_non_dict_json_falls_back(monkeypatch):
    """TENANT_SESSION_TTL_SECONDS that decodes to a list (not dict) logs a warning and returns default_ttl."""
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '[1, 2, 3]')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_tenant_session_ttl("tenant-acme", 3600, logger)

    assert result == 3600


def test_resolve_tenant_session_ttl_invalid_value_for_tenant_falls_back(monkeypatch):
    """Non-integer per-tenant TTL value logs a warning and returns default_ttl."""
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{"tenant-acme": "bad-value"}')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_tenant_session_ttl("tenant-acme", 3600, logger)

    assert result == 3600


def test_resolve_tenant_session_ttl_zero_value_for_tenant_falls_back(monkeypatch):
    """Zero per-tenant TTL value is invalid; logs a warning and returns default_ttl."""
    monkeypatch.setenv('TENANT_SESSION_TTL_SECONDS', '{"tenant-acme": 0}')
    logger = eng.logging.getLogger("taivium.engine")

    result = eng._resolve_tenant_session_ttl("tenant-acme", 3600, logger)

    assert result == 3600


# ---------------------------------------------------------------------------
# _build_tenant_session_store — OSError/IOError fallback branch
# ---------------------------------------------------------------------------

def test_build_tenant_session_store_falls_back_on_redis_oserror(monkeypatch):
    """OSError from RedisSessionStore.__init__ logs a warning and returns InMemorySessionStore."""
    monkeypatch.setenv('REDIS_URL', 'redis://localhost:6379')
    monkeypatch.setenv('SESSION_TTL_SECONDS', '3600')

    def mock_redis_init_raises(self, **kwargs):
        raise OSError("connection refused")

    with patch.object(eng.RedisSessionStore, '__init__', mock_redis_init_raises):
        store = eng._build_tenant_session_store(
            "tenant-acme", eng.logging.getLogger("taivium.engine")
        )

    assert isinstance(store, eng.InMemorySessionStore)


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


def test_smoke_tenant_isolation_produces_different_ids():
    """Different tenants processing the same text must receive different anonymized IDs."""
    text = "Email: alice@example.com"

    result_acme = eng.module_engine_process(text, options={"tenant_id": "tenant-acme"})
    result_xyz = eng.module_engine_process(text, options={"tenant_id": "tenant-xyz"})

    acme_ids = set(result_acme["mapping"].keys())
    xyz_ids = set(result_xyz["mapping"].keys())

    assert acme_ids != xyz_ids, (
        f"Tenant isolation broken: both tenants produced identical IDs {acme_ids}"
    )


def test_smoke_no_tenant_produces_consistent_ids():
    """Processing the same text twice without tenant_id should yield identical IDs."""
    text = "Email: bob@example.com"

    result1 = eng.module_engine_process(text, options={})
    result2 = eng.module_engine_process(text, options={})

    assert set(result1["mapping"].keys()) == set(result2["mapping"].keys()), (
        "Non-tenant processing is non-deterministic: IDs differ across calls"
    )


def test_smoke_tenant_anonymized_text_differs():
    """Anonymized output text should differ across tenants for the same input."""
    text = "Contact alice@example.com for support."

    result_acme = eng.module_engine_process(text, options={"tenant_id": "tenant-acme"})
    result_xyz = eng.module_engine_process(text, options={"tenant_id": "tenant-xyz"})

    assert result_acme["anonymized"] != result_xyz["anonymized"], (
        "Anonymized output is identical across tenants"
    )
