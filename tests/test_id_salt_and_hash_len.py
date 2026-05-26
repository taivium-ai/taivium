import pytest
from tarvium.engine import Tarvium, reverse_transform

def test_id_salt_scoping():
    text = "Alice and Bob are friends. Alice's email is alice@example.com."
    # Global (legacy) IDs
    engine_global = Tarvium()
    result_global = engine_global.process(text)
    ids_global = {e['id'] for e in result_global['entities']}

    # Tenant-scoped IDs
    engine_tenant = Tarvium(id_salt="tenantA")
    result_tenant = engine_tenant.process(text)
    ids_tenant = {e['id'] for e in result_tenant['entities']}

    # Session-scoped IDs
    engine_session = Tarvium(id_salt="sessionB")
    result_session = engine_session.process(text)
    ids_session = {e['id'] for e in result_session['entities']}

    # IDs should differ across salt scopes
    assert ids_global != ids_tenant
    assert ids_global != ids_session
    assert ids_tenant != ids_session

    # All should be able to reverse-transform
    for result in [result_global, result_tenant, result_session]:
        roundtrip = reverse_transform(result['anonymized'], result['mapping'])
        assert roundtrip == text

def test_id_hash_len():
    text = "Charlie works at Acme Corp."
    engine_short = Tarvium(id_hash_len=8)
    engine_long = Tarvium(id_hash_len=32)
    result_short = engine_short.process(text)
    result_long = engine_long.process(text)
    ids_short = [e['id'] for e in result_short['entities']]
    ids_long = [e['id'] for e in result_long['entities']]
    # All IDs should have the correct hash length
    for sid in ids_short:
        assert len(sid.split('_')[1]) == 8
    for lid in ids_long:
        assert len(lid.split('_')[1]) == 32
    # Both should roundtrip
    assert reverse_transform(result_short['anonymized'], result_short['mapping']) == text
    assert reverse_transform(result_long['anonymized'], result_long['mapping']) == text
