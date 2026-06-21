"""Additional tests for PrivacyClient to improve code coverage."""
import sys
from unittest.mock import MagicMock, patch
import pytest
import taivium.client as client_mod
from taivium.engine import PolicyAction


def test_privacyclient_with_policy_engine_and_reset():
    """PrivacyClient with default_action creates PolicyEngine and reset_session clears mapping."""
    # Mock openai.OpenAI to avoid needing real API key
    with patch("openai.OpenAI") as mock_openai_class:
        mock_instance = MagicMock()
        mock_instance.chat = MagicMock()
        mock_instance.chat.completions = MagicMock()
        mock_openai_class.return_value = mock_instance
        
        # Test creating PrivacyClient with default_action (should create PolicyEngine, line 183)
        c = client_mod.PrivacyClient(default_action=PolicyAction.ANONYMIZE)
        assert c._pipeline.policy is not None
        
        # Test reset_session (lines 179-180: InMemorySessionStore branch)
        c2 = client_mod.PrivacyClient()
        c2._pipeline.session_store.set("id1", {"label": "EMAIL", "text": "alice@acme.com"})
        assert len(c2.session_mapping) > 0
        c2.reset_session()
        assert len(c2.session_mapping) == 0
