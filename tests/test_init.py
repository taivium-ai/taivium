"""Tests for taivium.__init__.py version fallback."""
import sys
from unittest.mock import patch
import pytest


def test_version_fallback_on_package_not_found():
    """When PackageNotFoundError is raised, __version__ defaults to '0.0.0'."""
    # Remove taivium from sys.modules to force reimport
    if 'taivium' in sys.modules:
        del sys.modules['taivium']
    
    # Mock importlib.metadata.version to raise PackageNotFoundError
    with patch('importlib.metadata.version') as mock_version:
        from importlib.metadata import PackageNotFoundError
        mock_version.side_effect = PackageNotFoundError("taivium")
        
        # Reimport taivium
        import importlib
        import taivium
        importlib.reload(taivium)
        
        # Should have fallback version
        assert taivium.__version__ == "0.0.0"
