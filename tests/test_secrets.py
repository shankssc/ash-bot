"""Test secrets module instantiation."""

from app.core.secrets import Secrets, secrets


def test_secrets_instantiation():
    """Verify secrets module can be instantiated without errors."""
    test_secrets = Secrets()
    assert test_secrets is not None
    assert repr(test_secrets) == "Secrets(<redacted>)"
    assert str(test_secrets) == "Secrets(<redacted>)"


def test_global_secrets_instance():
    """Verify global secrets instance exists."""
    assert secrets is not None
    assert repr(secrets) == "Secrets(<redacted>)"


def test_secret_retrieval():
    """Verify secret retrieval method works."""
    test_secrets = Secrets(AWS_ACCESS_KEY_ID="test-key")
    value = test_secrets.get_secret_value("AWS_ACCESS_KEY_ID")
    assert value == "test-key"
