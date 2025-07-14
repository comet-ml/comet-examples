"""Pytest configuration and shared fixtures."""

import pytest
import os
import tempfile
import shutil
from unittest.mock import patch

from src.call_summarizer.models.models import CallCategoryConfig, VectorStoreConfig


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_categories_config_path(temp_dir):
    """Provide a temporary path for categories config file."""
    return os.path.join(temp_dir, "test_categories.json")


@pytest.fixture
def test_vector_store_path(temp_dir):
    """Provide a temporary path for vector store."""
    return os.path.join(temp_dir, "test_vector_store")


@pytest.fixture
def sample_category_config():
    """Provide a sample category configuration."""
    return CallCategoryConfig(
        name="test_category", description="A test category for unit tests", prompt_template="Summarize this test call: {transcript}"
    )


@pytest.fixture
def sample_vector_store_config(test_vector_store_path):
    """Provide a sample vector store configuration."""
    return VectorStoreConfig(persist_dir=test_vector_store_path, collection_name="test_collection")


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing when needed."""
    # Only mock if we're in a test that explicitly requests mocking
    # This allows real environment variables to be used for Opik integration
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key", "OPIK_API_KEY": "test-opik-key", "OPIK_WORKSPACE": "test-workspace"}):
        yield


@pytest.fixture
def mock_settings():
    """Mock application settings for testing when needed."""
    # Only mock if we're in a test that explicitly requests mocking
    with patch("src.call_summarizer.config.settings") as mock_settings:
        mock_settings.openai_api_key = "test-openai-key"
        mock_settings.opik_api_key = "test-opik-key"
        mock_settings.opik_workspace = "test-workspace"
        mock_settings.vector_store_path = "/tmp/test_vector_store"
        mock_settings.categories_config_path = "/tmp/test_categories.json"
        yield mock_settings


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Skip tests that require external services in CI
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # Add slow marker to tests that might be slow
        if "integration" in item.name or "workflow" in item.name:
            item.add_marker(pytest.mark.slow)
