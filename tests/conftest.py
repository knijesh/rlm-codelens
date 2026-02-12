"""
Pytest configuration and fixtures for RLM-Codelens tests.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock


@pytest.fixture
def sample_repository_data():
    """Fixture providing sample repository data for testing."""
    return pd.DataFrame(
        {
            "number": [1, 2, 3, 4, 5],
            "title": [
                "CUDA memory error on GPU",
                "Feature request: Add quantization",
                "Documentation typo",
                "Performance regression",
                "Bug in autograd",
            ],
            "body": [
                "Getting OOM error when using CUDA",
                "Would be great to have INT8 quantization",
                "Typo in the getting started guide",
                "Training is 50% slower in v2.0",
                "Gradient computation is wrong",
            ],
            "labels": [
                ["bug", "cuda"],
                ["feature", "performance"],
                ["documentation"],
                ["bug", "performance"],
                ["bug", "autograd"],
            ],
            "type": ["issue", "issue", "issue", "issue", "pr"],
            "author": ["user1", "user2", "user3", "user1", "user2"],
            "created_at": pd.date_range("2024-01-01", periods=5),
        }
    )


@pytest.fixture
def sample_cluster_data():
    """Fixture providing sample cluster data."""
    return pd.DataFrame(
        {
            "cluster_id": [0, 0, 0, 1, 1, 2],
            "number": [1, 2, 3, 4, 5, 6],
            "title": [
                "CUDA error",
                "GPU memory issue",
                "CUDA out of memory",
                "Quantization feature",
                "INT8 support",
                "Documentation update",
            ],
            "labels": [
                ["bug", "cuda"],
                ["bug", "cuda"],
                ["bug", "cuda"],
                ["feature"],
                ["feature"],
                ["docs"],
            ],
            "type": ["issue", "issue", "issue", "issue", "issue", "pr"],
            "author": ["user1", "user2", "user1", "user3", "user3", "user1"],
        }
    )


@pytest.fixture
def mock_db_manager():
    """Fixture providing mocked database manager."""
    mock = MagicMock()
    mock.load_dataframe.return_value = pd.DataFrame()
    mock.save_dataframe.return_value = None
    mock.list_tables.return_value = []
    return mock


@pytest.fixture
def test_config_dict():
    """Fixture providing test configuration."""
    return {
        "github_token": "ghp_test_token",
        "openai_api_key": "sk_test_key",
        "database_url": "postgresql://localhost/test_db",
        "budget_limit": 10.0,
        "model": "gpt-3.5-turbo",
    }


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Fixture providing temporary output directory for tests."""
    return tmp_path_factory.mktemp("test_outputs")
