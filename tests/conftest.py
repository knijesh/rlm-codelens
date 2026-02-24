"""
Pytest configuration and fixtures for RLM-Codelens tests.

This module provides shared fixtures and configuration for all tests.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture providing temporary output directory for tests."""
    path: Path = tmp_path_factory.mktemp("test_outputs")
    return path
