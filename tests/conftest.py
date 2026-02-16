"""
Pytest configuration and fixtures for RLM-Codelens tests.

This module provides shared fixtures and configuration for all tests.
"""

import pytest


@pytest.fixture(scope="session")
def test_output_dir(tmp_path_factory):
    """Fixture providing temporary output directory for tests."""
    return tmp_path_factory.mktemp("test_outputs")
