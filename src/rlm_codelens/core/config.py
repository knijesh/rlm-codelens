"""
Configuration module for RLM-Codelens.

This module provides centralized configuration management using
environment variables and configuration files.

Example:
    >>> from rlm_codelens.core.config import Config
    >>> config = Config()
    >>> print(config.github_token)
    >>> print(config.openai_api_key)
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Centralized configuration management.

    This class loads configuration from environment variables
    and provides a clean interface for accessing settings.

    All sensitive values (API keys, tokens) are loaded from
    environment variables and never hardcoded.

    Attributes:
        github_token: GitHub personal access token
        openai_api_key: OpenAI API key
        database_url: PostgreSQL connection string
        budget_limit: Maximum budget for API calls
        rlm_model: OpenAI model for RLM calls
        embedding_model: Model for embeddings

    Example:
        >>> config = Config()
        >>> print(f"Budget: ${config.budget_limit}")
        >>> print(f"Model: {config.rlm_model}")

    Environment Variables:
        GITHUB_TOKEN: GitHub personal access token
        OPENAI_API_KEY: OpenAI API key
        DATABASE_URL: PostgreSQL connection URL
        BUDGET_LIMIT: Maximum budget in USD (default: 50.0)
        RLM_MODEL: OpenAI model name (default: gpt-3.5-turbo)
        EMBEDDING_MODEL: Embedding model (default: text-embedding-3-small)
    """

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Load from environment
        self.github_token: str = os.getenv("GITHUB_TOKEN", "")
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.database_url: str = os.getenv(
            "DATABASE_URL", "postgresql://localhost/rlm_codelens"
        )

        # Analysis settings
        self.budget_limit: float = float(os.getenv("BUDGET_LIMIT", "50.0"))
        self.rlm_model: str = os.getenv("RLM_MODEL", "gpt-3.5-turbo")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # Derived paths
        self.base_dir: Path = Path(__file__).parent.parent.parent
        self.output_dir: Path = self.base_dir / "outputs"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> bool:
        """Validate that required configuration is present.

        Checks that all required environment variables are set.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If required configuration is missing

        Example:
            >>> config = Config()
            >>> try:
            ...     config.validate()
            ...     print("Configuration valid")
            ... except ValueError as e:
            ...     print(f"Invalid: {e}")
        """
        errors = []

        if not self.github_token:
            errors.append("GITHUB_TOKEN is required")

        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY is required")

        if self.budget_limit <= 0:
            errors.append("BUDGET_LIMIT must be positive")

        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))

        return True

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config("
            f"budget_limit=${self.budget_limit}, "
            f"rlm_model={self.rlm_model}, "
            f"db={self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url}"
            f")"
        )
