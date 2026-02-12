"""
Secure logging utilities for RLM-Codelens.

This module provides logging functionality that automatically
redacts sensitive information like API keys and tokens.

Example:
    >>> from rlm_codelens.utils.secure_logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing repository")  # Safe to log
    >>> logger.info(f"Config: {config}")  # Automatically redacts keys!
"""

import logging
import re
from typing import Any


class SensitiveDataFilter(logging.Filter):
    """Filter that redacts sensitive data from log records.

    Automatically redacts:
    - API keys (ghp_*, sk_*)
    - Tokens
    - Passwords
    - Database connection strings with credentials

    Example:
        >>> logger.addFilter(SensitiveDataFilter())
        >>> logger.info("Key: ghp_secret123")  # Logs: "Key: [REDACTED]"
    """

    # Patterns to redact
    SENSITIVE_PATTERNS = [
        (r"ghp_[a-zA-Z0-9]{36,}", "[GITHUB_TOKEN_REDACTED]"),
        (r"sk-[a-zA-Z0-9]{48}", "[OPENAI_KEY_REDACTED]"),
        (r"password[=:]\s*\S+", "password=[REDACTED]"),
        (r"passwd[=:]\s*\S+", "passwd=[REDACTED]"),
        (r"pwd[=:]\s*\S+", "pwd=[REDACTED]"),
        (r"postgresql://[^:]+:[^@]+@", "postgresql://[USER]:[PASS]@"),
        (r"mysql://[^:]+:[^@]+@", "mysql://[USER]:[PASS]@"),
        (r"api_key[=:]\s*\S+", "api_key=[REDACTED]"),
        (r"token[=:]\s*\S+", "token=[REDACTED]"),
        (r"secret[=:]\s*\S+", "secret=[REDACTED]"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact sensitive data from log record.

        Args:
            record: The log record to filter

        Returns:
            True to allow the record through
        """
        # Redact sensitive patterns in the message
        msg = str(record.getMessage())
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)

        # Update the record
        record.msg = msg
        record.args = ()  # Clear args since we've processed the message

        return True


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with sensitive data filtering.

    Creates a logger that automatically redacts sensitive information
    like API keys from log messages.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger with sensitive data filter

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting analysis")
        >>> # This will be redacted automatically:
        >>> logger.info(f"Using key: {config.github_token}")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add sensitive data filter
    logger.addFilter(SensitiveDataFilter())

    # Add console handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)

        # Format without sensitive data
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def safe_repr(obj: Any) -> str:
    """Create a string representation that redacts sensitive data.

    Use this instead of repr() or str() when logging objects that might
    contain sensitive information.

    Args:
        obj: Object to convert to string

    Returns:
        Safe string representation with sensitive data redacted

    Example:
        >>> config = Config()
        >>> print(safe_repr(config))
        # Shows config without API keys
    """
    obj_dict = getattr(obj, "__dict__", {})
    safe_dict = {}

    sensitive_keys = ["token", "key", "password", "secret", "api_key"]

    for key, value in obj_dict.items():
        if any(s in key.lower() for s in sensitive_keys):
            safe_dict[key] = "[REDACTED]"
        else:
            safe_dict[key] = value

    return f"{obj.__class__.__name__}({safe_dict})"


# Convenience function for common logging pattern
def log_config_summary(config: Any, logger: logging.Logger) -> None:
    """Log configuration summary without exposing sensitive data.

    Args:
        config: Configuration object
        logger: Logger to use

    Example:
        >>> log_config_summary(config, logger)
        # Logs: Budget: $50.0, Model: gpt-3.5-turbo
    """
    # Only log non-sensitive configuration
    safe_attrs = [
        "budget_limit",
        "rlm_model",
        "embedding_model",
        "max_clusters",
        "parallel_workers",
    ]

    summary = []
    for attr in safe_attrs:
        if hasattr(config, attr):
            value = getattr(config, attr)
            summary.append(f"{attr}={value}")

    logger.info(f"Configuration: {', '.join(summary)}")
