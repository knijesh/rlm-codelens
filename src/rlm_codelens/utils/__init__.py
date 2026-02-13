"""Utility functions and helpers for RLM-Codelens.

This module provides utility functions for cost tracking, database management,
and secure RLM client interactions.

Example:
    >>> from rlm_codelens.utils import CostTracker, SecureRLMClient
    >>> tracker = CostTracker(budget_limit=50.0)
    >>> client = SecureRLMClient(budget_limit=50.0)
"""

from rlm_codelens.utils.cost_tracker import CostTracker
from rlm_codelens.utils.cost_estimator import CostCalculator
from rlm_codelens.utils.database import get_db_manager, DatabaseManager
from rlm_codelens.utils.secure_rlm_client import (
    BudgetExceededError,
    CostEstimator,
    PromptSanitizer,
    RLMResult,
    SecureRLMClient,
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
)
from rlm_codelens.utils.secure_logging import (
    get_logger,
    safe_repr,
    log_config_summary,
    SensitiveDataFilter,
)

__all__ = [
    "CostTracker",
    "CostCalculator",
    "CostEstimator",
    "get_db_manager",
    "DatabaseManager",
    "BudgetExceededError",
    "PromptSanitizer",
    "RLMResult",
    "SecureRLMClient",
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "get_logger",
    "safe_repr",
    "log_config_summary",
    "SensitiveDataFilter",
]
