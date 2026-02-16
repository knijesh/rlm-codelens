"""Utility functions and helpers for RLM-Codelens.

This module provides utility functions for cost tracking and secure logging.

Example:
    >>> from rlm_codelens.utils import CostTracker
    >>> tracker = CostTracker(budget_limit=50.0)
"""

from rlm_codelens.utils.cost_tracker import CostTracker
from rlm_codelens.utils.secure_logging import (
    SensitiveDataFilter,
    get_logger,
    log_config_summary,
    safe_repr,
)

__all__ = [
    "CostTracker",
    "get_logger",
    "safe_repr",
    "log_config_summary",
    "SensitiveDataFilter",
]
