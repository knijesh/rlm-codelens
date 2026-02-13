"""
RLM-Codelens: A lens into your codebase

RLM-powered repository analysis using Recursive Language Models
with enterprise-grade cost control and security.

Example:
    >>> from rlm_codelens import RepositoryAnalyzer
    >>> analyzer = RepositoryAnalyzer(budget_limit=50.0)
    >>> results = analyzer.analyze_repository("pytorch/pytorch")
"""

__version__ = "0.1.0"
__author__ = "Nijesh Kanjinghat"
__email__ = "nijesh@example.com"
__license__ = "MIT"

from rlm_codelens.core.analyzer import RepositoryAnalyzer
from rlm_codelens.core.config import Config

# Import utility classes for convenience
from rlm_codelens.utils import (
    CostTracker,
    CostCalculator,
    CostEstimator,
    SecureRLMClient,
    DatabaseManager,
)

__all__ = [
    "RepositoryAnalyzer",
    "Config",
    "CostTracker",
    "CostCalculator",
    "CostEstimator",
    "SecureRLMClient",
    "DatabaseManager",
    "__version__",
    "__author__",
    "__license__",
]
