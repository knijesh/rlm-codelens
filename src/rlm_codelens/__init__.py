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
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"

from .core.analyzer import RepositoryAnalyzer
from .core.config import Config

__all__ = [
    "RepositoryAnalyzer",
    "Config",
    "__version__",
]
