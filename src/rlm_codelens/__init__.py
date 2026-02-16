"""
RLM-Codelens: Codebase architecture intelligence

Whole-codebase architecture analysis powered by Recursive Language Models.
Scan any Python repository, build module dependency graphs, detect anti-patterns,
and optionally run RLM-powered deep semantic analysis.

Example:
    >>> from rlm_codelens import RepositoryScanner, CodebaseGraphAnalyzer
    >>> scanner = RepositoryScanner("/path/to/repo")
    >>> structure = scanner.scan()
    >>> analyzer = CodebaseGraphAnalyzer(structure)
    >>> analysis = analyzer.analyze()
"""

__version__ = "0.2.1"
__author__ = "Nijesh Kanjinghat"
__email__ = "nijesh@example.com"
__license__ = "MIT"

from rlm_codelens.codebase_graph import CodebaseGraphAnalyzer
from rlm_codelens.repo_scanner import RepositoryScanner


def __getattr__(name: str) -> type:
    """Lazy imports for heavier modules to avoid circular imports."""
    if name == "CostTracker":
        from rlm_codelens.utils.cost_tracker import CostTracker

        return CostTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RepositoryScanner",
    "CodebaseGraphAnalyzer",
    "CostTracker",
    "__version__",
    "__author__",
    "__license__",
]
