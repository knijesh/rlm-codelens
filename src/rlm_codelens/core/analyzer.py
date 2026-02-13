"""
Core analyzer module for RLM-Codelens.

This module provides the main RepositoryAnalyzer class for analyzing
GitHub repositories using Recursive Language Models with cost control
and security features.

Classes:
    RepositoryAnalyzer: Main class for repository analysis
    AnalysisConfig: Configuration dataclass for analysis parameters
    AnalysisResult: Result container for analysis operations

Example:
    >>> from rlm_codelens import RepositoryAnalyzer
    >>> analyzer = RepositoryAnalyzer(budget_limit=50.0)
    >>> result = analyzer.analyze_repository("owner/repo")
    >>> print(f"Found {len(result.clusters)} topics")
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rlm_codelens.utils.secure_rlm_client import (
    BudgetExceededError,
    CostEstimator,
    PromptSanitizer,
    RLMResult,
    SecureRLMClient,
)
from rlm_codelens.utils.database import get_db_manager


@dataclass
class AnalysisConfig:
    """Configuration parameters for repository analysis.

    This dataclass holds all configuration options for the analysis pipeline,
    allowing fine-tuning of performance, cost, and quality trade-offs.

    Attributes:
        max_clusters: Maximum number of topic clusters to analyze (default: 100)
        sample_size: Number of items to sample per cluster (default: 10)
        parallel_workers: Number of parallel workers for cluster analysis (default: 4)
        enable_caching: Whether to cache RLM results (default: True)
        skip_if_over_budget: Whether to skip calls that exceed budget (default: True)
        prompt_optimization: Whether to optimize prompts for token efficiency (default: True)
        model: OpenAI model to use for RLM calls (default: "gpt-3.5-turbo")
        embedding_model: Model for generating embeddings (default: "text-embedding-3-small")

    Example:
        >>> config = AnalysisConfig(
        ...     max_clusters=50,
        ...     parallel_workers=8,
        ...     enable_caching=True
        ... )
    """

    max_clusters: int = 100
    sample_size: int = 10
    parallel_workers: int = 4
    enable_caching: bool = True
    skip_if_over_budget: bool = True
    prompt_optimization: bool = True
    model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_clusters < 1:
            raise ValueError("max_clusters must be at least 1")
        if self.sample_size < 1:
            raise ValueError("sample_size must be at least 1")
        if self.parallel_workers < 1:
            raise ValueError("parallel_workers must be at least 1")


@dataclass
class AnalysisResult:
    """Container for repository analysis results.

    This class holds all the data and metadata from a repository analysis,
    including clusters, correlations, and statistics.

    Attributes:
        repository: Name of the analyzed repository (e.g., "pytorch/pytorch")
        total_items: Total number of items analyzed (issues + PRs)
        clusters: DataFrame of identified topic clusters
        correlations: DataFrame of issue correlations
        statistics: Dict of analysis statistics
        cost_summary: Dict of cost tracking information
        execution_time: Total execution time in seconds
        timestamp: When the analysis was performed

    Example:
        >>> result = analyzer.analyze_repository("owner/repo")
        >>> print(f"Analyzed {result.total_items} items")
        >>> print(f"Found {len(result.clusters)} topics")
    """

    repository: str
    total_items: int
    clusters: pd.DataFrame = field(default_factory=pd.DataFrame)
    correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    statistics: Dict[str, Any] = field(default_factory=dict)
    cost_summary: Dict[str, float] = field(default_factory=dict)
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())

    def save(self, output_dir: str = "outputs") -> None:
        """Save analysis results to disk.

        Args:
            output_dir: Directory to save results (default: "outputs")

        Example:
            >>> result.save("./my_results")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save clusters
        if not self.clusters.empty:
            self.clusters.to_csv(output_path / "clusters.csv", index=False)

        # Save correlations
        if not self.correlations.empty:
            self.correlations.to_csv(output_path / "correlations.csv", index=False)

        # Save metadata
        metadata = {
            "repository": self.repository,
            "total_items": self.total_items,
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "cost_summary": self.cost_summary,
            "statistics": self.statistics,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)


class OptimizedPromptBuilder:
    """Builds optimized prompts to minimize token usage and cost.

    This class provides static methods for constructing prompts that are
    optimized for minimal token usage while maintaining effectiveness.

    The optimization strategies include:
    - Using abbreviated keys (e.g., "t" instead of "title")
    - Limiting sample sizes
    - Using compact JSON formatting
    - Truncating long text fields

    Example:
        >>> prompt = OptimizedPromptBuilder.build_cluster_analysis_prompt(
        ...     cluster_id=1,
        ...     sample_data=items,
        ...     total_size=150
        ... )
    """

    @staticmethod
    def build_cluster_analysis_prompt(
        cluster_id: int, sample_data: List[Dict], total_size: int
    ) -> str:
        """Build an optimized prompt for cluster topic analysis.

        Constructs a compact prompt that minimizes token usage while
        providing sufficient context for the LLM to analyze the cluster.

        Args:
            cluster_id: Unique identifier for the cluster
            sample_data: List of sample items from the cluster (should contain
                        'title', 'labels', 'type' keys)
            total_size: Total number of items in the cluster

        Returns:
            Optimized prompt string ready for LLM completion

        Example:
            >>> items = [
            ...     {"title": "CUDA memory error", "labels": ["bug"], "type": "issue"},
            ...     {"title": "GPU support needed", "labels": ["feature"], "type": "issue"}
            ... ]
            >>> prompt = OptimizedPromptBuilder.build_cluster_analysis_prompt(
            ...     1, items, 150
            ... )
            >>> print(len(prompt))  # Much shorter than naive approach
        """
        # Truncate and optimize sample data
        optimized_samples = []
        for item in sample_data[:5]:  # Limit to 5 samples
            optimized_samples.append(
                {
                    "t": item.get("title", "")[:60],  # Shortened key, truncated value
                    "l": item.get("labels", [])[:3],  # Limit labels
                    "y": item.get("type", "")[:2],  # Shortened type
                }
            )

        # Compact JSON to save tokens
        samples_json = json.dumps(optimized_samples, separators=(",", ":"))

        prompt = (
            f"Analyze cluster {cluster_id} ({total_size} items).\n"
            f"Samples: {samples_json}\n"
            f'Respond with JSON:{{"topic":"name","description":"brief",'
            f'"category":"Bug|Feature|Docs|Perf|API|Other",'
            f'"key_terms":["t1","t2","t3"],"grouping_reason":"why"}}'
        )

        return prompt

    @staticmethod
    def build_temporal_analysis_prompt(
        top_topics: List[str], monthly_data: pd.DataFrame
    ) -> str:
        """Build an optimized prompt for temporal trend analysis.

        Creates a compact summary of temporal data for trend analysis.

        Args:
            top_topics: List of top topic names by volume
            monthly_data: DataFrame with monthly activity data

        Returns:
            Optimized prompt string for temporal analysis
        """
        summary = {
            "topics": top_topics[:5],  # Limit to top 5
            "months": len(monthly_data),
            "trend": "increasing" if len(monthly_data) > 12 else "stable",
        }

        return (
            f"Analyze trends.\n"
            f"Data: {json.dumps(summary, separators=(',', ':'))}\n"
            f"Identify: emerging topics, declining topics, patterns (2 paragraphs)."
        )


class RepositoryAnalyzer:
    """Main class for analyzing GitHub repositories using RLM.

    This class provides a high-level interface for repository analysis,
    including data collection, clustering, correlation detection, and
    visualization generation. It includes built-in cost control and
    security features.

    The analysis pipeline follows these steps:
    1. Collect repository data from GitHub API
    2. Generate embeddings for items
    3. Cluster items by topic using HDBSCAN
    4. Analyze clusters using RLM (with cost control)
    5. Detect correlations between issues
    6. Generate visualizations and reports

    Attributes:
        config: AnalysisConfig instance with analysis parameters
        budget_limit: Maximum budget for API calls
        rlm_client: SecureRLMClient instance for RLM operations
        cost_estimator: CostEstimator for pre-flight cost estimation
        db: DatabaseManager for data persistence

    Example:
        >>> analyzer = RepositoryAnalyzer(budget_limit=50.0)
        >>> result = analyzer.analyze_repository("pytorch/pytorch")
        >>> print(f"Cost: ${result.cost_summary['total']:.2f}")
        >>> result.save("./analysis_results")

    Note:
        The analyzer automatically tracks costs and will halt execution
        if the budget would be exceeded. Set skip_if_over_budget=False
        to raise exceptions instead of graceful degradation.
    """

    def __init__(
        self, budget_limit: float = 50.0, config: Optional[AnalysisConfig] = None
    ):
        """Initialize the repository analyzer.

        Args:
            budget_limit: Maximum budget in USD for API calls (default: 50.0)
            config: AnalysisConfig instance with analysis parameters
                   (default: None, uses default configuration)

        Raises:
            ValueError: If budget_limit is not positive

        Example:
            >>> analyzer = RepositoryAnalyzer(budget_limit=20.0)
            >>> # Or with custom config
            >>> config = AnalysisConfig(max_clusters=50, parallel_workers=8)
            >>> analyzer = RepositoryAnalyzer(50.0, config)
        """
        if budget_limit <= 0:
            raise ValueError("budget_limit must be positive")

        self.config = config or AnalysisConfig()
        self.budget_limit = budget_limit
        self.db = get_db_manager()

        # Initialize secure RLM client with cost control
        self.rlm_client = SecureRLMClient(
            model=self.config.model,
            budget_limit=budget_limit,
            enable_caching=self.config.enable_caching,
            enable_circuit_breaker=True,
            max_retries=3,
        )

        self.cost_estimator = CostEstimator(self.config.model)

        # Track analysis metrics
        self._metrics = {
            "clusters_analyzed": 0,
            "correlations_found": 0,
            "cache_hits": 0,
            "api_calls": 0,
        }

    def analyze_repository(
        self, repository: str, limit: Optional[int] = None
    ) -> AnalysisResult:
        """Analyze a GitHub repository comprehensively.

        Performs the complete analysis pipeline on a repository:
        data collection, embedding generation, clustering, RLM analysis,
        correlation detection, and report generation.

        Args:
            repository: Repository name in "owner/repo" format
                       (e.g., "pytorch/pytorch")
            limit: Maximum number of items to analyze (optional,
                  useful for testing with smaller datasets)

        Returns:
            AnalysisResult containing all analysis data and metadata

        Raises:
            BudgetExceededError: If analysis would exceed budget
            ValueError: If repository format is invalid
            RuntimeError: If analysis fails critically

        Example:
            >>> analyzer = RepositoryAnalyzer(budget_limit=50.0)
            >>> result = analyzer.analyze_repository("pytorch/pytorch")
            >>> print(f"Found {len(result.clusters)} topics")
            >>> print(f"Cost: ${result.cost_summary['total']:.2f}")
            >>> result.save("./results")

        Note:
            This method can take significant time (1-4 hours) depending
            on repository size and configuration. Use the limit parameter
            for faster testing with smaller datasets.
        """
        start_time = time.time()

        print(f"🔍 Analyzing repository: {repository}")
        print(f"   Budget: ${self.budget_limit:.2f}")
        print(f"   Config: {self.config}")

        # TODO: Implement full analysis pipeline
        # This would call:
        # 1. self._collect_data(repository, limit)
        # 2. self._generate_embeddings()
        # 3. self._cluster_items()
        # 4. self._analyze_clusters_parallel()
        # 5. self._detect_correlations()
        # 6. self._generate_report()

        # For now, return empty result structure
        execution_time = time.time() - start_time

        return AnalysisResult(
            repository=repository,
            total_items=0,
            execution_time=execution_time,
            cost_summary=self.rlm_client.get_budget_summary(),
        )

    def estimate_cost(self, num_items: int) -> Dict[str, float]:
        """Estimate the cost to analyze a given number of items.

        Provides a pre-flight cost estimate before starting the analysis,
        helping users decide if they want to proceed.

        Args:
            num_items: Number of items (issues/PRs) to analyze

        Returns:
            Dictionary with cost breakdown:
            - embeddings: Cost for embedding generation
            - rlm_analysis: Cost for RLM cluster analysis
            - total: Total estimated cost
            - per_item: Average cost per item

        Example:
            >>> analyzer = RepositoryAnalyzer()
            >>> estimate = analyzer.estimate_cost(80000)
            >>> print(f"Estimated cost: ${estimate['total']:.2f}")
            >>> print(f"Per item: ${estimate['per_item']:.4f}")
        """
        # Estimate embedding cost (assume average 500 tokens per item)
        avg_tokens_per_item = 500
        embed_cost = self.cost_estimator.estimate_embeddings_cost(
            ["x" * avg_tokens_per_item] * num_items
        )

        # Estimate RLM analysis cost
        num_clusters = min(self.config.max_clusters, num_items // 50)
        rlm_cost = num_clusters * 0.03  # ~$0.03 per cluster

        total = embed_cost.estimated_cost + rlm_cost

        return {
            "embeddings": embed_cost.estimated_cost,
            "rlm_analysis": rlm_cost,
            "total": total,
            "per_item": total / num_items if num_items > 0 else 0,
            "within_budget": total <= self.budget_limit,
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get analysis metrics and statistics.

        Returns current metrics for monitoring analysis progress.

        Returns:
            Dictionary with metrics including:
            - clusters_analyzed: Number of clusters processed
            - correlations_found: Number of correlations detected
            - cache_hits: Number of cached responses used
            - api_calls: Total number of API calls made
            - cost_summary: Current cost tracking summary

        Example:
            >>> analyzer = RepositoryAnalyzer()
            >>> # ... run some analysis ...
            >>> metrics = analyzer.get_metrics()
            >>> print(f"Cache hit rate: {metrics['cache_hits']}")
        """
        return {
            **self._metrics,
            "cost_summary": self.rlm_client.get_budget_summary(),
        }
