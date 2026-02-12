"""
Redesigned RLM Analysis Module
Production-ready with cost controls, security, and performance optimizations
"""

import json
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dataclasses import dataclass
from pathlib import Path

from utils.database import get_db_manager
from utils.secure_rlm_client import (
    SecureRLMClient,
    CostEstimator,
    PromptSanitizer,
    BudgetExceededError,
)
from config import RLM_ROOT_MODEL, BUDGET_LIMIT


@dataclass
class AnalysisConfig:
    """Configuration for RLM analysis"""

    max_clusters: int = 100
    sample_size: int = 10  # Reduced from 20 to save tokens
    parallel_workers: int = 4
    enable_caching: bool = True
    skip_if_over_budget: bool = True
    prompt_optimization: bool = True


class OptimizedPromptBuilder:
    """Builds optimized prompts to minimize token usage"""

    @staticmethod
    def build_cluster_analysis_prompt(
        cluster_id: int, sample_data: List[Dict], total_size: int
    ) -> str:
        """Build optimized prompt for cluster analysis (minimized tokens)"""

        # Truncate and optimize sample data
        optimized_samples = []
        for item in sample_data[:5]:  # Use only 5 samples instead of 20
            optimized_samples.append(
                {
                    "t": item.get("title", "")[:60],  # Shortened key, truncated value
                    "l": item.get("labels", [])[:3],  # Limit labels
                    "y": item.get("type", "")[:2],  # Shortened
                }
            )

        # Compact JSON to save tokens
        samples_json = json.dumps(optimized_samples, separators=(",", ":"))

        prompt = f"""Analyze cluster {cluster_id} ({total_size} items).
Samples: {samples_json}
Respond with JSON:{{"topic":"name","description":"brief","category":"Bug|Feature|Docs|Perf|API|Other","key_terms":["t1","t2","t3"],"grouping_reason":"why"}}"""

        return prompt

    @staticmethod
    def build_temporal_analysis_prompt(
        top_topics: List[str], monthly_data: pd.DataFrame
    ) -> str:
        """Build optimized temporal analysis prompt"""
        # Aggregate data to reduce tokens
        summary = {
            "topics": top_topics[:5],  # Limit to top 5
            "months": len(monthly_data),
            "trend": "increasing" if len(monthly_data) > 12 else "stable",
        }

        return f"""Analyze trends in PyTorch repo.
Data: {json.dumps(summary, separators=(",", ":"))}
Identify: emerging topics, declining topics, patterns (2 paragraphs)."""


class ComparativeAnalyzer:
    """Compare RLM vs Non-RLM approaches"""

    def __init__(self):
        self.results = {
            "rlm": {
                "calls": 0,
                "tokens": 0,
                "cost": 0.0,
                "time": 0.0,
                "success_rate": 0.0,
            },
            "non_rlm": {
                "calls": 0,
                "tokens": 0,
                "cost": 0.0,
                "time": 0.0,
                "success_rate": 0.0,
            },
        }

    def record_rlm_call(self, tokens: int, cost: float, duration: float, success: bool):
        """Record RLM call metrics"""
        self.results["rlm"]["calls"] += 1
        self.results["rlm"]["tokens"] += tokens
        self.results["rlm"]["cost"] += cost
        self.results["rlm"]["time"] += duration
        if success:
            self.results["rlm"]["success_rate"] = (
                self.results["rlm"]["success_rate"] * (self.results["rlm"]["calls"] - 1)
                + 1
            ) / self.results["rlm"]["calls"]

    def record_non_rlm_call(
        self, tokens: int, cost: float, duration: float, success: bool
    ):
        """Record non-RLM call metrics"""
        self.results["non_rlm"]["calls"] += 1
        self.results["non_rlm"]["tokens"] += tokens
        self.results["non_rlm"]["cost"] += cost
        self.results["non_rlm"]["time"] += duration
        if success:
            self.results["non_rlm"]["success_rate"] = (
                self.results["non_rlm"]["success_rate"]
                * (self.results["non_rlm"]["calls"] - 1)
                + 1
            ) / self.results["non_rlm"]["calls"]

    def generate_report(self) -> Dict[str, Any]:
        """Generate comparison report"""
        rlm = self.results["rlm"]
        non_rlm = self.results["non_rlm"]

        # Calculate savings
        cost_savings = non_rlm["cost"] - rlm["cost"]
        cost_savings_pct = (
            (cost_savings / non_rlm["cost"] * 100) if non_rlm["cost"] > 0 else 0
        )

        time_diff = non_rlm["time"] - rlm["time"]

        return {
            "rlm_metrics": {
                "total_calls": rlm["calls"],
                "total_tokens": rlm["tokens"],
                "total_cost": rlm["cost"],
                "avg_cost_per_call": rlm["cost"] / rlm["calls"]
                if rlm["calls"] > 0
                else 0,
                "total_time_sec": rlm["time"],
                "success_rate": rlm["success_rate"] * 100,
            },
            "non_rlm_metrics": {
                "total_calls": non_rlm["calls"],
                "total_tokens": non_rlm["tokens"],
                "total_cost": non_rlm["cost"],
                "avg_cost_per_call": non_rlm["cost"] / non_rlm["calls"]
                if non_rlm["calls"] > 0
                else 0,
                "total_time_sec": non_rlm["time"],
                "success_rate": non_rlm["success_rate"] * 100,
            },
            "comparison": {
                "cost_savings": cost_savings,
                "cost_savings_percentage": cost_savings_pct,
                "time_difference_sec": time_diff,
                "winner": "RLM" if rlm["cost"] < non_rlm["cost"] else "Non-RLM",
            },
            "recommendation": self._generate_recommendation(
                cost_savings_pct, rlm, non_rlm
            ),
        }

    def _generate_recommendation(
        self, savings_pct: float, rlm: Dict, non_rlm: Dict
    ) -> str:
        """Generate recommendation based on comparison"""
        if savings_pct > 50:
            return (
                f"✅ RLM is significantly more cost-effective ({savings_pct:.1f}% savings). "
                f"Use RLM for this workload."
            )
        elif savings_pct > 20:
            return (
                f"✅ RLM provides moderate cost savings ({savings_pct:.1f}%). "
                f"Recommended for large-scale analysis."
            )
        elif savings_pct > -20:
            return (
                f"⚖️  Costs are comparable. Choose based on quality needs: "
                f"RLM for complex reasoning, Non-RLM for simple tasks."
            )
        else:
            return (
                f"❌ Non-RLM is more cost-effective for this workload. "
                f"Consider using direct LLM calls instead."
            )

    def print_report(self):
        """Print formatted comparison report"""
        report = self.generate_report()

        print("\n" + "=" * 70)
        print("RLM vs NON-RLM COMPARATIVE ANALYSIS")
        print("=" * 70)

        print("\n📊 RLM METRICS:")
        rlm_m = report["rlm_metrics"]
        print(f"  Calls: {rlm_m['total_calls']}")
        print(f"  Tokens: {rlm_m['total_tokens']:,}")
        print(f"  Cost: ${rlm_m['total_cost']:.2f}")
        print(f"  Avg per call: ${rlm_m['avg_cost_per_call']:.4f}")
        print(f"  Time: {rlm_m['total_time_sec']:.1f}s")
        print(f"  Success: {rlm_m['success_rate']:.1f}%")

        print("\n📊 NON-RLM METRICS:")
        non_m = report["non_rlm_metrics"]
        print(f"  Calls: {non_m['total_calls']}")
        print(f"  Tokens: {non_m['total_tokens']:,}")
        print(f"  Cost: ${non_m['total_cost']:.2f}")
        print(f"  Avg per call: ${non_m['avg_cost_per_call']:.4f}")
        print(f"  Time: {non_m['total_time_sec']:.1f}s")
        print(f"  Success: {non_m['success_rate']:.1f}%")

        print("\n🎯 COMPARISON:")
        comp = report["comparison"]
        print(
            f"  Cost Savings: ${comp['cost_savings']:.2f} ({comp['cost_savings_percentage']:.1f}%)"
        )
        print(f"  Time Diff: {comp['time_difference_sec']:+.1f}s")
        print(f"  Winner: {comp['winner']}")

        print(f"\n💡 RECOMMENDATION:")
        print(f"  {report['recommendation']}")

        print("=" * 70 + "\n")


class SecureRLMAnalyzer:
    """
    Production-ready secure RLM analyzer with:
    - Pre-flight cost estimation
    - Budget enforcement
    - Parallel processing
    - Result caching
    - Comparative analysis
    """

    def __init__(
        self,
        budget_limit: float = BUDGET_LIMIT,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = config or AnalysisConfig()
        self.db = get_db_manager()

        # Initialize secure RLM client
        self.rlm_client = SecureRLMClient(
            model=RLM_ROOT_MODEL,
            budget_limit=budget_limit,
            enable_caching=self.config.enable_caching,
            enable_circuit_breaker=True,
            max_retries=3,
        )

        # Comparative analyzer
        self.comparative_analyzer = ComparativeAnalyzer()

        # Prompt builder
        self.prompt_builder = OptimizedPromptBuilder()

        print("✅ Secure RLM Analyzer initialized")
        print(f"   Budget: ${budget_limit:.2f}")
        print(f"   Parallel workers: {self.config.parallel_workers}")
        print(f"   Caching: {'enabled' if self.config.enable_caching else 'disabled'}")

    def analyze_clusters_parallel(self) -> pd.DataFrame:
        """Analyze clusters in parallel with cost control"""
        print("\n🚀 Starting parallel cluster analysis...")

        # Load data
        clusters_df = self.db.load_dataframe("cluster_stats")
        items_df = self.db.load_dataframe("clustered_items")

        # Filter to real clusters (skip noise)
        clusters_to_analyze = clusters_df[clusters_df["cluster_id"] != -1].head(
            self.config.max_clusters
        )

        print(f"   Analyzing {len(clusters_to_analyze)} clusters...")

        # Estimate total cost before starting
        total_estimate = self._estimate_total_cost(clusters_to_analyze)
        print(f"   💰 Estimated total cost: ${total_estimate:.2f}")

        # Check if we should proceed
        budget_summary = self.rlm_client.get_budget_summary()
        if budget_summary["remaining"] < total_estimate:
            print(f"\n⛔ INSUFFICIENT BUDGET!")
            print(f"   Estimated: ${total_estimate:.2f}")
            print(f"   Remaining: ${budget_summary['remaining']:.2f}")
            print(f"   Skipping RLM analysis to stay within budget.")
            return pd.DataFrame()

        # Process in parallel
        results = []

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            future_to_cluster = {}
            for _, cluster_row in clusters_to_analyze.iterrows():
                cluster_id = cluster_row["cluster_id"]
                future = executor.submit(
                    self._analyze_single_cluster,
                    cluster_id,
                    items_df,
                    cluster_row.get("total_size", 0),
                )
                future_to_cluster[future] = cluster_id

            # Collect results as they complete
            for future in as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Progress update
                    if len(results) % 10 == 0:
                        print(
                            f"   Completed {len(results)}/{len(clusters_to_analyze)} clusters"
                        )
                        self.rlm_client.print_budget_summary()

                except Exception as e:
                    print(f"   ⚠️  Cluster {cluster_id} failed: {e}")
                    results.append({"cluster_id": cluster_id, "error": str(e)})

        # Save results
        results_df = pd.DataFrame(results)
        self.db.save_dataframe(results_df, "cluster_analyses", if_exists="replace")

        print(f"\n✅ Analyzed {len(results)} clusters")
        self.rlm_client.print_budget_summary()

        # Print comparative analysis
        self.comparative_analyzer.print_report()

        return results_df

    def _estimate_total_cost(self, clusters_df: pd.DataFrame) -> float:
        """Estimate total cost for all clusters"""
        # Rough estimate: each cluster costs ~$0.03
        return len(clusters_df) * 0.03

    def _analyze_single_cluster(
        self, cluster_id: int, items_df: pd.DataFrame, total_size: int
    ) -> Dict[str, Any]:
        """Analyze a single cluster"""

        # Get sample items
        cluster_items = items_df[items_df["cluster_id"] == cluster_id]
        sample = cluster_items.sample(min(self.config.sample_size, len(cluster_items)))

        sample_data = sample[["title", "labels", "type"]].to_dict("records")

        # Build optimized prompt
        prompt = self.prompt_builder.build_cluster_analysis_prompt(
            cluster_id, sample_data, total_size
        )

        # Execute with secure client (includes cost estimation)
        start_time = time.time()
        result = self.rlm_client.completion(
            prompt,
            expected_output_tokens=200,  # Optimized for short JSON response
            use_cache=self.config.enable_caching,
            skip_if_over_budget=self.config.skip_if_over_budget,
        )
        duration = time.time() - start_time

        # Record for comparative analysis
        self.comparative_analyzer.record_rlm_call(
            tokens=result.tokens_used,
            cost=result.cost,
            duration=duration,
            success=result.success,
        )

        # Parse result
        if result.success:
            try:
                # Extract JSON
                json_match = re.search(r"\{[^}]+\}", result.response)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = json.loads(result.response)

                analysis["cluster_id"] = cluster_id
                analysis["sample_size"] = len(sample_data)
                analysis["total_size"] = total_size
                analysis["cost"] = result.cost
                analysis["cache_hit"] = result.cache_hit

                return analysis

            except (json.JSONDecodeError, KeyError) as e:
                return {
                    "cluster_id": cluster_id,
                    "error": f"Parse error: {e}",
                    "raw_response": result.response[:200],
                    "cost": result.cost,
                }
        else:
            return {
                "cluster_id": cluster_id,
                "error": result.error or "Unknown error",
                "cost": result.cost,
            }

    def discover_correlations_safe(self) -> List[Dict]:
        """Discover correlations with cost control"""
        print("\n🔍 Discovering correlations...")

        query = """
        SELECT i.*, c.topic, c.category
        FROM clustered_items i
        JOIN cluster_analyses c ON i.cluster_id = c.cluster_id
        WHERE i.cluster_id != -1
        LIMIT 5000
        """

        df = self.db.load_dataframe(None, query)
        print(f"   Analyzing {len(df)} items...")

        # Limit to reduce cost
        if len(df) > 3000:
            df = df.sample(3000)
            print(f"   Sampled to {len(df)} items for cost control")

        correlations = []

        # Only analyze top 3 patterns to save cost
        print("   Analyzing temporal trends...")
        correlations.append(self._analyze_temporal_trends_safe(df))

        print("   Analyzing author patterns...")
        correlations.append(self._analyze_author_patterns_safe(df))

        # Save correlations
        correlations_df = pd.DataFrame(correlations)
        self.db.save_dataframe(correlations_df, "correlations", if_exists="replace")

        print(f"✅ Discovered {len(correlations)} correlation patterns")

        return correlations

    def _analyze_temporal_trends_safe(self, df: pd.DataFrame) -> Dict:
        """Analyze temporal trends with cost control"""
        df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M")

        # Get only recent 12 months (not 24) to save tokens
        recent_months = df["month"].unique()[-12:]
        recent_df = df[df["month"].isin(recent_months)]

        top_topics = recent_df["topic"].value_counts().head(5).index.tolist()

        # Build optimized prompt
        prompt = self.prompt_builder.build_temporal_analysis_prompt(
            top_topics, recent_df
        )

        start_time = time.time()
        result = self.rlm_client.completion(
            prompt, expected_output_tokens=400, use_cache=True
        )
        duration = time.time() - start_time

        self.comparative_analyzer.record_rlm_call(
            tokens=result.tokens_used,
            cost=result.cost,
            duration=duration,
            success=result.success,
        )

        return {
            "type": "temporal_trends",
            "analysis": result.response if result.success else f"Error: {result.error}",
            "topics_analyzed": len(top_topics),
            "months_analyzed": len(recent_months),
            "cost": result.cost,
        }

    def _analyze_author_patterns_safe(self, df: pd.DataFrame) -> Dict:
        """Analyze author patterns with cost control"""
        # Limit to top 20 authors
        author_stats = df.groupby(["author", "topic"]).size().reset_index(name="count")
        top_authors = author_stats.nlargest(20, "count")

        # Compact prompt
        summary = {
            "top_contributors": len(top_authors),
            "unique_authors": df["author"].nunique(),
            "unique_topics": df["topic"].nunique(),
        }

        prompt = f"""Analyze contributor patterns.
Data: {json.dumps(summary, separators=(",", ":"))}
Identify: specialists vs generalists, collaboration patterns (2 paragraphs)."""

        start_time = time.time()
        result = self.rlm_client.completion(
            prompt, expected_output_tokens=300, use_cache=True
        )
        duration = time.time() - start_time

        self.comparative_analyzer.record_rlm_call(
            tokens=result.tokens_used,
            cost=result.cost,
            duration=duration,
            success=result.success,
        )

        return {
            "type": "author_patterns",
            "analysis": result.response if result.success else f"Error: {result.error}",
            "authors_analyzed": summary["unique_authors"],
            "cost": result.cost,
        }


def main():
    """Run redesigned RLM analysis"""
    print("=" * 70)
    print("SECURE RLM ANALYSIS (REDESIGNED)")
    print("=" * 70)

    # Configure for cost efficiency
    config = AnalysisConfig(
        max_clusters=50,  # Limit for demo
        sample_size=5,  # Smaller samples
        parallel_workers=4,
        enable_caching=True,
        skip_if_over_budget=True,
        prompt_optimization=True,
    )

    # Initialize analyzer
    analyzer = SecureRLMAnalyzer(
        budget_limit=5.0,  # $5 budget for testing
        config=config,
    )

    try:
        # Analyze clusters in parallel
        cluster_analyses = analyzer.analyze_clusters_parallel()

        # Discover correlations
        correlations = analyzer.discover_correlations_safe()

        print("\n✅ Analysis complete!")
        print(f"   Clusters analyzed: {len(cluster_analyses)}")
        print(f"   Correlations found: {len(correlations)}")

        # Final budget summary
        analyzer.rlm_client.print_budget_summary()

        return cluster_analyses, correlations

    except BudgetExceededError as e:
        print(f"\n⛔ Budget exceeded: {e}")
        print("   Analysis halted to prevent cost overrun.")
        return None, None


if __name__ == "__main__":
    main()
