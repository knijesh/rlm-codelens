"""
Cost Comparison and Estimation Tool
Pre-flight cost estimation and RLM vs Non-RLM comparison
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class ModelTier(Enum):
    """Model tiers with pricing"""

    ECONOMY = "gpt-3.5-turbo"  # Cheapest
    BALANCED = "gpt-4o-mini"  # Good balance
    PREMIUM = "gpt-4o"  # Best quality


@dataclass
class CostScenario:
    """Cost scenario for analysis planning"""

    name: str
    num_items: int
    avg_tokens_per_item: int
    model: str
    approach: str  # "RLM" or "Non-RLM"

    def calculate(self) -> Dict[str, float]:
        """Calculate cost for this scenario"""
        pricing = CostCalculator.PRICING.get(
            self.model, {"input": 0.50, "output": 1.50}
        )

        if self.approach == "RLM":
            # RLM: Multiple small calls
            # Root call + sub-calls
            calls = self.num_items
            tokens_per_call = self.avg_tokens_per_item

            input_tokens = calls * tokens_per_call * 0.7
            output_tokens = calls * tokens_per_call * 0.3

        else:
            # Non-RLM: Fewer larger calls
            # Batch processing
            batch_size = 100
            calls = self.num_items // batch_size
            tokens_per_call = self.avg_tokens_per_item * batch_size

            input_tokens = calls * tokens_per_call * 0.7
            output_tokens = calls * tokens_per_call * 0.3

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_item": total_cost / self.num_items if self.num_items > 0 else 0,
        }


class CostCalculator:
    """
    Comprehensive cost calculator for LLM operations
    Provides pre-flight estimates and recommendations
    """

    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        # Embedding models
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
        # GPT-3.5 models
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-16k": {"input": 1.0, "output": 2.0},
        # GPT-4 models
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    }

    def __init__(self, budget_limit: float = 50.0):
        self.budget_limit = budget_limit

    def estimate_project_cost(
        self,
        num_issues: int = 80000,
        include_embeddings: bool = True,
        include_clustering: bool = True,
        include_rlm_analysis: bool = True,
        include_correlation: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate total project cost

        Returns detailed breakdown of all phases
        """
        estimates = {}

        # Phase 1: Data Collection (Free - GitHub API)
        estimates["data_collection"] = {
            "cost": 0.0,
            "description": "GitHub API (free tier)",
        }

        # Phase 2: Embeddings
        if include_embeddings:
            embed_cost = self._estimate_embeddings(num_issues)
            estimates["embeddings"] = embed_cost

        # Phase 3: Clustering (Free - local computation)
        if include_clustering:
            estimates["clustering"] = {
                "cost": 0.0,
                "description": "HDBSCAN clustering (local CPU)",
            }

        # Phase 4: RLM Analysis
        if include_rlm_analysis:
            rlm_cost = self._estimate_rlm_analysis(num_issues)
            estimates["rlm_analysis"] = rlm_cost

        # Phase 5: Correlation Analysis (Free - local computation)
        if include_correlation:
            estimates["correlation"] = {
                "cost": 0.0,
                "description": "Correlation analysis (local CPU)",
            }

        # Calculate totals
        total_cost = sum(e.get("cost", 0) for e in estimates.values())

        estimates["total"] = {
            "cost": total_cost,
            "within_budget": total_cost <= self.budget_limit,
            "budget_remaining": self.budget_limit - total_cost,
            "percentage_of_budget": (total_cost / self.budget_limit * 100),
        }

        return estimates

    def _estimate_embeddings(self, num_items: int) -> Dict[str, Any]:
        """Estimate embedding generation cost"""
        model = "text-embedding-3-small"
        pricing = self.PRICING[model]

        # Average 800 tokens per item (title + body excerpt)
        avg_tokens = 800
        total_tokens = num_items * avg_tokens

        cost = (total_tokens / 1_000_000) * pricing["input"]

        return {
            "cost": cost,
            "model": model,
            "num_items": num_items,
            "total_tokens": total_tokens,
            "description": f"Generate embeddings for {num_items:,} items",
        }

    def _estimate_rlm_analysis(
        self, num_items: int, num_clusters: int = 100
    ) -> Dict[str, Any]:
        """Estimate RLM analysis cost"""
        model = "gpt-3.5-turbo"
        pricing = self.PRICING[model]

        # Cluster analysis
        calls_per_cluster = 1
        tokens_per_cluster_call = 1200  # Optimized prompt

        cluster_input = num_clusters * calls_per_cluster * tokens_per_cluster_call * 0.7
        cluster_output = (
            num_clusters * calls_per_cluster * tokens_per_cluster_call * 0.3
        )

        # Correlation analysis (3 major calls)
        correlation_calls = 3
        tokens_per_corr_call = 1500

        corr_input = correlation_calls * tokens_per_corr_call * 0.7
        corr_output = correlation_calls * tokens_per_corr_call * 0.3

        # Total
        total_input = cluster_input + corr_input
        total_output = cluster_output + corr_output

        input_cost = (total_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return {
            "cost": total_cost,
            "model": model,
            "cluster_calls": num_clusters,
            "correlation_calls": correlation_calls,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "description": f"RLM analysis of {num_clusters} clusters + correlations",
        }

    def compare_rlm_vs_direct(
        self, num_items: int = 100, tokens_per_item: int = 1000
    ) -> Dict[str, Any]:
        """
        Compare RLM approach vs direct LLM approach

        Shows cost/benefit analysis
        """
        model = "gpt-3.5-turbo"

        # RLM Approach
        rlm_scenario = CostScenario(
            name="RLM (Recursive)",
            num_items=num_items,
            avg_tokens_per_item=tokens_per_item,
            model=model,
            approach="RLM",
        )
        rlm_calc = rlm_scenario.calculate()

        # Non-RLM (Direct) Approach
        non_rlm_scenario = CostScenario(
            name="Direct LLM",
            num_items=num_items,
            avg_tokens_per_item=tokens_per_item,
            model=model,
            approach="Non-RLM",
        )
        non_rlm_calc = non_rlm_scenario.calculate()

        # Comparison
        savings = non_rlm_calc["total_cost"] - rlm_calc["total_cost"]
        savings_pct = (
            (savings / non_rlm_calc["total_cost"] * 100)
            if non_rlm_calc["total_cost"] > 0
            else 0
        )

        return {
            "rlm": {
                "approach": "RLM (Recursive decomposition)",
                "calls": rlm_calc["calls"],
                "total_tokens": rlm_calc["input_tokens"] + rlm_calc["output_tokens"],
                "total_cost": rlm_calc["total_cost"],
                "cost_per_item": rlm_calc["cost_per_item"],
                "pros": [
                    "Handles unlimited context size",
                    "Better for complex reasoning",
                    "More granular analysis",
                    "Parallelizable",
                ],
                "cons": [
                    "More API calls to manage",
                    "Higher coordination overhead",
                    "Requires careful prompt design",
                ],
            },
            "direct": {
                "approach": "Direct LLM calls",
                "calls": non_rlm_calc["calls"],
                "total_tokens": non_rlm_calc["input_tokens"]
                + non_rlm_calc["output_tokens"],
                "total_cost": non_rlm_calc["total_cost"],
                "cost_per_item": non_rlm_calc["cost_per_item"],
                "pros": [
                    "Simpler implementation",
                    "Fewer API calls",
                    "Lower latency for small tasks",
                ],
                "cons": [
                    "Limited by context window",
                    "Expensive for large inputs",
                    "Cannot parallelize single call",
                ],
            },
            "comparison": {
                "cost_savings": savings,
                "cost_savings_percentage": savings_pct,
                "token_efficiency": (
                    (non_rlm_calc["input_tokens"] + non_rlm_calc["output_tokens"])
                    / (rlm_calc["input_tokens"] + rlm_calc["output_tokens"])
                ),
                "call_ratio": rlm_calc["calls"] / non_rlm_calc["calls"],
                "recommendation": self._generate_recommendation(
                    savings_pct, rlm_calc, non_rlm_calc
                ),
            },
        }

    def _generate_recommendation(
        self, savings_pct: float, rlm: Dict, direct: Dict
    ) -> str:
        """Generate recommendation based on comparison"""
        if savings_pct > 30:
            return (
                f"✅ STRONGLY RECOMMEND RLM: {savings_pct:.1f}% cost savings "
                f"(${direct['total_cost']:.2f} → ${rlm['total_cost']:.2f}). "
                f"Use RLM for large-scale analysis tasks."
            )
        elif savings_pct > 10:
            return (
                f"✅ RECOMMEND RLM: {savings_pct:.1f}% cost savings. "
                f"Good choice for complex multi-step analysis."
            )
        elif savings_pct > -10:
            return (
                f"⚖️  COMPARABLE COSTS: RLM provides better scalability "
                f"and unlimited context. Choose based on quality needs."
            )
        else:
            return (
                f"❌ NON-RLM MORE COST-EFFECTIVE: {abs(savings_pct):.1f}% cheaper. "
                f"Use direct LLM calls for this workload."
            )

    def get_budget_recommendation(
        self, available_budget: float, num_items: int = 80000
    ) -> Dict[str, Any]:
        """
        Get recommendations based on available budget
        """
        # Calculate minimum required budget
        min_required = self._estimate_embeddings(num_items)["cost"] + 2.0  # + buffer

        # Calculate recommended budget for full analysis
        estimates = self.estimate_project_cost(num_items)
        recommended = estimates["total"]["cost"]

        if available_budget < min_required:
            return {
                "feasible": False,
                "message": f"❌ Budget too low. Minimum required: ${min_required:.2f}",
                "recommendation": "Reduce number of items or request more budget",
                "alternatives": [
                    f"Analyze only {int(available_budget / 0.02)} items",
                    "Use cheaper models (gpt-3.5-turbo)",
                    "Skip RLM phase, use heuristics only",
                ],
            }
        elif available_budget < recommended:
            return {
                "feasible": True,
                "message": f"⚠️  Budget is tight. Recommended: ${recommended:.2f}",
                "recommendation": "Will proceed with optimizations enabled",
                "optimizations": [
                    "Enable aggressive caching",
                    "Reduce sample sizes",
                    "Limit number of clusters analyzed",
                    "Use cheaper models",
                ],
                "expected_cost": min(available_budget * 0.9, recommended),
            }
        else:
            return {
                "feasible": True,
                "message": f"✅ Budget is sufficient. Expected cost: ${recommended:.2f}",
                "recommendation": "Full analysis with all optimizations",
                "buffer": available_budget - recommended,
                "confidence": "high",
            }

    def print_project_estimate(
        self, num_items: int = 80000, repo_name: str = "Repository"
    ):
        """Print formatted project cost estimate"""
        estimates = self.estimate_project_cost(num_items)

        print("\n" + "=" * 70)
        print("PROJECT COST ESTIMATE")
        print("=" * 70)
        print(f"\nTarget: {num_items:,} issues/PRs from {repo_name}")
        print(f"Budget Limit: ${self.budget_limit:.2f}\n")

        for phase, data in estimates.items():
            if phase == "total":
                continue

            cost = data.get("cost", 0)
            desc = data.get("description", phase)

            if cost == 0:
                print(f"✓ {phase.replace('_', ' ').title()}: FREE")
            else:
                print(f"✓ {phase.replace('_', ' ').title()}: ${cost:.2f}")
            print(f"   {desc}")

        total = estimates["total"]
        print(f"\n{'=' * 70}")
        print(f"TOTAL ESTIMATED COST: ${total['cost']:.2f}")
        print(
            f"Budget Status: {'✅ Within budget' if total['within_budget'] else '❌ Exceeds budget'}"
        )
        print(f"Percentage of budget: {total['percentage_of_budget']:.1f}%")
        print(f"Remaining buffer: ${total['budget_remaining']:.2f}")
        print("=" * 70 + "\n")

    def print_comparison(self, num_items: int = 100):
        """Print RLM vs Non-RLM comparison"""
        comparison = self.compare_rlm_vs_direct(num_items)

        print("\n" + "=" * 70)
        print(f"RLM vs DIRECT LLM COMPARISON ({num_items} items)")
        print("=" * 70)

        for approach in ["rlm", "direct"]:
            data = comparison[approach]
            print(f"\n📊 {data['approach'].upper()}")
            print(f"   API Calls: {data['calls']:,}")
            print(f"   Total Tokens: {data['total_tokens']:,.0f}")
            print(f"   Total Cost: ${data['total_cost']:.2f}")
            print(f"   Cost per item: ${data['cost_per_item']:.4f}")

            print(f"\n   Pros:")
            for pro in data["pros"]:
                print(f"      + {pro}")

            print(f"   Cons:")
            for con in data["cons"]:
                print(f"      - {con}")

        comp = comparison["comparison"]
        print(f"\n🎯 COMPARISON")
        print(
            f"   Cost Savings: ${comp['cost_savings']:.2f} ({comp['cost_savings_percentage']:+.1f}%)"
        )
        print(f"   Token Efficiency: {comp['token_efficiency']:.2f}x")
        print(f"   Call Ratio (RLM:Direct): {comp['call_ratio']:.1f}:1")

        print(f"\n💡 {comp['recommendation']}")
        print("=" * 70 + "\n")


def main():
    """Demonstrate cost estimation capabilities"""
    print("=" * 70)
    print("COST ESTIMATION & COMPARISON TOOL")
    print("=" * 70)

    calculator = CostCalculator(budget_limit=50.0)

    # 1. Project cost estimate
    calculator.print_project_estimate(num_items=80000)

    # 2. RLM vs Non-RLM comparison
    calculator.print_comparison(num_items=100)

    # 3. Budget recommendation
    print("\n" + "=" * 70)
    print("BUDGET RECOMMENDATION")
    print("=" * 70)

    for budget in [5.0, 15.0, 50.0, 100.0]:
        rec = calculator.get_budget_recommendation(budget, num_items=80000)
        print(f"\n💰 Budget: ${budget:.2f}")
        print(f"   {rec['message']}")
        print(f"   {rec['recommendation']}")


if __name__ == "__main__":
    main()
