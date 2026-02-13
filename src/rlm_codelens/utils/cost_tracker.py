"""
Cost tracking utilities for monitoring API usage
Helps stay within budget limits
"""

import json
from datetime import datetime
from pathlib import Path
from rlm_codelens.config import BUDGET_LIMIT, BUDGET_ALERT_THRESHOLD, COSTS


class CostTracker:
    """Tracks API costs in real-time across different services"""

    def __init__(self, budget_limit=BUDGET_LIMIT, log_file=None):
        self.budget_limit = budget_limit
        # Handle case where BUDGET_ALERT_THRESHOLD might be None
        alert_pct = BUDGET_ALERT_THRESHOLD if BUDGET_ALERT_THRESHOLD is not None else 80
        self.alert_threshold = budget_limit * (alert_pct / 100)
        self.current_cost = 0.0
        self.cost_breakdown = {}
        self.calls_log = []
        self.log_file = log_file or "outputs/cost_log.json"

        # Ensure outputs directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)

    def add_embedding_call(self, tokens, model="text-embedding-3-small"):
        """Track embedding API cost"""
        cost_per_1m = COSTS.get(model, {}).get("input", 0.02)
        cost = (tokens / 1_000_000) * cost_per_1m

        self._add_cost("embeddings", model, tokens, cost)
        return cost

    def add_llm_call(self, input_tokens, output_tokens, model="gpt-3.5-turbo"):
        """Track LLM API cost"""
        model_costs = COSTS.get(model, {"input": 0.50, "output": 1.50})

        input_cost = (input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (output_tokens / 1_000_000) * model_costs["output"]
        total_cost = input_cost + output_cost

        self._add_cost(
            "llm",
            model,
            input_tokens + output_tokens,
            total_cost,
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
            },
        )
        return total_cost

    def add_rlm_call(self, result):
        """Track RLM call cost from result object"""
        if hasattr(result, "usage"):
            tokens = result.usage.total_tokens
            # Estimate based on typical input/output ratio
            input_tokens = int(tokens * 0.7)
            output_tokens = tokens - input_tokens
            return self.add_llm_call(input_tokens, output_tokens)
        return 0.0

    def _add_cost(self, category, model, tokens, cost, details=None):
        """Internal method to add cost entry"""
        self.current_cost += cost

        if category not in self.cost_breakdown:
            self.cost_breakdown[category] = {
                "total_cost": 0.0,
                "total_tokens": 0,
                "calls": 0,
                "models": {},
            }

        self.cost_breakdown[category]["total_cost"] += cost
        self.cost_breakdown[category]["total_tokens"] += tokens
        self.cost_breakdown[category]["calls"] += 1

        if model not in self.cost_breakdown[category]["models"]:
            self.cost_breakdown[category]["models"][model] = {
                "cost": 0.0,
                "tokens": 0,
                "calls": 0,
            }

        self.cost_breakdown[category]["models"][model]["cost"] += cost
        self.cost_breakdown[category]["models"][model]["tokens"] += tokens
        self.cost_breakdown[category]["models"][model]["calls"] += 1

        # Log call
        call_entry = {
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "model": model,
            "tokens": tokens,
            "cost": cost,
            "details": details or {},
        }
        self.calls_log.append(call_entry)

        # Check budget
        self._check_budget()

        # Save log
        self._save_log()

    def _check_budget(self):
        """Check if approaching budget limit"""
        percentage = (self.current_cost / self.budget_limit) * 100

        if self.current_cost >= self.budget_limit:
            raise BudgetExceededError(
                f"Budget exceeded! ${self.current_cost:.2f} / ${self.budget_limit:.2f}"
            )

        if self.current_cost >= self.alert_threshold and not hasattr(self, "_alerted"):
            print(
                f"\n⚠️  BUDGET ALERT: ${self.current_cost:.2f} / ${self.budget_limit:.2f} ({percentage:.1f}%)"
            )
            self._alerted = True

    def _save_log(self):
        """Save cost log to file"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "budget_limit": self.budget_limit,
            "current_cost": self.current_cost,
            "percentage_used": (self.current_cost / self.budget_limit) * 100,
            "breakdown": self.cost_breakdown,
            "calls": self.calls_log[-100:],  # Keep last 100 calls
        }

        with open(self.log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    def print_summary(self):
        """Print cost summary"""
        print("\n" + "=" * 60)
        print("COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost: ${self.current_cost:.2f} / ${self.budget_limit:.2f}")
        print(f"Percentage Used: {(self.current_cost / self.budget_limit) * 100:.1f}%")
        print("\nBreakdown:")

        for category, data in self.cost_breakdown.items():
            print(f"\n  {category.upper()}:")
            print(f"    Total: ${data['total_cost']:.2f}")
            print(f"    Tokens: {data['total_tokens']:,}")
            print(f"    Calls: {data['calls']}")

            for model, model_data in data["models"].items():
                print(
                    f"    - {model}: ${model_data['cost']:.2f} ({model_data['calls']} calls)"
                )

        print("=" * 60 + "\n")

    def get_summary_dict(self):
        """Get cost summary as dictionary"""
        return {
            "total_cost": self.current_cost,
            "budget_limit": self.budget_limit,
            "percentage_used": (self.current_cost / self.budget_limit) * 100,
            "breakdown": self.cost_breakdown,
            "remaining_budget": self.budget_limit - self.current_cost,
        }


class BudgetExceededError(Exception):
    """Exception raised when budget is exceeded"""

    pass


def format_cost(cost):
    """Format cost for display"""
    if cost < 0.01:
        return f"${cost * 100:.2f}¢"
    return f"${cost:.2f}"


if __name__ == "__main__":
    # Test cost tracker
    tracker = CostTracker(budget_limit=10.0)

    # Simulate some API calls
    tracker.add_embedding_call(1_000_000, "text-embedding-3-small")
    tracker.add_llm_call(1000, 500, "gpt-3.5-turbo")
    tracker.add_llm_call(2000, 1000, "gpt-3.5-turbo")

    tracker.print_summary()
