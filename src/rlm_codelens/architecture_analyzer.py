"""
RLM-powered architecture analysis for deep codebase intelligence.

Uses the real RLM API to perform semantic analysis that goes beyond static
AST parsing: module classification, hidden dependency discovery, architectural
pattern detection, and refactoring suggestions.

Requires: pip install 'rlm-codelens[rlm]'  (or the rlm package directly)

Example:
    >>> analyzer = ArchitectureRLMAnalyzer(scan_data, backend="openai", model="gpt-4o")
    >>> results = analyzer.run_all()
    >>> print(results["semantic_clusters"])

    # Using Ollama (local, free):
    >>> analyzer = ArchitectureRLMAnalyzer(
    ...     scan_data, backend="openai", model="llama3.1",
    ...     base_url="http://localhost:11434/v1", api_key="ollama",
    ... )
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rlm_codelens.repo_scanner import RepositoryStructure

try:
    from rlm import RLM  # type: ignore

    RLM_AVAILABLE = True
except ImportError:
    RLM_AVAILABLE = False


@dataclass
class RLMCostTracker:
    """Tracks RLM API usage and enforces a budget limit.

    Attributes:
        budget: Maximum allowed spend in USD
        total_cost: Accumulated cost so far
        calls: Number of RLM completion calls made
        call_log: List of per-call cost records
    """

    budget: float = 10.0
    total_cost: float = 0.0
    calls: int = 0
    call_log: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, result: Any, label: str = "") -> None:
        """Record usage from an RLM completion result.

        The RLM library returns RLMChatCompletion with a usage_summary
        containing per-model token counts. We estimate cost from tokens.
        """
        cost = 0.0
        total_tokens = 0

        # RLMChatCompletion has .usage_summary.model_usage_summaries
        usage_summary = getattr(result, "usage_summary", None)
        if usage_summary:
            model_summaries = getattr(usage_summary, "model_usage_summaries", {})
            if isinstance(model_summaries, dict):
                for model_usage in model_summaries.values():
                    input_tokens = getattr(model_usage, "total_input_tokens", 0) or 0
                    output_tokens = getattr(model_usage, "total_output_tokens", 0) or 0
                    total_tokens += input_tokens + output_tokens
                    # Rough cost estimate: $2.50/1M input, $10/1M output (gpt-4o rates)
                    cost += (input_tokens / 1_000_000) * 2.50
                    cost += (output_tokens / 1_000_000) * 10.0

        # Fallback: check for .usage.total_cost (legacy/mock interface)
        if cost == 0.0:
            usage = getattr(result, "usage", None)
            if usage:
                cost = getattr(usage, "total_cost", 0.0) or 0.0

        self.total_cost += cost
        self.calls += 1
        self.call_log.append(
            {
                "label": label,
                "cost": round(cost, 6),
                "tokens": total_tokens,
                "cumulative": round(self.total_cost, 6),
            }
        )

    def check_budget(self) -> None:
        """Raise if budget is exceeded."""
        if self.total_cost >= self.budget:
            raise BudgetExceededError(
                f"RLM budget exceeded: ${self.total_cost:.2f} >= ${self.budget:.2f}"
            )

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict."""
        return {
            "budget": self.budget,
            "total_cost": round(self.total_cost, 4),
            "calls": self.calls,
            "remaining": round(self.budget - self.total_cost, 4),
        }


class BudgetExceededError(Exception):
    """Raised when the RLM cost budget has been exceeded."""


_MARKDOWN_FENCE_RE = re.compile(
    r"```\w*\s*\n(.*?)\n\s*```",
    re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    """Extract JSON from an RLM response, handling markdown fences and extra text.

    Handles responses wrapped in ```json ... ```, ```repl ... ```,
    or any other fenced code block. Also handles bare JSON with
    surrounding prose.
    """
    # Try fenced code block first
    match = _MARKDOWN_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    # Try to extract bare JSON object or array from the text
    # Pick whichever bracket type appears first
    obj_start = text.find("{")
    arr_start = text.find("[")

    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        end = text.rfind("]")
        if end > arr_start:
            return text[arr_start : end + 1]

    if obj_start != -1:
        end = text.rfind("}")
        if end > obj_start:
            return text[obj_start : end + 1]

    return text.strip()


class ArchitectureRLMAnalyzer:
    """Performs deep architecture analysis using RLM.

    Args:
        structure: Scanned repository structure.
        backend: RLM backend name (e.g., "openai", "anthropic").
        model: Model name (e.g., "gpt-4o", "llama3.1").
        api_key: API key for the backend (reads from env if not provided).
        base_url: Override base URL for the backend API (e.g.,
            "http://localhost:11434/v1" for Ollama).
        environment: RLM environment ("local" or "docker").
        max_iterations: Max RLM iterations per completion call.
        budget: Maximum spend in USD.
        verbose: Print progress messages.
    """

    def __init__(
        self,
        structure: RepositoryStructure,
        backend: str = "openai",
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        environment: str = "local",
        max_iterations: int = 30,
        budget: float = 10.0,
        verbose: bool = True,
    ):
        if not RLM_AVAILABLE:
            raise ImportError(
                "RLM library not installed. Install with:\n"
                "  pip install 'rlm-codelens[rlm]'\n"
                "Or install directly:\n"
                "  pip install rlms"
            )

        self.structure = structure
        self.verbose = verbose
        self.cost_tracker = RLMCostTracker(budget=budget)

        # Build backend kwargs
        backend_kwargs: Dict[str, Any] = {"model_name": model}
        if api_key:
            backend_kwargs["api_key"] = api_key
        if base_url:
            backend_kwargs["base_url"] = base_url
            # Local servers (Ollama, vLLM) don't need a real API key,
            # but the OpenAI client library requires a non-empty string.
            if not api_key:
                backend_kwargs.setdefault("api_key", "ollama")

        self.rlm = RLM(
            backend=backend,
            backend_kwargs=backend_kwargs,
            environment=environment,
            max_depth=1,
            max_iterations=max_iterations,
            verbose=verbose,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [RLM] {msg}")

    def _build_module_summary(self) -> str:
        """Build a compact text summary of all modules for RLM context."""
        lines = []
        for path, mod in self.structure.modules.items():
            classes = ", ".join(c["name"] for c in mod.classes) or "none"
            functions = ", ".join(f["name"] for f in mod.functions) or "none"
            imports = mod.imports + [fi["module"] for fi in mod.from_imports]
            imports_str = ", ".join(imports[:10]) or "none"
            line = (
                f"- {path} ({mod.lines_of_code} LOC) | "
                f"pkg={mod.package} | classes=[{classes}] | "
                f"functions=[{functions}] | imports=[{imports_str}]"
            )
            if mod.docstring:
                doc_preview = mod.docstring[:80].replace("\n", " ")
                line += f' | doc="{doc_preview}"'
            lines.append(line)
        return "\n".join(lines)

    def classify_modules(self) -> Dict[str, str]:
        """Classify each module into an architectural layer.

        Layers: data, business, api, util, test, config

        Returns:
            Dict mapping module path to layer name.
        """
        self._log("Classifying modules into architectural layers...")
        self.cost_tracker.check_budget()

        module_summary = self._build_module_summary()

        prompt = f"""You are analyzing a Python codebase. Classify each module into exactly one architectural layer.

Layers:
- data: Models, schemas, ORM, database, migrations
- business: Core logic, services, domain, handlers, processors
- api: Routes, views, endpoints, controllers, REST/GraphQL
- util: Utilities, helpers, common, shared libraries
- test: Test files, fixtures, conftest
- config: Configuration, settings, constants

Modules:
{module_summary}

For each module, output a JSON object mapping the module path to its layer.
Example: {{"src/app/models.py": "data", "src/app/views.py": "api"}}

Output ONLY the JSON object, no other text."""

        result = self.rlm.completion(prompt=prompt)

        self.cost_tracker.record(result, "classify_modules")

        # Parse response
        try:
            classifications = json.loads(_strip_markdown_fences(result.response))
            if isinstance(classifications, dict):
                # Filter out keys not in structure.modules (unknown modules)
                known = set(self.structure.modules.keys())
                unknown = set(classifications.keys()) - known
                if unknown:
                    self._log(
                        f"Warning: Dropping {len(unknown)} unknown module(s) from classification: {sorted(unknown)[:5]}"
                    )
                return {k: v for k, v in classifications.items() if k in known}
        except (json.JSONDecodeError, TypeError):
            pass

        self._log(
            "Warning: Could not parse RLM classification response, using heuristic fallback"
        )
        return {}

    def discover_hidden_deps(self) -> List[Dict[str, Any]]:
        """Discover dynamic imports and hidden dependencies.

        Looks for importlib.import_module(), __import__(), plugin registries,
        getattr()-based module loading, etc.

        Returns:
            List of dicts with source, target, type, evidence keys.
        """
        self._log("Discovering hidden dependencies...")
        self.cost_tracker.check_budget()

        # Build source context for modules that have source
        source_modules = {}
        for path, mod in self.structure.modules.items():
            if mod.source:
                source_modules[path] = mod.source

        if not source_modules:
            self._log("No source code available for hidden dependency analysis")
            return []

        # Truncate if too much source
        max_chars = 50000
        context_lines = []
        total = 0
        for path, source in source_modules.items():
            if total + len(source) > max_chars:
                break
            context_lines.append(f"### {path}\n```python\n{source}\n```\n")
            total += len(source)

        context = "\n".join(context_lines)

        prompt = f"""Analyze these Python source files for hidden/dynamic dependencies that AST import analysis would miss.

Look for:
1. importlib.import_module() calls
2. __import__() calls
3. Plugin/registry patterns that load modules by string name
4. getattr() on modules
5. exec/eval that import modules
6. Dynamic class loading patterns

Source files:
{context}

Output a JSON array of hidden dependencies found. Each entry:
{{"source": "path/to/file.py", "target": "inferred.module", "type": "dynamic_import|plugin|registry|getattr", "evidence": "the code snippet"}}

If none found, output an empty array: []
Output ONLY the JSON array."""

        result = self.rlm.completion(prompt=prompt)

        self.cost_tracker.record(result, "discover_hidden_deps")

        try:
            deps = json.loads(_strip_markdown_fences(result.response))
            if isinstance(deps, list):
                required_keys = {"source", "target", "type", "evidence"}
                validated = []
                for item in deps:
                    if not isinstance(item, dict):
                        continue
                    missing = required_keys - set(item.keys())
                    if missing:
                        self._log(
                            f"Warning: Dropping hidden dep missing keys {missing}: {item}"
                        )
                        continue
                    if item["source"] == item["target"]:
                        self._log(
                            f"Warning: Dropping self-referencing hidden dep: {item['source']}"
                        )
                        continue
                    validated.append(item)
                return validated
        except (json.JSONDecodeError, TypeError):
            pass

        return []

    def detect_patterns(
        self, graph_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect architectural patterns and anti-patterns.

        Args:
            graph_metrics: Optional dict with cycles, hub_modules, coupling_metrics.

        Returns:
            Dict with detected_pattern, confidence, anti_patterns, reasoning.
        """
        self._log("Detecting architectural patterns...")
        self.cost_tracker.check_budget()

        module_summary = self._build_module_summary()

        metrics_context = ""
        if graph_metrics:
            metrics_context = f"""
Graph Metrics:
- Cycles: {json.dumps(graph_metrics.get('cycles', [])[:5])}
- Hub modules: {json.dumps(graph_metrics.get('hub_modules', [])[:5])}
- Total modules: {graph_metrics.get('total_modules', 'unknown')}
- Total edges: {graph_metrics.get('total_edges', 'unknown')}
"""

        prompt = f"""Analyze this Python codebase's architecture and identify its architectural pattern(s).

Modules:
{module_summary}
{metrics_context}

Identify:
1. Primary architectural pattern (e.g., MVC, layered, hexagonal, microkernel, pipe-and-filter, monolith, modular monolith)
2. Confidence level (0.0-1.0)
3. Anti-patterns present (e.g., circular deps, god modules, tight coupling, leaky abstractions)
4. Brief reasoning

Output a JSON object:
{{
  "detected_pattern": "pattern name",
  "confidence": 0.8,
  "anti_patterns": ["list", "of", "anti-patterns"],
  "reasoning": "brief explanation"
}}

Output ONLY the JSON."""

        result = self.rlm.completion(prompt=prompt)

        self.cost_tracker.record(result, "detect_patterns")

        try:
            patterns = json.loads(_strip_markdown_fences(result.response))
            if isinstance(patterns, dict):
                # Ensure required keys exist with sensible defaults
                patterns.setdefault("detected_pattern", "unknown")
                patterns.setdefault("confidence", 0.0)
                patterns.setdefault("anti_patterns", [])
                patterns.setdefault("reasoning", "")

                # Clamp confidence to [0.0, 1.0]
                try:
                    conf = float(patterns["confidence"])
                    patterns["confidence"] = max(0.0, min(1.0, conf))
                except (ValueError, TypeError):
                    patterns["confidence"] = 0.0

                # Ensure anti_patterns is a list
                if not isinstance(patterns["anti_patterns"], list):
                    patterns["anti_patterns"] = [patterns["anti_patterns"]]

                return patterns
        except (json.JSONDecodeError, TypeError):
            pass

        return {
            "detected_pattern": "unknown",
            "confidence": 0.0,
            "anti_patterns": [],
            "reasoning": "Could not parse RLM response",
        }

    def suggest_refactoring(
        self,
        classifications: Optional[Dict[str, str]] = None,
        anti_patterns: Optional[List[Dict[str, Any]]] = None,
        cycles: Optional[List[List[str]]] = None,
    ) -> List[str]:
        """Generate actionable refactoring suggestions.

        Args:
            classifications: Module layer classifications.
            anti_patterns: Detected anti-patterns.
            cycles: Circular import chains.

        Returns:
            List of refactoring suggestion strings.
        """
        self._log("Generating refactoring suggestions...")
        self.cost_tracker.check_budget()

        context_parts = []
        if classifications:
            context_parts.append(
                f"Module classifications: {json.dumps(classifications)}"
            )
        if anti_patterns:
            context_parts.append(f"Anti-patterns: {json.dumps(anti_patterns[:10])}")
        if cycles:
            context_parts.append(f"Circular imports: {json.dumps(cycles[:5])}")

        context = "\n".join(context_parts) or "No prior analysis available."

        module_summary = self._build_module_summary()

        prompt = f"""Based on the following codebase analysis, provide actionable refactoring suggestions.

Codebase modules:
{module_summary}

Analysis results:
{context}

Provide 3-7 specific, actionable refactoring suggestions. Each should:
- Name the specific module(s) involved
- Describe what to change
- Explain why it improves the architecture

Output a JSON array of strings, each a complete suggestion.
Output ONLY the JSON array."""

        result = self.rlm.completion(prompt=prompt)

        self.cost_tracker.record(result, "suggest_refactoring")

        try:
            suggestions = json.loads(_strip_markdown_fences(result.response))
            if isinstance(suggestions, list):
                return [str(s) for s in suggestions]
        except (json.JSONDecodeError, TypeError):
            pass

        return []

    def run_all(self, graph_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run all RLM analysis steps and return combined results.

        Args:
            graph_metrics: Optional static analysis metrics to inform RLM.

        Returns:
            Dict with semantic_clusters, hidden_dependencies,
            pattern_analysis, refactoring_suggestions, cost_summary.
        """
        results: Dict[str, Any] = {}

        # Step 1: Classify modules
        try:
            results["semantic_clusters"] = self.classify_modules()
        except (BudgetExceededError, Exception) as e:
            self._log(f"classify_modules failed: {e}")
            results["semantic_clusters"] = {}

        # Step 2: Discover hidden dependencies
        try:
            results["hidden_dependencies"] = self.discover_hidden_deps()
        except (BudgetExceededError, Exception) as e:
            self._log(f"discover_hidden_deps failed: {e}")
            results["hidden_dependencies"] = []

        # Step 3: Detect patterns
        try:
            results["pattern_analysis"] = self.detect_patterns(graph_metrics)
        except (BudgetExceededError, Exception) as e:
            self._log(f"detect_patterns failed: {e}")
            results["pattern_analysis"] = {}

        # Step 4: Suggest refactoring
        try:
            results["refactoring_suggestions"] = self.suggest_refactoring(
                classifications=results.get("semantic_clusters"),
                anti_patterns=(
                    graph_metrics.get("anti_patterns") if graph_metrics else None
                ),
                cycles=graph_metrics.get("cycles") if graph_metrics else None,
            )
        except (BudgetExceededError, Exception) as e:
            self._log(f"suggest_refactoring failed: {e}")
            results["refactoring_suggestions"] = []

        results["cost_summary"] = self.cost_tracker.summary()
        return results
