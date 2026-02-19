"""
Static codebase graph analysis using NetworkX.

Builds a directed graph of internal module imports and computes architecture
metrics: cycles, coupling, hub modules, layer detection, and anti-patterns.
No LLM calls — pure graph theory and heuristics.

Example:
    >>> analyzer = CodebaseGraphAnalyzer(structure)
    >>> analysis = analyzer.analyze()
    >>> print(f"Found {len(analysis.cycles)} circular import chains")
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from rlm_codelens.repo_scanner import RepositoryStructure

# Layer heuristic keywords mapped from path components
LAYER_PATTERNS = {
    "data": [
        "models",
        "model",
        "schema",
        "schemas",
        "orm",
        "db",
        "database",
        "entities",
        "migration",
        "migrations",
    ],
    "business": [
        "services",
        "service",
        "logic",
        "domain",
        "core",
        "engine",
        "handlers",
        "handler",
        "processors",
        "processor",
    ],
    "api": [
        "api",
        "views",
        "view",
        "routes",
        "route",
        "endpoints",
        "endpoint",
        "controllers",
        "controller",
        "rest",
        "graphql",
        "grpc",
    ],
    "util": ["utils", "util", "helpers", "helper", "common", "lib", "tools", "shared"],
    "test": ["tests", "test", "testing", "fixtures"],
    "config": ["config", "settings", "conf", "configuration", "constants"],
}


@dataclass
class CouplingMetrics:
    """Coupling metrics for a module.

    Attributes:
        module: Module path
        afferent: Number of modules that depend on this module (Ca)
        efferent: Number of modules this module depends on (Ce)
        instability: Ce / (Ca + Ce) — 0 = stable, 1 = unstable
    """

    module: str
    afferent: int = 0
    efferent: int = 0
    instability: float = 0.0


@dataclass
class AntiPattern:
    """Detected architectural anti-pattern.

    Attributes:
        type: Anti-pattern type (god_module, orphan, layer_violation, etc.)
        module: Module path involved
        details: Human-readable description
        severity: low, medium, or high
    """

    type: str
    module: str
    details: str
    severity: str = "medium"


@dataclass
class ArchitectureAnalysis:
    """Complete architecture analysis results.

    Attributes:
        repository: Repository name
        total_modules: Number of modules analyzed
        total_edges: Number of internal import edges
        cycles: List of circular import chains
        hub_modules: Top modules by connectivity
        coupling_metrics: Per-module coupling metrics
        layers: Heuristic layer assignments
        anti_patterns: Detected anti-patterns
        graph_data: D3.js-compatible graph structure (nodes + links)
        semantic_clusters: RLM-enriched layer classifications (optional)
        hidden_dependencies: RLM-discovered dynamic imports (optional)
        pattern_analysis: RLM architectural pattern analysis (optional)
        refactoring_suggestions: RLM refactoring suggestions (optional)
    """

    repository: str
    total_modules: int = 0
    total_edges: int = 0
    cycles: List[List[str]] = field(default_factory=list)
    hub_modules: List[Dict[str, Any]] = field(default_factory=list)
    coupling_metrics: List[Dict[str, Any]] = field(default_factory=list)
    layers: Dict[str, str] = field(default_factory=dict)
    anti_patterns: List[Dict[str, Any]] = field(default_factory=list)
    graph_data: Dict[str, Any] = field(default_factory=dict)

    # RLM-enriched fields (populated by architecture_analyzer.py)
    semantic_clusters: Optional[Dict[str, str]] = None
    hidden_dependencies: Optional[List[Dict[str, Any]]] = None
    pattern_analysis: Optional[Dict[str, Any]] = None
    refactoring_suggestions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    def save(self, output_path: str) -> None:
        """Save to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ArchitectureAnalysis":
        """Load from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CodebaseGraphAnalyzer:
    """Builds and analyzes a directed graph of module imports.

    Args:
        structure: A RepositoryStructure from the repo scanner.
    """

    def __init__(self, structure: RepositoryStructure):
        self.structure = structure
        self.graph = nx.DiGraph()
        self._internal_packages: Set[str] = set()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the directed graph from module imports."""
        # Collect all known internal package prefixes
        self._internal_packages = set(self.structure.packages)

        # Add nodes
        for path, module in self.structure.modules.items():
            self.graph.add_node(
                path,
                package=module.package,
                loc=module.lines_of_code,
                num_classes=len(module.classes),
                num_functions=len(module.functions),
                is_test=module.is_test,
                docstring=module.docstring or "",
            )

        # Build a lookup from package name to file path
        package_to_path: Dict[str, str] = {}
        for path, module in self.structure.modules.items():
            package_to_path[module.package] = path

        # Add edges for internal imports only
        for path, module in self.structure.modules.items():
            targets = set()

            # Direct imports: import foo.bar
            for imp in module.imports:
                target_path = self._resolve_import(imp, package_to_path)
                if target_path and target_path != path:
                    targets.add(target_path)

            # From imports: from foo.bar import baz
            for from_imp in module.from_imports:
                mod_name = from_imp.get("module", "")
                if mod_name:
                    target_path = self._resolve_import(mod_name, package_to_path)
                    if target_path and target_path != path:
                        targets.add(target_path)

            for target in targets:
                self.graph.add_edge(path, target)

        # Store stdlib/third-party imports as node attributes
        for path, module in self.structure.modules.items():
            external = []
            for imp in module.imports:
                if not self._is_internal(imp):
                    external.append(imp)
            for from_imp in module.from_imports:
                mod_name = from_imp.get("module", "")
                if mod_name and not self._is_internal(mod_name):
                    external.append(mod_name)
            if path in self.graph:
                self.graph.nodes[path]["external_imports"] = external

    def _is_internal(self, import_name: str) -> bool:
        """Check if an import refers to an internal module."""
        for pkg in self._internal_packages:
            if import_name == pkg or import_name.startswith(f"{pkg}."):
                return True
        # Also check direct module names
        for mod in self.structure.modules.values():
            if import_name == mod.package or import_name.startswith(f"{mod.package}."):
                return True
        return False

    def _resolve_import(
        self, import_name: str, package_to_path: Dict[str, str]
    ) -> Optional[str]:
        """Resolve an import name to a module file path, if internal."""
        # Direct match
        if import_name in package_to_path:
            return package_to_path[import_name]

        # Try prefix matching (e.g., 'from flask.app import Flask' -> flask/app.py)
        parts = import_name.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in package_to_path:
                return package_to_path[prefix]

        return None

    def analyze(self) -> ArchitectureAnalysis:
        """Run all analyses and return results."""
        layers = self.detect_layers()
        cycles = self.find_cycles()
        hubs = self.find_hub_modules()
        coupling = self.calculate_coupling_metrics()
        anti_patterns = self.detect_anti_patterns(layers)
        graph_data = self._build_graph_data(layers)

        return ArchitectureAnalysis(
            repository=self.structure.name,
            total_modules=self.graph.number_of_nodes(),
            total_edges=self.graph.number_of_edges(),
            cycles=cycles,
            hub_modules=hubs,
            coupling_metrics=[asdict(c) for c in coupling],
            layers=layers,
            anti_patterns=[asdict(a) for a in anti_patterns],
            graph_data=graph_data,
        )

    def find_cycles(self, max_length: int = 10) -> List[List[str]]:
        """Find circular import chains.

        Args:
            max_length: Maximum cycle length to report.

        Returns:
            List of cycles, each a list of module paths.
        """
        cycles = []
        try:
            for cycle in nx.simple_cycles(self.graph):
                if len(cycle) <= max_length:
                    cycles.append(cycle)
        except nx.NetworkXError:
            pass

        # Sort by length (shortest cycles are most actionable)
        cycles.sort(key=len)
        return cycles

    def find_hub_modules(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find modules with highest connectivity (fan-in + fan-out).

        Args:
            top_n: Number of top hub modules to return.

        Returns:
            List of dicts with module, fan_in, fan_out, total keys.
        """
        hubs = []
        for node in self.graph.nodes():
            fan_in = self.graph.in_degree(node)
            fan_out = self.graph.out_degree(node)
            hubs.append(
                {
                    "module": node,
                    "fan_in": fan_in,
                    "fan_out": fan_out,
                    "total": fan_in + fan_out,
                    "loc": self.graph.nodes[node].get("loc", 0),
                }
            )

        hubs.sort(key=lambda h: h["total"], reverse=True)
        return hubs[:top_n]

    def calculate_coupling_metrics(self) -> List[CouplingMetrics]:
        """Calculate afferent (Ca), efferent (Ce), and instability for each module."""
        metrics = []
        for node in self.graph.nodes():
            ca = self.graph.in_degree(node)  # afferent: who depends on me
            ce = self.graph.out_degree(node)  # efferent: who I depend on
            instability = ce / (ca + ce) if (ca + ce) > 0 else 0.0
            metrics.append(
                CouplingMetrics(
                    module=node,
                    afferent=ca,
                    efferent=ce,
                    instability=round(instability, 3),
                )
            )
        metrics.sort(key=lambda m: m.instability, reverse=True)
        return metrics

    def detect_layers(self) -> Dict[str, str]:
        """Assign each module to an architectural layer based on path heuristics.

        Returns:
            Dict mapping module path to layer name.
        """
        layers = {}
        for path in self.graph.nodes():
            layers[path] = self._classify_layer(path)
        return layers

    def _classify_layer(self, path: str) -> str:
        """Classify a single module path into a layer."""
        parts = Path(path).parts
        parts_lower = [p.lower() for p in parts]
        stem = Path(path).stem.lower()

        # Check path components against layer patterns
        for layer, keywords in LAYER_PATTERNS.items():
            for keyword in keywords:
                if keyword in parts_lower or keyword == stem:
                    return layer

        # Check if the module is a test (from AST analysis)
        node_data = self.graph.nodes.get(path, {})
        if node_data.get("is_test", False):
            return "test"

        # Default: classify as business logic
        return "business"

    def detect_anti_patterns(self, layers: Dict[str, str]) -> List[AntiPattern]:
        """Detect architectural anti-patterns.

        Currently detects:
        - God modules (high LOC + high fan-out)
        - Orphan modules (no imports or dependents)
        - Layer violations (lower layers importing higher layers)

        Args:
            layers: Module-to-layer mapping.

        Returns:
            List of detected anti-patterns.
        """
        patterns = []

        # Layer ordering (lower number = lower layer)
        layer_order = {
            "data": 0,
            "business": 1,
            "api": 2,
            "util": -1,
            "config": -1,
            "test": 99,
        }

        for node in self.graph.nodes():
            loc = self.graph.nodes[node].get("loc", 0)
            fan_out = self.graph.out_degree(node)
            fan_in = self.graph.in_degree(node)

            # God module: large file with many outgoing dependencies
            if loc > 500 and fan_out > 10:
                patterns.append(
                    AntiPattern(
                        type="god_module",
                        module=node,
                        details=f"Large module ({loc} LOC) with high fan-out ({fan_out}). Consider splitting.",
                        severity="high",
                    )
                )

            # Orphan module: no connections at all
            if fan_in == 0 and fan_out == 0:
                is_test = self.graph.nodes[node].get("is_test", False)
                if not is_test and not node.endswith("__init__.py"):
                    patterns.append(
                        AntiPattern(
                            type="orphan",
                            module=node,
                            details="Module has no internal imports or dependents.",
                            severity="low",
                        )
                    )

        # Layer violations
        for source, target in self.graph.edges():
            source_layer = layers.get(source, "business")
            target_layer = layers.get(target, "business")
            source_order = layer_order.get(source_layer, 1)
            target_order = layer_order.get(target_layer, 1)

            # Lower layer importing higher layer is a violation
            # (data importing api, for instance)
            if source_order >= 0 and target_order >= 0 and source_order < target_order:
                # data -> api or data -> business is a violation
                # but only if they're not both generic
                if source_layer != target_layer:
                    patterns.append(
                        AntiPattern(
                            type="layer_violation",
                            module=source,
                            details=f"Layer violation: {source_layer} layer ({source}) imports {target_layer} layer ({target}).",
                            severity="medium",
                        )
                    )

        return patterns

    def enrich_with_rlm(self, rlm_results: Dict[str, Any]) -> ArchitectureAnalysis:
        """Merge RLM semantic analysis into the architecture analysis.

        Args:
            rlm_results: Dict with optional keys: semantic_clusters,
                hidden_dependencies, pattern_analysis, refactoring_suggestions.

        Returns:
            Updated ArchitectureAnalysis.
        """
        analysis = self.analyze()
        analysis.semantic_clusters = rlm_results.get("semantic_clusters")
        analysis.hidden_dependencies = rlm_results.get("hidden_dependencies")
        analysis.pattern_analysis = rlm_results.get("pattern_analysis")
        analysis.refactoring_suggestions = rlm_results.get("refactoring_suggestions")

        # If RLM found hidden dependencies, add them as dashed edges in graph_data
        if analysis.hidden_dependencies:
            for dep in analysis.hidden_dependencies:
                analysis.graph_data.setdefault("links", []).append(
                    {
                        "source": dep.get("source", ""),
                        "target": dep.get("target", ""),
                        "type": "hidden",
                    }
                )

        return analysis

    def _build_graph_data(self, layers: Dict[str, str]) -> Dict[str, Any]:
        """Build D3.js-compatible graph data."""
        nodes = []
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            nodes.append(
                {
                    "id": node,
                    "package": data.get("package", ""),
                    "loc": data.get("loc", 0),
                    "num_classes": data.get("num_classes", 0),
                    "num_functions": data.get("num_functions", 0),
                    "is_test": data.get("is_test", False),
                    "layer": layers.get(node, "business"),
                    "fan_in": self.graph.in_degree(node),
                    "fan_out": self.graph.out_degree(node),
                    "docstring": data.get("docstring", ""),
                    "external_imports": data.get("external_imports", []),
                }
            )

        links = []
        for source, target in self.graph.edges():
            links.append(
                {
                    "source": source,
                    "target": target,
                    "type": "import",
                }
            )

        return {"nodes": nodes, "links": links}
