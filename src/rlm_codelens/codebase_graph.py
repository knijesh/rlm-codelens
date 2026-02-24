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
        self.graph: nx.DiGraph = nx.DiGraph()
        self._internal_packages: Set[str] = set()
        self._build_graph()

    def _build_graph(self) -> None:
        """Build the directed graph from module imports."""
        # Collect all known internal package prefixes
        self._internal_packages = set(self.structure.packages)
        # For src-layout projects, also consider non-prefixed package names
        for pkg in list(self._internal_packages):
            if pkg.startswith("src."):
                self._internal_packages.add(pkg[4:])

        # Build lookup tables for import resolution
        # package name -> file path (Python)
        self._package_to_path: Dict[str, str] = {}
        # file path without extension -> file path (multi-language)
        self._path_lookup: Dict[str, str] = {}
        # directory -> list of file paths (for directory-based imports like Go)
        self._dir_to_files: Dict[str, List[str]] = {}

        for path, module in self.structure.modules.items():
            self._package_to_path[module.package] = path
            # For src-layout projects, also register without "src." prefix
            # so that imports like "rlm_codelens.cli" resolve to
            # "src/rlm_codelens/cli.py" (package="src.rlm_codelens.cli").
            if module.package.startswith("src."):
                self._package_to_path[module.package[4:]] = path
            # Build path-based lookups for non-Python languages
            path_no_ext = str(Path(path).with_suffix(""))
            self._path_lookup[path_no_ext] = path
            self._path_lookup[path] = path
            # Directory-based lookup
            dir_path = str(Path(path).parent)
            self._dir_to_files.setdefault(dir_path, []).append(path)

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
                language=getattr(module, "language", "python"),
            )

        # Add edges for internal imports only
        for path, module in self.structure.modules.items():
            targets = set()
            lang = getattr(module, "language", "python")

            # Direct imports
            for imp in module.imports:
                resolved = self._resolve_import_multi(imp, path, lang)
                for target_path in resolved:
                    if target_path != path:
                        targets.add(target_path)

            # From imports (Python-specific)
            for from_imp in module.from_imports:
                mod_name = from_imp.get("module", "")
                if mod_name:
                    resolved_path = self._resolve_import(
                        mod_name, self._package_to_path
                    )
                    if resolved_path and resolved_path != path:
                        targets.add(resolved_path)

            for target in targets:
                self.graph.add_edge(path, target)

        # Store stdlib/third-party imports as node attributes
        for path, module in self.structure.modules.items():
            external = []
            lang = getattr(module, "language", "python")
            for imp in module.imports:
                if not self._is_internal_multi(imp, lang):
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

    def _resolve_import_multi(
        self, import_name: str, source_path: str, language: str
    ) -> List[str]:
        """Resolve an import to file path(s), handling language-specific conventions.

        Returns a list because some import styles (Go package imports) can
        resolve to multiple files in a directory.
        """
        if language == "python":
            result = self._resolve_import(import_name, self._package_to_path)
            return [result] if result else []

        if language == "go":
            return self._resolve_go_import(import_name, source_path)

        if language in ("javascript", "typescript"):
            return self._resolve_js_import(import_name, source_path)

        if language == "java":
            return self._resolve_java_import(import_name)

        if language == "rust":
            return self._resolve_rust_import(import_name, source_path)

        # Fallback: try path-based matching
        result = self._resolve_import(import_name, self._package_to_path)
        return [result] if result else []

    def _resolve_go_import(self, import_path: str, source_path: str) -> List[str]:
        """Resolve a Go import path to internal file(s).

        Go imports are package paths like "k8s.io/kubernetes/pkg/api".
        We match against directory suffixes in the repo.
        """
        # Strip the module prefix -- try matching directory suffixes
        parts = import_path.split("/")
        # Try progressively shorter suffixes
        for i in range(len(parts)):
            candidate_dir = "/".join(parts[i:])
            if candidate_dir in self._dir_to_files:
                return self._dir_to_files[candidate_dir]

        return []

    def _resolve_js_import(self, import_path: str, source_path: str) -> List[str]:
        """Resolve a JS/TS import path (relative imports only)."""
        if not import_path.startswith("."):
            return []  # npm package, not internal

        source_dir = str(Path(source_path).parent)
        # Resolve relative path
        resolved = str((Path(source_dir) / import_path).resolve())
        # Normalize -- strip any leading / or make relative to repo
        # Try with common extensions
        for suffix in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"):
            candidate = resolved + suffix
            # Try to match against known paths
            for known_path in self._path_lookup:
                if known_path.endswith(candidate) or candidate.endswith(known_path):
                    return [self._path_lookup[known_path]]

        # Direct match
        if resolved in self._path_lookup:
            return [self._path_lookup[resolved]]

        return []

    def _resolve_java_import(self, import_name: str) -> List[str]:
        """Resolve a Java import (e.g., com.example.Foo) to a file path."""
        # Java imports map to directory structure: com.example.Foo -> com/example/Foo.java
        path_candidate = import_name.replace(".", "/")
        # Try exact match
        if path_candidate in self._path_lookup:
            return [self._path_lookup[path_candidate]]
        # Try with .java suffix
        java_path = path_candidate + ".java"
        if java_path in self._path_lookup:
            return [self._path_lookup[java_path]]
        # Try matching suffix (package might be partial)
        for known in self._path_lookup:
            if known.endswith(path_candidate) or known.endswith(java_path):
                return [self._path_lookup[known]]
        return []

    def _resolve_rust_import(self, import_name: str, source_path: str) -> List[str]:
        """Resolve a Rust use path (e.g., crate::module::Type)."""
        parts = import_name.replace("::", "/").split("/")
        # Strip 'crate', 'self', 'super' prefixes
        while parts and parts[0] in ("crate", "self", "super"):
            parts.pop(0)
        if not parts:
            return []

        candidate = "/".join(parts)
        for suffix in ("", ".rs", "/mod.rs"):
            full = candidate + suffix
            for known in self._path_lookup:
                if known.endswith(full):
                    return [self._path_lookup[known]]
        return []

    def _is_internal_multi(self, import_name: str, language: str) -> bool:
        """Check if an import is internal, handling language conventions."""
        if language == "python":
            return self._is_internal(import_name)

        if language in ("javascript", "typescript"):
            return import_name.startswith(".")

        if language == "go":
            # Go imports that resolve to internal files are internal
            return bool(self._resolve_go_import(import_name, ""))

        if language == "java":
            return bool(self._resolve_java_import(import_name))

        if language == "rust":
            return import_name.startswith(("crate::", "self::", "super::"))

        return False

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

        # When edge density is very low relative to module count, import
        # resolution is not working for this codebase's primary language.
        # Orphan detection would just flag every module, which is noise.
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        low_edge_density = num_nodes > 20 and num_edges < num_nodes * 0.05

        # Count orphans (modules with zero connections)
        orphan_count = sum(
            1
            for n in self.graph.nodes()
            if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0
        )
        orphan_ratio = orphan_count / num_nodes if num_nodes > 0 else 0

        # Suppress individual orphan reporting when orphan ratio is too high.
        # In large repos, >50% orphans indicates import resolution limitations,
        # not actual architectural problems.
        suppress_orphans = low_edge_density or (num_nodes > 50 and orphan_ratio > 0.5)

        # Layer ordering (lower number = lower layer)
        layer_order = {
            "data": 0,
            "business": 1,
            "api": 2,
            "util": -1,
            "config": -1,
            "test": 99,
        }

        # Adaptive god-module thresholds — larger repos tend to have more
        # imports per file (especially Go with many internal packages).
        god_loc_threshold = 500 if num_nodes < 500 else 1000
        god_fanout_threshold = 10 if num_nodes < 500 else 20

        for node in self.graph.nodes():
            loc = self.graph.nodes[node].get("loc", 0)
            fan_out = self.graph.out_degree(node)
            fan_in = self.graph.in_degree(node)

            # God module: large file with many outgoing dependencies
            if loc > god_loc_threshold and fan_out > god_fanout_threshold:
                patterns.append(
                    AntiPattern(
                        type="god_module",
                        module=node,
                        details=f"Large module ({loc} LOC) with high fan-out ({fan_out}). Consider splitting.",
                        severity="high",
                    )
                )

            # Orphan module: no connections at all
            if fan_in == 0 and fan_out == 0 and not suppress_orphans:
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

        # When orphans are suppressed and there are orphans, emit a single
        # informational anti-pattern instead of thousands of individual entries.
        if suppress_orphans and orphan_count > 0:
            patterns.append(
                AntiPattern(
                    type="import_resolution_limited",
                    module="(repository-wide)",
                    details=(
                        f"{orphan_count} of {num_nodes} modules have no resolved "
                        f"imports. Import resolution may be incomplete for this "
                        f"language."
                    ),
                    severity="info",
                )
            )

        # Layer violations — only flag when at least one side has an explicit
        # (non-default) layer assignment.
        for source, target in self.graph.edges():
            source_layer = layers.get(source, "business")
            target_layer = layers.get(target, "business")
            source_order = layer_order.get(source_layer, 1)
            target_order = layer_order.get(target_layer, 1)

            # Lower layer importing higher layer is a violation
            # (data importing api, for instance)
            if source_order >= 0 and target_order >= 0 and source_order < target_order:
                if source_layer != target_layer:
                    # Require BOTH sides to have confident (explicit,
                    # non-default) layer classifications to reduce
                    # false positives in non-Python repos where path
                    # keywords like "api", "core", "service" are common.
                    source_explicit = self._has_explicit_layer(source)
                    target_explicit = self._has_explicit_layer(target)
                    if source_explicit and target_explicit:
                        patterns.append(
                            AntiPattern(
                                type="layer_violation",
                                module=source,
                                details=f"Layer violation: {source_layer} layer ({source}) imports {target_layer} layer ({target}).",
                                severity="medium",
                            )
                        )

        return patterns

    def _has_explicit_layer(self, path: str) -> bool:
        """Check if a module has an explicit (non-default) layer classification."""
        parts_lower = [p.lower() for p in Path(path).parts]
        stem = Path(path).stem.lower()
        for keywords in LAYER_PATTERNS.values():
            for keyword in keywords:
                if keyword in parts_lower or keyword == stem:
                    return True
        node_data = self.graph.nodes.get(path, {})
        if node_data.get("is_test", False):
            return True
        return False

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
            valid_nodes = {n["id"] for n in analysis.graph_data.get("nodes", [])}
            for dep in analysis.hidden_dependencies:
                source = dep.get("source", "")
                target = dep.get("target", "")
                if (
                    source
                    and target
                    and source in valid_nodes
                    and target in valid_nodes
                ):
                    analysis.graph_data.setdefault("links", []).append(
                        {
                            "source": source,
                            "target": target,
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
                    "language": data.get("language", "python"),
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
