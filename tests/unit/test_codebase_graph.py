"""Tests for the codebase graph analysis module."""

import tempfile
from pathlib import Path

import pytest

from rlm_codelens.codebase_graph import (
    ArchitectureAnalysis,
    CodebaseGraphAnalyzer,
)
from rlm_codelens.repo_scanner import ModuleInfo, RepositoryStructure


@pytest.fixture
def sample_structure():
    """Build a RepositoryStructure by hand for focused graph tests."""
    modules = {
        "src/app/models.py": ModuleInfo(
            path="src/app/models.py",
            package="src.app.models",
            imports=["dataclasses"],
            from_imports=[],
            classes=[{"name": "User", "bases": [], "methods": [], "line": 5}],
            functions=[],
            lines_of_code=30,
            docstring="Data models.",
            is_test=False,
        ),
        "src/app/service.py": ModuleInfo(
            path="src/app/service.py",
            package="src.app.service",
            imports=[],
            from_imports=[
                {"module": "src.app.models", "names": ["User"], "level": 0},
                {"module": "src.app.utils", "names": ["helper"], "level": 0},
            ],
            classes=[],
            functions=[
                {"name": "process", "args": ["data"], "decorators": [], "line": 3}
            ],
            lines_of_code=50,
            docstring="Business logic.",
            is_test=False,
        ),
        "src/app/api.py": ModuleInfo(
            path="src/app/api.py",
            package="src.app.api",
            imports=[],
            from_imports=[
                {"module": "src.app.service", "names": ["process"], "level": 0},
            ],
            classes=[],
            functions=[
                {"name": "handle_request", "args": ["req"], "decorators": [], "line": 5}
            ],
            lines_of_code=40,
            docstring="API layer.",
            is_test=False,
        ),
        "src/app/utils.py": ModuleInfo(
            path="src/app/utils.py",
            package="src.app.utils",
            imports=[],
            from_imports=[],
            classes=[],
            functions=[{"name": "helper", "args": ["x"], "decorators": [], "line": 1}],
            lines_of_code=15,
            docstring="Utilities.",
            is_test=False,
        ),
        "src/app/config.py": ModuleInfo(
            path="src/app/config.py",
            package="src.app.config",
            imports=["os"],
            from_imports=[],
            classes=[],
            functions=[],
            lines_of_code=10,
            docstring="Config.",
            is_test=False,
        ),
        "tests/test_service.py": ModuleInfo(
            path="tests/test_service.py",
            package="tests.test_service",
            imports=["pytest"],
            from_imports=[
                {"module": "src.app.service", "names": ["process"], "level": 0},
            ],
            classes=[],
            functions=[
                {"name": "test_process", "args": [], "decorators": [], "line": 5}
            ],
            lines_of_code=20,
            docstring=None,
            is_test=True,
        ),
    }

    return RepositoryStructure(
        root_path="/fake/repo",
        name="test-repo",
        modules=modules,
        packages=["src.app", "tests"],
        entry_points=[],
        total_files=len(modules),
        total_lines=sum(m.lines_of_code for m in modules.values()),
    )


@pytest.fixture
def cycle_structure():
    """Structure with a circular import."""
    modules = {
        "a.py": ModuleInfo(
            path="a.py",
            package="a",
            imports=[],
            from_imports=[{"module": "b", "names": ["x"], "level": 0}],
            classes=[],
            functions=[],
            lines_of_code=10,
        ),
        "b.py": ModuleInfo(
            path="b.py",
            package="b",
            imports=[],
            from_imports=[{"module": "a", "names": ["y"], "level": 0}],
            classes=[],
            functions=[],
            lines_of_code=10,
        ),
    }
    return RepositoryStructure(
        root_path="/fake",
        name="cycle-repo",
        modules=modules,
        packages=["a", "b"],
        entry_points=[],
        total_files=2,
        total_lines=20,
    )


class TestCodebaseGraphAnalyzer:
    """Tests for CodebaseGraphAnalyzer."""

    def test_graph_has_nodes(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        assert analyzer.graph.number_of_nodes() == len(sample_structure.modules)

    def test_graph_has_internal_edges(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        # service imports models and utils; api imports service; test imports service
        assert analyzer.graph.number_of_edges() >= 3

    def test_external_imports_not_edges(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        # 'os', 'dataclasses', 'pytest' should not be graph nodes
        for node in analyzer.graph.nodes():
            assert node not in ("os", "dataclasses", "pytest")

    def test_find_cycles_empty(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        cycles = analyzer.find_cycles()
        assert cycles == []

    def test_find_cycles_detected(self, cycle_structure):
        analyzer = CodebaseGraphAnalyzer(cycle_structure)
        cycles = analyzer.find_cycles()
        assert len(cycles) > 0
        # The cycle should contain both a.py and b.py
        cycle_modules = set()
        for cycle in cycles:
            cycle_modules.update(cycle)
        assert "a.py" in cycle_modules
        assert "b.py" in cycle_modules

    def test_find_hub_modules(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        hubs = analyzer.find_hub_modules(top_n=3)
        assert len(hubs) > 0
        assert all("module" in h for h in hubs)
        assert all("fan_in" in h for h in hubs)
        assert all("fan_out" in h for h in hubs)
        # service.py has the most connections (imports 2, imported by 2)
        hub_modules = [h["module"] for h in hubs]
        assert "src/app/service.py" in hub_modules

    def test_coupling_metrics(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        metrics = analyzer.calculate_coupling_metrics()
        assert len(metrics) == len(sample_structure.modules)
        for m in metrics:
            assert 0.0 <= m.instability <= 1.0

    def test_detect_layers(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        layers = analyzer.detect_layers()
        assert layers["tests/test_service.py"] == "test"
        assert layers["src/app/utils.py"] == "util"
        assert layers["src/app/config.py"] == "config"
        assert layers["src/app/api.py"] == "api"

    def test_detect_anti_patterns_orphan(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        layers = analyzer.detect_layers()
        patterns = analyzer.detect_anti_patterns(layers)
        # config.py has no internal imports/dependents = orphan
        orphans = [p for p in patterns if p.type == "orphan"]
        orphan_modules = [p.module for p in orphans]
        assert "src/app/config.py" in orphan_modules

    def test_analyze_returns_complete_results(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        analysis = analyzer.analyze()
        assert isinstance(analysis, ArchitectureAnalysis)
        assert analysis.total_modules > 0
        assert analysis.total_edges >= 0
        assert isinstance(analysis.cycles, list)
        assert isinstance(analysis.hub_modules, list)
        assert isinstance(analysis.coupling_metrics, list)
        assert isinstance(analysis.layers, dict)
        assert isinstance(analysis.anti_patterns, list)
        assert "nodes" in analysis.graph_data
        assert "links" in analysis.graph_data

    def test_graph_data_nodes_have_layer(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        analysis = analyzer.analyze()
        for node in analysis.graph_data["nodes"]:
            assert "layer" in node
            assert node["layer"] in (
                "data",
                "business",
                "api",
                "util",
                "test",
                "config",
            )

    def test_enrich_with_rlm(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        rlm_results = {
            "semantic_clusters": {"src/app/models.py": "data"},
            "hidden_dependencies": [
                {
                    "source": "src/app/api.py",
                    "target": "src/app/models.py",
                    "type": "dynamic_import",
                }
            ],
            "pattern_analysis": {"detected_pattern": "layered", "confidence": 0.8},
            "refactoring_suggestions": ["Split service.py into smaller modules"],
        }
        analysis = analyzer.enrich_with_rlm(rlm_results)
        assert analysis.semantic_clusters == {"src/app/models.py": "data"}
        assert len(analysis.hidden_dependencies) == 1
        # Hidden dep should be added to graph_data links
        hidden_links = [
            link
            for link in analysis.graph_data["links"]
            if link.get("type") == "hidden"
        ]
        assert len(hidden_links) == 1


class TestArchitectureAnalysis:
    """Tests for ArchitectureAnalysis serialization."""

    def test_save_and_load(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        analysis = analyzer.analyze()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        analysis.save(output_path)
        loaded = ArchitectureAnalysis.load(output_path)

        assert loaded.repository == analysis.repository
        assert loaded.total_modules == analysis.total_modules
        assert loaded.total_edges == analysis.total_edges
        assert len(loaded.cycles) == len(analysis.cycles)

        Path(output_path).unlink()

    def test_to_dict(self, sample_structure):
        analyzer = CodebaseGraphAnalyzer(sample_structure)
        analysis = analyzer.analyze()
        d = analysis.to_dict()
        assert isinstance(d, dict)
        assert "repository" in d
        assert "graph_data" in d


class TestAntiPatternDetection:
    """Tests for specific anti-pattern detection in detect_anti_patterns."""

    def test_detect_anti_patterns_god_module(self):
        """A module with LOC > 500 and fan_out > 10 should be detected as god_module."""
        # god.py imports 12 other modules
        modules = {
            "god.py": ModuleInfo(
                path="god.py",
                package="god",
                imports=[],
                from_imports=[
                    {"module": f"mod{i}", "names": ["x"], "level": 0} for i in range(12)
                ],
                classes=[],
                functions=[],
                lines_of_code=600,
                is_test=False,
            )
        }
        # Add the 12 target modules
        for i in range(12):
            modules[f"mod{i}.py"] = ModuleInfo(
                path=f"mod{i}.py",
                package=f"mod{i}",
                imports=[],
                from_imports=[],
                classes=[],
                functions=[],
                lines_of_code=20,
                is_test=False,
            )

        structure = RepositoryStructure(
            root_path="/fake",
            name="god-test",
            modules=modules,
            packages=["god"] + [f"mod{i}" for i in range(12)],
            entry_points=[],
            total_files=13,
            total_lines=840,
        )
        analyzer = CodebaseGraphAnalyzer(structure)
        layers = analyzer.detect_layers()
        patterns = analyzer.detect_anti_patterns(layers)
        god_patterns = [p for p in patterns if p.type == "god_module"]
        assert len(god_patterns) == 1
        assert god_patterns[0].module == "god.py"
        assert god_patterns[0].severity == "high"

    def test_detect_anti_patterns_layer_violation(self):
        """A 'data' layer module importing an 'api' layer module should be a layer_violation."""
        modules = {
            "src/models.py": ModuleInfo(
                path="src/models.py",
                package="src.models",
                imports=[],
                from_imports=[{"module": "src.api", "names": ["handler"], "level": 0}],
                classes=[],
                functions=[],
                lines_of_code=30,
                is_test=False,
            ),
            "src/api.py": ModuleInfo(
                path="src/api.py",
                package="src.api",
                imports=[],
                from_imports=[],
                classes=[],
                functions=[],
                lines_of_code=30,
                is_test=False,
            ),
        }
        structure = RepositoryStructure(
            root_path="/fake",
            name="layer-test",
            modules=modules,
            packages=["src.models", "src.api"],
            entry_points=[],
            total_files=2,
            total_lines=60,
        )
        analyzer = CodebaseGraphAnalyzer(structure)
        layers = analyzer.detect_layers()
        assert layers["src/models.py"] == "data"
        assert layers["src/api.py"] == "api"
        patterns = analyzer.detect_anti_patterns(layers)
        violations = [p for p in patterns if p.type == "layer_violation"]
        assert len(violations) >= 1
        assert violations[0].module == "src/models.py"
        assert violations[0].severity == "medium"

    def test_find_cycles_reports_correct_cycle(self, cycle_structure):
        """Verify the actual cycle path contains both a.py and b.py."""
        analyzer = CodebaseGraphAnalyzer(cycle_structure)
        cycles = analyzer.find_cycles()
        assert len(cycles) > 0
        # Each cycle should be a list containing both modules
        found_cycle = False
        for cycle in cycles:
            if "a.py" in cycle and "b.py" in cycle:
                found_cycle = True
                assert len(cycle) == 2
        assert found_cycle, f"Expected cycle with a.py and b.py, got: {cycles}"
