"""Tests for the architecture analyzer (RLM integration) module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from rlm_codelens.repo_scanner import ModuleInfo, RepositoryStructure


@pytest.fixture
def simple_structure():
    """Minimal RepositoryStructure for RLM analyzer tests."""
    modules = {
        "src/main.py": ModuleInfo(
            path="src/main.py",
            package="src.main",
            imports=["os"],
            from_imports=[{"module": "src.utils", "names": ["helper"], "level": 0}],
            classes=[],
            functions=[{"name": "main", "args": [], "decorators": [], "line": 1}],
            lines_of_code=20,
            docstring="Main entry point.",
            is_test=False,
            source='"""Main entry point."""\nimport os\nfrom src.utils import helper\ndef main(): pass\n',
        ),
        "src/utils.py": ModuleInfo(
            path="src/utils.py",
            package="src.utils",
            imports=[],
            from_imports=[],
            classes=[],
            functions=[{"name": "helper", "args": ["x"], "decorators": [], "line": 1}],
            lines_of_code=10,
            docstring="Utilities.",
            is_test=False,
            source='"""Utilities."""\ndef helper(x): return x + 1\n',
        ),
    }
    return RepositoryStructure(
        root_path="/fake/repo",
        name="test-repo",
        modules=modules,
        packages=["src"],
        entry_points=[],
        total_files=2,
        total_lines=30,
    )


class TestImportGuard:
    """Test that ImportError is handled gracefully when rlm is not installed."""

    def test_rlm_not_available_raises_import_error(self, simple_structure):
        """When rlms package is not installed, ArchitectureRLMAnalyzer should raise ImportError."""
        with patch.dict("sys.modules", {"rlms": None}):
            # Force reimport
            import rlm_codelens.architecture_analyzer as mod

            # Temporarily set RLM_AVAILABLE to False
            original = mod.RLM_AVAILABLE
            mod.RLM_AVAILABLE = False
            try:
                with pytest.raises(ImportError, match="RLM library not installed"):
                    mod.ArchitectureRLMAnalyzer(simple_structure)
            finally:
                mod.RLM_AVAILABLE = original


class TestBudgetTracker:
    """Test RLMCostTracker budget enforcement."""

    def test_budget_exceeded(self):
        from rlm_codelens.architecture_analyzer import (
            BudgetExceededError,
            RLMCostTracker,
        )

        tracker = RLMCostTracker(budget=1.0)
        tracker.total_cost = 1.5
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_budget_not_exceeded(self):
        from rlm_codelens.architecture_analyzer import RLMCostTracker

        tracker = RLMCostTracker(budget=10.0)
        tracker.total_cost = 5.0
        tracker.check_budget()  # Should not raise

    def test_record_tracks_calls(self):
        from rlm_codelens.architecture_analyzer import RLMCostTracker

        tracker = RLMCostTracker(budget=10.0)

        # Mock result with usage
        result = MagicMock()
        result.usage = MagicMock()
        result.usage.total_cost = 0.05

        tracker.record(result, "test_call")
        assert tracker.calls == 1
        assert tracker.total_cost == pytest.approx(0.05)
        assert len(tracker.call_log) == 1
        assert tracker.call_log[0]["label"] == "test_call"

    def test_summary(self):
        from rlm_codelens.architecture_analyzer import RLMCostTracker

        tracker = RLMCostTracker(budget=10.0)
        tracker.total_cost = 2.5
        tracker.calls = 3

        summary = tracker.summary()
        assert summary["budget"] == 10.0
        assert summary["total_cost"] == 2.5
        assert summary["calls"] == 3
        assert summary["remaining"] == pytest.approx(7.5)


class TestArchitectureRLMAnalyzerMocked:
    """Test ArchitectureRLMAnalyzer with mocked RLM."""

    @pytest.fixture
    def mock_rlm(self):
        """Create a mock RLM instance."""
        mock = MagicMock()
        mock.completion = MagicMock()
        return mock

    @pytest.fixture
    def analyzer(self, simple_structure, mock_rlm):
        """Create an analyzer with mocked RLM."""
        import rlm_codelens.architecture_analyzer as mod

        # Inject a fake RLM class into the module so the constructor works
        original_available = mod.RLM_AVAILABLE
        original_rlm = getattr(mod, "RLM", None)

        mod.RLM_AVAILABLE = True
        mod.RLM = MagicMock(return_value=mock_rlm)

        try:
            a = mod.ArchitectureRLMAnalyzer(
                simple_structure,
                backend="openai",
                model="gpt-4o",
                budget=10.0,
                verbose=False,
            )
            # Ensure the instance uses our mock
            a.rlm = mock_rlm
            yield a
        finally:
            mod.RLM_AVAILABLE = original_available
            if original_rlm is not None:
                mod.RLM = original_rlm
            elif hasattr(mod, "RLM"):
                delattr(mod, "RLM")

    def test_classify_modules_parses_json(self, analyzer, mock_rlm):
        result = MagicMock()
        result.response = json.dumps(
            {"src/main.py": "business", "src/utils.py": "util"}
        )
        result.usage = MagicMock()
        result.usage.total_cost = 0.01
        mock_rlm.completion.return_value = result

        classifications = analyzer.classify_modules()
        assert classifications == {"src/main.py": "business", "src/utils.py": "util"}
        mock_rlm.completion.assert_called_once()

    def test_classify_modules_bad_response(self, analyzer, mock_rlm):
        result = MagicMock()
        result.response = "not json"
        result.usage = MagicMock()
        result.usage.total_cost = 0.01
        mock_rlm.completion.return_value = result

        classifications = analyzer.classify_modules()
        assert classifications == {}  # Graceful fallback

    def test_discover_hidden_deps(self, analyzer, mock_rlm):
        result = MagicMock()
        result.response = json.dumps(
            [
                {
                    "source": "src/main.py",
                    "target": "src.plugin",
                    "type": "dynamic_import",
                    "evidence": "importlib",
                }
            ]
        )
        result.usage = MagicMock()
        result.usage.total_cost = 0.02
        mock_rlm.completion.return_value = result

        deps = analyzer.discover_hidden_deps()
        assert len(deps) == 1
        assert deps[0]["type"] == "dynamic_import"

    def test_detect_patterns(self, analyzer, mock_rlm):
        result = MagicMock()
        result.response = json.dumps(
            {
                "detected_pattern": "layered",
                "confidence": 0.85,
                "anti_patterns": ["tight coupling"],
                "reasoning": "Clear layer separation",
            }
        )
        result.usage = MagicMock()
        result.usage.total_cost = 0.01
        mock_rlm.completion.return_value = result

        patterns = analyzer.detect_patterns()
        assert patterns["detected_pattern"] == "layered"
        assert patterns["confidence"] == 0.85

    def test_suggest_refactoring(self, analyzer, mock_rlm):
        result = MagicMock()
        result.response = json.dumps(
            [
                "Extract database access from main.py into a data access layer",
                "Add dependency injection to service layer",
            ]
        )
        result.usage = MagicMock()
        result.usage.total_cost = 0.01
        mock_rlm.completion.return_value = result

        suggestions = analyzer.suggest_refactoring()
        assert len(suggestions) == 2
        assert "database" in suggestions[0].lower()

    def test_run_all(self, analyzer, mock_rlm):
        # Each call returns valid JSON
        responses = [
            json.dumps({"src/main.py": "business"}),  # classify
            json.dumps([]),  # hidden deps
            json.dumps(
                {
                    "detected_pattern": "monolith",
                    "confidence": 0.5,
                    "anti_patterns": [],
                    "reasoning": "",
                }
            ),  # patterns
            json.dumps(["Suggestion 1"]),  # refactoring
        ]
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            result = MagicMock()
            result.response = responses[min(call_count, len(responses) - 1)]
            result.usage = MagicMock()
            result.usage.total_cost = 0.01
            call_count += 1
            return result

        mock_rlm.completion.side_effect = side_effect

        results = analyzer.run_all()
        assert "semantic_clusters" in results
        assert "hidden_dependencies" in results
        assert "pattern_analysis" in results
        assert "refactoring_suggestions" in results
        assert "cost_summary" in results
        assert results["cost_summary"]["calls"] == 4

    def test_budget_enforcement(self, analyzer, mock_rlm):
        """After exceeding budget, run_all should still return partial results."""
        analyzer.cost_tracker.budget = 0.001  # Very low budget

        # First call succeeds but exceeds budget
        result = MagicMock()
        result.response = json.dumps({"src/main.py": "business"})
        result.usage = MagicMock()
        result.usage.total_cost = 0.01
        mock_rlm.completion.return_value = result

        results = analyzer.run_all()
        # Should still have results (run_all catches BudgetExceededError)
        assert "cost_summary" in results
