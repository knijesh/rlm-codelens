"""Tests for the HTML report generator module."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from rlm_codelens.report_generator import (
    _build_coverage_banner,
    _build_executive_summary_section,
    _build_pattern_analysis_section,
    _build_refactoring_section,
    _build_rlm_insights_section,
    _health_rating,
    _md_to_html,
    generate_analysis_report,
)


@pytest.fixture
def static_data() -> Dict[str, Any]:
    """Minimal valid architecture JSON with no RLM fields."""
    return {
        "repository": "test-repo",
        "total_modules": 5,
        "total_edges": 4,
        "cycles": [],
        "hub_modules": [
            {
                "module": "src/service.py",
                "fan_in": 2,
                "fan_out": 2,
                "total": 4,
                "loc": 100,
            }
        ],
        "coupling_metrics": [],
        "layers": {"src/service.py": "business", "src/api.py": "api"},
        "anti_patterns": [],
        "graph_data": {
            "nodes": [
                {"id": "src/service.py", "loc": 100, "layer": "business"},
                {"id": "src/api.py", "loc": 50, "layer": "api"},
            ],
            "links": [
                {"source": "src/api.py", "target": "src/service.py", "type": "import"}
            ],
        },
    }


@pytest.fixture
def rlm_data(static_data: Dict[str, Any]) -> Dict[str, Any]:
    """Full architecture JSON with RLM deep analysis fields."""
    data = dict(static_data)
    data["pattern_analysis"] = {
        "detected_pattern": "layered",
        "confidence": 0.85,
        "anti_patterns": ["tight coupling"],
        "reasoning": "Clear layer separation observed.",
    }
    data["semantic_clusters"] = {
        "src/service.py": "business",
        "src/api.py": "api",
    }
    data["hidden_dependencies"] = [
        {
            "source": "src/api.py",
            "target": "src/service.py",
            "type": "dynamic_import",
            "evidence": "importlib.import_module",
        }
    ]
    data["refactoring_suggestions"] = [
        "Split service.py into smaller focused modules",
        "Add dependency injection to reduce coupling",
    ]
    return data


class TestExecutiveSummary:
    def test_executive_summary_static(self, static_data: Dict[str, Any]) -> None:
        health = _health_rating(static_data)
        html = _build_executive_summary_section(static_data, health)
        assert "5</strong> modules" in html
        assert "/100" in html
        assert "RLM" not in html or "deep analysis" not in html.lower().split("rlm")[0]

    def test_executive_summary_rlm(self, rlm_data: Dict[str, Any]) -> None:
        health = _health_rating(rlm_data)
        html = _build_executive_summary_section(rlm_data, health)
        assert "layered" in html
        assert "2</strong> refactoring" in html


class TestPatternSection:
    def test_pattern_section_null(self, static_data: Dict[str, Any]) -> None:
        html = _build_pattern_analysis_section(static_data)
        assert "--deep" in html

    def test_pattern_section_populated(self, rlm_data: Dict[str, Any]) -> None:
        html = _build_pattern_analysis_section(rlm_data)
        assert "layered" in html
        assert "85" in html  # confidence percentage
        assert "Clear layer separation" in html


class TestRLMInsightsSection:
    def test_rlm_insights_null(self, static_data: Dict[str, Any]) -> None:
        html = _build_rlm_insights_section(static_data)
        assert "--deep" in html

    def test_rlm_insights_populated(self, rlm_data: Dict[str, Any]) -> None:
        html = _build_rlm_insights_section(rlm_data)
        # Semantic table
        assert "Semantic Classifications" in html
        assert "service" in html
        # Hidden deps table
        assert "Hidden Dependencies" in html
        assert "dynamic_import" in html

    def test_rlm_insights_deep_ran_but_empty(self) -> None:
        """When deep analysis ran but found nothing, show 'Deep analysis found no...' not '--deep'."""
        data = {
            "pattern_analysis": {"detected_pattern": "layered", "confidence": 0.5},
            "semantic_clusters": None,
            "hidden_dependencies": None,
        }
        html = _build_rlm_insights_section(data)
        assert "Deep analysis found no" in html
        assert "--deep" not in html


class TestRefactoringSection:
    def test_refactoring_null(self, static_data: Dict[str, Any]) -> None:
        html = _build_refactoring_section(static_data)
        assert "--deep" in html

    def test_refactoring_populated(self, rlm_data: Dict[str, Any]) -> None:
        html = _build_refactoring_section(rlm_data)
        assert "Split service.py" in html
        assert "dependency injection" in html
        # Check numbered items
        assert 'class="refactoring-number"' in html


class TestFullReport:
    def test_full_report_has_all_tabs(self, rlm_data: Dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(rlm_data, f)
            json_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as out:
                out_path = out.name

            generate_analysis_report(json_path, out_path, open_browser=False)
            html = Path(out_path).read_text()

            # Check tab buttons exist
            assert "tab-btn" in html
            assert "tab-overview" in html
            assert "tab-architecture" in html
            assert "tab-issues" in html
            assert "tab-deep" in html  # rlm_data has deep fields
            assert "tab-guidance" in html
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    def test_full_report_has_all_sections(self, rlm_data: Dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(rlm_data, f)
            json_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as out:
                out_path = out.name

            generate_analysis_report(json_path, out_path, open_browser=False)
            html = Path(out_path).read_text()

            section_ids = [
                "executive-summary",
                "summary",
                "health",
                "pattern",
                "rlm-insights",
                "fan-metrics",
                "hubs",
                "cycles",
                "antipatterns",
                "refactoring",
                "layers",
                "guidance",
            ]
            for sid in section_ids:
                assert f'id="{sid}"' in html, f"Missing section: {sid}"

            # Verify sections are inside correct tab panels
            overview_start = html.index('id="tab-overview"')
            arch_start = html.index('id="tab-architecture"')
            issues_start = html.index('id="tab-issues"')

            assert html.index('id="executive-summary"') > overview_start
            assert html.index('id="summary"') > overview_start
            assert html.index('id="health"') > overview_start

            assert html.index('id="pattern"') > arch_start
            assert html.index('id="layers"') > arch_start

            assert html.index('id="antipatterns"') > issues_start
            assert html.index('id="cycles"') > issues_start
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    def test_full_report_no_deep_tab_without_data(
        self, static_data: Dict[str, Any]
    ) -> None:
        """Deep Analysis tab should not appear when no deep data exists."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(static_data, f)
            json_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as out:
                out_path = out.name

            generate_analysis_report(json_path, out_path, open_browser=False)
            html = Path(out_path).read_text()

            assert "tab-deep" not in html
            assert "Deep Analysis" not in html
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)


class TestEmptyDict:
    def test_empty_pattern_analysis_shows_not_available(self) -> None:
        """pattern_analysis={} should render as 'not available', not 'Unknown 0%'."""
        data: Dict[str, Any] = {"pattern_analysis": {}}
        html = _build_pattern_analysis_section(data)
        assert "Unknown" not in html
        assert "0%" not in html
        assert "--deep" in html

    def test_empty_pattern_analysis_not_in_executive_summary(self) -> None:
        """pattern_analysis={} should not produce an RLM pattern sentence."""
        data = {
            "repository": "test",
            "total_modules": 5,
            "total_edges": 3,
            "cycles": [],
            "anti_patterns": [],
            "graph_data": {"nodes": [], "links": []},
            "pattern_analysis": {},
        }
        health = _health_rating(data)
        html = _build_executive_summary_section(data, health)
        assert "Unknown" not in html
        assert "0%" not in html


class TestCoverageBanner:
    def test_banner_appears_when_low_coverage(self) -> None:
        """Banner should appear when <50% of modules have connections."""
        data = {
            "total_modules": 10,
            "graph_data": {
                "nodes": [
                    {"id": f"m{i}", "fan_in": (1 if i < 3 else 0), "fan_out": 0}
                    for i in range(10)
                ]
            },
        }
        html = _build_coverage_banner(data)
        assert "coverage-banner" in html
        assert "30%" in html

    def test_banner_hidden_when_high_coverage(self) -> None:
        """Banner should not appear when >=50% of modules have connections."""
        data = {
            "total_modules": 10,
            "graph_data": {
                "nodes": [{"id": f"m{i}", "fan_in": 1, "fan_out": 0} for i in range(10)]
            },
        }
        html = _build_coverage_banner(data)
        assert html == ""


class TestMdToHtml:
    def test_md_to_html_converts_headers(self) -> None:
        result = _md_to_html("### My Header")
        assert "<strong>" in result
        assert "My Header" in result

    def test_md_to_html_converts_code_blocks(self) -> None:
        result = _md_to_html("```python\nprint('hi')\n```")
        assert "<pre><code>" in result
        assert "print" in result

    def test_md_to_html_escapes_html(self) -> None:
        result = _md_to_html('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestHealthRating:
    def test_healthy(self) -> None:
        data: Dict[str, Any] = {"cycles": [], "anti_patterns": []}
        label, color, score, explanation = _health_rating(data)
        assert label == "Healthy"
        assert score == 100

    def test_critical(self) -> None:
        data = {
            "cycles": [["a", "b"]] * 5,
            "anti_patterns": [{"severity": "high"}] * 3,
        }
        label, color, score, explanation = _health_rating(data)
        assert score <= 40
