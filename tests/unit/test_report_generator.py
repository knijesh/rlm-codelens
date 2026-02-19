"""Tests for the HTML report generator module."""

import json
import tempfile
from pathlib import Path

import pytest

from rlm_codelens.report_generator import (
    _build_executive_summary_section,
    _build_pattern_analysis_section,
    _build_refactoring_section,
    _build_rlm_insights_section,
    _health_rating,
    _md_to_html,
    generate_analysis_report,
)


@pytest.fixture
def static_data():
    """Minimal valid architecture JSON with no RLM fields."""
    return {
        "repository": "test-repo",
        "total_modules": 5,
        "total_edges": 4,
        "cycles": [],
        "hub_modules": [
            {"module": "src/service.py", "fan_in": 2, "fan_out": 2, "total": 4, "loc": 100}
        ],
        "coupling_metrics": [],
        "layers": {"src/service.py": "business", "src/api.py": "api"},
        "anti_patterns": [],
        "graph_data": {
            "nodes": [
                {"id": "src/service.py", "loc": 100, "layer": "business"},
                {"id": "src/api.py", "loc": 50, "layer": "api"},
            ],
            "links": [{"source": "src/api.py", "target": "src/service.py", "type": "import"}],
        },
    }


@pytest.fixture
def rlm_data(static_data):
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
    def test_executive_summary_static(self, static_data):
        health = _health_rating(static_data)
        html = _build_executive_summary_section(static_data, health)
        assert "5</strong> modules" in html
        assert "/100" in html
        assert "RLM" not in html or "deep analysis" not in html.lower().split("rlm")[0]

    def test_executive_summary_rlm(self, rlm_data):
        health = _health_rating(rlm_data)
        html = _build_executive_summary_section(rlm_data, health)
        assert "layered" in html
        assert "2</strong> refactoring" in html


class TestPatternSection:
    def test_pattern_section_null(self, static_data):
        html = _build_pattern_analysis_section(static_data)
        assert "--deep" in html

    def test_pattern_section_populated(self, rlm_data):
        html = _build_pattern_analysis_section(rlm_data)
        assert "layered" in html
        assert "85" in html  # confidence percentage
        assert "Clear layer separation" in html


class TestRLMInsightsSection:
    def test_rlm_insights_null(self, static_data):
        html = _build_rlm_insights_section(static_data)
        assert "--deep" in html

    def test_rlm_insights_populated(self, rlm_data):
        html = _build_rlm_insights_section(rlm_data)
        # Semantic table
        assert "Semantic Classifications" in html
        assert "service" in html
        # Hidden deps table
        assert "Hidden Dependencies" in html
        assert "dynamic_import" in html

    def test_rlm_insights_deep_ran_but_empty(self):
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
    def test_refactoring_null(self, static_data):
        html = _build_refactoring_section(static_data)
        assert "--deep" in html

    def test_refactoring_populated(self, rlm_data):
        html = _build_refactoring_section(rlm_data)
        assert "Split service.py" in html
        assert "dependency injection" in html
        # Check numbered items
        assert 'class="refactoring-number"' in html


class TestFullReport:
    def test_full_report_has_all_nav_links(self, rlm_data):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(rlm_data, f)
            json_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as out:
                out_path = out.name

            generate_analysis_report(json_path, out_path, open_browser=False)
            html = Path(out_path).read_text()

            nav_links = [
                "#executive-summary", "#summary", "#health", "#pattern",
                "#rlm-insights", "#fan-metrics", "#hubs", "#cycles",
                "#antipatterns", "#refactoring", "#layers", "#guidance",
            ]
            for link in nav_links:
                assert f'href="{link}"' in html, f"Missing nav link: {link}"
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)

    def test_full_report_has_all_sections(self, rlm_data):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(rlm_data, f)
            json_path = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as out:
                out_path = out.name

            generate_analysis_report(json_path, out_path, open_browser=False)
            html = Path(out_path).read_text()

            section_ids = [
                "executive-summary", "summary", "health", "pattern",
                "rlm-insights", "fan-metrics", "hubs", "cycles",
                "antipatterns", "refactoring", "layers", "guidance",
            ]
            for sid in section_ids:
                assert f'id="{sid}"' in html, f"Missing section: {sid}"
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)


class TestMdToHtml:
    def test_md_to_html_converts_headers(self):
        result = _md_to_html("### My Header")
        assert "<strong>" in result
        assert "My Header" in result

    def test_md_to_html_converts_code_blocks(self):
        result = _md_to_html("```python\nprint('hi')\n```")
        assert "<pre><code>" in result
        assert "print" in result

    def test_md_to_html_escapes_html(self):
        result = _md_to_html('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestHealthRating:
    def test_healthy(self):
        data = {"cycles": [], "anti_patterns": []}
        label, color, score, explanation = _health_rating(data)
        assert label == "Healthy"
        assert score == 100

    def test_critical(self):
        data = {
            "cycles": [["a", "b"]] * 5,
            "anti_patterns": [{"severity": "high"}] * 3,
        }
        label, color, score, explanation = _health_rating(data)
        assert score <= 40
