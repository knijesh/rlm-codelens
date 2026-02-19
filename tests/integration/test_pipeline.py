"""Integration smoke tests for the full analysis pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from rlm_codelens.codebase_graph import ArchitectureAnalysis, CodebaseGraphAnalyzer
from rlm_codelens.repo_scanner import RepositoryScanner
from rlm_codelens.report_generator import generate_analysis_report


@pytest.fixture(scope="module")
def self_scan():
    """Scan the rlm-codelens source tree itself."""
    repo_root = str(Path(__file__).resolve().parents[2])
    scanner = RepositoryScanner(repo_root)
    return scanner.scan()


class TestSelfScanAnalyzeReport:
    """End-to-end: scan -> analyze -> report."""

    def test_self_scan_analyze_report(self, self_scan):
        analyzer = CodebaseGraphAnalyzer(self_scan)
        analysis = analyzer.analyze()

        # Validate analysis structure
        assert isinstance(analysis.cycles, list)
        assert isinstance(analysis.anti_patterns, list)
        assert isinstance(analysis.hub_modules, list)
        assert isinstance(analysis.layers, dict)
        assert analysis.total_modules > 0

        # Generate HTML report
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as jf:
            json.dump(analysis.to_dict(), jf)
            json_path = jf.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as hf:
                html_path = hf.name

            generate_analysis_report(json_path, html_path, open_browser=False)
            html = Path(html_path).read_text()

            # Verify all section IDs present
            expected_sections = [
                "executive-summary", "summary", "health", "pattern",
                "rlm-insights", "fan-metrics", "hubs", "cycles",
                "antipatterns", "refactoring", "layers", "guidance",
            ]
            for sid in expected_sections:
                assert f'id="{sid}"' in html, f"Missing section: {sid}"

            # Basic HTML structure check
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html
            assert "<body>" in html
            assert "</body>" in html
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(html_path).unlink(missing_ok=True)


class TestAnalysisJsonRoundtrip:
    """Verify analysis can be saved to JSON and reloaded faithfully."""

    def test_analysis_json_roundtrip(self, self_scan):
        analyzer = CodebaseGraphAnalyzer(self_scan)
        analysis = analyzer.analyze()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            analysis.save(output_path)
            loaded = ArchitectureAnalysis.load(output_path)

            assert loaded.repository == analysis.repository
            assert loaded.total_modules == analysis.total_modules
            assert loaded.total_edges == analysis.total_edges
            assert len(loaded.cycles) == len(analysis.cycles)
            assert len(loaded.hub_modules) == len(analysis.hub_modules)
            assert len(loaded.coupling_metrics) == len(analysis.coupling_metrics)
            assert loaded.layers == analysis.layers
            assert len(loaded.anti_patterns) == len(analysis.anti_patterns)
        finally:
            Path(output_path).unlink(missing_ok=True)
