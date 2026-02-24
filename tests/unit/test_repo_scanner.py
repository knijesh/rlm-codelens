"""Tests for the repository scanner module."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from rlm_codelens.repo_scanner import (
    RepositoryScanner,
    RepositoryStructure,
)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal Python project for testing."""
    # Package structure
    pkg = tmp_path / "mypackage"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""My package."""\n')

    (pkg / "app.py").write_text(
        textwrap.dedent('''\
        """Main application module."""

        import os
        import sys
        from typing import List

        from mypackage.utils import helper
        from mypackage.models import User

        class App:
            """The main application."""

            def __init__(self):
                self.name = "test"

            def run(self):
                pass

        def main():
            app = App()
            app.run()
        ''')
    )

    (pkg / "models.py").write_text(
        textwrap.dedent('''\
        """Data models."""

        from dataclasses import dataclass

        @dataclass
        class User:
            name: str
            email: str

        @dataclass
        class Item:
            title: str
            value: int
        ''')
    )

    (pkg / "utils.py").write_text(
        textwrap.dedent('''\
        """Utility functions."""

        def helper(x):
            return x + 1

        def format_name(name):
            return name.strip().title()
        ''')
    )

    # Sub-package with relative import
    sub = pkg / "api"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    (sub / "routes.py").write_text(
        textwrap.dedent('''\
        """API routes."""

        from ..models import User
        from ..utils import helper

        def get_users():
            return []
        ''')
    )

    # Test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("")
    (tests_dir / "test_app.py").write_text(
        textwrap.dedent('''\
        """Tests for app module."""

        import pytest

        def test_something():
            assert True
        ''')
    )

    # Entry point
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "mypackage"

        [project.scripts]
        myapp = "mypackage.app:main"
        """)
    )

    # Directory that should be excluded
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "something.py").write_text("x = 1\n")

    return tmp_path


class TestRepositoryScanner:
    """Tests for RepositoryScanner."""

    def test_scan_finds_python_files(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        assert structure.total_files > 0

    def test_scan_excludes_venv(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        for path in structure.modules:
            assert ".venv" not in path

    def test_scan_detects_packages(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        assert "mypackage" in structure.packages
        assert "mypackage.api" in structure.packages

    def test_scan_extracts_imports(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules.get("mypackage/app.py")
        assert app_module is not None
        assert "os" in app_module.imports
        assert "sys" in app_module.imports

    def test_scan_extracts_from_imports(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        from_modules = [fi["module"] for fi in app_module.from_imports]
        assert "mypackage.utils" in from_modules
        assert "mypackage.models" in from_modules

    def test_scan_extracts_classes(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        class_names = [c["name"] for c in app_module.classes]
        assert "App" in class_names

    def test_scan_extracts_functions(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        func_names = [f["name"] for f in app_module.functions]
        assert "main" in func_names

    def test_scan_extracts_class_methods(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        app_class = next(c for c in app_module.classes if c["name"] == "App")
        assert "__init__" in app_class["methods"]
        assert "run" in app_class["methods"]

    def test_scan_extracts_docstrings(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        assert app_module.docstring == "Main application module."

    def test_scan_detects_test_files(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        test_module = structure.modules.get("tests/test_app.py")
        assert test_module is not None
        assert test_module.is_test is True

    def test_scan_non_test_files_not_marked(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        assert app_module.is_test is False

    def test_scan_resolves_relative_imports(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        routes_module = structure.modules.get("mypackage/api/routes.py")
        assert routes_module is not None
        from_modules = [fi["module"] for fi in routes_module.from_imports]
        # Relative imports should be resolved to absolute
        assert "mypackage.models" in from_modules
        assert "mypackage.utils" in from_modules

    def test_scan_detects_entry_points(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        entry_strs = " ".join(structure.entry_points)
        assert "pyproject.toml" in entry_strs

    def test_scan_counts_lines(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        assert structure.total_lines > 0
        for mod in structure.modules.values():
            assert mod.lines_of_code > 0

    def test_scan_custom_exclude(self, sample_repo: Path) -> None:
        # Create a directory to exclude
        custom = sample_repo / "generated"
        custom.mkdir()
        (custom / "auto.py").write_text("x = 1\n")

        scanner = RepositoryScanner(str(sample_repo), exclude_patterns=["generated"])
        structure = scanner.scan()
        for path in structure.modules:
            assert "generated" not in path

    def test_scan_include_source(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo), include_source=True)
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        assert app_module.source is not None
        assert "class App" in app_module.source

    def test_scan_no_source_by_default(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        app_module = structure.modules["mypackage/app.py"]
        assert app_module.source is None

    def test_invalid_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            RepositoryScanner("/nonexistent/path/that/does/not/exist")


class TestRepositoryStructure:
    """Tests for RepositoryStructure serialization."""

    def test_roundtrip_json(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        structure.save(output_path)
        loaded = RepositoryStructure.load(output_path)

        assert loaded.name == structure.name
        assert loaded.total_files == structure.total_files
        assert loaded.total_lines == structure.total_lines
        assert set(loaded.modules.keys()) == set(structure.modules.keys())
        assert loaded.packages == structure.packages

        Path(output_path).unlink()

    def test_to_dict(self, sample_repo: Path) -> None:
        scanner = RepositoryScanner(str(sample_repo))
        structure = scanner.scan()
        d = structure.to_dict()

        assert isinstance(d, dict)
        assert "modules" in d
        assert "packages" in d
        assert d["total_files"] == structure.total_files


class TestSelfScan:
    """Test scanning the rlm-codelens project itself."""

    def test_self_scan(self) -> None:
        """Scan the rlm-codelens repository itself — smoke test."""
        repo_root = Path(__file__).parent.parent.parent
        if not (repo_root / "pyproject.toml").exists():
            pytest.skip("Not running from within rlm-codelens repo")

        scanner = RepositoryScanner(str(repo_root))
        structure = scanner.scan()

        assert structure.total_files > 0
        assert "rlm_codelens" in " ".join(structure.packages)
        assert structure.total_lines > 0
