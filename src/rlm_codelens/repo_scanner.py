"""
Repository scanner for codebase architecture analysis.

Scans a local or remote repository, parsing source files to extract module
structure, imports, classes, functions, and entry points.  Supports Python
(via built-in ast), and Go, Java, Rust, C/C++, TypeScript/JavaScript (via
tree-sitter when installed).

Example:
    >>> scanner = RepositoryScanner("/path/to/repo")
    >>> structure = scanner.scan()
    >>> print(f"Found {structure.total_files} files in {len(structure.packages)} packages")
"""

import ast
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Directories to always skip
DEFAULT_EXCLUDE_PATTERNS = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".eggs",
    "vendor",  # Go vendored deps
    "target",  # Rust/Java build output
    "bin",  # compiled binaries
    "obj",  # C/C++ object files
    ".gradle",
    ".idea",
    ".vscode",
    "testdata",  # Go test fixtures (data, not code)
}

# Suffixes for egg-info dirs
EXCLUDE_SUFFIXES = (".egg-info",)


@dataclass
class ModuleInfo:
    """Information extracted from a single source module.

    Attributes:
        path: Relative path from repo root
        package: Dot-separated package name (e.g., "flask.app")
        imports: Absolute import targets
        from_imports: List of dicts with module, names, level keys
        classes: List of dicts with name, bases, methods, line keys
        functions: List of dicts with name, args, decorators, line keys
        lines_of_code: Total lines in the file
        docstring: Module-level docstring
        is_test: Whether this appears to be a test file
        source: Full source text (included when include_source=True)
        language: Programming language (e.g., "python", "go", "java")
    """

    path: str
    package: str
    imports: List[str] = field(default_factory=list)
    from_imports: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    lines_of_code: int = 0
    docstring: Optional[str] = None
    is_test: bool = False
    source: Optional[str] = None
    language: str = "python"


@dataclass
class RepositoryStructure:
    """Complete structure of a scanned repository.

    Attributes:
        root_path: Absolute path to the repository root
        name: Repository name (directory name)
        modules: Dict mapping relative path to ModuleInfo
        packages: List of discovered Python packages
        entry_points: List of detected entry point files
        total_files: Total number of Python files scanned
        total_lines: Total lines of code across all files
    """

    root_path: str
    name: str
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    packages: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    total_files: int = 0
    total_lines: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        data = asdict(self)
        return data

    def save(self, output_path: str) -> None:
        """Save to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepositoryStructure":
        """Deserialize from a dict (e.g. loaded from JSON)."""
        modules = {}
        for path, mod_data in data.get("modules", {}).items():
            modules[path] = ModuleInfo(**mod_data)
        return cls(
            root_path=data["root_path"],
            name=data["name"],
            modules=modules,
            packages=data.get("packages", []),
            entry_points=data.get("entry_points", []),
            total_files=data.get("total_files", 0),
            total_lines=data.get("total_lines", 0),
        )

    @classmethod
    def load(cls, path: str) -> "RepositoryStructure":
        """Load from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class RepositoryScanner:
    """Scans a repository and extracts module structure via AST/tree-sitter parsing.

    Supports Python (built-in ast), and Go, Java, Rust, C/C++, TypeScript/
    JavaScript via tree-sitter when the corresponding grammar is installed.

    Args:
        repo_path: Local path or remote git URL
        exclude_patterns: Additional directory names to skip
        include_source: Whether to include full source text in ModuleInfo
    """

    def __init__(
        self,
        repo_path: str,
        exclude_patterns: Optional[List[str]] = None,
        include_source: bool = False,
    ):
        self.original_path = repo_path
        self.include_source = include_source
        self._temp_dir: Optional[str] = None
        self._ts_parsers: Dict[str, Any] = {}  # cached UniversalParser instances

        # Merge exclude patterns
        self.exclude_patterns = set(DEFAULT_EXCLUDE_PATTERNS)
        if exclude_patterns:
            self.exclude_patterns.update(exclude_patterns)

        # Resolve repo path (clone if remote URL)
        self.repo_path = self._resolve_repo_path(repo_path)

    def _resolve_repo_path(self, repo_path: str) -> Path:
        """Resolve the repo path, cloning if it's a remote URL."""
        if repo_path.startswith(("http://", "https://", "git@")):
            return self._clone_repo(repo_path)
        path = Path(repo_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")
        return path

    def _clone_repo(self, url: str) -> Path:
        """Shallow-clone a remote repository to a temporary directory."""
        self._temp_dir = tempfile.mkdtemp(prefix="rlmc_scan_")
        clone_path = Path(self._temp_dir) / "repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(clone_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.cleanup()
            raise RuntimeError(f"Git clone timed out after 300s: {url}")
        except subprocess.CalledProcessError as e:
            self.cleanup()
            raise RuntimeError(
                f"Git clone failed for {url}: {e.stderr.strip() if e.stderr else e}"
            )
        return clone_path

    def cleanup(self) -> None:
        """Remove temporary clone directory if one was created."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def scan(self) -> RepositoryStructure:
        """Scan the repository and return its structure.

        Returns:
            RepositoryStructure containing all discovered modules.
        """
        try:
            return self._do_scan()
        finally:
            self.cleanup()

    def _do_scan(self) -> RepositoryStructure:
        """Internal scan implementation."""
        from rlm_codelens.language_support import EXTENSIONS, detect_language

        structure = RepositoryStructure(
            root_path=str(self.repo_path),
            name=self.repo_path.name,
        )

        # Find all supported source files
        source_files = self._find_source_files(set(EXTENSIONS.keys()))

        # Detect packages (Python-specific but harmless for other langs)
        structure.packages = self._detect_packages()

        # Parse each file
        for src_file in source_files:
            rel_path = str(src_file.relative_to(self.repo_path))
            lang = detect_language(rel_path)
            if lang == "python":
                module_info = self._parse_module(src_file, rel_path)
            else:
                module_info = self._parse_module_treesitter(
                    src_file, rel_path, lang or "unknown"
                )
            if module_info:
                structure.modules[rel_path] = module_info
                structure.total_lines += module_info.lines_of_code

        structure.total_files = len(structure.modules)

        # Detect entry points
        structure.entry_points = self._detect_entry_points()

        # Resolve relative imports (Python only)
        self._resolve_relative_imports(structure)

        return structure

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded from scanning."""
        for part in path.parts:
            if part in self.exclude_patterns:
                return True
            if any(part.endswith(suffix) for suffix in EXCLUDE_SUFFIXES):
                return True
        return False

    def _find_python_files(self) -> List[Path]:
        """Find all .py files in the repository, respecting exclude patterns."""
        py_files = []
        for py_file in self.repo_path.rglob("*.py"):
            rel = py_file.relative_to(self.repo_path)
            if not self._should_exclude(rel):
                py_files.append(py_file)
        return sorted(py_files)

    def _find_source_files(self, extensions: set) -> List[Path]:
        """Find all source files matching the given extensions."""
        files = []
        for path in self.repo_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                rel = path.relative_to(self.repo_path)
                if not self._should_exclude(rel):
                    files.append(path)
        return sorted(files)

    def _detect_packages(self) -> List[str]:
        """Detect Python packages (directories with __init__.py)."""
        packages = []
        for init_file in self.repo_path.rglob("__init__.py"):
            rel = init_file.relative_to(self.repo_path)
            if not self._should_exclude(rel):
                package_dir = rel.parent
                package_name = ".".join(package_dir.parts) if package_dir.parts else ""
                if package_name:
                    packages.append(package_name)
        return sorted(set(packages))

    def _path_to_package(self, rel_path: str) -> str:
        """Convert a relative file path to a dot-separated package name."""
        parts = Path(rel_path).with_suffix("").parts
        # Remove __init__ from the end
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _parse_module(self, file_path: Path, rel_path: str) -> Optional[ModuleInfo]:
        """Parse a single Python file using AST.

        Returns None if the file cannot be parsed.
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return None

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            # File has syntax errors — still count it but skip AST analysis
            return ModuleInfo(
                path=rel_path,
                package=self._path_to_package(rel_path),
                lines_of_code=source.count("\n") + 1,
                is_test=self._is_test_file(rel_path),
                source=source if self.include_source else None,
            )

        module_info = ModuleInfo(
            path=rel_path,
            package=self._path_to_package(rel_path),
            lines_of_code=source.count("\n") + 1,
            docstring=ast.get_docstring(tree),
            is_test=self._is_test_file(rel_path),
            source=source if self.include_source else None,
        )

        # Extract imports, classes, functions from top-level statements
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_info.imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                module_info.from_imports.append(
                    {
                        "module": node.module or "",
                        "names": [alias.name for alias in node.names],
                        "level": node.level,
                    }
                )

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "bases": [self._node_to_str(b) for b in node.bases],
                    "methods": [
                        n.name
                        for n in ast.iter_child_nodes(node)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ],
                    "line": node.lineno,
                }
                module_info.classes.append(class_info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                    "decorators": [self._node_to_str(d) for d in node.decorator_list],
                    "line": node.lineno,
                }
                module_info.functions.append(func_info)

        return module_info

    def _get_ts_parser(self, language: str) -> Optional[Any]:
        """Get or create a cached UniversalParser for the given language."""
        if language not in self._ts_parsers:
            from rlm_codelens.language_support import UniversalParser

            parser = UniversalParser(language)
            self._ts_parsers[language] = parser if parser.available else None
        return self._ts_parsers[language]

    def _parse_module_treesitter(
        self, file_path: Path, rel_path: str, language: str
    ) -> Optional[ModuleInfo]:
        """Parse a non-Python source file using tree-sitter.

        Returns None if tree-sitter is not available for this language.
        """
        parser = self._get_ts_parser(language)
        if parser is None:
            # No tree-sitter grammar -- create a minimal ModuleInfo with just LOC
            try:
                source = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                return None
            return ModuleInfo(
                path=rel_path,
                package=self._path_to_package(rel_path),
                lines_of_code=source.count("\n") + 1,
                is_test=self._is_test_file(rel_path),
                source=source if self.include_source else None,
                language=language,
            )

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return None

        parsed = parser.parse_module(source, rel_path, self.include_source)
        if parsed is None:
            return ModuleInfo(
                path=rel_path,
                package=self._path_to_package(rel_path),
                lines_of_code=source.count("\n") + 1,
                is_test=self._is_test_file(rel_path),
                source=source if self.include_source else None,
                language=language,
            )

        return ModuleInfo(
            path=rel_path,
            package=self._path_to_package(rel_path),
            imports=parsed.get("imports", []),
            from_imports=parsed.get("from_imports", []),
            classes=parsed.get("classes", []),
            functions=parsed.get("functions", []),
            lines_of_code=parsed.get("lines_of_code", 0),
            docstring=parsed.get("docstring"),
            is_test=self._is_test_file(rel_path),
            source=parsed.get("source"),
            language=language,
        )

    def _node_to_str(self, node: ast.AST) -> str:
        """Convert an AST node to a readable string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._node_to_str(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self._node_to_str(node.func)
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._node_to_str(node.value)
            slice_str = self._node_to_str(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._node_to_str(e) for e in node.elts)
            return f"({elements})"
        try:
            return ast.unparse(node)
        except Exception:
            return "<unknown>"

    def _is_test_file(self, rel_path: str) -> bool:
        """Determine if a file is a test file based on path conventions."""
        parts = Path(rel_path).parts
        name = Path(rel_path).stem
        # test directories or test_ prefix or _test suffix
        return (
            "tests" in parts
            or "test" in parts
            or "testdata" in parts
            or "__tests__" in parts
            or name.startswith("test_")
            or name.endswith("_test")
            or name.endswith("Test")  # Java: FooTest.java
            or name.endswith(".test")  # JS: foo.test.ts
            or name.endswith(".spec")  # JS: foo.spec.ts
            or name == "conftest"
        )

    def _detect_entry_points(self) -> List[str]:
        """Detect entry points from pyproject.toml, setup.py, and __main__.py."""
        entry_points = []

        # Check __main__.py files
        for main_file in self.repo_path.rglob("__main__.py"):
            rel = main_file.relative_to(self.repo_path)
            if not self._should_exclude(rel):
                entry_points.append(str(rel))

        # Check pyproject.toml
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                # Simple parsing for [project.scripts] entries
                in_scripts = False
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped == "[project.scripts]":
                        in_scripts = True
                        continue
                    if in_scripts:
                        if stripped.startswith("["):
                            break
                        if "=" in stripped:
                            entry_points.append(f"pyproject.toml:{stripped}")
            except OSError:
                pass

        # Check setup.py
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            entry_points.append("setup.py")

        return entry_points

    def _resolve_relative_imports(self, structure: RepositoryStructure) -> None:
        """Resolve relative imports to absolute package names."""
        for _path, module in structure.modules.items():
            resolved_from_imports = []
            for imp in module.from_imports:
                level = imp.get("level", 0)
                if level > 0:
                    # Relative import — resolve against current package
                    # package_parts includes the module name as the last element
                    # level=1 means current package (drop module name)
                    # level=2 means parent package (drop module name + 1 more)
                    package_parts = module.package.split(".")
                    if level <= len(package_parts):
                        base = ".".join(package_parts[: len(package_parts) - level])
                    else:
                        base = ""
                    abs_module = f"{base}.{imp['module']}" if imp["module"] else base
                    resolved_from_imports.append(
                        {
                            "module": abs_module,
                            "names": imp["names"],
                            "level": 0,  # Now absolute
                        }
                    )
                else:
                    resolved_from_imports.append(imp)
            module.from_imports = resolved_from_imports
