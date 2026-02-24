"""Universal multi-language support for repository scanning via tree-sitter.

Provides dynamic grammar loading, a universal CST-walking parser, and language
detection utilities.  Tree-sitter is an optional dependency -- when not installed,
the scanner falls back to Python's built-in ast module for .py files.

Example:
    >>> from rlm_codelens.language_support import detect_language, UniversalParser
    >>> detect_language("src/app.js")
    'javascript'
    >>> parser = UniversalParser("javascript")
    >>> parser.available
    True
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension registry
# ---------------------------------------------------------------------------

EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cs": "c_sharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".rb": "ruby",
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect the programming language of a file from its extension."""
    ext = Path(file_path).suffix.lower()
    return EXTENSIONS.get(ext)


def detect_repo_languages(repo_path: str) -> Dict[str, int]:
    """Scan a repository and count files per detected language.

    Returns:
        Dict mapping language name to file count, sorted by count descending.
    """
    counts: Dict[str, int] = {}
    root = Path(repo_path)
    if not root.is_dir():
        return counts

    skip_dirs = {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "build",
        "dist",
        ".tox",
    }

    for path in root.rglob("*"):
        if path.is_file() and not any(d in path.parts for d in skip_dirs):
            lang = detect_language(str(path))
            if lang:
                counts[lang] = counts.get(lang, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Dynamic grammar loading
# ---------------------------------------------------------------------------


def load_grammar(language: str) -> Optional[Any]:
    """Load a tree-sitter grammar dynamically.

    Convention: ``tree_sitter_{language}.language()`` returns the grammar.
    """
    try:
        import tree_sitter  # noqa: F401

        mod_name = f"tree_sitter_{language}"
        mod = importlib.import_module(mod_name)

        for func_name in ("language", f"language_{language}"):
            func = getattr(mod, func_name, None)
            if func is not None:
                return tree_sitter.Language(func())

        return None
    except (ImportError, Exception) as exc:
        logger.debug("Cannot load grammar for %s: %s", language, exc)
        return None


# ---------------------------------------------------------------------------
# Universal CST-walking parser
# ---------------------------------------------------------------------------

_IMPORT_KEYWORDS = {"import", "use", "include", "require"}
_CLASS_KEYWORDS = {"class", "struct", "interface", "enum", "trait", "impl"}
_FUNCTION_KEYWORDS = {"function", "method", "proc", "fun", "func", "def"}


def _node_type_matches(node_type: str, keywords: set) -> bool:
    """Check if a CST node type contains any of the given keywords."""
    parts = node_type.split("_")
    return bool(set(parts) & keywords)


class UniversalParser:
    """Parses source files using tree-sitter grammars via universal CST walking.

    Works with any language whose tree-sitter grammar is pip-installed.
    """

    def __init__(self, language: str) -> None:
        self.language = language
        self._parser: Any = None
        self._ts_language: Any = None
        self._available: Optional[bool] = None

    def _ensure_parser(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import tree_sitter  # noqa: F401

            ts_lang = load_grammar(self.language)
            if ts_lang is None:
                self._available = False
                return False

            self._ts_language = ts_lang
            self._parser = tree_sitter.Parser(ts_lang)
            self._available = True
        except (ImportError, Exception):
            self._available = False

        return self._available

    @property
    def available(self) -> bool:
        return self._ensure_parser()

    def parse_module(
        self,
        source: str,
        rel_path: str,
        include_source: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Parse a source file and extract module information.

        Returns:
            Dict with imports, from_imports, classes, functions, docstring,
            lines_of_code, source -- or None if parsing fails.
        """
        if not self._ensure_parser():
            return None

        try:
            tree = self._parser.parse(source.encode("utf-8"))
        except Exception:
            return None

        root = tree.root_node

        return {
            "imports": self._extract_imports(root, source),
            "from_imports": self._extract_from_imports(root, source),
            "classes": self._extract_classes(root, source),
            "functions": self._extract_functions(root, source),
            "docstring": self._extract_docstring(root, source),
            "lines_of_code": source.count("\n") + 1,
            "source": source if include_source else None,
        }

    @staticmethod
    def _node_text(node: Any, source: str) -> str:
        return source[node.start_byte : node.end_byte]

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_imports(self, root: Any, source: str) -> List[str]:
        imports: List[str] = []

        for child in root.children:
            if not _node_type_matches(child.type, _IMPORT_KEYWORDS):
                continue

            # Python from-imports handled separately
            if child.type == "import_from_statement":
                continue

            # JS/TS: import ... from 'source'
            source_node = child.child_by_field_name("source")
            if source_node is not None:
                mod = self._node_text(source_node, source).strip("'\"")
                if mod and mod not in imports:
                    imports.append(mod)
                continue

            # Python: import X
            name_node = child.child_by_field_name("name")
            if name_node is not None:
                mod = self._node_text(name_node, source)
                if mod and mod not in imports:
                    imports.append(mod)
                continue

            # Java, Rust, Go -- walk children for identifiers/strings
            self._collect_import_strings(child, source, imports)

        return imports

    def _collect_import_strings(
        self, node: Any, source: str, imports: List[str]
    ) -> None:
        for sub in node.children:
            if sub.type in (
                "scoped_identifier",
                "dotted_name",
                "identifier",
                "use_list",
                "scoped_use_list",
            ):
                mod = self._node_text(sub, source).strip("'\"")
                if mod and mod not in imports:
                    imports.append(mod)
            elif sub.type in (
                "string",
                "interpreted_string_literal",
                "raw_string_literal",
            ):
                mod = self._node_text(sub, source).strip("'\"`")
                if mod and mod not in imports:
                    imports.append(mod)
            elif sub.type in (
                "import_spec_list",
                "import_spec",
                "import_statement",
            ):
                self._collect_import_strings(sub, source, imports)

    # ------------------------------------------------------------------
    # Python from-imports
    # ------------------------------------------------------------------

    def _extract_from_imports(self, root: Any, source: str) -> List[Dict[str, Any]]:
        if self.language != "python":
            return []

        from_imports: List[Dict[str, Any]] = []
        for child in root.children:
            if child.type != "import_from_statement":
                continue

            module_name = ""
            names: List[str] = []
            level = 0
            past_import_keyword = False

            for sub in child.children:
                if sub.type == "import":
                    past_import_keyword = True
                elif sub.type == "relative_import":
                    for rsub in sub.children:
                        if rsub.type == "import_prefix":
                            level = self._node_text(rsub, source).count(".")
                        elif rsub.type == "dotted_name":
                            module_name = self._node_text(rsub, source)
                elif sub.type == "dotted_name":
                    text = self._node_text(sub, source)
                    if not past_import_keyword:
                        module_name = text
                    else:
                        names.append(text)
                elif sub.type == "aliased_import":
                    for asub in sub.children:
                        if asub.type in ("dotted_name", "identifier"):
                            names.append(self._node_text(asub, source))
                            break
                elif sub.type == "wildcard_import":
                    names.append("*")

            from_imports.append(
                {
                    "module": module_name,
                    "names": names,
                    "level": level,
                }
            )

        return from_imports

    # ------------------------------------------------------------------
    # Class extraction
    # ------------------------------------------------------------------

    def _extract_classes(self, root: Any, source: str) -> List[Dict[str, Any]]:
        classes: List[Dict[str, Any]] = []

        # Go method_declarations for receiver-based method association
        go_methods: Dict[str, List[str]] = {}
        if self.language == "go":
            for child in root.children:
                if child.type == "method_declaration":
                    receiver_type = self._go_receiver_type(child, source)
                    mname = child.child_by_field_name("name")
                    if receiver_type and mname is not None:
                        go_methods.setdefault(receiver_type, []).append(
                            self._node_text(mname, source)
                        )

        for child in root.children:
            target = child
            if child.type in ("export_statement", "decorated_definition"):
                for sub in child.children:
                    if _node_type_matches(sub.type, _CLASS_KEYWORDS):
                        target = sub
                        break
                else:
                    continue

            # Go: type_declaration -> type_spec -> struct_type/interface_type
            if target.type == "type_declaration":
                for spec in target.children:
                    if spec.type == "type_spec":
                        name_node = spec.child_by_field_name("name")
                        if name_node is None:
                            continue
                        has_body = any(
                            s.type in ("struct_type", "interface_type")
                            for s in spec.children
                        )
                        if has_body:
                            class_name = self._node_text(name_node, source)
                            methods = go_methods.get(class_name, [])
                            classes.append(
                                {
                                    "name": class_name,
                                    "bases": [],
                                    "methods": methods,
                                    "line": target.start_point[0] + 1,
                                }
                            )
                continue

            if not _node_type_matches(target.type, _CLASS_KEYWORDS):
                continue

            name_node = target.child_by_field_name("name")
            if name_node is None:
                continue

            class_name = self._node_text(name_node, source)
            body_methods: List[str] = []

            body_node = target.child_by_field_name("body")
            if body_node is not None:
                for body_child in body_node.children:
                    method_target = body_child
                    if body_child.type in ("decorated_definition", "export_statement"):
                        for sub in body_child.children:
                            if _node_type_matches(sub.type, _FUNCTION_KEYWORDS):
                                method_target = sub
                                break
                        else:
                            continue

                    if _node_type_matches(method_target.type, _FUNCTION_KEYWORDS):
                        mname_node = method_target.child_by_field_name("name")
                        if mname_node is not None:
                            body_methods.append(self._node_text(mname_node, source))

            classes.append(
                {
                    "name": class_name,
                    "bases": [],
                    "methods": body_methods,
                    "line": target.start_point[0] + 1,
                }
            )

        return classes

    @staticmethod
    def _go_receiver_type(method_node: Any, source: str) -> Optional[str]:
        receiver = method_node.child_by_field_name("receiver")
        if receiver is None:
            return None
        for param in receiver.children:
            if param.type == "parameter_declaration":
                for sub in param.children:
                    if sub.type == "pointer_type":
                        for inner in sub.children:
                            if inner.type == "type_identifier":
                                return source[inner.start_byte : inner.end_byte]
                    elif sub.type == "type_identifier":
                        return source[sub.start_byte : sub.end_byte]
        return None

    # ------------------------------------------------------------------
    # Function extraction
    # ------------------------------------------------------------------

    def _extract_functions(self, root: Any, source: str) -> List[Dict[str, Any]]:
        functions: List[Dict[str, Any]] = []
        seen_names: set = set()

        for child in root.children:
            target = child
            if child.type in ("export_statement", "decorated_definition"):
                for sub in child.children:
                    if _node_type_matches(sub.type, _FUNCTION_KEYWORDS):
                        target = sub
                        break
                else:
                    if child.type == "export_statement":
                        for sub in child.children:
                            if sub.type in (
                                "lexical_declaration",
                                "variable_declaration",
                            ):
                                target = sub
                                break
                        else:
                            continue
                    else:
                        continue

            if _node_type_matches(target.type, _FUNCTION_KEYWORDS):
                if target.type == "method_declaration":
                    continue
                name_node = target.child_by_field_name("name")
                if name_node is not None:
                    fname = self._node_text(name_node, source)
                    if fname not in seen_names:
                        seen_names.add(fname)
                        functions.append(
                            {
                                "name": fname,
                                "args": [],
                                "decorators": [],
                                "line": target.start_point[0] + 1,
                            }
                        )
                continue

            # JS/TS arrow functions: const foo = () => {}
            if target.type in ("lexical_declaration", "variable_declaration"):
                for declarator in target.children:
                    if declarator.type == "variable_declarator":
                        vname = declarator.child_by_field_name("name")
                        vvalue = declarator.child_by_field_name("value")
                        if (
                            vname is not None
                            and vvalue is not None
                            and vvalue.type == "arrow_function"
                        ):
                            fname = self._node_text(vname, source)
                            if fname not in seen_names:
                                seen_names.add(fname)
                                functions.append(
                                    {
                                        "name": fname,
                                        "args": [],
                                        "decorators": [],
                                        "line": target.start_point[0] + 1,
                                    }
                                )

        return functions

    # ------------------------------------------------------------------
    # Docstring extraction (Python-specific)
    # ------------------------------------------------------------------

    def _extract_docstring(self, root: Any, source: str) -> Optional[str]:
        if self.language != "python":
            return None

        for child in root.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        text = self._node_text(sub, source)
                        for quote in ('"""', "'''", '"', "'"):
                            if text.startswith(quote) and text.endswith(quote):
                                return text[len(quote) : -len(quote)].strip()
                        return text.strip()
            if child.type not in ("expression_statement", "comment"):
                break

        return None
