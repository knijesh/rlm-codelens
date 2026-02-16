# RLM-Codelens

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/knijesh/rlm-codelens/workflows/CI/badge.svg)](https://github.com/knijesh/rlm-codelens/actions)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Whole-codebase architecture intelligence powered by Recursive Language Models.**

## The Problem

Understanding a large codebase is one of the hardest problems in software engineering. Developers spend more time reading code than writing it, and the mental model of "how this system fits together" is usually locked inside the heads of senior engineers.

| Challenge | Why LLMs Fail | How RLM Works |
|-----------|--------------|---------------|
| Codebases are too large for a single context window | GPT-4 can see ~100K tokens; vLLM has 2,504 files | Recursively decomposes the codebase into manageable chunks |
| Imports form complex dependency graphs | LLMs can't reliably trace transitive dependencies | Builds a real graph with NetworkX, then reasons over it |
| Architecture has layers and patterns | LLMs hallucinate structure without seeing the full picture | Static analysis first, RLM enriches with semantic understanding |
| Anti-patterns hide in the connections | A single file looks fine; problems emerge from relationships | Graph algorithms detect cycles, hubs, and layering violations |

## How It Works

```
Repository  -->  AST Scanner  -->  Dependency Graph  -->  Architecture Analysis  -->  Interactive Visualization
  (input)        (parse every       (NetworkX graph       (anti-patterns, layers,     (D3.js force-directed
                  Python file)       of imports)           hub modules, cycles)        graph in HTML)
```

**Static analysis** works on any codebase with zero API calls. Add `--deep` to enable **RLM-powered semantic analysis** that classifies modules, discovers hidden dependencies, and suggests refactoring.

## Proven at Scale

| Repository | Files | LOC | Modules | Import Edges | Cycles | Anti-Patterns |
|-----------|-------|------|---------|--------------|--------|---------------|
| **Starlette** | 67 | 9,800 | 67 | 106 | 3 | 4 |
| **vLLM** | 2,504 | 483,000 | 2,504 | 7,412 | 127 | 89 |
| **rlm-codelens** (self) | 22 | 3,800 | 22 | 42 | 1 | 3 |

Example outputs are in [`outputs/examples/`](outputs/examples/).

## Quick Start

```bash
# Install
git clone https://github.com/knijesh/rlm-codelens.git
cd rlm-codelens
uv sync --extra dev

# Analyze any Python repository in 3 commands
uv run rlmc scan-repo /path/to/repo --output scan.json
uv run rlmc analyze-architecture scan.json --output arch.json
uv run rlmc visualize-arch arch.json
# Opens interactive visualization in your browser
```

Or use the pipeline script:

```bash
./run_analysis.sh /path/to/repo myproject
```

### Self-Scan Demo (no API keys needed)

```bash
./demo_analysis.sh
```

### Deep RLM Analysis (requires API key)

```bash
# Scan with source text included
uv run rlmc scan-repo /path/to/repo --include-source --output scan.json

# Analyze with RLM-powered insights
uv run rlmc analyze-architecture scan.json --deep --budget 5.0

# Supports OpenAI and Anthropic backends
uv run rlmc analyze-architecture scan.json --deep --backend anthropic --model claude-sonnet-4-5-20250929
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `rlmc scan-repo <path>` | Parse all Python files using AST; extract imports, classes, functions |
| `rlmc analyze-architecture <scan.json>` | Build dependency graph, detect cycles, layers, anti-patterns |
| `rlmc visualize-arch <analysis.json>` | Generate interactive D3.js visualization |

## Configuration

Create a `.env` file (see `.env.example`):

```env
# Required only for --deep RLM analysis
OPENAI_API_KEY=sk-xxxxxxxxxxxx    # if using openai backend
# ANTHROPIC_API_KEY=sk-ant-xxx    # if using anthropic backend

# Optional defaults
RLM_BACKEND=openai
RLM_MODEL=gpt-4o
BUDGET_LIMIT=50.0
```

## Project Structure

```
rlm-codelens/
├── src/rlm_codelens/
│   ├── cli.py                    # CLI entry point (rlmc)
│   ├── commands.py               # Command implementations
│   ├── config.py                 # Configuration from .env
│   ├── repo_scanner.py           # AST-based repository scanner
│   ├── codebase_graph.py         # Module dependency graph builder
│   ├── architecture_analyzer.py  # RLM-powered deep analysis
│   ├── visualizer.py             # D3.js visualization generator
│   └── utils/                    # Cost tracking, logging
├── tests/
│   └── unit/                     # Unit tests
├── outputs/examples/             # Pre-computed analysis results
├── visualization/                # Standalone HTML viewer
├── run_analysis.sh               # Architecture analysis pipeline
├── demo_analysis.sh              # Self-scan demo
└── pyproject.toml
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# With coverage
uv run pytest tests/ --cov=rlm_codelens --cov-report=html
```

## Python API

```python
from rlm_codelens import RepositoryScanner, CodebaseGraphAnalyzer

# Scan
scanner = RepositoryScanner("/path/to/repo")
structure = scanner.scan()
print(f"{structure.total_files} files, {structure.total_lines:,} LOC")

# Analyze
analyzer = CodebaseGraphAnalyzer(structure)
analysis = analyzer.analyze()
print(f"{len(analysis.cycles)} circular imports")
print(f"{len(analysis.anti_patterns)} anti-patterns")

# Save
analysis.save("architecture.json")
```
# Sample Output

[Sample_RLM_Output](sample_outputs/RLM_sample.png)
[Sample_RLM_Output_2](sample_output/RLM_Sample_2.png)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT License - see [LICENSE](LICENSE).

## Credits

- [alexzhang13/rlm](https://github.com/alexzhang13/rlm) - Recursive Language Models
- [NetworkX](https://networkx.org/) - Graph algorithms
- [D3.js](https://d3js.org/) - Interactive visualizations

---

**Author:** Nijesh Kanjinghat ([@knijesh](https://github.com/knijesh))
