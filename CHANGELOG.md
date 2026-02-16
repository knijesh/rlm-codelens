# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-02-16

### Added
- **Enhanced Architecture Visualization** with per-module graph tracer panel
  - Summary dashboard: modules, edges, cycles, hub modules, anti-patterns, total LOC, health indicator
  - Hierarchical force layout with curved edges and layer-based vertical bands
  - Module tracer panel (click any module): full path, layer, LOC, classes, functions, docstring
  - Dependency explorer: "Depends On" and "Used By" lists with click-to-navigate
  - Expandable transitive dependency tree (upstream/downstream, depth 3)
  - Full dependency tree highlighting on graph with graduated opacity by depth
  - Improved sidebar: expandable anti-pattern details, clickable hub modules and cycles
  - Search with zoom-to-fit, layer toggle checkboxes

### Changed
- `visualizer.py` now enriches analysis data with pre-computed per-node fields (depends_on, used_by, in_cycles, anti_patterns, upstream/downstream trees)
- Bumped version to 0.2.1

### Removed
- **All remaining legacy code** (~4,000+ lines removed):
  - 4 legacy source modules: `data_collection.py`, `dependency_analyzer.py`, `dependency_extractor.py`, `graph_analyzer.py`
  - `core/` package: `core/analyzer.py`, `core/config.py`
  - 3 legacy utils: `database.py`, `cost_estimator.py`, `secure_rlm_client.py`
  - 3 legacy visualization files: `dependency_graph_viewer.html`, `issue_graph_visualization.html`, `issue_graph.json`
  - 2 legacy docs: `API_KEYS.md`, `RLM_ANALYSIS_FINDINGS.md`
  - 4 legacy tests: `test_core.py`, `test_dependency_extractor.py`, `test_graph_analyzer.py`, `test_end_to_end.py`
- **3 secondary CLI commands**: `extract-deps`, `analyze-graph`, `visualize` (issue dependency pipeline)
- **8 unused dependencies**: pygithub, pandas, psycopg2-binary, sqlalchemy, tiktoken, openai, numpy, requests, tqdm
- **Legacy config variables**: `GITHUB_TOKEN`, `REPO_OWNER`, `REPO_NAME`, `DATABASE_URL`, `DEPENDENCY_*`, etc.

## [0.2.0] - 2026-02-16

### Added
- **Whole-Codebase Architecture Analysis** as primary feature
  - `scan-repo`: AST-based repository scanner for Python codebases
  - `analyze-architecture`: Module dependency graph, cycle detection, anti-pattern identification, layer classification
  - `visualize-arch`: Interactive D3.js architecture visualization
  - `--deep` flag for RLM-powered semantic analysis (module classification, hidden dependencies, refactoring suggestions)
- **Proven at scale**: Starlette (67 files), vLLM (2,504 files), self-scan (22 files)
- Example outputs in `outputs/examples/` for Starlette, vLLM, and self-analysis
- Anthropic backend support (`--backend anthropic`)

### Changed
- **Architecture analysis is now the primary feature**; issue dependency analysis is secondary
- Rewrote `README.md` centered on architecture analysis workflow
- Rewrote shell scripts: `run_analysis.sh` runs scan/analyze/visualize pipeline; `demo_analysis.sh` runs self-scan demo
- Updated `.env.example` to lead with architecture config, removed legacy sections
- `validate_config()` no longer requires `OPENAI_API_KEY` (only needed for `--deep`)
- Simplified `config.py`: removed legacy table names, embedding/clustering/visualization configs
- Simplified `commands.py`: removed 6 legacy command functions (~600 lines)
- Simplified `cli.py`: removed 6 legacy subcommands

### Removed
- **6 legacy source modules** (~2,800 lines of dead code):
  - `rlm_analysis.py`, `rlm_analyzer.py`, `issue_correlation.py`
  - `embeddings.py`, `clustering.py`, `report_generation.py`
- **Legacy CLI commands**: `estimate`, `analyze`, `compare`, `analyze-deps`, `rlm-impact`, `rlm-prioritize`
- **7 unused dependencies**: hdbscan, matplotlib, nltk, plotly, scikit-learn, seaborn, spacy
- **5 unused dev dependencies**: ipykernel, ipython, jupyter, sphinx, sphinx-rtd-theme
- **Stale files**: `starlette_analysis.db`, `rlm.md`, `test_pytorch_dependencies.py`, `test_rlm_behavior.py`
- **Stale docs**: `docs/FEATURES.md`, `docs/DEPENDENCY_ANALYZER_README.md`
- **Stale outputs**: demo and legacy visualization/analysis files
- **Empty directories**: `notebooks/`, `src/rlm_codelens/data/`
- **Legacy config**: `RLM_ROOT_MODEL`, `RLM_SUB_MODEL`, `RLM_MAX_DEPTH`, `EMBEDDING_*`, `MIN_CLUSTER_*`, `MAX_GRAPH_NODES`, `MIN_CORRELATION_STRENGTH`, `USE_SAMPLE_DATA`, `SAMPLE_SIZE`, `CORRELATION_TYPES`, `CATEGORIES`, `COSTS`, dynamic `TABLE_*` names

## [0.1.0] - 2025-02-13

### Added
- Initial release of RLM-Codelens
- Repository analysis using Recursive Language Models
- GitHub data collection for issues and PRs
- OpenAI embeddings generation (text-embedding-3-small)
- HDBSCAN topic clustering
- Cost tracking and budget management
- Secure RLM client with prompt sanitization

[0.2.1]: https://github.com/knijesh/rlm-codelens/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/knijesh/rlm-codelens/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/knijesh/rlm-codelens/releases/tag/v0.1.0
