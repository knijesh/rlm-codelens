# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Modern Python package layout following PyTorch standards
- CLI command `rlmc` with subcommands: `estimate`, `analyze`, `compare`
- Comprehensive type hints throughout the codebase
- GitHub Actions workflows for CI/CD
- Issue and PR templates
- Contributing guidelines and code of conduct
- Pre-commit hooks configuration
- Full test suite with pytest

### Changed
- Restructured package to use `src/` layout
- Updated all imports to use absolute package paths
- Migrated from requirements.txt to pyproject.toml with uv
- Improved CLI interface with argparse subcommands
- Enhanced error handling and user feedback

### Fixed
- Fixed import issues when running as installed package
- Updated relative imports to absolute imports

## [0.1.0] - 2025-02-13

### Added
- Initial release of RLM-Codelens
- Repository analysis using Recursive Language Models
- GitHub data collection for issues and PRs
- OpenAI embeddings generation
- HDBSCAN topic clustering
- Issue correlation discovery
- Interactive D3.js visualization
- Cost tracking and budget management
- Secure RLM client with prompt sanitization
- Circuit breaker pattern for reliability
- Report generation with visualizations

### Features
- Analyze any GitHub repository (not just PyTorch)
- Cost estimation before running analysis
- Parallel processing for performance
- Caching to reduce API costs
- Budget enforcement with hard limits
- Input sanitization for security
- Interactive force-directed graph visualization
- Export to JSON and Markdown formats

[Unreleased]: https://github.com/knijesh/rlm-codelens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/knijesh/rlm-codelens/releases/tag/v0.1.0
