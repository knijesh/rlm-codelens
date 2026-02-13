# Contributing to RLM-Codelens

First off, thank you for considering contributing to RLM-Codelens! It's people like you that make this project a great tool for the community.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include code samples and command outputs**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the enhancement**
- **Explain why this enhancement would be useful**

### Pull Requests

1. Fork the repository
2. Create a new branch from `main` for your feature or bug fix
3. Make your changes
4. Run the tests and ensure they pass
5. Update documentation if needed
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Git

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/rlm-codelens.git
cd rlm-codelens

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Install the package in editable mode
uv pip install -e .
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run tests with coverage
uv run pytest tests/ --cov=rlm_codelens --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_core.py -v
```

### Code Style

We use the following tools to maintain code quality:

- **ruff**: Linting and code style
- **black**: Code formatting
- **mypy**: Type checking

```bash
# Format code
uv run black src/ tests/

# Check linting
uv run ruff check src/ tests/

# Type checking
uv run mypy src/rlm_codelens/
```

### Pre-commit Hooks

We recommend setting up pre-commit hooks to automatically check your code:

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Project Structure

```
rlm-codelens/
├── src/rlm_codelens/     # Main source code
│   ├── cli.py            # CLI entry point
│   ├── commands.py       # Command implementations
│   ├── config.py         # Configuration
│   └── utils/            # Utility modules
├── tests/                # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── docs/                 # Documentation
└── .github/              # GitHub templates and workflows
```

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Documentation

- Update the README.md if you change functionality
- Add docstrings to new functions and classes
- Update type hints for new parameters

## Questions?

Feel free to open an issue with your question or contact the maintainers.

Thank you for contributing!
