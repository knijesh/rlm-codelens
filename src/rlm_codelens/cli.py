"""CLI entry point for rlmc command.

This module handles argument parsing and dispatches to command implementations.

Usage:
    rlmc scan-repo <path>
    rlmc analyze-architecture <scan.json> [--deep]
    rlmc visualize-arch <analysis.json>
"""

import argparse
import sys
from typing import List, Optional

from rlm_codelens import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for rlmc CLI."""
    parser = argparse.ArgumentParser(
        prog="rlmc",
        description="RLM-Codelens: Codebase architecture intelligence powered by Recursive Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan a local repository
  rlmc scan-repo /path/to/repo --output scan.json

  # Analyze architecture (static analysis)
  rlmc analyze-architecture scan.json --output arch.json

  # Analyze with RLM-powered deep analysis
  rlmc analyze-architecture scan.json --deep --budget 5.0

  # Analyze with Ollama (free, local)
  rlmc analyze-architecture scan.json --ollama --model deepseek-r1:latest

  # Generate interactive architecture visualization
  rlmc visualize-arch arch.json

  # One-step: scan + analyze
  rlmc analyze-architecture --repo /path/to/repo
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan-repo command
    scan_parser = subparsers.add_parser(
        "scan-repo",
        help="Scan a repository and extract module structure",
        description="Parse all Python files in a repository using AST to extract imports, classes, functions, and entry points.",
    )
    scan_parser.add_argument(
        "repo_path",
        help="Local path or remote git URL of the repository to scan",
    )
    scan_parser.add_argument(
        "--output",
        default="outputs/scan.json",
        help="Output JSON file path (default: outputs/scan.json)",
    )
    scan_parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Additional directory names to exclude from scanning",
    )
    scan_parser.add_argument(
        "--include-source",
        action="store_true",
        help="Include full source text in scan output (needed for --deep RLM analysis)",
    )
    scan_parser.add_argument(
        "--name",
        default=None,
        help="Override the repository name used in reports and visualizations",
    )

    # analyze-architecture command
    arch_parser = subparsers.add_parser(
        "analyze-architecture",
        help="Analyze codebase architecture from scan data",
        description="Build a module dependency graph and compute architecture metrics. Use --deep for RLM-powered semantic analysis.",
    )
    arch_parser.add_argument(
        "scan_file",
        nargs="?",
        default=None,
        help="Path to scan JSON file (from scan-repo command). Required unless --repo is provided.",
    )
    arch_parser.add_argument(
        "--repo",
        default=None,
        help="Repository path to scan and analyze in one step (skips separate scan-repo)",
    )
    arch_parser.add_argument(
        "--deep",
        action="store_true",
        help="Enable RLM-powered deep analysis (requires API key or --ollama)",
    )
    arch_parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama for deep analysis (shorthand for --backend openai --base-url http://localhost:11434/v1)",
    )
    arch_parser.add_argument(
        "--backend",
        default=None,
        help="RLM backend (default: from RLM_BACKEND env or 'openai')",
    )
    arch_parser.add_argument(
        "--model",
        default=None,
        help="RLM model name (default: from RLM_MODEL env or 'gpt-4o')",
    )
    arch_parser.add_argument(
        "--base-url",
        default=None,
        help="Override API base URL (e.g. http://localhost:11434/v1 for Ollama)",
    )
    arch_parser.add_argument(
        "--budget",
        type=float,
        default=10.0,
        help="RLM budget limit in USD (default: 10.0)",
    )
    arch_parser.add_argument(
        "--output",
        default="outputs/architecture.json",
        help="Output JSON file path (default: outputs/architecture.json)",
    )

    # list-models command
    models_parser = subparsers.add_parser(
        "list-models",
        help="List available models from a local Ollama instance",
        description="Query a running Ollama server for installed models. Useful for picking a --model for --deep analysis.",
    )
    models_parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    models_parser.add_argument(
        "--no-select",
        action="store_true",
        help="Just list models without interactive selection prompt",
    )

    # visualize-arch command
    viz_arch_parser = subparsers.add_parser(
        "visualize-arch",
        help="Visualize codebase architecture as interactive graph",
        description="Generate an interactive D3.js visualization of the architecture analysis.",
    )
    viz_arch_parser.add_argument(
        "analysis_file",
        help="Path to architecture analysis JSON file (from analyze-architecture command)",
    )
    viz_arch_parser.add_argument(
        "--output",
        default="outputs/architecture_visualization.html",
        help="Output HTML file path (default: outputs/architecture_visualization.html)",
    )
    viz_arch_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open in browser",
    )

    # generate-report command
    report_parser = subparsers.add_parser(
        "generate-report",
        help="Generate a standalone HTML analysis report",
        description="Produce a self-contained HTML report explaining the architecture analysis findings.",
    )
    report_parser.add_argument(
        "analysis_file",
        help="Path to architecture analysis JSON file (from analyze-architecture command)",
    )
    report_parser.add_argument(
        "--output",
        default="outputs/report.html",
        help="Output HTML file path (default: outputs/report.html)",
    )
    report_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open in browser",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for rlmc CLI.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 1

    try:
        if parsed_args.command == "scan-repo":
            from rlm_codelens.commands import scan_repository

            scan_repository(
                repo_path=parsed_args.repo_path,
                output=parsed_args.output,
                exclude=parsed_args.exclude,
                include_source=parsed_args.include_source,
                name=parsed_args.name,
            )
            return 0

        elif parsed_args.command == "analyze-architecture":
            from rlm_codelens.commands import analyze_architecture

            # --ollama implies --deep and sets backend/base-url
            backend = parsed_args.backend
            base_url = parsed_args.base_url
            deep = parsed_args.deep
            if parsed_args.ollama:
                deep = True
                backend = backend or "openai"
                base_url = base_url or "http://localhost:11434/v1"

            analyze_architecture(
                scan_file=parsed_args.scan_file,
                repo_path=parsed_args.repo,
                deep=deep,
                backend=backend,
                model=parsed_args.model,
                base_url=base_url,
                budget=parsed_args.budget,
                output=parsed_args.output,
            )
            return 0

        elif parsed_args.command == "list-models":
            from rlm_codelens.commands import list_ollama_models

            list_ollama_models(
                ollama_url=parsed_args.ollama_url,
                interactive=not parsed_args.no_select,
            )
            return 0

        elif parsed_args.command == "visualize-arch":
            from rlm_codelens.commands import visualize_architecture

            visualize_architecture(
                analysis_file=parsed_args.analysis_file,
                output=parsed_args.output,
                open_browser=not parsed_args.no_browser,
            )
            return 0

        elif parsed_args.command == "generate-report":
            from rlm_codelens.commands import generate_report

            generate_report(
                analysis_file=parsed_args.analysis_file,
                output=parsed_args.output,
                open_browser=not parsed_args.no_browser,
            )
            return 0

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
