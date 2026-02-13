"""CLI entry point for rlmc command.

This module handles argument parsing and dispatches to command implementations.

Usage:
    rlmc estimate [--items N]
    rlmc analyze [owner/repo] [--sample] [--limit N] [--budget N] [--phase PHASE]
    rlmc compare [--items N]
"""

import argparse
import sys
from typing import List, Optional

from rlm_codelens import __version__
from rlm_codelens.commands import compare_methods, estimate_costs, run_analysis
from rlm_codelens.config import REPO_OWNER, REPO_NAME


def get_default_repo() -> Optional[str]:
    """Get default repository from environment variables.

    Returns:
        Repository string in format owner/repo, or None if not configured.
    """
    if REPO_OWNER and REPO_NAME:
        return f"{REPO_OWNER}/{REPO_NAME}"
    return None


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for rlmc CLI."""
    default_repo = get_default_repo()
    default_repo_help = (
        f" (default from .env: {default_repo})"
        if default_repo
        else " (not configured in .env)"
    )

    parser = argparse.ArgumentParser(
        prog="rlmc",
        description="RLM-Codelens: Repository analysis with Recursive Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Estimate costs only (no API calls)
  rlmc estimate --items 1000
  
  # Analyze using repository from .env file
  rlmc analyze --sample --limit 20
  
  # Analyze a specific repository (overrides .env)
  rlmc analyze encode/starlette --sample --limit 20
  
  # Run full analysis with budget control
  rlmc analyze pytorch/pytorch --budget 20.0
  
  # Run specific phase only
  rlmc analyze owner/repo --phase collect --sample
  
  # Compare RLM vs Non-RLM approaches
  rlmc compare --items 100
  
Environment:
  Set REPO_OWNER and REPO_NAME in .env file to configure default repository.
  Command line argument will override .env values.{default_repo_help}
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # estimate command
    estimate_parser = subparsers.add_parser(
        "estimate",
        help="Estimate costs without making API calls",
        description="Run pre-flight cost estimation to check if analysis is feasible with current budget.",
    )
    estimate_parser.add_argument(
        "--items",
        type=int,
        default=80000,
        help="Number of items to estimate for (default: 80000)",
    )

    # analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run repository analysis",
        description="Analyze a GitHub repository using Recursive Language Models. Uses REPO_OWNER and REPO_NAME from .env if not specified.",
    )
    analyze_parser.add_argument(
        "repo",
        nargs="?",
        default=None,
        help=f"Repository to analyze (format: owner/repo). Defaults to {default_repo} from .env if set."
        if default_repo
        else "Repository to analyze (format: owner/repo). Set REPO_OWNER and REPO_NAME in .env to use as default.",
    )
    analyze_parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample data (faster, cheaper)",
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of items to process",
    )
    analyze_parser.add_argument(
        "--budget",
        type=float,
        help="Budget limit in USD",
    )
    analyze_parser.add_argument(
        "--phase",
        choices=["all", "collect", "embed", "cluster", "rlm", "correlate", "report"],
        default="all",
        help="Which phase to run (default: all)",
    )
    analyze_parser.add_argument(
        "--skip-estimate",
        action="store_true",
        help="Skip pre-flight cost estimation",
    )

    # compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare RLM vs Non-RLM approaches",
        description="Show cost and performance comparison between RLM and traditional approaches.",
    )
    compare_parser.add_argument(
        "--items",
        type=int,
        default=100,
        help="Number of items to compare (default: 100)",
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
        if parsed_args.command == "estimate":
            feasible = estimate_costs(num_items=parsed_args.items)
            return 0 if feasible else 1

        elif parsed_args.command == "analyze":
            # Use command line repo if provided, otherwise use .env default
            repo = parsed_args.repo if parsed_args.repo else get_default_repo()

            if not repo:
                print("❌ Error: No repository specified.")
                print("\nPlease either:")
                print("  1. Provide repository as argument: rlmc analyze owner/repo")
                print("  2. Set REPO_OWNER and REPO_NAME in .env file")
                print("\nExample .env configuration:")
                print("  REPO_OWNER=encode")
                print("  REPO_NAME=starlette")
                return 1

            print(f"📦 Analyzing repository: {repo}")
            if parsed_args.repo:
                print("   (from command line argument)")
            else:
                print("   (from .env configuration)")
            print()

            run_analysis(
                repo=repo,
                sample=parsed_args.sample,
                limit=parsed_args.limit,
                budget=parsed_args.budget,
                phase=parsed_args.phase,
                skip_estimate=parsed_args.skip_estimate,
            )
            return 0

        elif parsed_args.command == "compare":
            compare_methods(num_items=parsed_args.items)
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
