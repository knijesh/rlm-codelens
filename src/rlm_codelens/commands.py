"""Command implementations for rlmc CLI.

This module contains the actual implementation of CLI commands,
separated from argument parsing for better testability.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from rlm_codelens.config import (
    BUDGET_LIMIT,
    REPO_FULL_NAME,
    REPO_NAME,
    REPO_OWNER,
    REPO_SLUG,
    SAMPLE_SIZE,
    USE_SAMPLE_DATA,
    print_config,
    set_repo,
    validate_config,
)
from rlm_codelens.utils.cost_estimator import CostCalculator
from rlm_codelens.utils.cost_tracker import CostTracker


def run_phase(phase_name: str, phase_func, *args, **kwargs):
    """Helper to run a phase with error handling and monitoring."""
    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"🚀 PHASE: {phase_name}")
    print("=" * 70)

    try:
        result = phase_func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"\n✅ {phase_name} completed successfully ({duration:.1f}s)")
        return result
    except Exception as e:
        print(f"\n❌ {phase_name} failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def estimate_costs(num_items: int = 80000) -> bool:
    """Run cost estimation without making API calls.

    Args:
        num_items: Number of items to estimate for

    Returns:
        True if feasible, False otherwise
    """
    print("\n" + "=" * 70)
    print("💰 PRE-FLIGHT COST ESTIMATION")
    print("=" * 70)

    calculator = CostCalculator(budget_limit=BUDGET_LIMIT)

    # Project estimate
    repo_full_name = f"{REPO_OWNER}/{REPO_NAME}"
    calculator.print_project_estimate(num_items=num_items, repo_name=repo_full_name)

    # RLM vs Non-RLM comparison
    calculator.print_comparison(num_items=100)

    # Budget recommendation
    rec = calculator.get_budget_recommendation(BUDGET_LIMIT, num_items)
    print(f"\n📋 RECOMMENDATION:")
    print(f"   {rec['message']}")

    if not rec.get("feasible", True):
        print(f"\n⛔ Analysis not recommended with current budget.")
        print(f"   {rec['recommendation']}")
        return False

    return True


def compare_methods(num_items: int = 100) -> None:
    """Show RLM vs Non-RLM comparison.

    Args:
        num_items: Number of items to compare
    """
    calculator = CostCalculator(budget_limit=BUDGET_LIMIT)
    calculator.print_comparison(num_items=num_items)


def run_analysis(
    repo: str,
    sample: bool = False,
    limit: Optional[int] = None,
    budget: Optional[float] = None,
    phase: str = "all",
    skip_estimate: bool = False,
) -> None:
    """Run the complete repository analysis pipeline.

    Args:
        repo: Repository to analyze (owner/repo format)
        sample: Whether to use sample data
        limit: Maximum number of items to process
        budget: Budget limit in USD
        phase: Which phase to run (all, collect, embed, cluster, rlm, correlate, report)
        skip_estimate: Whether to skip pre-flight cost estimation
    """
    # Parse repository string first so DB and table names match the repo
    try:
        repo_owner, repo_name = repo.split("/")
    except ValueError:
        print(f"❌ Error: Invalid repository format '{repo}'")
        print("   Expected format: owner/repo")
        print("   Example: encode/starlette")
        return

    # Set config to this repo so DATABASE_URL and table names use repo name (e.g. encode_starlette_analysis.db)
    set_repo(repo_owner, repo_name)

    # Validate configuration
    print("\n🔧 Validating configuration...")
    validate_config()
    print_config()

    # Use config default when budget not specified (e.g. when --budget omitted in CLI)
    if budget is None:
        budget = BUDGET_LIMIT

    # Determine number of items
    num_items = limit or (SAMPLE_SIZE if sample or USE_SAMPLE_DATA else 80000)

    # Estimate costs first
    if not skip_estimate:
        feasible = estimate_costs(num_items)
        if not feasible:
            print("\n⚠️  Continuing anyway (use --skip-estimate to bypass)")
            response = input("Continue? (y/n): ")
            if response.lower() != "y":
                print("Exiting.")
                return

    # Initialize cost tracker with user-specified budget
    cost_tracker = CostTracker(budget_limit=budget)

    # Track start time
    start_time = datetime.now()

    try:
        # Phase 1: Data Collection
        if phase in ["all", "collect"]:
            from rlm_codelens.data_collection import RepositoryDataCollector

            item_limit = limit or (SAMPLE_SIZE if sample or USE_SAMPLE_DATA else None)

            collector = run_phase(
                "Data Collection",
                lambda: RepositoryDataCollector(
                    repo_owner=repo_owner, repo_name=repo_name
                ),
            )
            df = run_phase("Collecting Data", collector.collect_all, limit=item_limit)

            print(f"\n📊 Collected {len(df)} items from {repo}")
            cost_tracker.print_summary()

        # Phase 2: Embeddings
        if phase in ["all", "embed"]:
            from rlm_codelens.embeddings import EmbeddingGenerator

            generator = run_phase("Initializing Embeddings", EmbeddingGenerator)
            df = run_phase("Generating Embeddings", generator.generate_embeddings)

            # Track embedding costs
            cost_tracker.add_embedding_call(
                len(df) * 800,  # Estimated tokens
                "text-embedding-3-small",
            )
            cost_tracker.print_summary()

        # Phase 3: Clustering
        if phase in ["all", "cluster"]:
            from rlm_codelens.clustering import TopicClusterer

            clusterer = run_phase("Initializing Clustering", TopicClusterer)
            df, stats = run_phase("Clustering Items", clusterer.cluster)

            print(f"\n📊 Created {len(stats)} clusters")

        # Phase 4: RLM Analysis
        if phase in ["all", "rlm"]:
            print("\n" + "=" * 70)
            print("🤖 RLM ANALYSIS")
            print("=" * 70)

            try:
                from rlm_codelens.rlm_analysis import SecureRLMAnalyzer, AnalysisConfig

                # Configure for cost efficiency
                config = AnalysisConfig(
                    max_clusters=50 if sample else 100,
                    sample_size=5,
                    parallel_workers=4,
                    enable_caching=True,
                    skip_if_over_budget=True,
                    prompt_optimization=True,
                )

                analyzer = SecureRLMAnalyzer(budget_limit=budget, config=config)

                # Analyze in parallel with cost control
                cluster_analyses = analyzer.analyze_clusters_parallel()
                correlations = analyzer.discover_correlations_safe()

                print(f"\n📊 Analyzed {len(cluster_analyses)} clusters")
                print(f"📊 Discovered {len(correlations)} correlations")

            except ImportError as e:
                print(f"⚠️  Could not import RLM analyzer: {e}")
                print("   Please ensure rlm_analysis.py is properly configured")
                raise

            cost_tracker.print_summary()

        # Phase 5: Issue Correlation
        if phase in ["all", "correlate"]:
            from rlm_codelens.issue_correlation import IssueCorrelationAnalyzer

            analyzer = run_phase(
                "Initializing Correlation Analysis", IssueCorrelationAnalyzer
            )

            df = run_phase("Loading Data", analyzer.load_data)
            correlations = run_phase(
                "Finding Correlations", analyzer.find_correlations, df
            )
            G = run_phase("Building Graph", analyzer.build_graph, correlations, df)

            # Export for visualization
            graph_data = run_phase(
                "Exporting Graph", analyzer.export_for_d3, G, "outputs/issue_graph.json"
            )
            # Also copy to visualization directory
            import shutil

            shutil.copy("outputs/issue_graph.json", "visualization/issue_graph.json")

            # Get central issues
            central = analyzer.analyze_central_issues(G, top_n=20)
            print(f"\n📊 Top 10 most central issues:")
            for issue in central[:10]:
                print(
                    f"  #{issue['number']}: {issue['title'][:60]}... (score: {issue['composite_score']:.3f})"
                )

            # Save correlation analysis JSON
            import json

            correlation_results = {
                "central_issues": central,
                "total_correlations": len(correlations),
                "correlation_breakdown": graph_data["statistics"]["correlation_types"],
            }
            Path("outputs").mkdir(exist_ok=True)
            with open("outputs/correlation_analysis.json", "w") as f:
                json.dump(correlation_results, f, indent=2)
            print(
                f"  📄 Saved correlation analysis to outputs/correlation_analysis.json"
            )

        # Phase 6: Report Generation
        if phase in ["all", "report"]:
            from rlm_codelens.report_generation import ReportGenerator

            generator = run_phase("Initializing Report Generator", ReportGenerator)

            report = run_phase(
                "Generating Report", generator.generate_executive_summary
            )
            run_phase("Creating Visualizations", generator.generate_visualizations)

            print(f"\n📄 Report generated: outputs/{REPO_SLUG}_analysis_report.md")

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\n⏱️  Total time: {duration}")
        print(f"💰 Total cost: ${cost_tracker.current_cost:.2f} / ${budget:.2f}")
        print(f"📁 Outputs directory: outputs/")
        print("\nGenerated files:")
        print("  - outputs/issue_graph.json (for visualization)")
        print("  - outputs/correlation_analysis.json")
        print(f"  - outputs/{REPO_SLUG}_analysis_report.md")
        print("  - outputs/*.png (charts)")
        print("  - outputs/cost_log.json (detailed cost tracking)")
        print("\n🌐 Open visualization/issue_graph_visualization.html in your browser")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cost_tracker.print_summary()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed: {e}")
        cost_tracker.print_summary()
        sys.exit(1)
