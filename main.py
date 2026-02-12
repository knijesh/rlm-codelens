"""
Main Orchestrator - Redesigned with Security, Cost Control, and Performance
Runs the complete PyTorch analysis pipeline with production-ready features
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    validate_config,
    print_config,
    USE_SAMPLE_DATA,
    SAMPLE_SIZE,
    BUDGET_LIMIT,
    REPO_OWNER,
    REPO_NAME,
)
from utils.cost_tracker import CostTracker
from utils.cost_estimator import CostCalculator
from utils.secure_rlm_client import SecureRLMClient


def run_phase(phase_name, phase_func, *args, **kwargs):
    """Helper to run a phase with error handling and monitoring"""
    import time

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


def estimate_costs_only(num_items=80000):
    """Run cost estimation without making API calls"""
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


def main():
    """Main entry point with production-ready orchestration"""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="PyTorch Repository Analysis with Secure RLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate costs only (no API calls)
  python main.py --estimate-only
  
  # Run with sample data (fast, cheap)
  python main.py --sample --limit 1000
  
  # Run full analysis with tight budget
  python main.py --budget 20.0
  
  # Run specific phase
  python main.py --phase rlm --sample
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["all", "collect", "embed", "cluster", "rlm", "correlate", "report"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Use sample data (faster, cheaper)"
    )
    parser.add_argument("--limit", type=int, help="Limit number of items to process")
    parser.add_argument(
        "--budget",
        type=float,
        default=BUDGET_LIMIT,
        help=f"Budget limit in USD (default: ${BUDGET_LIMIT})",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate costs, don't run analysis",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Show RLM vs Non-RLM comparison"
    )
    parser.add_argument(
        "--skip-estimate", action="store_true", help="Skip pre-flight cost estimation"
    )

    args = parser.parse_args()

    # Validate configuration
    print("\n🔧 Validating configuration...")
    validate_config()
    print_config()

    # Estimate costs first
    if args.estimate_only:
        estimate_costs_only(args.limit or 80000)
        return

    # Determine number of items
    num_items = args.limit or (SAMPLE_SIZE if args.sample or USE_SAMPLE_DATA else 80000)

    if not args.skip_estimate:
        feasible = estimate_costs_only(num_items)
        if not feasible:
            print("\n⚠️  Continuing anyway (use --skip-estimate to bypass)")
            response = input("Continue? (y/n): ")
            if response.lower() != "y":
                print("Exiting.")
                return

    if args.compare:
        calculator = CostCalculator(budget_limit=args.budget)
        calculator.print_comparison(num_items=100)
        return

    # Initialize cost tracker with user-specified budget
    cost_tracker = CostTracker(budget_limit=args.budget)

    # Track start time
    from datetime import datetime

    start_time = datetime.now()

    try:
        # Phase 1: Data Collection
        if args.phase in ["all", "collect"]:
            from data_collection import PyTorchDataCollector

            limit = args.limit or (
                SAMPLE_SIZE if args.sample or USE_SAMPLE_DATA else None
            )

            collector = run_phase("Data Collection", PyTorchDataCollector)
            df = run_phase("Collecting Data", collector.collect_all, limit=limit)

            print(f"\n📊 Collected {len(df)} items")
            cost_tracker.print_summary()

        # Phase 2: Embeddings
        if args.phase in ["all", "embed"]:
            from embeddings import EmbeddingGenerator

            generator = run_phase("Initializing Embeddings", EmbeddingGenerator)
            df = run_phase("Generating Embeddings", generator.generate_embeddings)

            # Track embedding costs
            cost_tracker.add_embedding_call(
                len(df) * 800,  # Estimated tokens
                "text-embedding-3-small",
            )
            cost_tracker.print_summary()

        # Phase 3: Clustering
        if args.phase in ["all", "cluster"]:
            from clustering import TopicClusterer

            clusterer = run_phase("Initializing Clustering", TopicClusterer)
            df, stats = run_phase("Clustering Items", clusterer.cluster)

            print(f"\n📊 Created {len(stats)} clusters")

        # Phase 4: RLM Analysis (SECURE VERSION)
        if args.phase in ["all", "rlm"]:
            print("\n" + "=" * 70)
            print("🤖 RLM ANALYSIS (SECURE VERSION)")
            print("=" * 70)

            # Import redesigned module
            try:
                from rlm_analysis_v2 import SecureRLMAnalyzer, AnalysisConfig

                # Configure for cost efficiency
                config = AnalysisConfig(
                    max_clusters=50 if args.sample else 100,
                    sample_size=5,  # Reduced from 20
                    parallel_workers=4,
                    enable_caching=True,
                    skip_if_over_budget=True,
                    prompt_optimization=True,
                )

                analyzer = SecureRLMAnalyzer(budget_limit=args.budget, config=config)

                # Analyze in parallel with cost control
                cluster_analyses = analyzer.analyze_clusters_parallel()
                correlations = analyzer.discover_correlations_safe()

                print(f"\n📊 Analyzed {len(cluster_analyses)} clusters")
                print(f"📊 Discovered {len(correlations)} correlations")

                # Print comparative analysis
                analyzer.comparative_analyzer.print_report()

            except ImportError as e:
                print(f"⚠️  Could not import secure RLM: {e}")
                print("   Falling back to original implementation...")

                from rlm_analysis import PyTorchRLMAnalyzer

                analyzer = PyTorchRLMAnalyzer()
                cluster_analyses = analyzer.analyze_clusters()
                correlations = analyzer.discover_correlations()

            cost_tracker.print_summary()

        # Phase 5: Issue Correlation
        if args.phase in ["all", "correlate"]:
            from issue_correlation import IssueCorrelationAnalyzer

            analyzer = run_phase(
                "Initializing Correlation Analysis", IssueCorrelationAnalyzer
            )

            df = run_phase("Loading Data", analyzer.load_data)
            correlations = run_phase(
                "Finding Correlations", analyzer.find_correlations, df
            )
            G = run_phase("Building Graph", analyzer.build_graph, correlations, df)

            # Export for visualization
            run_phase(
                "Exporting Graph", analyzer.export_for_d3, G, "outputs/issue_graph.json"
            )

            # Get central issues
            central = analyzer.analyze_central_issues(G, top_n=20)
            print(f"\n📊 Top 10 most central issues:")
            for issue in central[:10]:
                print(
                    f"  #{issue['number']}: {issue['title'][:60]}... (score: {issue['composite_score']:.3f})"
                )

        # Phase 6: Report Generation
        if args.phase in ["all", "report"]:
            from report_generation import ReportGenerator

            generator = run_phase("Initializing Report Generator", ReportGenerator)

            report = run_phase(
                "Generating Report", generator.generate_executive_summary
            )
            run_phase("Creating Visualizations", generator.generate_visualizations)

            print("\n📄 Report generated: outputs/pytorch_analysis_report.md")

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("🎉 ANALYSIS COMPLETE!")
        print("=" * 70)
        print(f"\n⏱️  Total time: {duration}")
        print(f"💰 Total cost: ${cost_tracker.current_cost:.2f} / ${args.budget:.2f}")
        print(f"📁 Outputs directory: outputs/")
        print("\nGenerated files:")
        print("  - outputs/issue_graph.json (for visualization)")
        print("  - outputs/correlation_analysis.json")
        print("  - outputs/pytorch_analysis_report.md")
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


if __name__ == "__main__":
    main()
