"""Command implementations for rlmc CLI.

This module contains the actual implementation of CLI commands,
separated from argument parsing for better testability.
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def run_phase(
    phase_name: str, phase_func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
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


def _fetch_ollama_models(ollama_url: str = "http://localhost:11434") -> Optional[list]:
    """Query Ollama for installed models.

    Args:
        ollama_url: Base URL for the Ollama server.

    Returns:
        List of model dicts, or None if Ollama is unreachable.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return sorted(data.get("models", []), key=lambda x: x.get("name", ""))
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def _print_ollama_table(models: list) -> None:
    """Print a numbered table of Ollama models."""
    print(f"\n  {'#':<4} {'Model':<35} {'Size':>10}  {'Modified'}")
    print("  " + "-" * 66)
    for i, m in enumerate(models, 1):
        name = m.get("name", "unknown")
        size_gb = m.get("size", 0) / (1024**3)
        modified = m.get("modified_at", "")[:10]
        print(f"  {i:<4} {name:<35} {size_gb:>7.1f} GB  {modified}")


def _select_ollama_model(
    ollama_url: str = "http://localhost:11434",
) -> Optional[str]:
    """Interactively select an Ollama model. Returns model name or None."""
    models = _fetch_ollama_models(ollama_url)
    if models is None:
        print(f"\n❌ Could not connect to Ollama at {ollama_url}")
        print("   Make sure Ollama is running:  ollama serve")
        return None
    if not models:
        print("\nNo models installed. Pull one with:  ollama pull llama3.1")
        return None

    _print_ollama_table(models)
    print()

    while True:
        try:
            choice = input(f"  Select model [1-{len(models)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            selected = str(models[int(choice) - 1]["name"])
            print(f"  → Selected: {selected}")
            return selected
        # Also accept a model name typed directly
        names = [m["name"] for m in models]
        if choice in names:
            print(f"  → Selected: {choice}")
            return choice
        print(f"  Invalid choice. Enter a number 1-{len(models)} or a model name.")


def list_ollama_models(
    ollama_url: str = "http://localhost:11434",
    interactive: bool = True,
) -> None:
    """List models available on a local Ollama instance.

    Args:
        ollama_url: Base URL for the Ollama server.
        interactive: If True, prompt user to select a model.
    """
    print(f"\nQuerying Ollama at {ollama_url} ...")

    models = _fetch_ollama_models(ollama_url)
    if models is None:
        print(f"\n❌ Could not connect to Ollama at {ollama_url}")
        print("   Make sure Ollama is running:  ollama serve")
        return
    if not models:
        print("\nNo models installed. Pull one with:  ollama pull llama3.1")
        return

    _print_ollama_table(models)
    print(f"\n  {len(models)} model(s) available.")

    if not interactive:
        return

    print()
    selected = _select_ollama_model(ollama_url)
    if selected:
        print("\n  Run deep analysis with this model:")
        print("  rlmc analyze-architecture scan.json --deep \\")
        print(f"    --backend openai --model {selected} \\")
        print(f"    --base-url {ollama_url}/v1")


def scan_repository(
    repo_path: str,
    output: str = "outputs/scan.json",
    exclude: Optional[list] = None,
    include_source: bool = False,
    name: Optional[str] = None,
) -> None:
    """Scan a repository and extract module structure.

    Args:
        repo_path: Local path or remote git URL
        output: Output JSON file path
        exclude: Additional directory names to exclude
        include_source: Whether to include source text
        name: Override the repository name
    """
    from rlm_codelens.repo_scanner import RepositoryScanner

    print("\n" + "=" * 70)
    print("📂 REPOSITORY SCAN")
    print("=" * 70)
    print(f"Repository: {repo_path}")
    print(f"Output: {output}")
    if exclude:
        print(f"Extra excludes: {', '.join(exclude)}")
    print("=" * 70)

    try:
        print("\n🔍 Scanning repository...")
        scanner = RepositoryScanner(
            repo_path=repo_path,
            exclude_patterns=exclude,
            include_source=include_source,
        )
        structure = scanner.scan()

        if name:
            structure.name = name

        # Print summary
        print("\n📊 Scan Summary:")
        print(f"   Repository: {structure.name}")
        print(f"   Python files: {structure.total_files}")
        print(f"   Total lines: {structure.total_lines:,}")
        print(f"   Packages: {len(structure.packages)}")
        print(f"   Entry points: {len(structure.entry_points)}")

        if structure.packages:
            print("\n📦 Packages:")
            for pkg in structure.packages[:15]:
                print(f"   - {pkg}")
            if len(structure.packages) > 15:
                print(f"   ... and {len(structure.packages) - 15} more")

        if structure.entry_points:
            print("\n🚀 Entry Points:")
            for ep in structure.entry_points:
                print(f"   - {ep}")

        # Save
        structure.save(output)

        print(f"\n{'=' * 70}")
        print(f"✅ Scan saved to: {output}")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error during scan: {e}")
        import traceback

        traceback.print_exc()


def analyze_architecture(
    scan_file: Optional[str] = None,
    repo_path: Optional[str] = None,
    deep: bool = False,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    budget: float = 10.0,
    output: str = "outputs/architecture.json",
) -> None:
    """Analyze codebase architecture from scan data.

    Args:
        scan_file: Path to scan JSON (from scan-repo)
        repo_path: Repository path to scan inline (alternative to scan_file)
        deep: Enable RLM-powered deep analysis
        backend: RLM backend name
        model: RLM model name
        base_url: Override API base URL (e.g. http://localhost:11434/v1 for Ollama)
        budget: RLM budget limit
        output: Output JSON file path
    """
    from rlm_codelens.codebase_graph import CodebaseGraphAnalyzer
    from rlm_codelens.config import RLM_BACKEND, RLM_BASE_URL, RLM_MODEL
    from rlm_codelens.repo_scanner import RepositoryScanner, RepositoryStructure

    print("\n" + "=" * 70)
    print("🏗️  ARCHITECTURE ANALYSIS")
    print("=" * 70)

    # Get the repository structure
    if scan_file:
        print(f"Loading scan: {scan_file}")
        structure = RepositoryStructure.load(scan_file)
    elif repo_path:
        print(f"Scanning repository: {repo_path}")
        scanner = RepositoryScanner(repo_path, include_source=deep)
        structure = scanner.scan()
    else:
        print("❌ Error: Provide either a scan file or --repo path")
        return

    print(
        f"Repository: {structure.name} ({structure.total_files} files, {structure.total_lines:,} LOC)"
    )
    print(f"Deep analysis: {'enabled' if deep else 'disabled'}")
    print(f"Output: {output}")
    print("=" * 70)

    # Build and analyze graph
    print("\n📊 Building module dependency graph...")
    graph_analyzer = CodebaseGraphAnalyzer(structure)
    analysis = graph_analyzer.analyze()

    # Print static analysis summary
    print("\n📈 Static Analysis Results:")
    print(f"   Modules: {analysis.total_modules}")
    print(f"   Import edges: {analysis.total_edges}")
    print(f"   Circular imports: {len(analysis.cycles)}")
    print(f"   Anti-patterns: {len(analysis.anti_patterns)}")

    if analysis.cycles:
        print("\n🔄 Circular Imports:")
        for cycle in analysis.cycles[:5]:
            names = [Path(p).stem for p in cycle]
            print(f"   {' -> '.join(names)} -> {names[0]}")
        if len(analysis.cycles) > 5:
            print(f"   ... and {len(analysis.cycles) - 5} more")

    if analysis.hub_modules:
        print("\n🔗 Hub Modules (highest connectivity):")
        for hub in analysis.hub_modules[:5]:
            print(
                f"   {hub['module']}: fan_in={hub['fan_in']}, fan_out={hub['fan_out']}, LOC={hub['loc']}"
            )

    if analysis.anti_patterns:
        print("\n⚠️  Anti-Patterns:")
        for ap in analysis.anti_patterns[:5]:
            print(f"   [{ap['severity']}] {ap['type']}: {ap['details'][:80]}")
        if len(analysis.anti_patterns) > 5:
            print(f"   ... and {len(analysis.anti_patterns) - 5} more")

    # Layer distribution
    layer_counts: Dict[str, int] = {}
    for layer in analysis.layers.values():
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
    print("\n📐 Layer Distribution:")
    for layer, count in sorted(layer_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {layer}: {count} modules")

    # Deep RLM analysis
    if deep:
        print(f"\n{'=' * 70}")
        print("🤖 RLM DEEP ANALYSIS")
        print("=" * 70)

        try:
            from rlm_codelens.architecture_analyzer import ArchitectureRLMAnalyzer

            rlm_backend = backend or RLM_BACKEND
            rlm_base_url = base_url or RLM_BASE_URL or None
            rlm_model = model or RLM_MODEL

            # Interactive model selection when using Ollama without explicit --model
            if rlm_base_url and not model and "11434" in rlm_base_url:
                ollama_base = rlm_base_url.rstrip("/").removesuffix("/v1")
                print("\nOllama detected — select a model:\n")
                selected = _select_ollama_model(ollama_base)
                if not selected:
                    print("\n❌ No model selected. Aborting deep analysis.")
                    return
                rlm_model = selected
                print()

            print(f"Backend: {rlm_backend}")
            print(f"Model: {rlm_model}")
            if rlm_base_url:
                print(f"Base URL: {rlm_base_url}")
            print(f"Budget: ${budget:.2f}")

            rlm_analyzer = ArchitectureRLMAnalyzer(
                structure=structure,
                backend=rlm_backend,
                model=rlm_model,
                base_url=rlm_base_url,
                budget=budget,
            )

            graph_metrics = {
                "cycles": analysis.cycles,
                "hub_modules": analysis.hub_modules,
                "anti_patterns": analysis.anti_patterns,
                "total_modules": analysis.total_modules,
                "total_edges": analysis.total_edges,
            }

            rlm_results = rlm_analyzer.run_all(graph_metrics=graph_metrics)

            # Merge into analysis
            analysis = graph_analyzer.enrich_with_rlm(rlm_results)

            # Print RLM results
            if rlm_results.get("semantic_clusters"):
                print(
                    f"\n🏷️  RLM Module Classifications: {len(rlm_results['semantic_clusters'])} modules classified"
                )

            if rlm_results.get("hidden_dependencies"):
                print(
                    f"\n🔍 Hidden Dependencies Found: {len(rlm_results['hidden_dependencies'])}"
                )
                for dep in rlm_results["hidden_dependencies"][:3]:
                    print(
                        f"   {dep.get('source', '?')} -> {dep.get('target', '?')} ({dep.get('type', '?')})"
                    )

            if rlm_results.get("pattern_analysis"):
                pa = rlm_results["pattern_analysis"]
                print(
                    f"\n🏛️  Detected Pattern: {pa.get('detected_pattern', 'unknown')} (confidence: {pa.get('confidence', 0):.0%})"
                )

            if rlm_results.get("refactoring_suggestions"):
                print("\n💡 Refactoring Suggestions:")
                for suggestion in rlm_results["refactoring_suggestions"][:3]:
                    print(f"   - {suggestion[:100]}")

            cost = rlm_results.get("cost_summary", {})
            print(
                f"\n💰 RLM Cost: ${cost.get('total_cost', 0):.4f} / ${cost.get('budget', budget):.2f}"
            )

        except ImportError as e:
            print(f"\n❌ RLM not available: {e}")
            print("   Install with: pip install rlm")
        except Exception as e:
            print(f"\n❌ RLM analysis failed: {e}")
            import traceback

            traceback.print_exc()

    # Save results
    analysis.save(output)

    print(f"\n{'=' * 70}")
    print(f"✅ Architecture analysis saved to: {output}")
    print("=" * 70)


def visualize_architecture(
    analysis_file: str,
    output: str = "outputs/architecture_visualization.html",
    open_browser: bool = True,
) -> None:
    """Generate interactive architecture visualization.

    Args:
        analysis_file: Path to architecture analysis JSON
        output: Output HTML file path
        open_browser: Whether to open in browser
    """
    from rlm_codelens.visualizer import generate_architecture_visualization

    print("\n" + "=" * 70)
    print("🎨 ARCHITECTURE VISUALIZATION")
    print("=" * 70)
    print(f"Input: {analysis_file}")
    print(f"Output: {output}")
    print("=" * 70 + "\n")

    try:
        output_path = generate_architecture_visualization(
            analysis_file=analysis_file,
            output_file=output,
            open_browser=open_browser,
        )

        print(f"\n{'=' * 70}")
        print("✅ Architecture visualization generated!")
        print(f"📄 File: {output_path}")
        if open_browser:
            print("🌐 Opening in your default browser...")
        else:
            print(f"💡 Open manually: open {output_path}")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you've run 'analyze-architecture' first")
    except Exception as e:
        print(f"\n❌ Error during visualization: {e}")
        import traceback

        traceback.print_exc()


def generate_report(
    analysis_file: str,
    output: str = "outputs/report.html",
    open_browser: bool = True,
) -> None:
    """Generate a standalone HTML analysis report.

    Args:
        analysis_file: Path to architecture analysis JSON
        output: Output HTML file path
        open_browser: Whether to open in browser
    """
    from rlm_codelens.report_generator import generate_analysis_report

    print("\n" + "=" * 70)
    print("📋 ANALYSIS REPORT")
    print("=" * 70)
    print(f"Input: {analysis_file}")
    print(f"Output: {output}")
    print("=" * 70 + "\n")

    try:
        output_path = generate_analysis_report(
            analysis_file=analysis_file,
            output_file=output,
            open_browser=open_browser,
        )

        print(f"\n{'=' * 70}")
        print("✅ Analysis report generated!")
        print(f"📄 File: {output_path}")
        if open_browser:
            print("🌐 Opening in your default browser...")
        else:
            print(f"💡 Open manually: open {output_path}")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Make sure you've run 'analyze-architecture' first")
    except Exception as e:
        print(f"\n❌ Error during report generation: {e}")
        import traceback

        traceback.print_exc()
