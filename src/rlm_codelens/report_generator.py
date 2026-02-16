"""HTML analysis report generator for rlm-codelens.

Produces a standalone, self-contained HTML report that explains
the architecture analysis findings with embedded CSS (dark theme).
"""

import json
import webbrowser
from pathlib import Path
from typing import Any, Dict, List


def _health_rating(data: Dict[str, Any]) -> tuple:
    """Compute a health rating (label, color, explanation) from analysis data."""
    penalty = 0

    # Circular dependencies are the worst
    cycles = data.get("cycles", [])
    penalty += len(cycles) * 15

    # Anti-patterns by severity
    for ap in data.get("anti_patterns", []):
        sev = ap.get("severity", "low")
        if sev == "high":
            penalty += 10
        elif sev == "medium":
            penalty += 5
        else:
            penalty += 2

    # Normalize: 0 penalty = 100 score, cap at 0
    score = max(0, 100 - penalty)

    if score >= 80:
        return ("Healthy", "#4ade80", score, "The codebase has a clean architecture with few issues.")
    elif score >= 60:
        return ("Fair", "#facc15", score, "Some architectural issues exist that should be addressed.")
    elif score >= 40:
        return ("Needs Attention", "#fb923c", score, "Multiple architectural issues detected. Prioritize refactoring.")
    else:
        return ("Critical", "#ef4444", score, "Significant structural problems. Immediate refactoring recommended.")


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _module_short(path: str) -> str:
    """Shorten a module path for display."""
    return Path(path).stem


def _build_summary_section(data: Dict[str, Any], health: tuple) -> str:
    total_loc = 0
    for node in data.get("graph_data", {}).get("nodes", []):
        total_loc += node.get("loc", 0)

    label, color, score, _ = health
    badge = f'<span class="badge" style="background:{color};color:#000">{label} ({score}/100)</span>'

    return f"""
    <section id="summary">
      <h2>Repository Summary</h2>
      <div class="grid-2">
        <div class="card">
          <div class="stat-label">Repository</div>
          <div class="stat-value">{_escape(data.get("repository", "unknown"))}</div>
        </div>
        <div class="card">
          <div class="stat-label">Health</div>
          <div class="stat-value">{badge}</div>
        </div>
        <div class="card">
          <div class="stat-label">Modules</div>
          <div class="stat-value">{data.get("total_modules", 0)}</div>
        </div>
        <div class="card">
          <div class="stat-label">Import Edges</div>
          <div class="stat-value">{data.get("total_edges", 0)}</div>
        </div>
        <div class="card">
          <div class="stat-label">Total LOC</div>
          <div class="stat-value">{total_loc:,}</div>
        </div>
        <div class="card">
          <div class="stat-label">Circular Dependencies</div>
          <div class="stat-value">{len(data.get("cycles", []))}</div>
        </div>
      </div>
    </section>"""


def _build_health_section(health: tuple) -> str:
    label, color, score, explanation = health
    bar_width = score

    return f"""
    <section id="health">
      <h2>Health Assessment</h2>
      <div class="card">
        <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
          <span class="badge" style="background:{color};color:#000;font-size:1.1em">{label}</span>
          <span style="color:#94a3b8">Score: {score} / 100</span>
        </div>
        <div class="bar-bg">
          <div class="bar-fill" style="width:{bar_width}%;background:{color}"></div>
        </div>
        <p style="margin-top:12px;color:#cbd5e1">{explanation}</p>
        <p style="color:#94a3b8;font-size:0.85em;margin-top:8px">
          Scoring: starts at 100, penalised &minus;15 per circular dependency,
          &minus;10 per high-severity anti-pattern, &minus;5 per medium, &minus;2 per low.
        </p>
      </div>
    </section>"""


def _build_fanin_fanout_section() -> str:
    return """
    <section id="fan-metrics">
      <h2>Understanding Fan-In &amp; Fan-Out</h2>
      <div class="card">
        <p><strong>Fan-In</strong> (afferent coupling) &mdash; the number of other modules that
        <em>depend on</em> this module. High fan-in means the module is widely used; changes to
        it ripple outward.</p>

        <p><strong>Fan-Out</strong> (efferent coupling) &mdash; the number of modules this module
        <em>depends on</em>. High fan-out means the module has many dependencies and is sensitive
        to changes elsewhere.</p>

        <p><strong>Instability (I)</strong> = Fan-Out / (Fan-In + Fan-Out). Ranges from 0 (maximally
        stable, hard to change) to 1 (maximally unstable, easy to change but volatile).</p>

        <table>
          <thead><tr><th>Metric</th><th>Low Value</th><th>High Value</th></tr></thead>
          <tbody>
            <tr><td>Fan-In</td><td>Module is not widely depended on</td><td>Module is a core dependency &mdash; changes are risky</td></tr>
            <tr><td>Fan-Out</td><td>Module is self-contained</td><td>Module is tightly coupled to many others</td></tr>
            <tr><td>Instability</td><td>Stable foundation (good for core libraries)</td><td>Volatile leaf (good for UI/CLI layers)</td></tr>
          </tbody>
        </table>
      </div>
    </section>"""


def _build_hub_modules_section(data: Dict[str, Any]) -> str:
    hubs = data.get("hub_modules", [])
    if not hubs:
        return """
    <section id="hubs">
      <h2>Hub Modules</h2>
      <div class="card"><p style="color:#94a3b8">No hub modules detected.</p></div>
    </section>"""

    rows = ""
    for h in hubs:
        name = _escape(_module_short(h.get("module", "")))
        rows += f"""
            <tr>
              <td>{name}</td>
              <td>{h.get("fan_in", 0)}</td>
              <td>{h.get("fan_out", 0)}</td>
              <td>{h.get("total", 0)}</td>
              <td>{h.get("loc", 0):,}</td>
            </tr>"""

    return f"""
    <section id="hubs">
      <h2>Hub Modules</h2>
      <div class="card">
        <p style="color:#94a3b8;margin-bottom:12px">Modules with the highest total connectivity (fan-in + fan-out).
        These are critical integration points.</p>
        <table>
          <thead><tr><th>Module</th><th>Fan-In</th><th>Fan-Out</th><th>Total</th><th>LOC</th></tr></thead>
          <tbody>{rows}
          </tbody>
        </table>
      </div>
    </section>"""


def _build_cycles_section(data: Dict[str, Any]) -> str:
    cycles = data.get("cycles", [])
    if not cycles:
        return """
    <section id="cycles">
      <h2>Circular Dependencies</h2>
      <div class="card"><p style="color:#4ade80">No circular dependencies detected. Great!</p></div>
    </section>"""

    items = ""
    for cycle in cycles:
        names = [_module_short(p) for p in cycle]
        chain = " &rarr; ".join(_escape(n) for n in names)
        chain += f" &rarr; {_escape(names[0])}"
        items += f'<li class="cycle-item">{chain}</li>\n'

    return f"""
    <section id="cycles">
      <h2>Circular Dependencies</h2>
      <div class="card">
        <p style="color:#fb923c;margin-bottom:12px">
          {len(cycles)} circular dependency chain{"s" if len(cycles) != 1 else ""} detected.
          Cycles make the code harder to test, refactor, and reason about.
        </p>
        <ol class="cycle-list">{items}</ol>
      </div>
    </section>"""


def _build_antipatterns_section(data: Dict[str, Any]) -> str:
    patterns = data.get("anti_patterns", [])
    if not patterns:
        return """
    <section id="antipatterns">
      <h2>Anti-Patterns</h2>
      <div class="card"><p style="color:#4ade80">No anti-patterns detected. Excellent!</p></div>
    </section>"""

    type_explanations = {
        "god_module": "A module that does too much &mdash; high fan-out and large LOC. Consider splitting into focused sub-modules.",
        "orphan": "A module with no incoming or outgoing dependencies. It may be dead code or missing integration.",
        "layer_violation": "A module in a lower layer imports from a higher layer, breaking the dependency rule.",
        "unstable_dependency": "A stable module depends on an unstable one, creating fragile foundations.",
    }

    severity_order = {"high": 0, "medium": 1, "low": 2}
    patterns_sorted = sorted(patterns, key=lambda p: severity_order.get(p.get("severity", "low"), 3))

    severity_colors = {"high": "#ef4444", "medium": "#fb923c", "low": "#facc15"}

    items = ""
    for ap in patterns_sorted:
        sev = ap.get("severity", "low")
        color = severity_colors.get(sev, "#94a3b8")
        ap_type = ap.get("type", "unknown")
        explanation = type_explanations.get(ap_type, "")
        details = _escape(ap.get("details", ""))

        items += f"""
        <div class="ap-item">
          <div class="ap-header">
            <span class="badge" style="background:{color};color:#000">{sev.upper()}</span>
            <strong>{_escape(ap_type)}</strong>
            <span style="color:#94a3b8">&mdash; {_escape(ap.get("module", ""))}</span>
          </div>
          <p class="ap-details">{details}</p>
          {"<p class='ap-explain'>" + explanation + "</p>" if explanation else ""}
        </div>"""

    high_count = sum(1 for p in patterns if p.get("severity") == "high")
    med_count = sum(1 for p in patterns if p.get("severity") == "medium")
    low_count = sum(1 for p in patterns if p.get("severity") == "low")

    return f"""
    <section id="antipatterns">
      <h2>Anti-Patterns</h2>
      <div class="card">
        <p style="color:#94a3b8;margin-bottom:12px">
          {len(patterns)} anti-pattern{"s" if len(patterns) != 1 else ""} found:
          <span style="color:#ef4444">{high_count} high</span>,
          <span style="color:#fb923c">{med_count} medium</span>,
          <span style="color:#facc15">{low_count} low</span>.
        </p>
        {items}
      </div>
    </section>"""


def _build_layers_section(data: Dict[str, Any]) -> str:
    layers = data.get("layers", {})
    if not layers:
        return """
    <section id="layers">
      <h2>Layer Distribution</h2>
      <div class="card"><p style="color:#94a3b8">No layer information available.</p></div>
    </section>"""

    layer_counts: Dict[str, int] = {}
    for layer in layers.values():
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = max(layer_counts.values()) if layer_counts else 1

    layer_colors = {
        "presentation": "#60a5fa",
        "application": "#a78bfa",
        "domain": "#4ade80",
        "infrastructure": "#fb923c",
        "utility": "#94a3b8",
        "config": "#facc15",
    }

    bars = ""
    for layer_name, count in sorted_layers:
        pct = (count / max_count) * 100
        color = layer_colors.get(layer_name, "#64748b")
        bars += f"""
        <div class="layer-row">
          <div class="layer-label">{_escape(layer_name)}</div>
          <div class="bar-bg" style="flex:1">
            <div class="bar-fill" style="width:{pct}%;background:{color}"></div>
          </div>
          <div class="layer-count">{count}</div>
        </div>"""

    return f"""
    <section id="layers">
      <h2>Layer Distribution</h2>
      <div class="card">{bars}</div>
    </section>"""


def _build_guidance_section() -> str:
    return """
    <section id="guidance">
      <h2>How to Use This Analysis</h2>
      <div class="card">
        <ol class="guidance-list">
          <li><strong>Break circular dependencies first.</strong> Cycles are the highest-impact
          issue. Introduce interfaces, move shared code to a common module, or invert the
          dependency direction.</li>
          <li><strong>Split god modules.</strong> Extract cohesive groups of functions into
          separate modules. Aim for single-responsibility.</li>
          <li><strong>Fix layer violations.</strong> Lower layers should never import from higher
          layers. Use dependency injection or event-based patterns to invert the flow.</li>
          <li><strong>Review orphan modules.</strong> If an orphan is dead code, remove it.
          Otherwise, integrate it properly into the dependency graph.</li>
          <li><strong>Protect hub modules.</strong> High fan-in modules are critical &mdash;
          add thorough tests, keep interfaces stable, and avoid unnecessary changes.</li>
        </ol>
      </div>
    </section>"""


_CSS = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #0f172a; color: #e2e8f0; line-height: 1.6;
      padding: 32px 24px; max-width: 960px; margin: 0 auto;
    }
    h1 { font-size: 1.8em; color: #f8fafc; margin-bottom: 4px; }
    .subtitle { color: #64748b; font-size: 0.95em; margin-bottom: 32px; }
    h2 {
      font-size: 1.25em; color: #f1f5f9; margin-bottom: 16px;
      padding-bottom: 8px; border-bottom: 1px solid #1e293b;
    }
    section { margin-bottom: 36px; }
    .card {
      background: #1e293b; border-radius: 8px; padding: 20px;
      border: 1px solid #334155;
    }
    .grid-2 {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
      gap: 12px;
    }
    .stat-label { color: #94a3b8; font-size: 0.85em; margin-bottom: 4px; }
    .stat-value { font-size: 1.3em; font-weight: 600; color: #f1f5f9; }
    .badge {
      display: inline-block; padding: 2px 10px; border-radius: 12px;
      font-weight: 600; font-size: 0.85em;
    }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #334155; }
    th { color: #94a3b8; font-weight: 600; font-size: 0.85em; text-transform: uppercase; }
    td { color: #cbd5e1; }
    tr:last-child td { border-bottom: none; }
    .bar-bg {
      background: #334155; border-radius: 4px; height: 12px; overflow: hidden;
    }
    .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
    .layer-row {
      display: flex; align-items: center; gap: 12px; margin-bottom: 8px;
    }
    .layer-label { width: 120px; color: #cbd5e1; font-size: 0.9em; text-align: right; }
    .layer-count { width: 30px; color: #94a3b8; font-size: 0.85em; }
    .cycle-list { padding-left: 20px; }
    .cycle-item { color: #fb923c; margin-bottom: 6px; font-family: monospace; font-size: 0.9em; }
    .ap-item { margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #334155; }
    .ap-item:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
    .ap-header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; flex-wrap: wrap; }
    .ap-details { color: #cbd5e1; font-size: 0.9em; }
    .ap-explain { color: #94a3b8; font-size: 0.85em; margin-top: 4px; font-style: italic; }
    .guidance-list { padding-left: 20px; }
    .guidance-list li { margin-bottom: 12px; color: #cbd5e1; }
    .guidance-list strong { color: #f1f5f9; }
    p { margin-bottom: 8px; }
    nav { margin-bottom: 32px; }
    nav a {
      color: #60a5fa; text-decoration: none; margin-right: 16px;
      font-size: 0.9em;
    }
    nav a:hover { text-decoration: underline; }
"""


def generate_analysis_report(
    analysis_file: str,
    output_file: str = "outputs/report.html",
    open_browser: bool = True,
) -> str:
    """Generate a standalone HTML analysis report.

    Args:
        analysis_file: Path to architecture analysis JSON (from analyze-architecture)
        output_file: Output HTML file path
        open_browser: Whether to open the report in the default browser

    Returns:
        Absolute path to the generated HTML file
    """
    analysis_path = Path(analysis_file)
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {analysis_file}")

    with open(analysis_path, "r") as f:
        data = json.load(f)

    health = _health_rating(data)
    repo_name = _escape(data.get("repository", "unknown"))

    nav = """
    <nav>
      <a href="#summary">Summary</a>
      <a href="#health">Health</a>
      <a href="#fan-metrics">Fan-In/Out</a>
      <a href="#hubs">Hubs</a>
      <a href="#cycles">Cycles</a>
      <a href="#antipatterns">Anti-Patterns</a>
      <a href="#layers">Layers</a>
      <a href="#guidance">Guidance</a>
    </nav>"""

    body = (
        _build_summary_section(data, health)
        + _build_health_section(health)
        + _build_fanin_fanout_section()
        + _build_hub_modules_section(data)
        + _build_cycles_section(data)
        + _build_antipatterns_section(data)
        + _build_layers_section(data)
        + _build_guidance_section()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{repo_name} — Architecture Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>{repo_name} — Architecture Report</h1>
  <p class="subtitle">Generated by RLM-Codelens</p>
  {nav}
  {body}
</body>
</html>
"""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    abs_path = str(output_path.resolve())

    if open_browser:
        webbrowser.open(f"file://{abs_path}")

    return abs_path
