"""HTML analysis report generator for rlm-codelens.

Produces a standalone, self-contained HTML report that explains
the architecture analysis findings with embedded CSS (dark theme).
"""

import json
import webbrowser
from pathlib import Path
from typing import Any, Dict


def _has_data(val: Any) -> bool:
    """Return True if val is present and non-empty (not None, {}, or [])."""
    if val is None:
        return False
    if isinstance(val, (dict, list)) and len(val) == 0:
        return False
    return True


def _health_rating(data: Dict[str, Any]) -> tuple:
    """Compute a health rating (label, color, explanation) from analysis data."""
    penalty = 0

    # Circular dependencies — capped so large repos aren't auto-critical
    cycles = data.get("cycles", [])
    penalty += min(len(cycles) * 15, 40)

    # Anti-patterns by severity — cap each severity tier so heuristic-based
    # detections don't overwhelm the score in large repos.
    # "info" severity carries zero penalty (purely informational).
    high_penalty = 0
    medium_penalty = 0
    low_penalty = 0
    for ap in data.get("anti_patterns", []):
        sev = ap.get("severity", "low")
        if sev == "high":
            high_penalty += 10
        elif sev == "medium":
            medium_penalty += 5
        elif sev == "info":
            pass  # no penalty for informational items
        else:
            low_penalty += 2
    penalty += min(high_penalty, 30)  # cap god modules at 30
    penalty += min(medium_penalty, 20)  # cap layer violations at 20
    penalty += min(low_penalty, 10)  # cap orphan-like penalties at 10

    # Normalize: 0 penalty = 100 score, cap at 0
    score = max(0, 100 - penalty)

    if score >= 80:
        return (
            "Healthy",
            "#4ade80",
            score,
            "The codebase has a clean architecture with few issues.",
        )
    elif score >= 60:
        return (
            "Fair",
            "#facc15",
            score,
            "Some architectural issues exist that should be addressed.",
        )
    elif score >= 40:
        return (
            "Needs Attention",
            "#fb923c",
            score,
            "Multiple architectural issues detected. Prioritize refactoring.",
        )
    else:
        return (
            "Critical",
            "#ef4444",
            score,
            "Significant structural problems. Immediate refactoring recommended.",
        )


def _escape(text: str) -> str:
    """HTML-escape a string."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _md_to_html(text: str) -> str:
    """Convert basic Markdown to HTML for LLM-generated content."""
    import re

    # Escape HTML entities first
    text = _escape(text)

    # Code blocks (```...```) → <pre><code>
    text = re.sub(
        r"```\w*\n(.*?)```",
        r"<pre><code>\1</code></pre>",
        text,
        flags=re.DOTALL,
    )

    # Inline code `...` → <code>
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

    # Headers ### → <strong> (keep it subtle inside cards)
    text = re.sub(r"^#{1,4}\s+(.+)$", r"<strong>\1</strong>", text, flags=re.MULTILINE)

    # Bold **...**
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    # Convert markdown list items (- or numbered) to HTML list items
    # First handle unordered lists
    text = re.sub(r"^\s*[-*]\s+(.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    # Wrap consecutive <li> blocks (from unordered) in <ul>
    text = re.sub(
        r"((?:<li>.*?</li>\n?)+)",
        r"<ul>\1</ul>",
        text,
    )
    # Avoid nested <ul> inside existing <ol>/<ul> — keep it simple

    # Paragraphs: double newlines → <br><br>
    text = re.sub(r"\n{2,}", "<br><br>", text)
    # Single newlines within text → <br>
    text = re.sub(r"\n", "<br>", text)

    return text.strip()


def _module_short(path: str) -> str:
    """Shorten a module path for display."""
    return Path(path).stem


def _build_summary_section(data: Dict[str, Any], health: tuple) -> str:
    total_loc = 0
    lang_counts: Dict[str, int] = {}
    for node in data.get("graph_data", {}).get("nodes", []):
        total_loc += node.get("loc", 0)
        lang = node.get("language", "unknown")
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    label, color, score, _ = health
    badge = f'<span class="badge" style="background:{color};color:#000">{label} ({score}/100)</span>'

    # Language badges
    lang_html = ""
    if lang_counts:
        sorted_langs = sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
        lang_badges = " ".join(
            f'<span class="badge" style="background:#334155;color:#e2e8f0">'
            f"{_escape(lang.capitalize())} ({count:,})</span>"
            for lang, count in sorted_langs
        )
        lang_html = f"""
        <div class="card">
          <div class="stat-label">Languages</div>
          <div style="margin-top:4px">{lang_badges}</div>
        </div>"""

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
          <div class="stat-value">{data.get("total_modules", 0):,}</div>
        </div>
        <div class="card">
          <div class="stat-label">Import Edges</div>
          <div class="stat-value">{data.get("total_edges", 0):,}</div>
        </div>
        <div class="card">
          <div class="stat-label">Total LOC</div>
          <div class="stat-value">{total_loc:,}</div>
        </div>
        <div class="card">
          <div class="stat-label">Circular Dependencies</div>
          <div class="stat-value">{len(data.get("cycles", []))}</div>
        </div>
        {lang_html}
      </div>
    </section>"""


def _build_health_section(health: tuple, data: Dict[str, Any] | None = None) -> str:
    label, color, score, explanation = health
    bar_width = score

    resolution_note = ""
    if data:
        total_modules = data.get("total_modules", 0)
        if total_modules > 0:
            nodes = data.get("graph_data", {}).get("nodes", [])
            connected = sum(
                1 for n in nodes if n.get("fan_in", 0) > 0 or n.get("fan_out", 0) > 0
            )
            resolution_rate = connected / total_modules
            if resolution_rate < 0.5:
                pct = resolution_rate * 100
                resolution_note = (
                    f'<p style="color:#60a5fa;font-size:0.85em;margin-top:8px">'
                    f"Import resolution coverage: {pct:.0f}%. Health score may not "
                    f"fully reflect the codebase. Install language grammars for "
                    f"more accurate analysis.</p>"
                )

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
          Scoring: starts at 100. Penalties capped per category &mdash;
          cycles (max &minus;40), high anti-patterns (max &minus;30),
          medium (max &minus;20), low (max &minus;10).
        </p>
        {resolution_note}
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
        "import_resolution_limited": "Import resolution could not resolve all module dependencies for this language.",
    }

    severity_colors = {
        "high": "#ef4444",
        "medium": "#fb923c",
        "low": "#facc15",
        "info": "#60a5fa",
    }

    # Count by severity
    high_count = sum(1 for p in patterns if p.get("severity") == "high")
    med_count = sum(1 for p in patterns if p.get("severity") == "medium")
    low_count = sum(1 for p in patterns if p.get("severity") == "low")
    info_count = sum(1 for p in patterns if p.get("severity") == "info")
    actionable_count = high_count + med_count + low_count

    # Filter pills
    filter_pills = f"""
      <div class="filter-bar">
        <span class="filter-pill active" data-severity="all">All ({actionable_count})</span>
        <span class="filter-pill" data-severity="high" style="border-color:#ef4444">High ({high_count})</span>
        <span class="filter-pill" data-severity="medium" style="border-color:#fb923c">Medium ({med_count})</span>
        <span class="filter-pill" data-severity="low" style="border-color:#facc15">Low ({low_count})</span>
      </div>"""

    # Separate info items
    info_patterns = [p for p in patterns if p.get("severity") == "info"]
    actionable_patterns = [p for p in patterns if p.get("severity") != "info"]

    # Group by type
    groups: Dict[str, list] = {}
    for ap in actionable_patterns:
        ap_type = ap.get("type", "unknown")
        groups.setdefault(ap_type, []).append(ap)

    # Sort groups: highest max severity first, then by count descending
    severity_order = {"high": 0, "medium": 1, "low": 2}

    def group_sort_key(item: tuple) -> tuple:
        _, items = item
        max_sev = min(severity_order.get(p.get("severity", "low"), 2) for p in items)
        return (max_sev, -len(items))

    sorted_groups = sorted(groups.items(), key=group_sort_key)

    # Build accordion groups
    cap_per_group = 10
    groups_html = ""
    for ap_type, items in sorted_groups:
        max_sev = "low"
        for p in items:
            s = p.get("severity", "low")
            if severity_order.get(s, 2) < severity_order.get(max_sev, 2):
                max_sev = s
        max_sev_color = severity_colors.get(max_sev, "#94a3b8")
        start_open = max_sev == "high"
        open_class = " open" if start_open else ""

        explanation = type_explanations.get(ap_type, "")
        explain_html = (
            f'<p class="ap-explain" style="margin-bottom:8px">{explanation}</p>'
            if explanation
            else ""
        )

        items_html = ""
        for i, ap in enumerate(items):
            sev = ap.get("severity", "low")
            color = severity_colors.get(sev, "#94a3b8")
            details = _escape(ap.get("details", ""))
            hidden_class = " hidden-item" if i >= cap_per_group else ""
            hidden_style = ' style="display:none"' if i >= cap_per_group else ""
            items_html += f"""
            <div class="ap-item{hidden_class}" data-severity="{sev}"{hidden_style}>
              <div class="ap-header">
                <span class="badge" style="background:{color};color:#000">{sev.upper()}</span>
                <span style="color:#94a3b8">{_escape(ap.get("module", ""))}</span>
              </div>
              <p class="ap-details">{details}</p>
            </div>"""

        show_more_html = ""
        if len(items) > cap_per_group:
            show_more_html = f'<button class="show-more-btn" style="background:none;border:1px solid #475569;color:#60a5fa;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.85em;margin-top:8px">Show all {len(items)}</button>'

        groups_html += f"""
        <div class="ap-group">
          <div class="ap-group-header{open_class}">
            <span class="chevron">&#9654;</span>
            <strong>{_escape(ap_type)}</strong>
            <span class="badge" style="background:{max_sev_color};color:#000;font-size:0.75em">{max_sev.upper()}</span>
            <span style="color:#94a3b8;font-size:0.85em">({len(items)})</span>
          </div>
          <div class="ap-group-body{open_class}">
            {explain_html}
            <div class="ap-group-items">
              {items_html}
            </div>
            {show_more_html}
          </div>
        </div>"""

    # Info items rendered as a separate callout box
    info_html = ""
    if info_patterns:
        info_items = ""
        for ap in info_patterns:
            details = _escape(ap.get("details", ""))
            ap_type = ap.get("type", "unknown")
            explanation = type_explanations.get(ap_type, "")
            info_items += f"""
            <div style="margin-bottom:8px">
              <span class="badge" style="background:#60a5fa;color:#000">INFO</span>
              <strong style="margin-left:6px">{_escape(ap_type)}</strong>
              <p class="ap-details" style="margin-top:4px">{details}</p>
              {"<p class='ap-explain'>" + explanation + "</p>" if explanation else ""}
            </div>"""
        info_html = f"""
        <div class="card" style="margin-top:16px;border-color:#60a5fa;background:#1a2744">
          <p style="color:#60a5fa;font-weight:600;margin-bottom:8px">Informational Notes</p>
          {info_items}
        </div>"""

    summary_parts = []
    if high_count:
        summary_parts.append(f'<span style="color:#ef4444">{high_count} high</span>')
    if med_count:
        summary_parts.append(f'<span style="color:#fb923c">{med_count} medium</span>')
    if low_count:
        summary_parts.append(f'<span style="color:#facc15">{low_count} low</span>')
    if info_count:
        summary_parts.append(f'<span style="color:#60a5fa">{info_count} info</span>')
    summary_str = ", ".join(summary_parts)

    return f"""
    <section id="antipatterns">
      <h2>Anti-Patterns</h2>
      <div class="card">
        <p style="color:#94a3b8;margin-bottom:12px">
          {actionable_count} anti-pattern{"s" if actionable_count != 1 else ""} found{" (" + summary_str + ")" if summary_str else ""}.
        </p>
        {filter_pills}
        {groups_html}
      </div>
      {info_html}
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
        "data": "#42a5f5",
        "business": "#66bb6a",
        "api": "#ffa726",
        "util": "#bdbdbd",
        "test": "#ab47bc",
        "config": "#ffee58",
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


def _deep_was_run(data: Dict[str, Any]) -> bool:
    """Return True if the analysis JSON contains any RLM deep analysis fields."""
    return any(
        _has_data(data.get(k))
        for k in (
            "pattern_analysis",
            "semantic_clusters",
            "hidden_dependencies",
            "refactoring_suggestions",
        )
    )


def _build_executive_summary_section(data: Dict[str, Any], health: tuple) -> str:
    """Build an executive summary paragraph synthesizing key findings."""
    label, color, score, _ = health

    total_modules = data.get("total_modules", 0)
    total_edges = data.get("total_edges", 0)
    total_loc = sum(
        n.get("loc", 0) for n in data.get("graph_data", {}).get("nodes", [])
    )
    num_cycles = len(data.get("cycles", []))
    num_antipatterns = len(data.get("anti_patterns", []))

    sentences = [
        f"This repository contains <strong>{total_modules}</strong> modules "
        f"({total_loc:,} lines of code) with <strong>{total_edges}</strong> import edges."
    ]

    sentences.append(
        f'The overall health score is <span style="color:{color}">'
        f"<strong>{score}/100 ({label})</strong></span>."
    )

    if num_cycles:
        sentences.append(
            f'<span style="color:#fb923c"><strong>{num_cycles}</strong> circular '
            f"dependency chain{'s' if num_cycles != 1 else ''}</span> detected."
        )
    else:
        sentences.append("No circular dependencies were found.")

    if num_antipatterns:
        sentences.append(
            f"<strong>{num_antipatterns}</strong> anti-pattern"
            f"{'s' if num_antipatterns != 1 else ''} identified."
        )

    # RLM deep analysis extras
    pattern_analysis = data.get("pattern_analysis")
    refactoring = data.get("refactoring_suggestions")

    if _has_data(pattern_analysis):
        assert pattern_analysis is not None
        pname = _escape(pattern_analysis.get("detected_pattern", "Unknown"))
        conf = pattern_analysis.get("confidence", 0)
        sentences.append(
            f"RLM deep analysis detected a <strong>{pname}</strong> architectural "
            f"pattern with {conf:.0%} confidence."
        )

    if _has_data(refactoring):
        assert refactoring is not None
        sentences.append(
            f"<strong>{len(refactoring)}</strong> refactoring suggestion"
            f"{'s' if len(refactoring) != 1 else ''} available."
        )

    # Top concern
    if num_cycles:
        sentences.append(
            "<em>Key concern: resolve circular dependencies to improve modularity.</em>"
        )
    elif num_antipatterns:
        sentences.append(
            "<em>Key concern: address detected anti-patterns to strengthen architecture.</em>"
        )
    else:
        sentences.append("<em>The codebase architecture is in good shape.</em>")

    paragraph = " ".join(sentences)

    return f"""
    <section id="executive-summary">
      <h2>Executive Summary</h2>
      <div class="card executive-summary">
        <p>{paragraph}</p>
      </div>
    </section>"""


def _build_pattern_analysis_section(data: Dict[str, Any]) -> str:
    """Build the architectural pattern analysis section from RLM deep data."""
    pattern_analysis = data.get("pattern_analysis")

    if not _has_data(pattern_analysis):
        msg = (
            "No architectural pattern detected in this codebase."
            if _deep_was_run(data)
            else "No deep pattern analysis available. Run with <code>--deep</code> to enable."
        )
        return f"""
    <section id="pattern">
      <h2>Architectural Pattern</h2>
      <div class="card">
        <p style="color:#94a3b8">{msg}</p>
      </div>
    </section>"""

    assert pattern_analysis is not None
    pname = _escape(pattern_analysis.get("detected_pattern", "Unknown"))
    confidence = pattern_analysis.get("confidence", 0)
    conf_pct = confidence * 100
    anti_patterns = pattern_analysis.get("anti_patterns", [])
    reasoning = _escape(pattern_analysis.get("reasoning", ""))

    # Confidence bar color
    if conf_pct >= 70:
        bar_color = "#4ade80"
    elif conf_pct >= 40:
        bar_color = "#facc15"
    else:
        bar_color = "#fb923c"

    ap_list = ""
    if anti_patterns:
        items = "".join(f"<li>{_escape(str(ap))}</li>" for ap in anti_patterns)
        ap_list = f"""
        <div style="margin-top:12px">
          <strong style="color:#f1f5f9">Anti-Patterns Detected:</strong>
          <ul style="padding-left:20px;margin-top:6px;color:#cbd5e1">{items}</ul>
        </div>"""

    reasoning_html = ""
    if reasoning:
        reasoning_html = f"""
        <div style="margin-top:12px">
          <strong style="color:#f1f5f9">Reasoning:</strong>
          <p style="color:#cbd5e1;margin-top:4px">{reasoning}</p>
        </div>"""

    return f"""
    <section id="pattern">
      <h2>Architectural Pattern</h2>
      <div class="card">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
          <span class="insight-badge">{pname}</span>
          <span style="color:#94a3b8">Confidence: {conf_pct:.0f}%</span>
        </div>
        <div class="bar-bg">
          <div class="bar-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
        </div>
        {ap_list}
        {reasoning_html}
      </div>
    </section>"""


def _build_rlm_insights_section(data: Dict[str, Any]) -> str:
    """Build the RLM insights section with semantic classifications and hidden dependencies."""
    semantic_clusters = data.get("semantic_clusters")
    hidden_deps = data.get("hidden_dependencies")

    if not semantic_clusters and not hidden_deps:
        msg = (
            "Deep analysis found no semantic classifications or hidden dependencies for this codebase."
            if _deep_was_run(data)
            else "No RLM insights available. Run with <code>--deep</code> to enable."
        )
        return f"""
    <section id="rlm-insights">
      <h2>RLM Insights</h2>
      <div class="card">
        <p style="color:#94a3b8">{msg}</p>
      </div>
    </section>"""

    parts = ""

    # Semantic Classifications sub-card
    if semantic_clusters:
        static_layers = data.get("layers", {})
        rows = ""
        for module_path, rlm_layer in sorted(semantic_clusters.items()):
            mod_name = _escape(_module_short(module_path))
            rlm_layer_esc = _escape(str(rlm_layer))
            static_layer = static_layers.get(module_path, "")
            diff = ""
            if static_layer and static_layer != rlm_layer:
                diff = f' <span style="color:#fb923c" title="Static analysis assigned: {_escape(static_layer)}">(static: {_escape(static_layer)})</span>'
            rows += f"<tr><td>{mod_name}</td><td>{rlm_layer_esc}{diff}</td></tr>"

        parts += f"""
        <div class="card" style="margin-bottom:16px">
          <h3 style="color:#f1f5f9;font-size:1em;margin-bottom:8px">Semantic Classifications</h3>
          <p style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">
            RLM-assigned layer for each module based on semantic analysis of code content.
          </p>
          <table>
            <thead><tr><th>Module</th><th>RLM Layer</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # Hidden Dependencies sub-card
    if hidden_deps:
        rows = ""
        for dep in hidden_deps:
            src = _escape(str(dep.get("source", "")))
            tgt = _escape(str(dep.get("target", "")))
            dep_type = _escape(str(dep.get("type", "")))
            evidence = _escape(str(dep.get("evidence", "")))
            rows += f"<tr><td>{src}</td><td>{tgt}</td><td>{dep_type}</td><td>{evidence}</td></tr>"

        parts += f"""
        <div class="card">
          <h3 style="color:#f1f5f9;font-size:1em;margin-bottom:8px">Hidden Dependencies</h3>
          <p style="color:#94a3b8;font-size:0.85em;margin-bottom:8px">
            Dependencies detected through semantic analysis that are not visible in import statements.
          </p>
          <table>
            <thead><tr><th>Source</th><th>Target</th><th>Type</th><th>Evidence</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return f"""
    <section id="rlm-insights">
      <h2>RLM Insights</h2>
      {parts}
    </section>"""


def _build_refactoring_section(data: Dict[str, Any]) -> str:
    """Build the refactoring suggestions section from RLM deep data."""
    suggestions = data.get("refactoring_suggestions")

    if not suggestions:
        msg = (
            "Deep analysis found no refactoring suggestions for this codebase."
            if _deep_was_run(data)
            else "No refactoring suggestions available. Run with <code>--deep</code> to enable."
        )
        return f"""
    <section id="refactoring">
      <h2>Refactoring Recommendations</h2>
      <div class="card">
        <p style="color:#94a3b8">{msg}</p>
      </div>
    </section>"""

    items = ""
    for i, suggestion in enumerate(suggestions, 1):
        items += f'<div class="refactoring-item"><div class="refactoring-number">{i}</div><div class="refactoring-content">{_md_to_html(str(suggestion))}</div></div>\n'

    return f"""
    <section id="refactoring">
      <h2>Refactoring Recommendations</h2>
      <div class="card">
        <p style="color:#94a3b8;margin-bottom:12px">
          {len(suggestions)} suggestion{"s" if len(suggestions) != 1 else ""} from RLM deep analysis:
        </p>
        {items}
      </div>
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


def _build_coverage_banner(data: Dict[str, Any]) -> str:
    """Build a coverage warning banner if import resolution is below 50%."""
    total_modules = data.get("total_modules", 0)
    if total_modules == 0:
        return ""
    nodes = data.get("graph_data", {}).get("nodes", [])
    connected = sum(
        1 for n in nodes if n.get("fan_in", 0) > 0 or n.get("fan_out", 0) > 0
    )
    ratio = connected / total_modules
    if ratio >= 0.5:
        return ""
    pct = ratio * 100
    return f"""
    <div class="coverage-banner">
      <span style="font-size:1.3em">&#9432;</span>
      <div>
        <strong style="color:#60a5fa">Import Resolution: {pct:.0f}%</strong>
        <p style="color:#94a3b8;margin:4px 0 0 0;font-size:0.85em">
          Only {pct:.0f}% of modules have resolved import connections.
          Health scores and anti-pattern counts may be inflated.
        </p>
      </div>
    </div>"""


def _get_antipattern_count(data: Dict[str, Any]) -> int:
    """Return total count of actionable anti-patterns (excluding info)."""
    return sum(1 for p in data.get("anti_patterns", []) if p.get("severity") != "info")


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
    .executive-summary p { font-size: 1em; line-height: 1.7; color: #cbd5e1; }
    .insight-badge {
      display: inline-block; padding: 4px 14px; border-radius: 12px;
      background: #3b82f6; color: #fff; font-weight: 600; font-size: 0.95em;
    }
    .refactoring-item {
      display: flex; gap: 14px; margin-bottom: 20px; padding-bottom: 20px;
      border-bottom: 1px solid #334155;
    }
    .refactoring-item:last-child { margin-bottom: 0; padding-bottom: 0; border-bottom: none; }
    .refactoring-number {
      flex-shrink: 0; width: 28px; height: 28px; border-radius: 50%;
      background: #334155; color: #60a5fa; font-weight: 700; font-size: 0.85em;
      display: flex; align-items: center; justify-content: center; margin-top: 2px;
    }
    .refactoring-content { flex: 1; color: #cbd5e1; font-size: 0.9em; line-height: 1.6; }
    .refactoring-content strong { color: #f1f5f9; }
    .refactoring-content code {
      background: #334155; padding: 1px 5px; border-radius: 3px;
      font-size: 0.9em; color: #e2e8f0;
    }
    .refactoring-content pre {
      background: #0f172a; border: 1px solid #334155; border-radius: 6px;
      padding: 12px; margin: 8px 0; overflow-x: auto;
    }
    .refactoring-content pre code { background: none; padding: 0; }
    .refactoring-content ul { padding-left: 18px; margin: 6px 0; }
    .refactoring-content li { margin-bottom: 4px; }
    p { margin-bottom: 8px; }

    /* Tabs */
    .tab-bar { display:flex; gap:0; border-bottom:2px solid #334155; margin-bottom:24px; overflow-x:auto; }
    .tab-btn { padding:10px 20px; background:none; border:none; border-bottom:2px solid transparent;
      color:#94a3b8; font-size:0.9em; font-weight:600; cursor:pointer; white-space:nowrap; margin-bottom:-2px; }
    .tab-btn:hover { color:#e2e8f0; }
    .tab-btn.active { color:#60a5fa; border-bottom-color:#60a5fa; }
    .tab-panel { display:none; }
    .tab-panel.active { display:block; }

    /* Severity filter pills */
    .filter-bar { display:flex; gap:8px; margin-bottom:16px; flex-wrap:wrap; }
    .filter-pill { padding:4px 12px; border-radius:16px; font-size:0.8em; font-weight:600;
      cursor:pointer; border:1px solid #475569; background:#1e293b; color:#94a3b8; }
    .filter-pill.active { background:#334155; color:#e2e8f0; }

    /* Accordion groups */
    .ap-group { margin-bottom:4px; }
    .ap-group-header { display:flex; align-items:center; gap:8px; cursor:pointer;
      padding:10px 0; border-bottom:1px solid #334155; user-select:none; }
    .ap-group-header .chevron { transition:transform 0.2s; color:#94a3b8; }
    .ap-group-header.open .chevron { transform:rotate(90deg); }
    .ap-group-body { display:none; padding-top:8px; }
    .ap-group-body.open { display:block; }

    /* Coverage warning banner */
    .coverage-banner { background:#1e293b; border:1px solid #60a5fa; border-radius:8px;
      padding:12px 16px; margin-bottom:16px; display:flex; align-items:center; gap:12px; }

    /* Print: show all tabs */
    @media print { .tab-panel { display:block !important; } .tab-bar { display:none; } }
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

    # Build tab contents
    has_deep = _deep_was_run(data)
    issue_count = _get_antipattern_count(data) + len(data.get("cycles", []))

    # Tab 1: Overview
    tab_overview = (
        _build_coverage_banner(data)
        + _build_executive_summary_section(data, health)
        + _build_summary_section(data, health)
        + _build_health_section(health, data)
    )

    # Tab 2: Architecture
    tab_architecture = (
        _build_pattern_analysis_section(data)
        + _build_layers_section(data)
        + _build_hub_modules_section(data)
        + _build_fanin_fanout_section()
    )

    # Tab 3: Issues
    tab_issues = _build_antipatterns_section(data) + _build_cycles_section(data)

    # Tab 4: Deep Analysis (conditional)
    tab_deep = ""
    if has_deep:
        tab_deep = _build_rlm_insights_section(data) + _build_refactoring_section(data)

    # Tab 5: Guidance
    tab_guidance = _build_guidance_section()

    # Build tab bar
    issue_badge = (
        f' <span class="badge" style="background:#475569;color:#e2e8f0;font-size:0.75em;margin-left:4px">{issue_count}</span>'
        if issue_count
        else ""
    )
    tab_buttons = f"""
    <div class="tab-bar">
      <button class="tab-btn active" data-tab="tab-overview">Overview</button>
      <button class="tab-btn" data-tab="tab-architecture">Architecture</button>
      <button class="tab-btn" data-tab="tab-issues">Issues{issue_badge}</button>"""
    if has_deep:
        tab_buttons += """
      <button class="tab-btn" data-tab="tab-deep">Deep Analysis</button>"""
    tab_buttons += """
      <button class="tab-btn" data-tab="tab-guidance">Guidance</button>
    </div>"""

    # Build tab panels
    tab_panels = f"""
    <div id="tab-overview" class="tab-panel active">{tab_overview}</div>
    <div id="tab-architecture" class="tab-panel">{tab_architecture}</div>
    <div id="tab-issues" class="tab-panel">{tab_issues}</div>"""
    if has_deep:
        tab_panels += f"""
    <div id="tab-deep" class="tab-panel">{tab_deep}</div>"""
    tab_panels += f"""
    <div id="tab-guidance" class="tab-panel">{tab_guidance}</div>"""

    # JavaScript for interactivity
    js = """
<script>
// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  });
});
// Accordion toggle
document.querySelectorAll('.ap-group-header').forEach(h => {
  h.addEventListener('click', () => {
    h.classList.toggle('open');
    h.nextElementSibling.classList.toggle('open');
  });
});
// Severity filter
document.querySelectorAll('.filter-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    const isAll = pill.dataset.severity === 'all';
    if (isAll) {
      document.querySelectorAll('.filter-pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
    } else {
      document.querySelector('.filter-pill[data-severity="all"]').classList.remove('active');
      pill.classList.toggle('active');
      if (!document.querySelector('.filter-pill.active')) {
        document.querySelector('.filter-pill[data-severity="all"]').classList.add('active');
      }
    }
    const activePills = [...document.querySelectorAll('.filter-pill.active')].map(p => p.dataset.severity);
    const showAll = activePills.includes('all');
    document.querySelectorAll('.ap-item[data-severity]').forEach(item => {
      const visible = showAll || activePills.includes(item.dataset.severity);
      if (visible) {
        if (!item.classList.contains('hidden-item')) item.style.display = '';
      } else {
        item.style.display = 'none';
      }
    });
    document.querySelectorAll('.ap-group').forEach(group => {
      const items = group.querySelectorAll('.ap-item[data-severity]');
      const anyVisible = [...items].some(i => i.style.display !== 'none');
      group.style.display = anyVisible ? '' : 'none';
    });
  });
});
// Show more
document.querySelectorAll('.show-more-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.previousElementSibling.querySelectorAll('.ap-item.hidden-item').forEach(i => i.style.display = '');
    btn.style.display = 'none';
  });
});
</script>"""

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
  {tab_buttons}
  {tab_panels}
  {js}
</body>
</html>
"""

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    abs_path = str(output_path.resolve())

    if open_browser:
        webbrowser.open(f"file://{abs_path}")

    return abs_path
