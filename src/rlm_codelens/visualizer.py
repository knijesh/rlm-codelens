"""
Generate interactive HTML visualization of codebase architecture.

This module creates a standalone HTML file with an interactive D3.js
visualization of the architecture analysis results, including a
per-module tracer panel for exploring dependency trees.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def _enrich_analysis_data(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich analysis data with pre-computed fields for the tracer panel.

    Adds per-node: depends_on, used_by, in_cycles, anti_patterns,
    upstream_tree (depth 3), downstream_tree (depth 3).
    """
    graph_data = analysis_data.get("graph_data", {})
    nodes = graph_data.get("nodes", [])
    links = graph_data.get("links", [])
    cycles = analysis_data.get("cycles", [])
    anti_patterns = analysis_data.get("anti_patterns", [])

    # Build adjacency maps
    depends_on_map: Dict[str, List[str]] = defaultdict(list)  # source -> targets
    used_by_map: Dict[str, List[str]] = defaultdict(list)  # target -> sources

    for link in links:
        src = link.get("source", "")
        tgt = link.get("target", "")
        if isinstance(src, dict):
            src = src.get("id", "")
        if isinstance(tgt, dict):
            tgt = tgt.get("id", "")
        if src and tgt:
            depends_on_map[src].append(tgt)
            used_by_map[tgt].append(src)

    # Build cycle membership
    cycle_membership: Dict[str, List[int]] = defaultdict(list)
    for i, cycle in enumerate(cycles):
        for module in cycle:
            cycle_membership[module].append(i)

    # Build anti-pattern membership
    ap_membership: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ap in anti_patterns:
        module = ap.get("module", "")
        if module:
            ap_membership[module].append(ap)

    def _build_tree(
        start: str, adjacency: Dict[str, List[str]], max_depth: int = 3
    ) -> Dict[str, Any]:
        """Build a transitive dependency tree up to max_depth."""
        visited: Set[str] = set()

        def _recurse(node: str, depth: int) -> List[Dict[str, Any]]:
            if depth >= max_depth or node in visited:
                return []
            visited.add(node)
            children = []
            for child in adjacency.get(node, []):
                if child not in visited:
                    children.append(
                        {
                            "id": child,
                            "children": _recurse(child, depth + 1),
                        }
                    )
            return children

        return {"id": start, "children": _recurse(start, 0)}

    # Compute total LOC
    total_loc = sum(n.get("loc", 0) for n in nodes)

    # Enrich each node
    for node in nodes:
        nid = node["id"]
        node["depends_on"] = depends_on_map.get(nid, [])
        node["used_by"] = used_by_map.get(nid, [])
        node["in_cycles"] = cycle_membership.get(nid, [])
        node["node_anti_patterns"] = ap_membership.get(nid, [])
        node["upstream_tree"] = _build_tree(nid, depends_on_map)
        node["downstream_tree"] = _build_tree(nid, used_by_map)
        # Ensure fan_in/fan_out are present
        if "fan_in" not in node:
            node["fan_in"] = len(used_by_map.get(nid, []))
        if "fan_out" not in node:
            node["fan_out"] = len(depends_on_map.get(nid, []))

    analysis_data["total_loc"] = total_loc

    return analysis_data


def generate_architecture_visualization(
    analysis_file: str,
    output_file: Optional[str] = None,
    open_browser: bool = True,
) -> str:
    """
    Generate interactive HTML visualization of codebase architecture.

    Args:
        analysis_file: Path to architecture analysis JSON file
        output_file: Path for output HTML file
        open_browser: Whether to open the visualization in browser

    Returns:
        Path to generated HTML file
    """
    if output_file is None:
        output_file = "outputs/architecture_visualization.html"

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get template path
    template_path = (
        Path(__file__).parent.parent.parent
        / "visualization"
        / "architecture_graph_viewer.html"
    )

    if not template_path.exists():
        raise FileNotFoundError(
            f"Architecture visualization template not found: {template_path}"
        )

    # Read template
    with open(template_path, "r") as f:
        html_content = f.read()

    # Read analysis data
    with open(analysis_file, "r") as f:
        analysis_data = json.load(f)

    # Enrich data with pre-computed fields for the tracer panel
    analysis_data = _enrich_analysis_data(analysis_data)

    # Inject data before </head>
    data_script = f"""
    <script>
        const EMBEDDED_DATA = {json.dumps(analysis_data, indent=2)};
    </script>
    """
    html_content = html_content.replace("</head>", f"{data_script}</head>")

    # Write output file
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"✅ Architecture visualization generated: {output_path}")

    # Open in browser if requested
    if open_browser:
        import webbrowser

        webbrowser.open(f"file://{output_path.absolute()}")
        print("🌐 Opening in browser...")

    return str(output_path)
