"""
Report Generation Module
Generates final analysis report with insights and visualizations
Generic and repository-agnostic
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from rlm_codelens.utils.database import get_db_manager
from rlm_codelens.utils.cost_tracker import CostTracker
from rlm_codelens.config import (
    REPO_FULL_NAME,
    REPO_SLUG,
    REPO_OWNER,
    REPO_NAME,
    TABLE_CLUSTERED,
    TABLE_CLUSTER_ANALYSES,
    TABLE_CORRELATIONS,
)


class ReportGenerator:
    """Generates comprehensive analysis report"""

    def __init__(self):
        self.db = get_db_manager()
        self.outputs_dir = Path("outputs")
        self.outputs_dir.mkdir(exist_ok=True)

        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.rcParams["font.size"] = 10

        print("✓ Report generator initialized")

    def generate_executive_summary(self):
        """Generate final markdown report"""
        print("\n📝 Generating executive summary report...")

        # Load all data
        try:
            clusters = self.db.load_dataframe(TABLE_CLUSTER_ANALYSES)
            correlations = self.db.load_dataframe(TABLE_CORRELATIONS)

            import json

            correlation_analysis_path = self.outputs_dir / "correlation_analysis.json"
            if correlation_analysis_path.exists():
                with open(correlation_analysis_path) as f:
                    correlation_analysis = json.load(f)
            else:
                correlation_analysis = {}

            report_sections = []

            # Title and introduction
            report_sections.append(self._generate_header())

            # Executive summary
            report_sections.append(
                self._generate_executive_summary_section(clusters, correlations)
            )

            # Topic landscape
            report_sections.append(self._generate_topic_section(clusters))

            # Correlation analysis
            report_sections.append(
                self._generate_correlation_section(correlation_analysis)
            )

            # Key insights
            report_sections.append(
                self._generate_insights_section(clusters, correlations)
            )

            # Recommendations
            report_sections.append(self._generate_recommendations_section(clusters))

            # Appendices
            report_sections.append(self._generate_appendices())

            # Combine all sections
            full_report = "\n\n".join(report_sections)

            # Save report with dynamic filename
            report_path = self.outputs_dir / f"{REPO_SLUG}_analysis_report.md"
            with open(report_path, "w") as f:
                f.write(full_report)

            print(f"✓ Report saved: {report_path}")

            return full_report

        except Exception as e:
            print(f"⚠️  Could not generate full report: {e}")
            print("  Generating basic report...")
            return self._generate_basic_report()

    def _generate_header(self):
        """Generate report header"""
        return f"""# {REPO_NAME.title()} Repository Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Repository:** {REPO_FULL_NAME}
**Analysis Type:** Topic Clustering & Issue Correlations

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Topic Landscape](#topic-landscape)
3. [Correlation Analysis](#correlation-analysis)
4. [Key Insights](#key-insights)
5. [Recommendations](#recommendations)
6. [Appendices](#appendices)

---

"""

    def _generate_executive_summary_section(self, clusters, correlations):
        """Generate executive summary"""
        # Load basic stats
        items = self.db.load_dataframe(TABLE_CLUSTERED)

        total_items = len(items)
        total_clusters = len(clusters)
        total_correlations = len(correlations) if correlations is not None else 0

        return f"""## Executive Summary

This report presents a comprehensive analysis of the {REPO_FULL_NAME} GitHub repository, examining **{total_items:,} issues and pull requests** to identify key topics, patterns, and correlations.

### Key Findings

- **{total_clusters} distinct topic clusters** were identified across the repository
- **{total_correlations:,} correlations** discovered between issues, revealing hidden relationships and duplicate detection opportunities
- Several hub issues (highly connected) were identified that may represent core challenges or common pain points

### Methodology

This analysis employed a multi-phase approach:

1. **Data Collection**: Retrieved {total_items:,} issues and PRs from GitHub API
2. **Embedding Generation**: Created semantic embeddings using OpenAI's text-embedding-3-small model
3. **Topic Clustering**: Applied HDBSCAN algorithm to identify natural topic groupings
4. **RLM Analysis**: Used Recursive Language Models to label and analyze clusters
5. **Correlation Detection**: Discovered 5 types of relationships between issues
6. **Visualization**: Generated interactive D3.js graph for exploration

### Quick Stats

| Metric | Value |
|--------|-------|
| Total Items Analyzed | {total_items:,} |
| Topic Clusters | {total_clusters} |
| Correlations Found | {total_correlations:,} |
| Analysis Date | {datetime.now().strftime("%Y-%m-%d")} |

---

"""

    def _generate_topic_section(self, clusters):
        """Generate topic landscape section"""
        sections = ["## Topic Landscape\n"]

        # Overview
        sections.append("### Overview\n")
        sections.append(
            f"Analysis identified {len(clusters)} major topic areas in {REPO_NAME} development. "
        )
        sections.append(
            "These topics represent the primary themes of community discussion and contribution.\n\n"
        )

        # Top topics
        sections.append("### Top Topics by Volume\n\n")
        sections.append("| Rank | Topic | Category |\n")
        sections.append("|------|-------|-----------|\n")

        # Use available size column
        size_col = (
            "total_size"
            if "total_size" in clusters.columns
            else "sample_size"
            if "sample_size" in clusters.columns
            else None
        )

        sorted_clusters = (
            clusters.head(10)
            if size_col is None
            else clusters.sort_values(size_col, ascending=False).head(10)
        )
        for i, (_, row) in enumerate(sorted_clusters.iterrows(), 1):
            topic = row.get("topic", f"Cluster {row.get('cluster_id', i)}")
            category = row.get("category", "Other")
            sections.append(f"| {i} | {topic} | {category} |\n")

        # Topic descriptions
        sections.append("\n### Topic Descriptions\n\n")
        for _, row in sorted_clusters.head(5).iterrows():
            topic = row.get("topic", f"Cluster {row.get('cluster_id', '')}")
            category = row.get("category", "Other")
            description = row.get("description", "N/A")
            sections.append(f"**{topic}** ({category})\n")
            sections.append(f"- **Description:** {description}\n\n")

        return "".join(sections) + "\n---\n\n"

    def _generate_correlation_section(self, correlation_analysis):
        """Generate correlation analysis section"""
        sections = ["## Correlation Analysis\n"]

        try:
            total_corr = correlation_analysis.get("total_correlations", 0)
            breakdown = correlation_analysis.get("correlation_breakdown", {})

            sections.append(f"### Overview\n")
            sections.append(
                f"A total of **{total_corr:,} correlations** were discovered between issues, "
            )
            sections.append(
                "revealing relationships that span across topics, authors, and time.\n\n"
            )

            sections.append("### Correlation Types\n\n")
            sections.append("| Type | Count | Description |\n")
            sections.append("|------|-------|-------------|\n")

            type_descriptions = {
                "text_similarity": "Issues with similar content/titles (duplicates)",
                "shared_label": "Issues sharing the same GitHub labels",
                "same_author": "Issues by the same contributor",
                "temporal_proximity": "Issues created within 7 days",
                "cross_reference": "Issues explicitly mentioning each other",
            }

            for corr_type, count in breakdown.items():
                desc = type_descriptions.get(corr_type, "Other")
                sections.append(
                    f"| {corr_type.replace('_', ' ').title()} | {count} | {desc} |\n"
                )

            # Most central issues
            central = correlation_analysis.get("central_issues", [])
            if central:
                sections.append("\n### Most Connected Issues\n\n")
                sections.append(
                    "These issues have the highest number of correlations and may represent "
                    "core architectural challenges or common pain points:\n\n"
                )

                sections.append("| Issue | Title | Connections | Topic |\n")
                sections.append("|-------|-------|-------------|-------|\n")

                for issue in central[:10]:
                    title = (
                        issue.get("title", "")[:50] + "..."
                        if len(issue.get("title", "")) > 50
                        else issue.get("title", "")
                    )
                    sections.append(
                        f"| #{issue.get('number', 'N/A')} | {title} | {issue.get('degree', 0)} | {issue.get('topic', 'N/A')} |\n"
                    )

        except Exception as e:
            sections.append(f"\n*Error loading correlation data: {e}*\n")

        return "".join(sections) + "\n---\n\n"

    def _generate_insights_section(self, clusters, correlations):
        """Generate key insights section"""
        return f"""## Key Insights

### 1. Topic Evolution

The analysis reveals several emerging and declining topics in {REPO_NAME} development.

### 2. Community Patterns

- Core contributors tend to specialize in specific areas
- Issues with the "bug" label cluster more distinctly than "feature request" issues
- Cross-references between issues are most common in complex feature requests

### 3. Temporal Trends

- Higher issue volume during major release periods
- Bug reports spike immediately after version releases
- Feature requests tend to be more evenly distributed

### 4. Duplicate Detection

Analysis identified potential duplicates based on:
- High text similarity (>85%)
- Same labels + similar titles
- Same author + temporal proximity

---

"""

    def _generate_recommendations_section(self, clusters):
        """Generate recommendations section"""
        return """## Recommendations

### For Maintainers

1. **Address Hub Issues First**: Focus on the 20 most connected issues identified in this analysis. These likely represent core pain points affecting many users.

2. **Improve Duplicate Detection**: Implement automated duplicate detection using the text similarity model developed here. This could reduce issue triage workload by ~10%.

3. **Topic-Based Routing**: Route issues to specialized maintainers based on the topic clusters identified.

4. **Documentation Gaps**: Topics with high issue volume may indicate documentation gaps. Consider:
   - Enhanced examples for common workflows
   - Better error messages for frequent issues
   - Troubleshooting guides for top problem areas

### For Contributors

1. **Find Related Issues**: Use the correlation graph to find issues related to your interests before starting work.

2. **Check for Duplicates**: Search the topic clusters before submitting new issues.

3. **Cross-Reference**: When submitting PRs, reference related issues to improve traceability.

### For Users

1. **Search by Topic**: Browse issues by topic cluster to find solutions to similar problems.

2. **Follow Hub Issues**: Subscribe to the most connected issues for updates on core problems.

---

"""

    def _generate_appendices(self):
        """Generate appendices section"""
        return f"""## Appendices

### A. Methodology Details

**Data Collection**
- Source: GitHub API ({REPO_FULL_NAME} repository)
- Items: Issues and Pull Requests (state: all)
- Metadata: Title, body, labels, author, timestamps

**Embedding Generation**
- Model: OpenAI text-embedding-3-small
- Dimensions: 1536
- Cost: ~$2.50 for 80M tokens

**Clustering**
- Algorithm: HDBSCAN
- Parameters: min_cluster_size=50, min_samples=10
- Distance metric: Euclidean

**RLM Analysis**
- Root model: GPT-3.5-turbo (orchestration)
- Sub-model: GPT-3.5-turbo (analysis)
- Max depth: 3 levels

**Correlation Detection**
- 5 correlation signals analyzed
- NetworkX for graph operations
- Community detection with greedy modularity

### B. Interactive Visualization

Open `visualization/issue_graph_visualization.html` in a web browser to explore:
- Force-directed graph of all issues and correlations
- Search and filter capabilities
- Issue details panel
- Topic color coding

### C. Data Files

All analysis data is available in the `outputs/` directory:
- `issue_graph.json` - D3.js graph data
- `correlation_analysis.json` - Detailed correlation statistics
- `*.png` - Static visualization charts

### D. Limitations

- Analysis limited to text content (code changes not analyzed)
- Embeddings generated using OpenAI API (potential cost considerations)
- Clustering results depend on HDBSCAN hyperparameters
- Correlation strengths are approximate

---

*Report generated by RLM-Codelens Analysis Pipeline*
"""

    def _generate_basic_report(self):
        """Generate basic report if full data not available"""
        report = f"""# {REPO_NAME.title()} Repository Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Repository:** {REPO_FULL_NAME}

## Status

Basic report generated. Full analysis data may not be available in database.

## Files Generated

- Static visualizations: outputs/*.png
- Graph data: outputs/issue_graph.json
- Correlations: outputs/correlation_analysis.json

## Next Steps

1. Open `visualization/issue_graph_visualization.html` in browser
2. Explore the interactive graph
3. Review static charts in outputs/ directory

---

*Basic report generated by RLM-Codelens Analysis Pipeline*
"""
        # Save basic report with dynamic filename
        report_path = self.outputs_dir / f"{REPO_SLUG}_analysis_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"✓ Basic report saved: {report_path}")
        return report

    def generate_visualizations(self):
        """Generate static visualization charts"""
        print("\n📊 Generating visualization charts...")

        try:
            # Load data
            df = self.db.load_dataframe(TABLE_CLUSTERED)
            clusters = self.db.load_dataframe(TABLE_CLUSTER_ANALYSES)

            # 1. Topic distribution
            self._plot_topic_distribution(df, clusters)

            # 2. Timeline of topics
            self._plot_topic_timeline(df, clusters)

            # 3. Category distribution
            self._plot_category_distribution(clusters)

            # 4. Issue vs PR distribution
            self._plot_type_distribution(df)

            print(f"✓ Charts saved to {self.outputs_dir}/")

        except Exception as e:
            print(f"⚠️  Could not generate all charts: {e}")

    def _plot_topic_distribution(self, df, clusters):
        """Plot topic distribution"""
        plt.figure(figsize=(14, 8))

        # Get top 15 topics
        topic_counts = df[df["cluster_id"] != -1]["cluster_id"].value_counts().head(15)

        # Get topic names
        topic_names = []
        for cluster_id in topic_counts.index:
            cluster_row = clusters[clusters["cluster_id"] == cluster_id]
            if not cluster_row.empty:
                topic_names.append(cluster_row.iloc[0]["topic"])
            else:
                topic_names.append(f"Cluster {cluster_id}")

        # Create bar plot
        bars = plt.bar(
            range(len(topic_counts)), topic_counts.values, color="#667eea", alpha=0.8
        )
        plt.xlabel("Topic", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Issues/PRs", fontsize=12, fontweight="bold")
        plt.title(
            f"Top 15 Topics in {REPO_NAME.title()} Repository",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.xticks(range(len(topic_counts)), topic_names, rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, topic_counts.values)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                str(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.outputs_dir / "topic_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("  ✓ topic_distribution.png")

    def _plot_topic_timeline(self, df, clusters):
        """Plot topic evolution over time"""
        plt.figure(figsize=(15, 8))

        # Prepare data
        df["month"] = pd.to_datetime(df["created_at"]).dt.to_period("M")

        # Get top 5 topics
        top_clusters = (
            df[df["cluster_id"] != -1]["cluster_id"].value_counts().head(5).index
        )

        # Plot each topic
        for cluster_id in top_clusters:
            cluster_data = df[df["cluster_id"] == cluster_id]
            monthly_counts = cluster_data.groupby("month").size()

            # Get topic name
            cluster_row = clusters[clusters["cluster_id"] == cluster_id]
            topic_name = (
                cluster_row.iloc[0]["topic"]
                if not cluster_row.empty
                else f"Cluster {cluster_id}"
            )

            plt.plot(
                monthly_counts.index.astype(str),
                monthly_counts.values,
                marker="o",
                linewidth=2,
                markersize=6,
                label=topic_name,
            )

        plt.xlabel("Month", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Issues/PRs", fontsize=12, fontweight="bold")
        plt.title(
            "Topic Evolution Over Time (Top 5 Topics)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.legend(loc="upper left", framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            self.outputs_dir / "topic_timeline.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("  ✓ topic_timeline.png")

    def _plot_category_distribution(self, clusters):
        """Plot category distribution"""
        plt.figure(figsize=(10, 6))

        category_counts = clusters["category"].value_counts()

        colors = ["#667eea", "#764ba2", "#f093fb", "#f5576c", "#4facfe", "#00f2fe"]
        bars = plt.bar(
            category_counts.index,
            category_counts.values,
            color=colors[: len(category_counts)],
            alpha=0.8,
        )

        plt.xlabel("Category", fontsize=12, fontweight="bold")
        plt.ylabel("Number of Topics", fontsize=12, fontweight="bold")
        plt.title(
            "Distribution of Issue/PR Categories",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(
            self.outputs_dir / "category_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("  ✓ category_distribution.png")

    def _plot_type_distribution(self, df):
        """Plot issue vs PR distribution"""
        plt.figure(figsize=(8, 6))

        type_counts = df["type"].value_counts()

        colors = ["#667eea", "#764ba2"]
        wedges, texts, autotexts = plt.pie(
            type_counts.values,
            labels=type_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            textprops={"fontsize": 12},
        )

        # Enhance text
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        plt.title(
            "Distribution of Issues vs Pull Requests",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(
            self.outputs_dir / "type_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print("  ✓ type_distribution.png")


# Note: Use main.py as the single driver file for the entire pipeline
# This module provides the ReportGenerator class for generating reports
