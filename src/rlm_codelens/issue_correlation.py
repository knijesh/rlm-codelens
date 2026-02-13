"""
Issue Correlation Analysis Module
Finds relationships between GitHub issues/PRs using multiple correlation signals
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import networkx as nx
from typing import List, Dict, Tuple
import json


class IssueCorrelationAnalyzer:
    """Analyzes correlations between issues using multiple signals"""

    def __init__(self, db_url=None):
        from rlm_codelens.config import (
            DATABASE_URL,
            TABLE_CLUSTERED,
            TABLE_CLUSTER_ANALYSES,
        )

        self.engine = create_engine(db_url or DATABASE_URL)
        self.table_clustered = TABLE_CLUSTERED
        self.table_cluster_analyses = TABLE_CLUSTER_ANALYSES
        self.correlations = []
        self.graph = nx.Graph()

    def load_data(self) -> pd.DataFrame:
        """Load issue data with embeddings"""
        # Try query with cluster_analyses join first
        query_with_analysis = f"""
        SELECT
            i.number,
            i.title,
            i.body,
            i.labels,
            i.author,
            i.created_at,
            i.type,
            i.cluster_id,
            c.topic,
            c.category,
            i.embedding
        FROM {self.table_clustered} i
        LEFT JOIN {self.table_cluster_analyses} c ON i.cluster_id = c.cluster_id
        WHERE i.type = 'issue'
        """

        # Fallback query without cluster_analyses
        query_simple = f"""
        SELECT
            i.number,
            i.title,
            i.body,
            i.labels,
            i.author,
            i.created_at,
            i.type,
            i.cluster_id,
            NULL as topic,
            NULL as category,
            i.embedding
        FROM {self.table_clustered} i
        WHERE i.type = 'issue'
        """

        try:
            df = pd.read_sql(query_with_analysis, self.engine)
        except Exception:
            # If cluster_analyses doesn't exist, use simple query
            df = pd.read_sql(query_simple, self.engine)

        # Parse JSON embeddings back to lists
        import json

        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: json.loads(x)
                if isinstance(x, str) and x.startswith("[")
                else x
            )

        # Parse labels from comma-separated strings to lists
        if "labels" in df.columns:
            df["labels"] = df["labels"].apply(
                lambda x: [l.strip() for l in x.split(",") if l.strip()]
                if isinstance(x, str)
                else (x if isinstance(x, list) else [])
            )

        # Parse dates to datetime objects
        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])

        return df

    def find_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find correlations between issues using multiple signals:
        1. Text similarity (embeddings)
        2. Shared labels
        3. Same author
        4. Temporal proximity
        5. Cross-references in body
        """
        correlations = []

        # Convert embeddings to matrix
        embeddings = np.array(df["embedding"].tolist())

        # 1. Text similarity correlations
        print("Finding text similarity correlations...")
        text_corr = self._find_text_correlations(df, embeddings)
        correlations.extend(text_corr)

        # 2. Label-based correlations
        print("Finding label correlations...")
        label_corr = self._find_label_correlations(df)
        correlations.extend(label_corr)

        # 3. Author correlations
        print("Finding author correlations...")
        author_corr = self._find_author_correlations(df)
        correlations.extend(author_corr)

        # 4. Temporal correlations
        print("Finding temporal correlations...")
        temporal_corr = self._find_temporal_correlations(df)
        correlations.extend(temporal_corr)

        # 5. Cross-reference correlations
        print("Finding cross-reference correlations...")
        xref_corr = self._find_cross_reference_correlations(df)
        correlations.extend(xref_corr)

        return correlations

    def _find_text_correlations(
        self, df: pd.DataFrame, embeddings: np.ndarray
    ) -> List[Dict]:
        """Find issues with high text similarity"""
        correlations = []

        # Calculate cosine similarity matrix (sample for performance)
        sample_size = min(5000, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        sample_embeddings = embeddings[sample_indices]

        similarity_matrix = cosine_similarity(sample_embeddings)

        # Find pairs with similarity > 0.7
        threshold = 0.7
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                if similarity_matrix[i][j] > threshold:
                    idx1, idx2 = sample_indices[i], sample_indices[j]
                    correlations.append(
                        {
                            "source": int(df.iloc[idx1]["number"]),
                            "target": int(df.iloc[idx2]["number"]),
                            "type": "text_similarity",
                            "strength": float(similarity_matrix[i][j]),
                            "source_title": df.iloc[idx1]["title"],
                            "target_title": df.iloc[idx2]["title"],
                        }
                    )

        return correlations

    def _find_label_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Find issues sharing the same labels"""
        correlations = []

        # Group by labels
        label_groups = defaultdict(list)
        for idx, row in df.iterrows():
            labels = row["labels"]
            if labels and len(labels) > 0:
                for label in labels:
                    label_groups[label].append(idx)

        # Create correlations for issues sharing labels
        for label, indices in label_groups.items():
            if (
                len(indices) > 1 and len(indices) < 100
            ):  # Limit to avoid complete graphs
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        correlations.append(
                            {
                                "source": int(df.loc[idx1, "number"]),
                                "target": int(df.loc[idx2, "number"]),
                                "type": "shared_label",
                                "strength": 0.8,
                                "label": label,
                                "source_title": df.loc[idx1, "title"],
                                "target_title": df.loc[idx2, "title"],
                            }
                        )

        return correlations

    def _find_author_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Find issues by the same author"""
        correlations = []

        # Group by author
        author_groups = df.groupby("author").groups

        for author, indices in author_groups.items():
            if author and len(indices) > 1 and len(indices) < 50:
                indices = list(indices)
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        idx1, idx2 = indices[i], indices[j]
                        correlations.append(
                            {
                                "source": int(df.loc[idx1, "number"]),
                                "target": int(df.loc[idx2, "number"]),
                                "type": "same_author",
                                "strength": 0.6,
                                "author": author,
                                "source_title": df.loc[idx1, "title"],
                                "target_title": df.loc[idx2, "title"],
                            }
                        )

        return correlations

    def _find_temporal_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Find issues created close in time (within 7 days)"""
        correlations = []

        df_sorted = df.sort_values("created_at")

        # Sliding window approach
        window_days = 7
        for i in range(len(df_sorted)):
            current_time = df_sorted.iloc[i]["created_at"]
            current_cluster = df_sorted.iloc[i]["cluster_id"]

            # Look ahead in window
            for j in range(i + 1, min(i + 50, len(df_sorted))):
                time_diff = (df_sorted.iloc[j]["created_at"] - current_time).days

                if time_diff > window_days:
                    break

                # Only correlate if same cluster or very close time
                if df_sorted.iloc[j]["cluster_id"] == current_cluster or time_diff <= 1:
                    correlations.append(
                        {
                            "source": int(df_sorted.iloc[i]["number"]),
                            "target": int(df_sorted.iloc[j]["number"]),
                            "type": "temporal_proximity",
                            "strength": 1.0 - (time_diff / window_days),
                            "days_diff": int(time_diff),
                            "source_title": df_sorted.iloc[i]["title"],
                            "target_title": df_sorted.iloc[j]["title"],
                        }
                    )

        return correlations

    def _find_cross_reference_correlations(self, df: pd.DataFrame) -> List[Dict]:
        """Find issues that reference each other in body text"""
        correlations = []

        # Create issue number lookup
        issue_numbers = set(df["number"].tolist())

        for idx, row in df.iterrows():
            if pd.isna(row["body"]):
                continue

            body = str(row["body"])

            # Look for references like #1234, issue #1234, etc.
            import re

            refs = re.findall(r"(?:issue|pr|#)\s*(\d+)", body.lower())

            for ref in refs:
                ref_num = int(ref)
                if ref_num in issue_numbers and ref_num != row["number"]:
                    target_row = df[df["number"] == ref_num].iloc[0]
                    correlations.append(
                        {
                            "source": int(row["number"]),
                            "target": ref_num,
                            "type": "cross_reference",
                            "strength": 1.0,
                            "source_title": row["title"],
                            "target_title": target_row["title"],
                        }
                    )

        return correlations

    def build_graph(self, correlations: List[Dict], df: pd.DataFrame) -> nx.Graph:
        """Build NetworkX graph from correlations"""
        G = nx.Graph()

        # Add nodes (issues)
        for _, row in df.iterrows():
            G.add_node(
                int(row["number"]),
                title=row["title"][:100] + "..."
                if len(row["title"]) > 100
                else row["title"],
                author=row["author"],
                created_at=str(row["created_at"]),
                cluster_id=int(row["cluster_id"])
                if pd.notna(row["cluster_id"])
                else -1,
                topic=row["topic"] if pd.notna(row["topic"]) else "Unknown",
                category=row["category"] if pd.notna(row["category"]) else "Other",
                labels=row["labels"] if row["labels"] else [],
            )

        # Add edges (correlations)
        for corr in correlations:
            # Avoid duplicate edges
            if not G.has_edge(corr["source"], corr["target"]):
                G.add_edge(
                    corr["source"],
                    corr["target"],
                    type=corr["type"],
                    strength=corr["strength"],
                )

        self.graph = G
        return G

    def export_for_d3(self, G: nx.Graph, output_path: str = "issue_graph.json"):
        """Export graph in D3.js compatible format"""

        # Convert to D3 format
        nodes = []
        for node_id, attrs in G.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "title": attrs.get("title", ""),
                    "author": attrs.get("author", ""),
                    "created_at": attrs.get("created_at", ""),
                    "cluster_id": attrs.get("cluster_id", -1),
                    "topic": attrs.get("topic", "Unknown"),
                    "category": attrs.get("category", "Other"),
                    "labels": attrs.get("labels", []),
                    "degree": G.degree(node_id),
                }
            )

        links = []
        for source, target, attrs in G.edges(data=True):
            links.append(
                {
                    "source": source,
                    "target": target,
                    "type": attrs.get("type", "unknown"),
                    "strength": attrs.get("strength", 0.5),
                }
            )

        # Calculate cluster statistics for coloring
        cluster_stats = defaultdict(int)
        for node in nodes:
            cluster_stats[node["topic"]] += 1

        # Assign colors based on topic
        import random

        random.seed(42)
        topics = list(cluster_stats.keys())
        colors = {}
        for i, topic in enumerate(topics):
            hue = (i * 137.508) % 360  # Golden angle approximation
            colors[topic] = f"hsl({hue}, 70%, 60%)"

        for node in nodes:
            node["color"] = colors.get(node["topic"], "#999")

        graph_data = {
            "nodes": nodes,
            "links": links,
            "statistics": {
                "total_nodes": len(nodes),
                "total_links": len(links),
                "cluster_distribution": dict(cluster_stats),
                "correlation_types": {},
            },
        }

        # Count correlation types
        for link in links:
            corr_type = link["type"]
            if corr_type not in graph_data["statistics"]["correlation_types"]:
                graph_data["statistics"]["correlation_types"][corr_type] = 0
            graph_data["statistics"]["correlation_types"][corr_type] += 1

        # Save to file
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"Graph exported to {output_path}")
        print(f"Nodes: {len(nodes)}, Links: {len(links)}")

        return graph_data

    def analyze_central_issues(self, G: nx.Graph, top_n: int = 20) -> List[Dict]:
        """Find most central/connected issues"""

        # Calculate centrality metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Fallback for small or disconnected graphs
            eigenvector_centrality = {node: 0.0 for node in G.nodes()}

        # Combine metrics
        issues = []
        for node_id in G.nodes():
            # Handle None values in centrality metrics
            deg_cent = degree_centrality.get(node_id, 0.0) or 0.0
            bet_cent = betweenness_centrality.get(node_id, 0.0) or 0.0
            eig_cent = eigenvector_centrality.get(node_id, 0.0) or 0.0

            issues.append(
                {
                    "number": node_id,
                    "title": G.nodes[node_id].get("title", ""),
                    "topic": G.nodes[node_id].get("topic", "Unknown"),
                    "degree": G.degree(node_id),
                    "degree_centrality": deg_cent,
                    "betweenness_centrality": bet_cent,
                    "eigenvector_centrality": eig_cent,
                    "composite_score": (
                        deg_cent * 0.4 + bet_cent * 0.3 + eig_cent * 0.3
                    ),
                }
            )

        # Sort by composite score
        issues.sort(key=lambda x: x["composite_score"], reverse=True)

        return issues[:top_n]

    def find_issue_clusters(self, G: nx.Graph) -> List[Dict]:
        """Find natural clusters in the issue correlation graph"""

        # Use community detection
        from networkx.algorithms.community import greedy_modularity_communities

        communities = list(greedy_modularity_communities(G))

        cluster_info = []
        for i, community in enumerate(communities):
            subgraph = G.subgraph(community)

            # Get most common topic in this cluster
            topics = [G.nodes[n].get("topic", "Unknown") for n in community]
            from collections import Counter

            most_common_topic = Counter(topics).most_common(1)[0][0]

            cluster_info.append(
                {
                    "cluster_id": i,
                    "size": len(community),
                    "issues": list(community),
                    "dominant_topic": most_common_topic,
                    "internal_edges": subgraph.number_of_edges(),
                    "density": nx.density(subgraph),
                }
            )

        # Sort by size
        cluster_info.sort(key=lambda x: x["size"], reverse=True)

        return cluster_info


def main():
    """Run issue correlation analysis"""

    print("Initializing Issue Correlation Analyzer...")
    analyzer = IssueCorrelationAnalyzer()

    print("Loading data...")
    df = analyzer.load_data()
    print(f"Loaded {len(df)} issues")

    print("Finding correlations...")
    correlations = analyzer.find_correlations(df)
    print(f"Found {len(correlations)} correlations")

    print("Building graph...")
    G = analyzer.build_graph(correlations, df)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Exporting for D3.js...")
    graph_data = analyzer.export_for_d3(G, "issue_graph.json")

    print("Analyzing central issues...")
    central = analyzer.analyze_central_issues(G)
    print(f"\nTop 10 most central issues:")
    for issue in central[:10]:
        print(
            f"  #{issue['number']}: {issue['title'][:60]}... (score: {issue['composite_score']:.3f})"
        )

    print("\nFinding issue clusters...")
    clusters = analyzer.find_issue_clusters(G)
    print(f"Found {len(clusters)} natural clusters")
    for cluster in clusters[:5]:
        print(
            f"  Cluster {cluster['cluster_id']}: {cluster['size']} issues, topic: {cluster['dominant_topic']}"
        )

    # Save analysis results
    results = {
        "central_issues": central,
        "clusters": clusters,
        "total_correlations": len(correlations),
        "correlation_breakdown": graph_data["statistics"]["correlation_types"],
    }

    with open("correlation_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nAnalysis complete! Files generated:")
    print("  - issue_graph.json (for D3.js visualization)")
    print("  - correlation_analysis.json (analysis results)")


if __name__ == "__main__":
    main()
