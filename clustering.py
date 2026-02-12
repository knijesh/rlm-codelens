"""
Clustering Module
Clusters issues/PRs by topic using HDBSCAN on embeddings
"""

import pandas as pd
import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from utils.database import get_db_manager
from config import MIN_CLUSTER_SIZE, MIN_SAMPLES


class TopicClusterer:
    """Clusters GitHub items by topic using embeddings"""

    def __init__(self, min_cluster_size=None, min_samples=None):
        self.min_cluster_size = min_cluster_size or MIN_CLUSTER_SIZE
        self.min_samples = min_samples or MIN_SAMPLES
        self.db = get_db_manager()

        print(f"✓ Topic clusterer initialized")
        print(f"  Algorithm: HDBSCAN")
        print(f"  Min cluster size: {self.min_cluster_size}")
        print(f"  Min samples: {self.min_samples}")

    def cluster(self, table_name="pytorch_items_with_embeddings"):
        """
        Cluster items by topic

        Args:
            table_name: Table with embeddings

        Returns:
            DataFrame with cluster assignments
        """
        print(f"\n🎯 Clustering items by topic...")

        # Load data with embeddings
        df = self.db.load_dataframe(table_name)
        print(f"  Loaded {len(df)} items")

        # Filter out items without embeddings
        df_valid = df[df["embedding"].notna()].copy()
        print(f"  Valid embeddings: {len(df_valid)} items")

        if len(df_valid) == 0:
            raise ValueError("No valid embeddings found. Run embeddings.py first.")

        # Convert embeddings to numpy array
        print("  Preparing embedding matrix...")
        embeddings = np.array(df_valid["embedding"].tolist())

        # Run HDBSCAN clustering
        print(f"  Running HDBSCAN...")
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass
        )

        cluster_labels = clusterer.fit_predict(embeddings)

        # Add cluster labels to dataframe
        df_valid["cluster_id"] = cluster_labels

        # Handle items without embeddings (assign to noise cluster -1)
        df_invalid = df[df["embedding"].isna()].copy()
        df_invalid["cluster_id"] = -1

        # Combine back
        df_clustered = pd.concat([df_valid, df_invalid], ignore_index=True)

        # Analyze clusters
        self._analyze_clusters(df_clustered)

        # Save to database
        self._save_to_database(df_clustered)

        return df_clustered

    def _analyze_clusters(self, df):
        """Analyze and print cluster statistics"""
        print("\n📊 Cluster Analysis:")

        # Count items per cluster
        cluster_counts = df["cluster_id"].value_counts().sort_index()

        # Separate noise cluster (-1) from real clusters
        noise_count = cluster_counts.get(-1, 0)
        real_clusters = cluster_counts[cluster_counts.index != -1]

        print(f"  Total clusters: {len(real_clusters)}")
        print(
            f"  Noise points (unclustered): {noise_count} ({noise_count / len(df) * 100:.1f}%)"
        )
        print(
            f"  Clustered items: {len(df) - noise_count} ({(len(df) - noise_count) / len(df) * 100:.1f}%)"
        )

        print(f"\n  Top 10 largest clusters:")
        for cluster_id, count in real_clusters.head(10).items():
            print(f"    Cluster {cluster_id}: {count} items")

        # Analyze cluster sizes
        print(f"\n  Cluster size statistics:")
        print(f"    Min: {real_clusters.min()}")
        print(f"    Max: {real_clusters.max()}")
        print(f"    Mean: {real_clusters.mean():.1f}")
        print(f"    Median: {real_clusters.median():.1f}")

    def get_cluster_stats(self, df):
        """Generate detailed statistics for each cluster"""
        stats = []

        for cluster_id in df["cluster_id"].unique():
            if cluster_id == -1:
                continue

            cluster_items = df[df["cluster_id"] == cluster_id]

            # Get most common labels
            all_labels = [
                label for labels in cluster_items["labels"] for label in labels
            ]
            top_labels = Counter(all_labels).most_common(5)

            # Get most common authors
            top_authors = cluster_items["author"].value_counts().head(5).to_dict()

            # Get date range
            date_range = {
                "earliest": cluster_items["created_at"].min(),
                "latest": cluster_items["created_at"].max(),
            }

            stats.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(cluster_items),
                    "issue_count": len(cluster_items[cluster_items["type"] == "issue"]),
                    "pr_count": len(cluster_items[cluster_items["type"] == "pr"]),
                    "top_labels": top_labels,
                    "top_authors": top_authors,
                    "date_range": date_range,
                    "sample_titles": cluster_items["title"].head(5).tolist(),
                }
            )

        return pd.DataFrame(stats)

    def _save_to_database(self, df):
        """Save clustered data to database"""
        print("\n💾 Saving clustered data to database...")

        # Save main table
        self.db.save_dataframe(df, "clustered_items", if_exists="replace")

        # Generate and save cluster statistics
        cluster_stats = self.get_cluster_stats(df)
        self.db.save_dataframe(cluster_stats, "cluster_stats", if_exists="replace")

        # Create indexes
        self.db.create_indexes("clustered_items", ["cluster_id", "number", "type"])

        print("✓ Clustered data saved successfully")

    def get_cluster_items(self, cluster_id):
        """Get all items in a specific cluster"""
        df = self.db.load_dataframe("clustered_items")
        return df[df["cluster_id"] == cluster_id]

    def get_noise_items(self):
        """Get all unclustered items (noise)"""
        df = self.db.load_dataframe("clustered_items")
        return df[df["cluster_id"] == -1]


def main():
    """Run clustering"""
    print("=" * 60)
    print("TOPIC CLUSTERING")
    print("=" * 60)

    # Initialize clusterer
    clusterer = TopicClusterer()

    # Run clustering
    df = clusterer.cluster()

    # Get statistics
    stats = clusterer.get_cluster_stats(df)

    print("\n✓ Clustering complete!")
    print(f"  Total clusters: {len(stats)}")
    print(f"  Total items clustered: {stats['size'].sum()}")

    # Show sample of largest cluster
    if len(stats) > 0:
        largest = stats.loc[stats["size"].idxmax()]
        print(f"\n📌 Largest cluster (ID {largest['cluster_id']}):")
        print(f"  Size: {largest['size']} items")
        print(f"  Sample titles:")
        for title in largest["sample_titles"][:3]:
            print(f"    - {title[:80]}...")

    return df, stats


if __name__ == "__main__":
    main()
