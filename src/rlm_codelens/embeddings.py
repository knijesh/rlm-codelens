"""
Embeddings Generation Module
Generates OpenAI embeddings for issue/PR text
"""

import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from rlm_codelens.utils.database import get_db_manager
from rlm_codelens.utils.cost_tracker import CostTracker, format_cost
from rlm_codelens.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    TABLE_ITEMS,
    TABLE_EMBEDDINGS,
)


class EmbeddingGenerator:
    """Generates embeddings for text data using OpenAI API"""

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model or EMBEDDING_MODEL
        self.db = get_db_manager()
        self.cost_tracker = CostTracker()

        print(f"✓ Embedding generator initialized")
        print(f"  Model: {self.model}")
        print(f"  Batch size: {EMBEDDING_BATCH_SIZE}")

    def prepare_text(self, row):
        """Prepare text for embedding from a row"""
        # Combine title and body (truncated)
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))[:1000]  # Limit body length

        # Add metadata context
        text = f"Title: {title}\n\nDescription: {body}"

        # Add labels context if available
        labels = row.get("labels", [])
        if labels and len(labels) > 0:
            text += f"\n\nLabels: {', '.join(labels)}"

        return text.strip()

    def generate_embeddings(self, table_name=None, batch_size=None):
        """
        Generate embeddings for all items in database

        Args:
            table_name: Name of table with items (defaults to TABLE_ITEMS from config)
            batch_size: Batch size for API calls

        Returns:
            DataFrame with embeddings added
        """
        table_name = table_name or TABLE_ITEMS
        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        print(f"\n🔄 Generating embeddings...")
        print(f"  Source table: {table_name}")

        # Load data
        df = self.db.load_dataframe(table_name)
        print(f"  Loaded {len(df)} items")

        # Check if embeddings already exist
        if "embedding" in df.columns and df["embedding"].notna().any():
            print(f"  ⚠️  Embeddings already exist. Skipping generation.")
            print(f"     Use force=True to regenerate.")
            return df

        # Prepare texts
        print("  Preparing texts...")
        df["text_for_embedding"] = df.apply(self.prepare_text, axis=1)

        # Generate embeddings in batches
        all_embeddings = []
        total_tokens = 0

        num_batches = (len(df) + batch_size - 1) // batch_size

        with tqdm(total=len(df), desc="Generating embeddings") as pbar:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                texts = batch["text_for_embedding"].tolist()

                try:
                    # Call OpenAI API
                    response = self.client.embeddings.create(
                        model=self.model, input=texts
                    )

                    # Extract embeddings
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)

                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    batch_tokens = sum(len(text) // 4 for text in texts)
                    total_tokens += batch_tokens

                    # Track cost
                    cost = self.cost_tracker.add_embedding_call(
                        batch_tokens, self.model
                    )

                    # Update progress
                    pbar.update(len(batch))

                    # Save checkpoint every 10 batches
                    if (i // batch_size) % 10 == 0 and i > 0:
                        self._save_checkpoint(
                            df.iloc[: i + len(batch)], all_embeddings, i
                        )

                except Exception as e:
                    print(f"\n  ❌ Error in batch {i // batch_size + 1}: {e}")
                    # Fill with None for failed batch
                    all_embeddings.extend([None] * len(batch))
                    pbar.update(len(batch))

        # Add embeddings to dataframe
        df["embedding"] = all_embeddings
        df["embedding_model"] = self.model

        # Calculate total cost
        total_cost = self.cost_tracker.current_cost

        print(f"\n✓ Embeddings generated!")
        print(f"  Total items: {len(df)}")
        print(f"  Successful: {df['embedding'].notna().sum()}")
        print(f"  Failed: {df['embedding'].isna().sum()}")
        print(f"  Estimated tokens: {total_tokens:,}")
        print(f"  Cost: {format_cost(total_cost)}")

        # Save to database
        self._save_to_database(df)

        # Print cost summary
        self.cost_tracker.print_summary()

        return df

    def generate_for_subset(self, df_subset, batch_size=None):
        """
        Generate embeddings for a subset of data

        Useful for testing or incremental updates
        """
        batch_size = batch_size or EMBEDDING_BATCH_SIZE

        print(f"\n🔄 Generating embeddings for subset ({len(df_subset)} items)...")

        # Prepare texts
        df_subset["text_for_embedding"] = df_subset.apply(self.prepare_text, axis=1)

        all_embeddings = []

        with tqdm(total=len(df_subset), desc="Generating") as pbar:
            for i in range(0, len(df_subset), batch_size):
                batch = df_subset.iloc[i : i + batch_size]
                texts = batch["text_for_embedding"].tolist()

                try:
                    response = self.client.embeddings.create(
                        model=self.model, input=texts
                    )

                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)

                    # Track cost
                    batch_tokens = sum(len(text) // 4 for text in texts)
                    self.cost_tracker.add_embedding_call(batch_tokens, self.model)

                    pbar.update(len(batch))

                except Exception as e:
                    print(f"\n  ❌ Error: {e}")
                    all_embeddings.extend([None] * len(batch))
                    pbar.update(len(batch))

        df_subset["embedding"] = all_embeddings
        df_subset["embedding_model"] = self.model

        return df_subset

    def _save_checkpoint(self, df, embeddings, start_idx):
        """Save checkpoint to database"""
        try:
            checkpoint_df = df.iloc[: start_idx + len(embeddings)].copy()
            checkpoint_df.loc[checkpoint_df.index[: len(embeddings)], "embedding"] = (
                embeddings
            )

            temp_table = "embeddings_checkpoint"
            self.db.save_dataframe(
                checkpoint_df[["number", "type", "embedding", "embedding_model"]],
                temp_table,
                if_exists="replace",
            )

            print(f"  💾 Checkpoint saved ({len(embeddings)} embeddings)")
        except Exception as e:
            print(f"  ⚠️  Could not save checkpoint: {e}")

    def _save_to_database(self, df):
        """Save embeddings to database"""
        print("\n💾 Saving embeddings to database...")

        # Convert embedding vectors to JSON strings for SQLite compatibility
        import json

        df_to_save = df.copy()
        if "embedding" in df_to_save.columns:
            df_to_save["embedding"] = df_to_save["embedding"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else x
            )

        # Save full dataframe with dynamic table name
        self.db.save_dataframe(df_to_save, TABLE_EMBEDDINGS, if_exists="replace")

        # Create indexes (only on columns that exist)
        available_columns = [
            col for col in ["number", "type", "cluster_id"] if col in df_to_save.columns
        ]
        if available_columns:
            self.db.create_indexes(TABLE_EMBEDDINGS, available_columns)

        print("✓ Embeddings saved successfully")

    def get_embedding_stats(self):
        """Get statistics about generated embeddings"""
        df = self.db.load_dataframe(TABLE_EMBEDDINGS, parse_embeddings=True)

        stats = {
            "total_items": len(df),
            "with_embeddings": df["embedding"].notna().sum(),
            "without_embeddings": df["embedding"].isna().sum(),
            "embedding_dimension": len(df[df["embedding"].notna()].iloc[0]["embedding"])
            if df["embedding"].notna().any()
            else 0,
            "model": df["embedding_model"].iloc[0]
            if "embedding_model" in df.columns
            else None,
        }

        return stats


# Note: Use main.py as the single driver file for the entire pipeline
# This module provides the EmbeddingGenerator class for generating embeddings
