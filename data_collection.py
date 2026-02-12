"""
Data Collection Module
Fetches PRs and Issues from PyTorch GitHub repository
"""

import os
from github import Github
import pandas as pd
from datetime import datetime, timedelta
from utils.database import get_db_manager
from config import (
    GITHUB_TOKEN,
    REPO_OWNER,
    REPO_NAME,
    DAYS_LIMIT,
    USE_SAMPLE_DATA,
    SAMPLE_SIZE,
)


class PyTorchDataCollector:
    """Collects data from PyTorch GitHub repository"""

    def __init__(self, github_token=None):
        self.token = github_token or GITHUB_TOKEN
        if not self.token:
            raise ValueError(
                "GitHub token is required. Set GITHUB_TOKEN environment variable."
            )

        self.g = Github(self.token)
        self.repo = self.g.get_repo(f"{REPO_OWNER}/{REPO_NAME}")
        self.db = get_db_manager()

        print(f"✓ Connected to repository: {self.repo.full_name}")
        print(f"  Stars: {self.repo.stargazers_count:,}")
        print(f"  Forks: {self.repo.forks_count:,}")
        print(f"  Open Issues: {self.repo.open_issues_count:,}")

    def collect_issues(self, limit=None, days_limit=None):
        """
        Collect all issues with metadata

        Args:
            limit: Maximum number of issues to collect (for testing)
            days_limit: Only collect issues from last N days (None = all)
        """
        print(f"\n📥 Collecting issues...")

        issues = []
        issue_iterator = self.repo.get_issues(state="all")

        # Calculate cutoff date if days_limit specified
        cutoff_date = None
        if days_limit or DAYS_LIMIT:
            days = days_limit or int(DAYS_LIMIT)
            cutoff_date = datetime.now() - timedelta(days=days)
            print(f"  Filtering: Only issues from last {days} days")

        count = 0
        for issue in issue_iterator:
            # Check limit
            if limit and count >= limit:
                print(f"  Reached limit of {limit} issues")
                break

            # Check date cutoff
            if cutoff_date and issue.created_at < cutoff_date:
                continue

            # Extract issue data
            issue_data = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "labels": [l.name for l in issue.labels],
                "state": issue.state,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at,
                "closed_at": issue.closed_at,
                "comments_count": issue.comments,
                "author": issue.user.login if issue.user else None,
                "author_id": issue.user.id if issue.user else None,
                "type": "issue",
                "is_pr": False,
                "url": issue.html_url,
            }

            issues.append(issue_data)
            count += 1

            # Progress update
            if count % 1000 == 0:
                print(f"  Collected {count} issues...")
                # Save checkpoint
                self._save_checkpoint(pd.DataFrame(issues), "issues_checkpoint")

        print(f"✓ Collected {len(issues)} issues total")
        return pd.DataFrame(issues)

    def collect_pull_requests(self, limit=None, days_limit=None):
        """
        Collect all pull requests with metadata

        Args:
            limit: Maximum number of PRs to collect
            days_limit: Only collect PRs from last N days
        """
        print(f"\n📥 Collecting pull requests...")

        prs = []
        pr_iterator = self.repo.get_pulls(state="all")

        # Calculate cutoff date
        cutoff_date = None
        if days_limit or DAYS_LIMIT:
            days = days_limit or int(DAYS_LIMIT)
            cutoff_date = datetime.now() - timedelta(days=days)
            print(f"  Filtering: Only PRs from last {days} days")

        count = 0
        for pr in pr_iterator:
            # Check limit
            if limit and count >= limit:
                print(f"  Reached limit of {limit} PRs")
                break

            # Check date cutoff
            if cutoff_date and pr.created_at < cutoff_date:
                continue

            # Extract PR data
            pr_data = {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body or "",
                "labels": [l.name for l in pr.labels],
                "state": pr.state,
                "created_at": pr.created_at,
                "updated_at": pr.updated_at,
                "closed_at": pr.closed_at,
                "merged_at": pr.merged_at,
                "comments_count": pr.comments,
                "author": pr.user.login if pr.user else None,
                "author_id": pr.user.id if pr.user else None,
                "type": "pr",
                "is_pr": True,
                "merged": pr.merged,
                "merge_commit_sha": pr.merge_commit_sha,
                "additions": pr.additions,
                "deletions": pr.deletions,
                "changed_files": pr.changed_files,
                "url": pr.html_url,
            }

            prs.append(pr_data)
            count += 1

            # Progress update
            if count % 1000 == 0:
                print(f"  Collected {count} PRs...")
                self._save_checkpoint(pd.DataFrame(prs), "prs_checkpoint")

        print(f"✓ Collected {len(prs)} PRs total")
        return pd.DataFrame(prs)

    def collect_all(self, limit=None):
        """
        Collect both issues and PRs

        Returns:
            Combined DataFrame with all items
        """
        if USE_SAMPLE_DATA:
            limit = limit or SAMPLE_SIZE
            print(f"\n🧪 SAMPLE MODE: Collecting {limit} items only")

        # Collect issues
        issues_df = self.collect_issues(limit=limit)

        # Collect PRs
        prs_df = self.collect_pull_requests(limit=limit)

        # Combine
        combined = pd.concat([issues_df, prs_df], ignore_index=True)

        print(f"\n📊 Collection Summary:")
        print(f"  Issues: {len(issues_df)}")
        print(f"  PRs: {len(prs_df)}")
        print(f"  Total: {len(combined)}")

        # Convert labels list to string for SQLite compatibility
        if "labels" in combined.columns:
            combined["labels"] = combined["labels"].apply(
                lambda x: ", ".join(x) if isinstance(x, list) else str(x)
            )

        # Save to database
        self.db.save_dataframe(combined, "pytorch_items", if_exists="replace")

        # Create indexes
        self.db.create_indexes(
            "pytorch_items", ["number", "type", "author", "created_at"]
        )

        return combined

    def _save_checkpoint(self, df, table_name):
        """Save intermediate results to database"""
        try:
            self.db.save_dataframe(df, table_name, if_exists="replace")
        except Exception as e:
            print(f"  Warning: Could not save checkpoint: {e}")

    def get_rate_limit(self):
        """Check current rate limit status"""
        rate_limit = self.g.get_rate_limit()
        print(f"\n⏱️  GitHub API Rate Limit:")
        print(f"  Remaining: {rate_limit.core.remaining} / {rate_limit.core.limit}")
        print(f"  Resets at: {rate_limit.core.reset}")
        return rate_limit


def main():
    """Run data collection"""
    print("=" * 60)
    print("PYTORCH DATA COLLECTION")
    print("=" * 60)

    # Initialize collector
    collector = PyTorchDataCollector()

    # Check rate limit
    collector.get_rate_limit()

    # Collect data
    df = collector.collect_all()

    print("\n✓ Data collection complete!")
    print(f"  Total items: {len(df)}")

    # Show sample
    print("\n📋 Sample data:")
    print(df.head(3)[["number", "type", "title", "author"]].to_string())

    return df


if __name__ == "__main__":
    main()
