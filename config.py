"""
Configuration module for PyTorch RLM Analysis
Loads settings from environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.absolute()
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# API Keys
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/pytorch_analysis")

# Budget
BUDGET_LIMIT = float(os.getenv("BUDGET_LIMIT", "50.0"))
BUDGET_ALERT_THRESHOLD = int(os.getenv("BUDGET_ALERT_THRESHOLD", "80"))

# RLM Configuration
RLM_ROOT_MODEL = os.getenv("RLM_ROOT_MODEL", "gpt-3.5-turbo")
RLM_SUB_MODEL = os.getenv("RLM_SUB_MODEL", "gpt-3.5-turbo")
RLM_MAX_DEPTH = int(os.getenv("RLM_MAX_DEPTH", "3"))

# Repository Configuration
REPO_OWNER = os.getenv("REPO_OWNER", "pytorch")
REPO_NAME = os.getenv("REPO_NAME", "pytorch")
DAYS_LIMIT = os.getenv("DAYS_LIMIT")  # Optional: None means all data

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1000"))

# Clustering Configuration
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "50"))
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "10"))

# Visualization Configuration
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", "1000"))
MIN_CORRELATION_STRENGTH = float(os.getenv("MIN_CORRELATION_STRENGTH", "0.5"))

# Development/Testing
USE_SAMPLE_DATA = os.getenv("USE_SAMPLE_DATA", "False").lower() == "true"
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "1000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(OUTPUTS_DIR / "analysis.log"))

# Cost estimates (per 1M tokens as of 2024)
COSTS = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
}

# Correlation types and their default weights
CORRELATION_TYPES = {
    "text_similarity": 1.0,
    "shared_label": 0.8,
    "same_author": 0.6,
    "temporal_proximity": 0.7,
    "cross_reference": 1.0,
}

# Categories for classification
CATEGORIES = ["Bug", "Feature", "Documentation", "Performance", "API", "Other"]


def validate_config():
    """Validate that required configuration is set"""
    errors = []

    if not GITHUB_TOKEN:
        errors.append(
            "GITHUB_TOKEN is required. Get it from https://github.com/settings/tokens"
        )

    if not OPENAI_API_KEY:
        errors.append(
            "OPENAI_API_KEY is required. Get it from https://platform.openai.com/api-keys"
        )

    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    print("✓ Configuration validated successfully")
    return True


def print_config():
    """Print current configuration (without sensitive data)"""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Repository: {REPO_OWNER}/{REPO_NAME}")
    print(
        f"Database: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}"
    )
    print(f"Budget Limit: ${BUDGET_LIMIT:.2f}")
    print(f"RLM Root Model: {RLM_ROOT_MODEL}")
    print(f"RLM Sub Model: {RLM_SUB_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Sample Mode: {USE_SAMPLE_DATA}")
    if USE_SAMPLE_DATA:
        print(f"Sample Size: {SAMPLE_SIZE}")
    print(f"Outputs Directory: {OUTPUTS_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    validate_config()
    print_config()
