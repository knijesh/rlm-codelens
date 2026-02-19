"""
Configuration module for RLM-Codelens
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

# API Keys (for --deep RLM analysis)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Budget
BUDGET_LIMIT = float(os.getenv("BUDGET_LIMIT", "50.0"))
BUDGET_ALERT_THRESHOLD = int(os.getenv("BUDGET_ALERT_THRESHOLD", "80"))

# RLM Architecture Analyzer Configuration
RLM_BACKEND = os.getenv("RLM_BACKEND", "openai")
RLM_MODEL = os.getenv("RLM_MODEL", "gpt-4o")
RLM_BASE_URL = os.getenv("RLM_BASE_URL", "")  # e.g. http://localhost:11434/v1 for Ollama
RLM_ENVIRONMENT = os.getenv("RLM_ENVIRONMENT", "local")
RLM_MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "30"))

# Repository Scan Configuration
SCAN_EXCLUDE_PATTERNS = [
    p.strip() for p in os.getenv("SCAN_EXCLUDE_PATTERNS", "").split(",") if p.strip()
]

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(OUTPUTS_DIR / "analysis.log"))
