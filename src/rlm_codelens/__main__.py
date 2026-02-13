"""Entry point for running rlm_codelens as a module.

Usage:
    python -m rlm_codelens [args...]

Example:
    python -m rlm_codelens estimate --items 100
    python -m rlm_codelens analyze encode/starlette --sample
"""

import sys

from rlm_codelens.cli import main

if __name__ == "__main__":
    sys.exit(main())
