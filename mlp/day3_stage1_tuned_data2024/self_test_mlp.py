"""
Self-test wrapper for the tuned stage1 MLP model on ./data_2024.

This keeps the same train / validation / test methodology and embargo logic as
the current stage1 winner, while changing only the underlying historical data
source.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp.day3_stage1_tuned import self_test_mlp as base  # noqa: E402
try:  # noqa: E402
    from .paths import DATA_2024_DIR
except ImportError:  # noqa: E402
    from paths import DATA_2024_DIR


base.DATA_DIR = DATA_2024_DIR


if __name__ == "__main__":
    base.main()
