"""
Wrapper around the current tuned stage1 MLP model that reads from ./data_2024.

The model logic, feature engineering, teacher-imposed portfolio constraints,
and weighting search are intentionally unchanged. This experiment isolates the
effect of extending the historical training window.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp.day3_stage1_tuned import mlp_model as base  # noqa: E402
try:  # noqa: E402
    from .paths import DATA_2024_DIR
except ImportError:  # noqa: E402
    from paths import DATA_2024_DIR


base.DATA_DIR = DATA_2024_DIR


if __name__ == "__main__":
    base.main()
