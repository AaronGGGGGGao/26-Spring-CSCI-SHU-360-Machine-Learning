"""
Wrapper around the current tuned stage1 MLP model.

This keeps the exact same model and development workflow. It exists so the
robust-split self-test line has its own isolated directory and artifact paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlp.day3_stage1_tuned import mlp_model as base  # noqa: E402


if __name__ == "__main__":
    base.main()
