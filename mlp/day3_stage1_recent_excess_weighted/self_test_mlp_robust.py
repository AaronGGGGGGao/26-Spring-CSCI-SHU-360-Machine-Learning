"""Compatibility entrypoint for robust self-test."""
from __future__ import annotations

try:
    from .self_test_robust_main import main
except ImportError:
    from self_test_robust_main import main


if __name__ == "__main__":
    main()
