"""Stage 2 5-day provided XGBoost baseline entrypoint."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline.baseline_xgboost import *  # noqa: F401,F403
from baseline.baseline_xgboost import main


if __name__ == "__main__":
    main()
