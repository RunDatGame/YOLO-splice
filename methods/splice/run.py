import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "splice.txt"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline_core.runner import process_task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="拼接建模独立入口")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--work_dir", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    process_task(
        args.csv,
        args.video,
        args.work_dir,
        model_mode="library",
        config_path=args.config,
    )
