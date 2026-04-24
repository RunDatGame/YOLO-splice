from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline_splice.core.runner import process_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline-splice", description="管道缺陷检测与模型拼接")
    parser.add_argument("--config", type=str, default="config/splice.txt", help="配置文件路径")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--csv", type=str, required=True, help="里程 CSV 路径")
    parser.add_argument("--work-dir", type=str, default=".", help="工作目录")
    parser.add_argument(
        "--mode",
        type=str,
        default="library",
        choices=["library", "splice", "reconstruction"],
        help="建模模式 (默认: library)",
    )
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = work_dir / config_path

    result = process_task(
        csv_path=args.csv,
        video_path=args.video,
        work_dir=work_dir,
        model_mode=args.mode,
        config_path=config_path,
    )

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
