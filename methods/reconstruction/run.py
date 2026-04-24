import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "reconstruction.txt"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline_core.config import parse_config_file
from pipeline_core.runner import process_reconstruction_only, process_task


def load_defaults(config_path: Path) -> dict[str, str | int | float]:
    if not config_path.exists():
        return {}
    return parse_config_file(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三维重建独立入口")
    parser.add_argument("--csv", help="里程 CSV；默认读取配置中的 raw")
    parser.add_argument("--video", help="视频路径；默认读取配置中的 video")
    parser.add_argument("--work_dir", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force-extract", action="store_true", help="忽略已有切帧，强制重新抽帧")
    parser.add_argument("--skip-export", action="store_true", help="只跑重建 mesh，不调用 Blender 导出 GLB")
    parser.add_argument("--full-pipeline", action="store_true", help="仍然执行完整的检测+重建流程")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    config_path = Path(args.config).resolve()
    defaults = load_defaults(config_path)

    video = args.video or str(defaults.get("video", "")).strip()
    csv = args.csv or str(defaults.get("raw", "")).strip() or video

    if not video:
        raise SystemExit("缺少视频路径。请通过 --video 指定，或在 config.txt 中配置 video。")

    if args.full_pipeline:
        result = process_task(
            csv,
            video,
            work_dir,
            model_mode="reconstruction",
            config_path=config_path,
        )
    else:
        result = process_reconstruction_only(
            csv,
            video,
            work_dir,
            force_extract=args.force_extract,
            export_glb=not args.skip_export,
            config_path=config_path,
        )

    if not result.success:
        print("流程失败。")
        for step in result.steps:
            if not step.success:
                print(f"[{step.name}] {step.details}")
        raise SystemExit(1)
