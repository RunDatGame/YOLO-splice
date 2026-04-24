from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

from pipeline_splice.core.config import (
    build_task_input,
    load_pipeline_config,
    normalize_model_mode,
    validate_runtime_config,
)

CORE_MODULES = [
    "cv2",
    "torch",
    "torchvision",
    "pandas",
    "numpy",
    "watchdog",
    "trimesh",
    "pygltflib",
]


def check_imports() -> list[str]:
    errors: list[str] = []
    for module_name in CORE_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"缺少依赖 {module_name}: {exc}")
    return errors


def check_blender(blender_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        result = subprocess.run(
            [str(blender_path), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        first_line = (result.stdout or result.stderr).splitlines()[0]
        print(f"Blender: {first_line}")
    except Exception as exc:
        errors.append(f"Blender 无法启动: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight check for the YOLO pipeline")
    parser.add_argument("--csv", required=True, help="里程 CSV 路径")
    parser.add_argument("--video", required=True, help="视频路径")
    parser.add_argument("--work_dir", default=".", help="项目工作目录")
    parser.add_argument("--mode", help="本次运行的建模方式: library / reconstruction")
    args = parser.parse_args()

    print(f"Python: {sys.version.splitlines()[0]}")

    task = build_task_input(args.csv, args.video, args.work_dir)
    config_path = task.work_dir / "config" / "splice.txt"
    if not config_path.exists():
        config_path = task.work_dir / "config.txt"
    config = load_pipeline_config(config_path, task, model_mode_override=normalize_model_mode(args.mode))

    errors = []
    errors.extend(check_imports())

    try:
        validate_runtime_config(config, task)
    except Exception as exc:
        errors.append(str(exc))

    if config.blender_path and config.blender_path.exists():
        errors.extend(check_blender(config.blender_path))

    if errors:
        print("预检失败:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("预检通过")
    print(f"视频: {task.video_path}")
    print(f"CSV: {task.csv_path}")
    print(f"输出目录: {task.output_dir}")
    print(f"Blender: {config.blender_path}")
    if config.model_mode == "library":
        print(f"缺陷模型库: {config.dataset_path}")
        print(f"井室模型: {config.manhole_path}")
    else:
        print(f"Meshroom: {config.meshroom_path}")
        print(f"重建命令: {config.reconstruction_command or '<内建 Meshroom/直接 mesh>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
