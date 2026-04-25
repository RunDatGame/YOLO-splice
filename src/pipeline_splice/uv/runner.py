from __future__ import annotations

from pathlib import Path

from .texture_builder import build_uv_pipeline

from pipeline_splice.core.config import build_task_input, load_pipeline_config
from pipeline_splice.core.detect_stage import run_detection_stage


def process_task(
    csv_path: str | Path,
    video_path: str | Path,
    work_dir: str | Path = ".",
    config_path: str | Path | None = None,
) -> dict:
    work_dir_path = Path(work_dir).resolve()
    video_path_obj = Path(video_path).resolve()
    csv_path_obj = Path(csv_path).resolve()

    runtime_config_path = None
    if config_path is not None:
        runtime_config_path = Path(config_path).resolve()
        if not runtime_config_path.is_absolute():
            runtime_config_path = work_dir_path / runtime_config_path

    task = build_task_input(csv_path_obj, video_path_obj, work_dir_path)
    config = load_pipeline_config(runtime_config_path, task, model_mode_override=None)
    config.__dict__["model_mode"] = "uv"
    config.__dict__["use_depth"] = False

    output_dir = work_dir_path / "outputs" / "uv_texture" / video_path_obj.stem
    info_txt = work_dir_path / "video_Config.txt"

    print("=" * 40)
    print("UV 圆柱纹理重建任务启动")
    print(f"视频路径: {video_path_obj}")
    print(f"CSV 路径: {csv_path_obj}")
    print(f"工作目录: {work_dir_path}")

    print(">>> [1/2] 运行病害检测...")
    artifacts = run_detection_stage(config, work_dir_path)
    print(f"  检测完成，frame_dir: {artifacts.frame_dir}")
    print(f"  有效帧数: {artifacts.valid_frame_count}")

    print(">>> [2/2] 生成 video_Config.txt...")
    info_txt.parent.mkdir(parents=True, exist_ok=True)
    info_txt.write_text(
        f"管节外径: {config.outer}\n建模模式: uv_texture\n",
        encoding="utf-8",
    )
    print(f"  信息文件: {info_txt}")

    result = build_uv_pipeline(
        video_path=video_path_obj,
        csv_path=csv_path_obj,
        work_dir=work_dir_path,
        config_path=runtime_config_path,
    )

    print(f"[OK] 重建完成")
    print(f"  总长度: {result['total_length']:.3f} m")
    print(f"  半径: {result['radius']:.3f} m")
    print(f"  采样帧数: {result['sample_count']}")
    print(f"  纹理: {result['texture_path']}")
    print(f"  OBJ: {result['obj_path']}")
    print(f"  MTL: {result['mtl_path']}")
    print(f"  病害CSV: {artifacts.detect_csv_local}")
    print(f"  外参配置: {info_txt}")
    print("=" * 40)
    return result