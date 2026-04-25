from __future__ import annotations

from pathlib import Path

from .texture_builder import build_uv_pipeline


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

    print("=" * 40)
    print("UV 圆柱纹理重建任务启动")
    print(f"视频路径: {video_path_obj}")
    print(f"CSV 路径: {csv_path_obj}")
    print(f"工作目录: {work_dir_path}")

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
    print("=" * 40)
    return result
