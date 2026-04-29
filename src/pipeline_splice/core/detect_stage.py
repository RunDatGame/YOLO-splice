from __future__ import annotations

from pathlib import Path

from pipeline_splice.detection import engine as detect_engine

from .contracts import DetectArtifacts, PipelineConfig


def run_detection_stage(
    config: PipelineConfig,
    work_dir: Path,
    visual_callback=None,
    output_dir: Path | None = None,
) -> DetectArtifacts:
    video_path = str(config.video_path)
    result_dir = Path(video_path).stem
    frame_dir = work_dir / "runs" / "detect" / result_dir / "frames"

    count, width, height, fps = detect_engine.save_frames(video_path, str(frame_dir), config.interval)
    if not count or not width or not height or not fps:
        raise RuntimeError("抽帧失败或视频不可读")

    valid_frames = detect_engine.filter_valid_frames(video_path, str(frame_dir), str(config.raw_csv_path), fps)
    if not valid_frames:
        raise RuntimeError("里程同步失败或没有有效帧")

    pipe_params = (config.inner, config.outer, config.length, config.segment)
    best_defects = detect_engine.filter_best_defects(
        valid_frames,
        width,
        height,
        pipe_params,
        device=detect_engine.device,
        visual_callback=visual_callback,
        pixel_per_meter=config.pixel_per_meter,
    )

    detect_engine.process_best_defects(
        best_defects,
        result_dir,
        pipe_params,
        output_dir=output_dir,
        use_depth=config.use_depth,
        wall_thickness=config.wall_thickness,
        rebar_spacing=config.rebar_spacing,
    )

    detect_csv_local = (output_dir or work_dir) / "defect_results_full.csv"
    if not detect_csv_local.exists():
        import pandas as pd
        pd.DataFrame(columns=[
            "编号", "模型类型", "管节序号",
            "管节内径", "管节外径", "管节长度", "管道壁厚", "钢筋间距",
            "节内里程", "偏移距", "轴线偏角",
            "病害长", "病害宽", "病害高",
            "严重等级", "模型路径", "数据截图",
        ]).to_csv(detect_csv_local, index=False, encoding="utf-8-sig")
        print(f"  [INFO] 未检测到病害，已创建空结果: {detect_csv_local}")

    return DetectArtifacts(
        result_dir_name=result_dir,
        frame_dir=frame_dir,
        frame_count=count,
        valid_frame_count=len(valid_frames),
        detect_csv_local=detect_csv_local,
    )
