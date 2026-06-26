"""第三方程序调用入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .core.pipeline import (
    CounterResult,
    FrameResult,
    PipeJointPipeline,
    PipelineConfig,
    save_result,
)
from .core.report import InspectionReport, build_inspection_report


def count_pipe_joints(
    video_path: str | Path,
    csv_path: str | Path,
    config: PipelineConfig | None = None,
    on_frame: Callable[[FrameResult], bool] | None = None,
    output_json: str | Path | None = None,
    save_report: bool = True,
) -> CounterResult:
  video_path = Path(video_path)
  cfg = config or PipelineConfig()
  cfg.csv_path = str(csv_path)

  pipeline = PipeJointPipeline(cfg)
  result = pipeline.process_video(video_path, on_frame=on_frame)

  if output_json is not None or save_report:
    json_path = Path(output_json) if output_json else video_path.with_suffix(".joint_count.json")
    save_result(result, json_path, pipeline.calibration_result)

  return result


def build_report(
    joint_count: int,
    csv_path: str | Path,
    video_path: str | Path | None = None,
) -> InspectionReport:
  return build_inspection_report(
      joint_count=joint_count,
      csv_path=csv_path,
      video_path=video_path,
  )


__all__ = [
    "CounterResult",
    "FrameResult",
    "InspectionReport",
    "PipelineConfig",
    "build_report",
    "count_pipe_joints",
]
