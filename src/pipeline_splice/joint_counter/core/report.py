"""管节检测报告：单节长度、管径计算与文件输出。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .csv_mileage import CsvMileageInfo, read_mileage_from_csv

DEFAULT_SEGMENT_LENGTH_M = 2.0


@dataclass
class InspectionReport:
  detection_date: str
  detection_time: str
  joint_count: int
  segment_length_m: float | None
  total_pipe_length_m: float | None
  segment_diameter_dn: int | None
  csv_total_mileage_m: float | None
  csv_path: str | None = None
  used_default_segment_length: bool = False
  detected_joint_count: int | None = None
  raw_segment_length_m: float | None = None
  mileage_column_index: int | None = None

  def to_dict(self) -> dict:
    return asdict(self)


def detection_timestamp_from_video(video_path: str | Path) -> tuple[str, str]:
  """检测日期、时间取视频文件的修改时间。"""
  path = Path(video_path)
  mtime = path.stat().st_mtime
  dt = datetime.fromtimestamp(mtime)
  return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")


def quantize_segment_length(raw_segment_m: float) -> tuple[float, bool]:
  """
  单管节长度量化（record 51）：
  <2 → 计算异常，取 2 m；[2,2.5)→2 m；[2.5,3.0)→2.5 m；≥3.0 → 3.0 m。
  """
  if raw_segment_m < 2.0:
    return DEFAULT_SEGMENT_LENGTH_M, True
  if raw_segment_m < 2.5:
    return 2.0, False
  if raw_segment_m < 3.0:
    return 2.5, False
  return 3.0, False


def compute_default_diameter_dn(segment_length_m: float) -> int:
  return int(round(600.0 + 1400.0 * (segment_length_m - 2.0)))


def compute_segment_total_and_diameter(
    total_mileage_m: float, detected_joint_count: int
) -> tuple[float, float, int, bool, float, int]:
  if detected_joint_count <= 0:
    raise ValueError("detected_joint_count must be positive")
  raw_segment = total_mileage_m / detected_joint_count
  segment, used_default = quantize_segment_length(raw_segment)
  refined_count = max(1, int(round(total_mileage_m / segment)))
  total = segment * refined_count
  diameter = compute_default_diameter_dn(segment)
  return segment, total, diameter, used_default, raw_segment, refined_count


def build_inspection_report(
    joint_count: int,
    csv_path: str | Path,
    video_path: str | Path | None = None,
    mileage_info: CsvMileageInfo | None = None,
) -> InspectionReport:
  info = mileage_info or read_mileage_from_csv(csv_path)
  detected_count = joint_count

  if video_path is not None:
    detection_date, detection_time = detection_timestamp_from_video(video_path)
  else:
    detection_date, detection_time = "N/A", "N/A"

  segment_length: float | None = None
  total_length: float | None = None
  diameter_dn: int | None = None
  used_default = False
  raw_segment: float | None = None
  refined_count = detected_count

  if detected_count > 0 and info.total_mileage_m > 0:
    segment_length, total_length, diameter_dn, used_default, raw_segment, refined_count = (
        compute_segment_total_and_diameter(info.total_mileage_m, detected_count)
    )
  elif detected_count > 0:
    segment_length = DEFAULT_SEGMENT_LENGTH_M
    total_length = segment_length * detected_count
    diameter_dn = compute_default_diameter_dn(segment_length)
    used_default = True

  return InspectionReport(
      detection_date=detection_date,
      detection_time=detection_time,
      joint_count=refined_count,
      segment_length_m=segment_length,
      total_pipe_length_m=total_length,
      segment_diameter_dn=diameter_dn,
      csv_total_mileage_m=info.total_mileage_m,
      csv_path=str(csv_path),
      used_default_segment_length=used_default,
      detected_joint_count=detected_count,
      raw_segment_length_m=raw_segment,
      mileage_column_index=info.mileage_column_index,
  )


def format_inspection_calculation_lines(report: InspectionReport) -> list[str]:
  """输出里程校正计算过程（record 50）。"""
  if (
      report.detected_joint_count is None
      or report.csv_total_mileage_m is None
      or report.detected_joint_count <= 0
      or report.csv_total_mileage_m <= 0
      or report.segment_length_m is None
  ):
    return []

  detected = report.detected_joint_count
  total_m = report.csv_total_mileage_m
  raw = report.raw_segment_length_m if report.raw_segment_length_m is not None else total_m / detected
  segment = report.segment_length_m
  refined = report.joint_count
  floored = int(math.floor(raw))

  if raw < 2.0:
    segment_line = (
        f"{_fmt_num(total_m)}/{detected}={_fmt_num(raw)}，小于 2 m，单管节长度按 {_fmt_num(segment)} m 处理"
    )
  elif 2.0 <= raw < 2.5 and segment == 2.0:
    segment_line = (
        f"{_fmt_num(total_m)}/{detected}={_fmt_num(raw)}，向下取整为 {floored} m，"
        f"单管节长度取 {_fmt_num(segment)} m"
    )
  else:
    segment_line = (
        f"{_fmt_num(total_m)}/{detected}={_fmt_num(raw)}，单管节长度取 {_fmt_num(segment)} m"
    )

  lines = [
      f"本次检测出 {detected} 个管节，总里程 {_fmt_num(total_m)} m",
      segment_line,
      f"{_fmt_num(total_m)}/{_fmt_num(segment)}={refined}，实际管节数 {refined}",
  ]
  if report.mileage_column_index is not None:
    lines.insert(1, f"CSV 总里程取自第 {report.mileage_column_index + 1} 列")
  if report.used_default_segment_length:
    lines.append("说明: 计算单节长度小于 2 m，已按 2 m 处理")
  return lines


def format_inspection_result_lines(report: InspectionReport) -> list[str]:
  """完整检测结果，供日志、控制台、弹窗与 txt 报告统一输出（record 54）。"""
  lines = [
      f"检测日期: {report.detection_date}",
      f"检测时间: {report.detection_time}",
  ]
  lines.extend(format_inspection_calculation_lines(report))
  if (
      report.detected_joint_count is not None
      and report.joint_count != report.detected_joint_count
  ):
    lines.append(f"检测识别管节数: {report.detected_joint_count}")
  lines.extend([
      f"管节总长度: {_fmt_m(report.total_pipe_length_m)}",
      f"管节总数: {report.joint_count}",
      f"单管节长度: {_fmt_m(report.segment_length_m)}",
      f"单管节管径: {_fmt_dn(report.segment_diameter_dn)}",
  ])
  return lines


def save_inspection_report(report: InspectionReport, output_path: str | Path) -> Path:
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  lines = format_inspection_result_lines(report)
  with path.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return path


def save_report_json(report: InspectionReport, output_path: str | Path) -> Path:
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
  return path


def _fmt_num(value: float) -> str:
  if abs(value - round(value)) < 1e-6:
    return str(int(round(value)))
  return f"{value:.2f}"


def _fmt_m(value: float | None) -> str:
  if value is None:
    return "N/A"
  if abs(value - round(value)) < 1e-6:
    return f"{int(round(value))} m"
  return f"{value:.1f} m"


def _fmt_dn(value: int | None) -> str:
  if value is None:
    return "N/A"
  return f"DN{value}"
