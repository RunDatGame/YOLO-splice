"""圆环数量曲线与包络线 CSV 导出。"""

from __future__ import annotations

import csv
from pathlib import Path

from .counter import CounterResult


def curve_csv_path(output_json_path: str | Path) -> Path:
  """与 joint_count.json 同目录的曲线 CSV 路径。"""
  path = Path(output_json_path)
  if path.name.endswith(".joint_count.json"):
    name = path.name.replace(".joint_count.json", ".curve.csv")
    return path.with_name(name)
  return path.with_suffix(".curve.csv")


def save_curve_csv(
    result: CounterResult,
    output_json_path: str | Path,
) -> Path | None:
  """
  导出移动帧曲线：frame, raw_ring_count, envelope, is_joint_peak。
  文件自动保存到与 output_json_path 相同目录。
  """
  if not result.moving_frames or not result.moving_ring_counts:
    return None

  out_path = curve_csv_path(output_json_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)

  envelope = result.smoothed_ring_counts
  if not envelope or len(envelope) != len(result.moving_frames):
    envelope = [float(c) for c in result.moving_ring_counts]

  joint_set = set(result.joint_peak_frames)
  period = result.estimated_period_samples

  with out_path.open("w", newline="", encoding="utf-8-sig") as f:
    if period > 0:
      f.write(f"# estimated_period_samples={period}\n")
    writer = csv.writer(f)
    writer.writerow(
        ["sample_index", "frame", "raw_ring_count", "envelope", "is_joint_peak"]
    )
    for i, (frame, raw) in enumerate(
        zip(result.moving_frames, result.moving_ring_counts)
    ):
      env_val = envelope[i] if i < len(envelope) else float(raw)
      writer.writerow(
          [
              i,
              frame,
              raw,
              round(float(env_val), 6),
              1 if frame in joint_set else 0,
          ]
      )

  return out_path
