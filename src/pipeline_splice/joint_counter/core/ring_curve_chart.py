"""移动帧圆环数量曲线绘制（ASCII 标签，自适应画布尺寸）。"""

from __future__ import annotations

import cv2
import numpy as np


def render_ring_count_curve(
    moving_frames: list[int],
    moving_counts: list[int],
    joint_frames: list[int] | None = None,
    smoothed_counts: list[float] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
  """绘制曲线；width/height 为 None 时按数据长度估算最小尺寸。"""
  n_pts = len(moving_frames) if moving_frames else 0
  canvas_w = max(320, int(width)) if width and width > 1 else max(480, min(1200, 48 + n_pts * 8))
  canvas_h = max(120, int(height)) if height and height > 1 else 220

  canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)

  if not moving_frames or not moving_counts:
    cv2.putText(
        canvas, "Waiting for data...", (16, canvas_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA,
    )
    return canvas

  margin_l, margin_r, margin_t, margin_b = 48, 12, 28, 32
  plot_w = max(40, canvas_w - margin_l - margin_r)
  plot_h = max(40, canvas_h - margin_t - margin_b)

  xs = np.array(moving_frames, dtype=np.float32)
  ys = np.array(moving_counts, dtype=np.float32)
  smooth = (
      np.array(smoothed_counts, dtype=np.float32)
      if smoothed_counts and len(smoothed_counts) == len(moving_counts)
      else ys
  )

  all_y = np.concatenate([ys, smooth])
  x_min, x_max = float(xs.min()), float(xs.max())
  y_min, y_max = float(all_y.min()), float(all_y.max())
  if x_max <= x_min:
    x_max = x_min + 1.0
  if y_max <= y_min:
    y_max = y_min + 1.0
  y_pad = max(1.0, (y_max - y_min) * 0.12)
  y_min = max(0.0, y_min - y_pad)
  y_max = y_max + y_pad

  def to_px(x: float, y: float) -> tuple[int, int]:
    px = int(margin_l + (x - x_min) / (x_max - x_min) * plot_w)
    py = int(margin_t + plot_h - (y - y_min) / (y_max - y_min) * plot_h)
    return px, py

  cv2.rectangle(
      canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h),
      (210, 210, 210), 1,
  )
  cv2.putText(
      canvas, "Ring Count Curve", (margin_l, 20),
      cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 50, 50), 1, cv2.LINE_AA,
  )
  cv2.putText(
      canvas, "Frame", (margin_l + plot_w // 2 - 22, canvas_h - 6),
      cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA,
  )
  cv2.putText(
      canvas, "Cnt", (6, margin_t + plot_h // 2 + 4),
      cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1, cv2.LINE_AA,
  )

  raw_points = [to_px(float(x), float(y)) for x, y in zip(xs, ys)]
  smooth_points = [to_px(float(x), float(y)) for x, y in zip(xs, smooth)]

  for i in range(1, len(raw_points)):
    cv2.line(canvas, raw_points[i - 1], raw_points[i], (200, 205, 215), 1, cv2.LINE_AA)
  for i in range(1, len(smooth_points)):
    cv2.line(canvas, smooth_points[i - 1], smooth_points[i], (60, 110, 210), 1, cv2.LINE_AA)

  joint_set = set(joint_frames or [])
  for x, y, frame in zip(xs, ys, moving_frames):
    px, py = to_px(float(x), float(y))
    if frame in joint_set:
      cv2.circle(canvas, (px, py), 3, (0, 190, 60), -1, cv2.LINE_AA)
    else:
      cv2.circle(canvas, (px, py), 1, (140, 150, 170), -1, cv2.LINE_AA)

  cv2.putText(
      canvas, "gray=raw blue=envelope green=joint",
      (margin_l, canvas_h - 6),
      cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 120, 40), 1, cv2.LINE_AA,
  )
  return canvas
