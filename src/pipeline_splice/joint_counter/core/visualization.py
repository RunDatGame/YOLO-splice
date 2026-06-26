"""环状候选提取步骤可视化。"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import JointDetection
from .text_overlay import put_text


def _is_selected(det: JointDetection, selected: list[JointDetection]) -> bool:
  for s in selected:
    dist = np.hypot(det.x - s.x, det.y - s.y)
    tol = max(det.radius, s.radius) * 0.2
    if dist < tol and abs(det.radius - s.radius) < tol:
      return True
  return False


def render_combined_display(
    frame: np.ndarray,
    candidates: list[JointDetection],
    selected: list[JointDetection],
    joint_count: int,
    motion_state: str,
    frame_idx: int,
    max_draw: int = 10,
    joint_ready: bool = False,
) -> np.ndarray:
  """在同一视频帧上叠加环状候选与管节计数信息。"""
  vis = frame.copy()
  h, w = vis.shape[:2]
  selected = selected or []
  total = len(candidates)
  draw_list = sorted(candidates, key=lambda d: d.confidence, reverse=True)[:max_draw]

  for i, det in enumerate(draw_list):
    thickness = max(2, int(det.ring_thickness or det.radius * 0.07))
    inner_r = max(1, int(det.radius - thickness))
    outer_r = int(det.radius + thickness)
    picked = _is_selected(det, selected)

    if picked:
      ring_color = (0, 255, 80)
      center_color = (0, 255, 255)
    else:
      ring_color = (0, 140, 255)
      center_color = (0, 180, 255)

    cv2.circle(vis, (det.x, det.y), inner_r, ring_color, 2)
    cv2.circle(vis, (det.x, det.y), outer_r, ring_color, 2)
    cv2.circle(vis, (det.x, det.y), 4, center_color, -1)

    if i < 8:
      label = f"#{i + 1} {det.confidence:.2f}"
      cv2.putText(
          vis,
          label,
          (max(5, det.x - 30), max(18, det.y - outer_r - 6)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.42,
          ring_color,
          1,
          cv2.LINE_AA,
      )

  overlay = vis.copy()
  cv2.rectangle(overlay, (8, 8), (min(w - 8, 520), 148), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
  put_text(vis, f"管节段数: {joint_count}", (18, 42), 32, (0, 255, 0), stroke_width=1, stroke_color_bgr=(0, 80, 0))
  put_text(vis, f"Frame: {frame_idx}  Motion: {motion_state}", (18, 72), 19, (255, 255, 255))
  joint_tag = "是" if joint_ready else "否"
  joint_color = (0, 255, 120) if joint_ready else (0, 140, 255)
  put_text(
      vis,
      f"候选: {total} (显示前{len(draw_list)})  已选: {len(selected)}  接口峰: {joint_tag}",
      (18, 102),
      16,
      joint_color,
  )
  put_text(vis, "绿色=已选  橙色=候选  (峰值=接口)", (18, 128), 14, (160, 160, 160))
  return vis


def render_ring_candidates(
    frame: np.ndarray,
    candidates: list[JointDetection],
    selected: list[JointDetection] | None = None,
    frame_idx: int = 0,
    title: str = "Ring Candidates",
) -> np.ndarray:
  """绘制环状候选：绿色=通过阈值，橙色=未通过，标注置信度等指标。"""
  vis = frame.copy()
  h, w = vis.shape[:2]
  selected = selected or []

  for i, det in enumerate(candidates):
    thickness = max(2, int(det.ring_thickness or det.radius * 0.07))
    inner_r = max(1, int(det.radius - thickness))
    outer_r = int(det.radius + thickness)
    picked = _is_selected(det, selected)

    if picked:
      ring_color = (0, 255, 80)
      center_color = (0, 255, 255)
    else:
      ring_color = (0, 140, 255)
      center_color = (0, 180, 255)

    cv2.circle(vis, (det.x, det.y), inner_r, ring_color, 2)
    cv2.circle(vis, (det.x, det.y), outer_r, ring_color, 2)
    cv2.circle(vis, (det.x, det.y), 4, center_color, -1)

    label = f"#{i + 1} c={det.confidence:.2f}"
    if det.contrast > 0 or det.completeness > 0:
      label += f" t={det.contrast:.2f} p={det.completeness:.2f}"
    cv2.putText(
        vis,
        label,
        (max(5, det.x - 40), max(20, det.y - outer_r - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        ring_color,
        1,
        cv2.LINE_AA,
    )

  overlay = vis.copy()
  box_h = 95 if selected else 75
  cv2.rectangle(overlay, (8, 8), (min(w - 8, 420), box_h), (0, 0, 0), -1)
  cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)
  cv2.putText(
      vis,
      title,
      (16, 32),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.75,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      vis,
      f"Frame {frame_idx}  Total: {len(candidates)}  Selected: {len(selected)}",
      (16, 58),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (200, 200, 200),
      1,
  )
  cv2.putText(
      vis,
      "Green=selected  Orange=candidate",
      (16, 82),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.48,
      (180, 180, 180),
      1,
  )
  return vis


def compose_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
  """左右拼接两路可视化（计数结果 | 环状候选）。"""
  h = max(left.shape[0], right.shape[0])
  w1, w2 = left.shape[1], right.shape[1]

  def _pad(img: np.ndarray) -> np.ndarray:
    if img.shape[0] == h and img.shape[1] == w1:
      return img
    out = np.zeros((h, img.shape[1], 3), dtype=np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

  left_p = _pad(left)
  right_p = _pad(right)
  sep = np.full((h, gap, 3), 40, dtype=np.uint8)
  return np.hstack([left_p, sep, right_p])


def compose_vertical(top: np.ndarray, bottom: np.ndarray, gap: int = 6) -> np.ndarray:
  """上下拼接两路可视化。"""
  w = max(top.shape[1], bottom.shape[1])

  def _pad(img: np.ndarray) -> np.ndarray:
    out = np.zeros((img.shape[0], w, 3), dtype=np.uint8)
    out[: img.shape[0], : img.shape[1]] = img
    return out

  sep = np.full((gap, w, 3), 40, dtype=np.uint8)
  return np.vstack([_pad(top), sep, _pad(bottom)])
