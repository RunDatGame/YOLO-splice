"""管节检测：环状区域检测 + 可选 YOLO。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class JointDetection:
  x: int
  y: int
  radius: float
  confidence: float
  bbox: tuple[int, int, int, int] | None = None
  ring_thickness: float = 0.0
  contrast: float = 0.0
  completeness: float = 0.0

  @property
  def center(self) -> tuple[int, int]:
    return self.x, self.y


@dataclass
class DetectorParams:
  num_angles: int = 36
  min_radius_ratio: float = 0.10
  max_radius_ratio: float = 0.58
  gradient_threshold: float = 0.30
  min_confidence: float = 0.30
  min_contrast: float = 0.08
  min_completeness: float = 0.45
  ring_thickness_ratio: float = 0.07
  max_detections: int = 1
  max_candidates_per_frame: int = 16
  min_score_floor: float = 0.16
  hough_param2: int = 28
  max_center_offset_x_ratio: float = 0.10
  max_center_offset_y_ratio: float = 0.30


class BaseJointDetector(ABC):
  @abstractmethod
  def detect(self, frame: np.ndarray) -> List[JointDetection]:
    candidates = self.detect_candidates(frame)
    filtered = [
        det
        for det in candidates
        if det.confidence >= self.params.min_confidence
        and det.contrast >= self.params.min_contrast
        and det.completeness >= self.params.min_completeness
    ]
    filtered.sort(key=lambda d: d.confidence, reverse=True)
    return self._nms_rings(filtered)[: self.params.max_detections]

  def detect_candidates(self, frame: np.ndarray) -> List[JointDetection]:
    """返回所有环状候选（供标定阶段复用，避免重复计算）。"""
    ...

  def apply_params(self, params: DetectorParams) -> None:
    pass


class AnnularRingDetector(BaseJointDetector):
  """检测偏心环状管节：基于环形区域边缘、对比度与完整性评分。"""

  def __init__(self, params: DetectorParams | None = None) -> None:
    self.params = params or DetectorParams()

  def apply_params(self, params: DetectorParams) -> None:
    self.params = params

  def detect(self, frame: np.ndarray) -> List[JointDetection]:
    return self._filter_and_nms(self.detect_candidates(frame))

  def detect_candidates(self, frame: np.ndarray) -> List[JointDetection]:
    """返回全部环状候选（标定阶段复用，避免重复计算）。"""
    h, w = frame.shape[:2]
    short_side = min(h, w)
    min_r = max(8, int(short_side * self.params.min_radius_ratio))
    max_r = int(short_side * self.params.max_radius_ratio)
    if max_r <= min_r:
      return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    candidates = self._collect_hough_candidates(edges, min_r, max_r)
    if len(candidates) < 8:
      candidates.extend(
          self._collect_gradient_candidates(gray, grad_mag, min_r, max_r, h, w)
      )

    candidates = self._dedupe_raw_candidates(candidates)[:30]
    candidates = self._filter_raw_by_center(candidates, w, h)

    scored: list[JointDetection] = []
    seen: set[tuple[int, int, int]] = set()
    for cx, cy, radius in candidates:
      if not self._is_center_in_range(cx, cy, w, h):
        continue
      key = (cx // 12, cy // 12, radius // 12)
      if key in seen:
        continue
      seen.add(key)

      det = self._score_annular_ring(gray, edges, grad_mag, cx, cy, radius, h, w)
      if det is not None and self._is_center_in_range(det.x, det.y, w, h):
        scored.append(det)

    return self._limit_candidates(scored, self.params.max_candidates_per_frame)

  def _is_center_in_range(self, cx: int, cy: int, w: int, h: int) -> bool:
    """管节环心相对图像中心：水平偏差 ≤ 宽度10%，垂直偏差 ≤ 高度30%。"""
    img_cx, img_cy = w / 2.0, h / 2.0
    dx = abs(cx - img_cx) / max(w, 1)
    dy = abs(cy - img_cy) / max(h, 1)
    return (
        dx <= self.params.max_center_offset_x_ratio
        and dy <= self.params.max_center_offset_y_ratio
    )

  def _filter_raw_by_center(
      self, raw: list[tuple[int, int, int]], w: int, h: int
  ) -> list[tuple[int, int, int]]:
    return [(cx, cy, r) for cx, cy, r in raw if self._is_center_in_range(cx, cy, w, h)]

  @staticmethod
  def _dedupe_raw_candidates(
      raw: list[tuple[int, int, int]],
  ) -> list[tuple[int, int, int]]:
    kept: list[tuple[int, int, int]] = []
    for cx, cy, r in raw:
      duplicate = False
      for ex, ey, er in kept:
        if np.hypot(cx - ex, cy - ey) < max(r, er) * 0.2 and abs(r - er) < max(r, er) * 0.2:
          duplicate = True
          break
      if not duplicate:
        kept.append((cx, cy, r))
    return kept

  @staticmethod
  def _limit_candidates(
      scored: list[JointDetection], max_count: int
  ) -> list[JointDetection]:
    if not scored:
      return []
    ordered = sorted(scored, key=lambda d: d.confidence, reverse=True)
    kept: list[JointDetection] = []
    for det in ordered:
      duplicate = False
      for existing in kept:
        dist = np.hypot(det.x - existing.x, det.y - existing.y)
        radius_tol = max(det.radius, existing.radius) * 0.22
        if dist < radius_tol and abs(det.radius - existing.radius) < radius_tol:
          duplicate = True
          break
      if not duplicate:
        kept.append(det)
      if len(kept) >= max_count:
        break
    return kept

  @staticmethod
  def filter_candidates(
      candidates: list[JointDetection],
      params: DetectorParams,
      max_detections: int = 3,
      frame_shape: tuple[int, int] | None = None,
  ) -> List[JointDetection]:
    if frame_shape is not None:
      h, w = frame_shape
      img_cx, img_cy = w / 2.0, h / 2.0
      candidates = [
          det
          for det in candidates
          if abs(det.x - img_cx) / max(w, 1) <= params.max_center_offset_x_ratio
          and abs(det.y - img_cy) / max(h, 1) <= params.max_center_offset_y_ratio
      ]
    filtered = [
        det
        for det in candidates
        if det.confidence >= params.min_confidence
        and det.contrast >= params.min_contrast
        and det.completeness >= params.min_completeness
    ]
    filtered.sort(key=lambda d: d.confidence, reverse=True)
    kept: list[JointDetection] = []
    for det in filtered:
      duplicate = False
      for existing in kept:
        dist = np.hypot(det.x - existing.x, det.y - existing.y)
        radius_tol = max(det.radius, existing.radius) * 0.25
        if dist < radius_tol and abs(det.radius - existing.radius) < radius_tol:
          duplicate = True
          break
      if not duplicate:
        kept.append(det)
    return kept[:max_detections]

  def _filter_and_nms(self, scored: list[JointDetection]) -> List[JointDetection]:
    return self.filter_candidates(scored, self.params, self.params.max_detections)

  def _collect_hough_candidates(
      self, edges: np.ndarray, min_r: int, max_r: int
  ) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=max(24, min_r // 2),
        param1=100,
        param2=self.params.hough_param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
      return candidates
    for circle in circles[0]:
      x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
      candidates.append((x, y, r))
    return candidates

  def _collect_gradient_candidates(
      self,
      gray: np.ndarray,
      grad_mag: np.ndarray,
      min_r: int,
      max_r: int,
      h: int,
      w: int,
  ) -> list[tuple[int, int, int]]:
    """在少量偏心锚点做径向扫描，补充 Hough 未覆盖的环。"""
    candidates: list[tuple[int, int, int]] = []
    anchors: list[tuple[int, int]] = [(w // 2, h // 2)]
    for fx, fy in ((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)):
      anchors.append((int(w * fx), int(h * fy)))

    angles = np.linspace(0, 2 * np.pi, self.params.num_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    r_step = 2

    for cx, cy in anchors:
      radius_votes: dict[int, list[float]] = {}
      for ai in range(len(angles)):
        profile: list[float] = []
        radii: list[int] = []
        for r in range(min_r, max_r, r_step):
          x = int(cx + r * cos_a[ai])
          y = int(cy + r * sin_a[ai])
          if 0 <= x < w and 0 <= y < h:
            profile.append(float(grad_mag[y, x]))
            radii.append(r)
        if len(profile) < 5:
          continue
        arr = np.array(profile, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        gradient = np.abs(np.diff(arr))
        if gradient.size == 0:
          continue
        peak_idx = int(np.argmax(gradient))
        peak_val = float(gradient[peak_idx])
        if peak_val >= 0.18:
          radius_votes.setdefault(radii[peak_idx], []).append(peak_val)

      if not radius_votes:
        continue
      best_r = max(radius_votes, key=lambda r: np.mean(radius_votes[r]))
      candidates.append((cx, cy, best_r))
    return candidates

  def _score_annular_ring(
      self,
      gray: np.ndarray,
      edges: np.ndarray,
      grad_mag: np.ndarray,
      cx: int,
      cy: int,
      radius: int,
      h: int,
      w: int,
  ) -> JointDetection | None:
    thickness = max(3, int(radius * self.params.ring_thickness_ratio))
    n_samples = self.params.num_angles
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)

    edge_hits = 0
    grad_vals: list[float] = []
    ring_vals: list[float] = []
    inner_vals: list[float] = []
    outer_vals: list[float] = []

    for angle in angles:
      cos_a, sin_a = np.cos(angle), np.sin(angle)
      x = int(cx + radius * cos_a)
      y = int(cy + radius * sin_a)
      if not (0 <= x < w and 0 <= y < h):
        continue
      ring_vals.append(float(gray[y, x]))
      grad_vals.append(float(grad_mag[y, x]))
      if edges[y, x] > 0:
        edge_hits += 1

      xi = int(cx + max(1, radius - thickness) * cos_a)
      yi = int(cy + max(1, radius - thickness) * sin_a)
      if 0 <= xi < w and 0 <= yi < h:
        inner_vals.append(float(gray[yi, xi]))

      xo = int(cx + (radius + thickness) * cos_a)
      yo = int(cy + (radius + thickness) * sin_a)
      if 0 <= xo < w and 0 <= yo < h:
        outer_vals.append(float(gray[yo, xo]))

    if not ring_vals:
      return None

    completeness = edge_hits / n_samples
    ring_std = float(np.std(ring_vals))
    inner_mean = float(np.mean(inner_vals)) if inner_vals else float(np.mean(ring_vals))
    outer_mean = float(np.mean(outer_vals)) if outer_vals else float(np.mean(ring_vals))
    contrast = (
        abs(inner_mean - outer_mean) / 255.0 * 0.6 + min(ring_std / 64.0, 1.0) * 0.4
    )
    grad_norm = float(np.mean(grad_vals))
    grad_score = min(grad_norm / 80.0, 1.0)

    confidence = (
        0.30 * completeness + 0.35 * min(contrast, 1.0) + 0.35 * grad_score
    )
    if confidence < self.params.min_score_floor:
      return None

    bbox = self._bbox_from_annulus(cx, cy, radius, thickness, w, h)
    return JointDetection(
        x=cx,
        y=cy,
        radius=float(radius),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        bbox=bbox,
        ring_thickness=float(thickness),
        contrast=float(contrast),
        completeness=float(completeness),
    )

  def _nms_rings(self, detections: list[JointDetection]) -> list[JointDetection]:
    kept: list[JointDetection] = []
    for det in detections:
      duplicate = False
      for existing in kept:
        dist = np.hypot(det.x - existing.x, det.y - existing.y)
        radius_tol = max(det.radius, existing.radius) * 0.25
        if dist < radius_tol and abs(det.radius - existing.radius) < radius_tol:
          duplicate = True
          break
      if not duplicate:
        kept.append(det)
    return kept

  @staticmethod
  def _bbox_from_annulus(
      cx: int, cy: int, radius: int, thickness: int, w: int, h: int
  ) -> tuple[int, int, int, int]:
    outer = radius + thickness
    x1 = max(0, cx - outer)
    y1 = max(0, cy - outer)
    x2 = min(w - 1, cx + outer)
    y2 = min(h - 1, cy + outer)
    return x1, y1, x2, y2


# 向后兼容别名
TraditionalJointDetector = AnnularRingDetector


class YoloJointDetector(BaseJointDetector):
  """YOLOv8 管节检测（需提供训练权重）。"""

  def __init__(
      self,
      weights_path: str | Path,
      conf_threshold: float = 0.35,
      device: str = "cpu",
  ) -> None:
    from ultralytics import YOLO

    self.model = YOLO(str(weights_path))
    self.conf_threshold = conf_threshold
    self.device = device

  def detect(self, frame: np.ndarray) -> List[JointDetection]:
    results = self.model.predict(
        frame, conf=self.conf_threshold, device=self.device, verbose=False
    )
    detections: list[JointDetection] = []
    for result in results:
      if result.boxes is None:
        continue
      for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = max(x2 - x1, y2 - y1) / 2
        detections.append(
            JointDetection(
                x=cx,
                y=cy,
                radius=float(radius),
                confidence=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return detections


def create_detector(
    mode: str = "traditional",
    yolo_weights: Optional[str | Path] = None,
    device: str = "cpu",
    params: DetectorParams | None = None,
    **kwargs,
) -> BaseJointDetector:
  if mode == "yolo":
    if not yolo_weights or not Path(yolo_weights).exists():
      raise FileNotFoundError(f"YOLO 模式需要有效权重文件: {yolo_weights}")
    return YoloJointDetector(weights_path=yolo_weights, device=device, **kwargs)
  return AnnularRingDetector(params=params)
