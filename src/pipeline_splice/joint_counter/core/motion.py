"""光流运动分析：区分前进通过与旋转停留。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class MotionState(str, Enum):
  FORWARD = "forward"
  ROTATION = "rotation"
  STATIC = "static"
  UNKNOWN = "unknown"


@dataclass
class MotionMetrics:
  translation: float
  rotation: float
  state: MotionState


class MotionAnalyzer:
  """基于 Farneback 光流的运动模式识别。"""

  def __init__(
      self,
      forward_threshold: float = 1.2,
      rotation_threshold: float = 0.8,
      static_threshold: float = 0.3,
  ) -> None:
    self.forward_threshold = forward_threshold
    self.rotation_threshold = rotation_threshold
    self.static_threshold = static_threshold
    self._prev_gray: np.ndarray | None = None

  def reset(self) -> None:
    self._prev_gray = None

  def analyze(self, frame: np.ndarray) -> MotionMetrics:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if self._prev_gray is None:
      self._prev_gray = gray
      return MotionMetrics(0.0, 0.0, MotionState.UNKNOWN)

    flow = cv2.calcOpticalFlowFarneback(
        self._prev_gray,
        gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    self._prev_gray = gray

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    translation = float(np.mean(magnitude))
    rotation = self._estimate_rotation_strength(flow, frame.shape)

    state = self._classify(translation, rotation)
    return MotionMetrics(translation=translation, rotation=rotation, state=state)

  def _estimate_rotation_strength(
      self, flow: np.ndarray, shape: tuple[int, ...]
  ) -> float:
    h, w = shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    rx = x_coords - cx
    ry = y_coords - cy
    norm = np.sqrt(rx**2 + ry**2) + 1e-6

    tangential_x = -ry / norm
    tangential_y = rx / norm
    tangential_flow = flow[..., 0] * tangential_x + flow[..., 1] * tangential_y
    radial_flow = flow[..., 0] * (rx / norm) + flow[..., 1] * (ry / norm)

    tangential_mean = float(np.mean(np.abs(tangential_flow)))
    radial_mean = float(np.mean(np.abs(radial_flow)))
    if radial_mean < 1e-6:
      return tangential_mean
    return tangential_mean / (radial_mean + 1e-6)

  def _classify(self, translation: float, rotation: float) -> MotionState:
    if translation < self.static_threshold:
      return MotionState.STATIC
    if rotation >= self.rotation_threshold and translation < self.forward_threshold * 1.5:
      return MotionState.ROTATION
    if translation >= self.forward_threshold:
      return MotionState.FORWARD
    return MotionState.UNKNOWN
