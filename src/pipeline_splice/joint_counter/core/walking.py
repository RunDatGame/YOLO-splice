"""行走状态检测：帧差 + 光流融合。"""

from __future__ import annotations

import cv2
import numpy as np

from .motion import MotionState


class WalkingAnalyzer:
  """通过帧间变化与光流判断是否在管道内行走。"""

  def __init__(
      self,
      diff_threshold: float = 5.0,
      frame_interval: int = 10,
  ) -> None:
    self.diff_threshold = diff_threshold
    self.frame_interval = max(1, frame_interval)
    self._prev_gray: np.ndarray | None = None
    self.walking_frames_since_joint = 0
    self.total_walking_frames = 0
    self._is_walking = False

  def reset(self) -> None:
    self._prev_gray = None
    self.walking_frames_since_joint = 0
    self.total_walking_frames = 0
    self._is_walking = False

  @property
  def is_walking(self) -> bool:
    return self._is_walking

  def update(self, frame: np.ndarray, motion_state: MotionState) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    frame_diff = 0.0
    if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
      frame_diff = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
    self._prev_gray = gray

    walking_by_diff = frame_diff >= self.diff_threshold
    walking_by_flow = motion_state == MotionState.FORWARD
    at_joint_stop = motion_state in (MotionState.ROTATION, MotionState.STATIC)

    self._is_walking = (walking_by_diff or walking_by_flow) and not (
        at_joint_stop and frame_diff < self.diff_threshold * 0.6
    )

    if self._is_walking:
      self.walking_frames_since_joint += self.frame_interval
      self.total_walking_frames += self.frame_interval

    return self._is_walking

  def mark_joint_passed(self) -> int:
    """记录通过一个接口，返回该接口前的行走帧数。"""
    gap = self.walking_frames_since_joint
    self.walking_frames_since_joint = 0
    return gap
