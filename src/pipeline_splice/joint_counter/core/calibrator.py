"""检测参数与视频预扫描标定。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import AnnularRingDetector, DetectorParams, JointDetection
from .spacing import JointEvent, estimate_joint_count_from_events, enforce_joint_count_range
from .video_bounds import DEFAULT_TAIL_TRIM_FRAMES, effective_frame_limit


@dataclass
class CalibrationResult:
  params: DetectorParams
  estimated_joints: int
  sampled_frames: int
  detection_events: int
  score: float = 0.0
  median_walking_gap: float = 0.0
  total_candidates: int = 0


@dataclass
class CalibrationConfig:
  frame_interval: int = 20
  scale_factor: float = 0.35
  expected_min: int = 6
  expected_max: int = 50
  merge_radius_ratio: float = 0.12
  merge_frame_gap: int = 60
  walking_diff_threshold: float = 5.0
  max_candidates_per_frame: int = 12
  max_total_candidates: int = 500
  tail_trim_frames: int = DEFAULT_TAIL_TRIM_FRAMES


class ThresholdCalibrator:
  """快速预扫描视频，结合行走帧与等间隔先验搜索检测阈值。"""

  GRADIENT_THRESHOLDS = (0.24, 0.32, 0.40)
  MIN_CONFIDENCES = (0.32, 0.40, 0.48)
  MIN_CONTRASTS = (0.08, 0.11, 0.14)
  MIN_COMPLETENESS = (0.42, 0.52, 0.62)

  def __init__(self, config: CalibrationConfig | None = None) -> None:
    self.config = config or CalibrationConfig()

  def calibrate(
      self,
      video_path: str | Path,
      on_progress: Callable[[int, int, str], None] | None = None,
      on_sample: Callable[[int, np.ndarray, list[JointDetection], int], None] | None = None,
  ) -> CalibrationResult:
    samples = self._sample_frames(video_path, on_progress)
    if not samples:
      return CalibrationResult(
          params=DetectorParams(),
          estimated_joints=0,
          sampled_frames=0,
          detection_events=0,
      )

    probe_params = DetectorParams(
        max_candidates_per_frame=self.config.max_candidates_per_frame,
        num_angles=36,
    )
    probe = AnnularRingDetector(probe_params)
    cached: list[tuple[int, list[JointDetection], bool, tuple[int, int]]] = []
    prev_gray: np.ndarray | None = None
    total_candidates = 0

    for i, (frame_idx, frame) in enumerate(samples):
      if on_progress:
        on_progress(i + 1, len(samples), "提取环状候选")

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (5, 5), 0)
      frame_diff = 0.0
      if prev_gray is not None and prev_gray.shape == gray.shape:
        frame_diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
      prev_gray = gray

      is_walking = frame_diff >= self.config.walking_diff_threshold
      candidates = probe.detect_candidates(frame)
      total_candidates += len(candidates)
      cached.append((frame_idx, candidates, is_walking, frame.shape[:2]))

      if on_sample is not None:
        on_sample(frame_idx, frame, candidates, total_candidates)

    best: CalibrationResult | None = None
    total_combos = (
        len(self.GRADIENT_THRESHOLDS)
        * len(self.MIN_CONFIDENCES)
        * len(self.MIN_CONTRASTS)
        * len(self.MIN_COMPLETENESS)
    )
    combo_idx = 0

    for grad in self.GRADIENT_THRESHOLDS:
      for conf in self.MIN_CONFIDENCES:
        for contrast in self.MIN_CONTRASTS:
          for completeness in self.MIN_COMPLETENESS:
            combo_idx += 1
            if on_progress:
              on_progress(combo_idx, total_combos, "搜索阈值组合")

            params = DetectorParams(
                gradient_threshold=grad,
                min_confidence=conf,
                min_contrast=contrast,
                min_completeness=completeness,
                max_candidates_per_frame=self.config.max_candidates_per_frame,
                num_angles=36,
            )
            events = self._build_events_from_samples(cached, params)
            estimated = estimate_joint_count_from_events(
                events, self.config.expected_min, self.config.expected_max
            )
            events = enforce_joint_count_range(
                events, self.config.expected_min, self.config.expected_max
            )
            gaps = [e.walking_frames_since_prev for e in events[1:] if e.walking_frames_since_prev > 0]
            median_gap = float(np.median(gaps)) if gaps else 0.0
            raw_events = sum(
                1
                for _, candidates, walking, shape in cached
                if walking
                and AnnularRingDetector.filter_candidates(
                    candidates, params, frame_shape=shape
                )
            )

            target_mid = (self.config.expected_min + self.config.expected_max) / 2
            if self.config.expected_min <= estimated <= self.config.expected_max:
              distance = abs(estimated - target_mid)
              score = 2000.0 - distance * 10.0
            else:
              distance = min(
                  abs(estimated - self.config.expected_min),
                  abs(estimated - self.config.expected_max),
              )
              score = max(0.0, 800.0 - distance * 30.0)

            score -= max(0, raw_events - estimated * 3) * 5.0
            score -= max(0, total_candidates - self.config.max_total_candidates) * 2.0

            result = CalibrationResult(
                params=params,
                estimated_joints=estimated,
                sampled_frames=len(samples),
                detection_events=raw_events,
                score=score,
                median_walking_gap=median_gap,
                total_candidates=total_candidates,
            )
            if best is None or result.score > best.score:
              best = result

    assert best is not None
    return best

  def _build_events_from_samples(
      self,
      cached: list[tuple[int, list[JointDetection], bool]],
      params: DetectorParams,
  ) -> list[JointEvent]:
    events: list[JointEvent] = []
    walking_since_prev = 0
    last_frame = -99999

    for frame_idx, candidates, is_walking, shape in cached:
      if is_walking:
        walking_since_prev += self.config.frame_interval

      dets = AnnularRingDetector.filter_candidates(
          candidates, params, frame_shape=shape
      )
      if not dets or not is_walking:
        continue
      if frame_idx - last_frame < self.config.merge_frame_gap // 2:
        continue

      best = max(dets, key=lambda d: d.confidence)
      events.append(
          JointEvent(
              frame_idx=frame_idx,
              confidence=best.confidence,
              track_id=len(events) + 1,
              walking_frames_since_prev=walking_since_prev if events else 0,
          )
      )
      walking_since_prev = 0
      last_frame = frame_idx

    return events

  def _sample_frames(
      self,
      video_path: str | Path,
      on_progress: Callable[[int, int, str], None] | None,
  ) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise FileNotFoundError(f"无法打开视频: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_limit = effective_frame_limit(total, self.config.tail_trim_frames)
    samples: list[tuple[int, np.ndarray]] = []
    frame_idx = 0

    try:
      while frame_idx < frame_limit:
        if frame_idx % self.config.frame_interval == 0:
          ret, frame = cap.read()
          if not ret:
            break
          working = self._resize(frame, self.config.scale_factor)
          samples.append((frame_idx, working))
          if on_progress and total > 0:
            on_progress(frame_idx, frame_limit, "预扫描采样")
        else:
          if not cap.grab():
            break
        frame_idx += 1
    finally:
      cap.release()

    return samples

  @staticmethod
  def _resize(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0 or scale >= 1.0:
      return frame
    h, w = frame.shape[:2]
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
