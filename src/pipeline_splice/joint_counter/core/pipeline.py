"""管节计数处理流水线。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional

import cv2
import numpy as np

from .calibrator import CalibrationConfig, CalibrationResult, ThresholdCalibrator
from .counter import CounterResult, JointCounter
from .detector import DetectorParams, JointDetection, AnnularRingDetector, create_detector
from .motion import MotionAnalyzer, MotionState
from .preprocessor import FramePreprocessor
from .csv_mileage import find_best_matching_csv
from .video_bounds import DEFAULT_TAIL_TRIM_FRAMES, effective_frame_limit, tail_trim_summary
from .tracker import InterfaceRingTracker
from .report import InspectionReport, build_inspection_report, save_inspection_report
from .ring_curve_chart import render_ring_count_curve
from .curve_csv import save_curve_csv
from .text_overlay import put_text
from .visualization import render_combined_display, render_ring_candidates


@dataclass
class PipelineConfig:
  detector_mode: str = "traditional"
  yolo_weights: Optional[str] = None
  csv_path: Optional[str] = None
  frame_interval: int = 10
  scale_factor: float = 0.35
  show_overlay: bool = True
  preprocess: bool = False
  device: str = "cpu"
  auto_calibrate: bool = True
  calibration_interval: int = 20
  expected_joints_min: int = 6
  expected_joints_max: int = 50
  show_candidates: bool = True
  min_rings_at_joint: int = 4
  min_consecutive_hits: int = 2
  max_interruption_frames: int = 30
  interface_segment_window: int = 120
  post_joint_walking_cooldown: int = 200
  watermark_frame_count: int = 30
  tail_trim_frames: int = DEFAULT_TAIL_TRIM_FRAMES


@dataclass
class FrameResult:
  frame_idx: int
  frame: np.ndarray
  detections: list[JointDetection]
  joint_count: int
  motion_state: MotionState
  fps: float = 0.0
  candidates_frame: np.ndarray | None = None
  candidate_count: int = 0
  selected_count: int = 0
  is_moving_frame: bool = True
  moving_frames: list[int] = field(default_factory=list)
  moving_ring_counts: list[int] = field(default_factory=list)
  joint_peak_frames: list[int] = field(default_factory=list)
  smoothed_ring_counts: list[float] = field(default_factory=list)
  detrended_ring_counts: list[float] = field(default_factory=list)
  curve_image: np.ndarray | None = None


class PipeJointPipeline:
  """视频管节计数主流程。"""

  def __init__(self, config: PipelineConfig | None = None) -> None:
    self.config = config or PipelineConfig()
    if self.config.frame_interval < 1:
      self.config.frame_interval = 1
    self.preprocessor = FramePreprocessor()
    self.detector = create_detector(
        mode=self.config.detector_mode,
        yolo_weights=self.config.yolo_weights,
        device=self.config.device,
    )
    self.motion = MotionAnalyzer()
    self.ring_tracker = InterfaceRingTracker(
        frame_interval=self.config.frame_interval,
        min_rings_at_joint=self.config.min_rings_at_joint,
        min_consecutive_hits=self.config.min_consecutive_hits,
        max_interruption_frames=self.config.max_interruption_frames,
        interface_segment_window=self.config.interface_segment_window,
    )
    self.counter = JointCounter(
        expected_min=self.config.expected_joints_min,
        expected_max=self.config.expected_joints_max,
        frame_interval=self.config.frame_interval,
    )
    self._stop_requested = False
    self.calibration_result: CalibrationResult | None = None

  @staticmethod
  def describe_video_bounds(
      video_path: str | Path, tail_trim_frames: int = DEFAULT_TAIL_TRIM_FRAMES
  ) -> tuple[int, int, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise FileNotFoundError(f"无法打开视频: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    limit = effective_frame_limit(total_frames, tail_trim_frames)
    return total_frames, limit, tail_trim_summary(total_frames, tail_trim_frames)

  def calibrate(
      self,
      video_path: str | Path,
      on_progress: Callable[[int, int, str], None] | None = None,
      on_sample: Callable[[int, np.ndarray, list[JointDetection], int], None] | None = None,
  ) -> CalibrationResult:
    if self.config.detector_mode != "traditional":
      self.calibration_result = CalibrationResult(
          params=DetectorParams(),
          estimated_joints=0,
          sampled_frames=0,
          detection_events=0,
      )
      return self.calibration_result

    calibrator = ThresholdCalibrator(
        CalibrationConfig(
            frame_interval=self.config.calibration_interval,
            scale_factor=self.config.scale_factor,
            expected_min=self.config.expected_joints_min,
            expected_max=self.config.expected_joints_max,
            merge_frame_gap=max(40, self.config.frame_interval * 4),
            tail_trim_frames=self.config.tail_trim_frames,
        )
    )
    self.calibration_result = calibrator.calibrate(
        video_path, on_progress, on_sample=on_sample
    )
    self.detector.apply_params(self.calibration_result.params)
    return self.calibration_result

  def stop(self) -> None:
    self._stop_requested = True

  def reset(self) -> None:
    self.motion.reset()
    self.ring_tracker.reset()
    self.counter.reset()
    self._stop_requested = False

  def process_video(
      self,
      video_path: str | Path,
      on_frame: Callable[[FrameResult], bool] | None = None,
  ) -> CounterResult:
    self.reset()
    if (
        self.config.auto_calibrate
        and self.config.detector_mode == "traditional"
        and self.calibration_result is None
    ):
      self.calibrate(video_path)

    csv_path = self._resolve_csv_path(video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = effective_frame_limit(total_frames, self.config.tail_trim_frames)
    frame_idx = 0
    processed = 0
    motion_history: list[str] = []
    t0 = time.perf_counter()

    try:
      while not self._stop_requested and frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
          break

        is_moving = self.counter.observe_motion(frame)

        if frame_idx % self.config.frame_interval == 0:
          result = self._process_frame(frame, frame_idx, fps, is_moving)
          motion_history.append(result.motion_state.value)
          processed += 1

          if on_frame is not None:
            should_continue = on_frame(result)
            if should_continue is False:
              break

        frame_idx += 1
    finally:
      cap.release()

    final = self.counter.finalize()
    final.frame_count = processed
    final.motion_history = motion_history
    final.total_video_frames = total_frames
    final.effective_frame_limit = frame_limit
    final.tail_trim_frames = self.config.tail_trim_frames
    elapsed = time.perf_counter() - t0
    if elapsed > 0:
      final.fps = processed / elapsed

    final.inspection_report = build_inspection_report(
        joint_count=final.count,
        csv_path=csv_path,
        video_path=video_path,
    )
    return final

  def _resolve_csv_path(self, video_path: Path) -> Path:
    if self.config.csv_path:
      csv_path = Path(self.config.csv_path)
    else:
      matched = find_best_matching_csv(video_path)
      if matched is None:
        csv_path = video_path.with_suffix(".csv")
      else:
        csv_path = matched
    if not csv_path.exists():
      raise FileNotFoundError(
          f"未找到 CSV 里程文件: {csv_path}，请通过 --csv 指定或在同目录放置匹配的 .csv"
      )
    return csv_path

  def iter_frames(
      self, video_path: str | Path
  ) -> Generator[FrameResult, None, CounterResult]:
    self.reset()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
      raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = effective_frame_limit(total_frames, self.config.tail_trim_frames)
    frame_idx = 0

    try:
      while not self._stop_requested and frame_idx < frame_limit:
        ret, frame = cap.read()
        if not ret:
          break
        is_moving = self.counter.observe_motion(frame)
        if frame_idx % self.config.frame_interval == 0:
          yield self._process_frame(frame, frame_idx, fps, is_moving)
        frame_idx += 1
    finally:
      cap.release()

    return self.counter.finalize()

  def _process_frame(
      self, frame: np.ndarray, frame_idx: int, fps: float, is_moving: bool
  ) -> FrameResult:
    working, scale = self._prepare_working_frame(frame)
    if self.config.preprocess:
      working = self.preprocessor.process(working)

    all_candidates: list[JointDetection] = []
    detections: list[JointDetection] = []

    all_candidates_working: list[JointDetection] = []
    if isinstance(self.detector, AnnularRingDetector):
      all_candidates_working = self.detector.detect_candidates(working)
    else:
      all_candidates_working = list(self.detector.detect(working))
    all_candidates = self._scale_detections(all_candidates_working, scale)
    self.ring_tracker.update(all_candidates, frame_idx, frame.shape)

    count = self.counter.count
    if is_moving:
      count = self.counter.record_sample(frame_idx, len(all_candidates))
    joint_peaks = self.counter.peak_counter.joint_frames
    if is_moving:
      detections = (
          self._select_peak_detections(all_candidates)
          if frame_idx in joint_peaks
          else []
      )

    motion = self.motion.analyze(working)
    curve = render_ring_count_curve(
        self.counter.peak_counter.moving_frames,
        self.counter.peak_counter.moving_counts,
        joint_peaks,
        smoothed_counts=self.counter.peak_counter.smoothed_counts,
    )

    display = frame.copy()
    candidates_vis: np.ndarray | None = None
    if self.config.show_candidates and is_moving:
      display = render_combined_display(
          frame,
          all_candidates,
          detections,
          count,
          motion.state.value,
          frame_idx,
          joint_ready=frame_idx in joint_peaks,
      )
      candidates_vis = display.copy()
    elif self.config.show_overlay and is_moving:
      display = self._draw_overlay(display, detections, count, motion.state, frame_idx)
    elif not is_moving:
      display = self._draw_static_overlay(display, frame_idx, count)

    return FrameResult(
        frame_idx=frame_idx,
        frame=display,
        detections=detections,
        joint_count=count,
        motion_state=motion.state,
        fps=fps,
        candidates_frame=candidates_vis,
        candidate_count=len(all_candidates),
        selected_count=len(detections),
        is_moving_frame=is_moving,
        moving_frames=list(self.counter.peak_counter.moving_frames),
        moving_ring_counts=list(self.counter.peak_counter.moving_counts),
        joint_peak_frames=list(joint_peaks),
        smoothed_ring_counts=list(self.counter.peak_counter.smoothed_counts),
        detrended_ring_counts=list(self.counter.peak_counter.detrended_counts),
        curve_image=curve,
    )

  @staticmethod
  def _select_peak_detections(candidates: list[JointDetection]) -> list[JointDetection]:
    if not candidates:
      return []
    best = max(candidates, key=lambda d: d.confidence)
    return [best]

  def _prepare_working_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
    scale = self.config.scale_factor
    if scale <= 0 or scale >= 1.0:
      return frame.copy(), 1.0
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    working = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return working, scale

  @staticmethod
  def _scale_detections(
      detections: list[JointDetection], scale: float
  ) -> list[JointDetection]:
    if scale <= 0 or scale >= 1.0:
      return detections
    inv = 1.0 / scale
    scaled: list[JointDetection] = []
    for det in detections:
      bbox = None
      if det.bbox:
        x1, y1, x2, y2 = det.bbox
        bbox = (int(x1 * inv), int(y1 * inv), int(x2 * inv), int(y2 * inv))
      scaled.append(
          JointDetection(
              x=int(det.x * inv),
              y=int(det.y * inv),
              radius=det.radius * inv,
              confidence=det.confidence,
              bbox=bbox,
              ring_thickness=det.ring_thickness * inv,
              contrast=det.contrast,
              completeness=det.completeness,
          )
      )
    return scaled

  @staticmethod
  def _draw_overlay(
      frame: np.ndarray,
      detections: list[JointDetection],
      count: int,
      motion: MotionState,
      frame_idx: int,
  ) -> np.ndarray:
    h, w = frame.shape[:2]
    for det in detections:
      thickness = max(2, int(det.ring_thickness or det.radius * 0.07))
      inner_r = max(1, int(det.radius - thickness))
      outer_r = int(det.radius + thickness)
      cv2.circle(frame, (det.x, det.y), inner_r, (0, 180, 255), 2)
      cv2.circle(frame, (det.x, det.y), outer_r, (0, 255, 120), 2)
      if det.bbox:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 360), 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    put_text(frame, f"管节段数: {count}", (20, 45), 32, (0, 255, 0), stroke_width=1, stroke_color_bgr=(0, 80, 0))
    put_text(frame, f"Frame: {frame_idx}  Motion: {motion.value}", (20, 85), 20, (255, 255, 255))
    return frame

  @staticmethod
  def _draw_static_overlay(
      frame: np.ndarray, frame_idx: int, count: int
  ) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (min(w - 10, 380), 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    put_text(frame, f"管节段数: {count}", (20, 45), 32, (0, 255, 0), stroke_width=1, stroke_color_bgr=(0, 80, 0))
    put_text(frame, f"Frame: {frame_idx}  静止帧(已跳过)", (20, 78), 18, (180, 180, 180))
    return frame


def save_result(
    result: CounterResult,
    output_path: str | Path,
    calibration: CalibrationResult | None = None,
) -> tuple[Path, Path | None, Path | None]:
  data = {
      "joint_count": result.count,
      "frame_count": result.frame_count,
      "tracks": [
          {
              "track_id": t.track_id,
              "start_frame": t.start_frame,
              "last_frame": t.last_frame,
              "confidence": round(t.mean_confidence, 4),
          }
          for t in result.tracks
      ],
  }
  if calibration is not None:
    data["calibration"] = {
        "estimated_joints": calibration.estimated_joints,
        "sampled_frames": calibration.sampled_frames,
        "median_walking_gap": round(calibration.median_walking_gap, 1),
        "total_candidates": calibration.total_candidates,
        "params": {
            "gradient_threshold": calibration.params.gradient_threshold,
            "min_confidence": calibration.params.min_confidence,
            "min_contrast": calibration.params.min_contrast,
            "min_completeness": calibration.params.min_completeness,
        },
    }
  if result.joint_events:
    data["joint_events"] = [
        {
            "frame_idx": e.frame_idx,
            "confidence": round(e.confidence, 4),
            "walking_frames_since_prev": e.walking_frames_since_prev,
        }
        for e in result.joint_events
    ]
  if result.walking_frames_total:
    data["walking_frames_total"] = result.walking_frames_total
  if result.moving_frames:
    data["ring_count_curve"] = {
        "moving_frames": result.moving_frames,
        "moving_ring_counts": result.moving_ring_counts,
        "smoothed_ring_counts": result.smoothed_ring_counts,
        "detrended_ring_counts": result.detrended_ring_counts,
        "joint_peak_frames": result.joint_peak_frames,
    }
  if result.inspection_report is not None:
    data["inspection_report"] = result.inspection_report.to_dict()
  if result.total_video_frames > 0:
    data["video_bounds"] = {
        "total_video_frames": result.total_video_frames,
        "effective_frame_limit": result.effective_frame_limit,
        "tail_trim_frames": result.tail_trim_frames,
        "tail_trim_summary": tail_trim_summary(
            result.total_video_frames, result.tail_trim_frames
        ),
    }
  path = Path(output_path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

  report_path: Path | None = None
  if result.inspection_report is not None:
    if path.name.endswith(".joint_count.json"):
      report_name = path.name.replace(".joint_count.json", ".joint_report.txt")
    else:
      report_name = f"{path.stem}.joint_report.txt"
    report_path = path.with_name(report_name)
    save_inspection_report(result.inspection_report, report_path)

  curve_path = save_curve_csv(result, path)
  return path, report_path, curve_path
