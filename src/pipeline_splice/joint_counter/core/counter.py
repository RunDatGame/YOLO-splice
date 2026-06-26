"""管节计数：移动帧圆环数量峰值检测。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .report import InspectionReport
from .ring_peak_detector import PeakDetectionResult, RingPeakJointCounter
from .spacing import JointEvent
from .tracker import RingCandidateTrack


@dataclass
class CounterResult:
  count: int
  tracks: list[RingCandidateTrack] = field(default_factory=list)
  frame_count: int = 0
  motion_history: list[str] = field(default_factory=list)
  joint_events: list[JointEvent] = field(default_factory=list)
  walking_frames_total: int = 0
  inspection_report: InspectionReport | None = None
  moving_frames: list[int] = field(default_factory=list)
  moving_ring_counts: list[int] = field(default_factory=list)
  smoothed_ring_counts: list[float] = field(default_factory=list)
  detrended_ring_counts: list[float] = field(default_factory=list)
  joint_peak_frames: list[int] = field(default_factory=list)
  estimated_period_samples: int = 0
  total_video_frames: int = 0
  effective_frame_limit: int = 0
  tail_trim_frames: int = 0


class JointCounter:
  """基于移动帧圆环数量峰值与等间隔二次过滤的管节计数。"""

  def __init__(
      self,
      expected_min: int = 6,
      expected_max: int = 50,
      frame_interval: int = 10,
      motion_diff_threshold: float = 0.45,
  ) -> None:
    self.expected_min = expected_min
    self.expected_max = expected_max
    self.frame_interval = max(1, frame_interval)
    self.peak_counter = RingPeakJointCounter(
        expected_min=expected_min,
        expected_max=expected_max,
        frame_interval=frame_interval,
    )
    self.peak_counter.motion_filter.diff_threshold = motion_diff_threshold

  def reset(self) -> None:
    self.peak_counter.reset()

  @property
  def count(self) -> int:
    return self.peak_counter.count

  def observe_motion(self, frame_bgr) -> bool:
    return self.peak_counter.observe_motion(frame_bgr)

  def record_sample(self, frame_idx: int, ring_count: int) -> int:
    return self.peak_counter.record_sample(frame_idx, ring_count)

  def update(self, frame_idx: int, ring_count: int, frame_bgr=None) -> int:
    return self.peak_counter.update(frame_idx, ring_count, frame_bgr)

  def finalize(self) -> CounterResult:
    peak_result = self.peak_counter.finalize()
    return CounterResult(
        count=len(peak_result.joint_events),
        joint_events=peak_result.joint_events,
        moving_frames=peak_result.moving_frames,
        moving_ring_counts=peak_result.moving_counts,
        smoothed_ring_counts=peak_result.smoothed_counts,
        detrended_ring_counts=peak_result.detrended_counts,
        joint_peak_frames=peak_result.joint_frames,
        estimated_period_samples=peak_result.estimated_period_samples,
        walking_frames_total=len(peak_result.moving_frames) * self.frame_interval,
    )
