"""管节接口环状候选的时序跟踪与选中。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detector import JointDetection


@dataclass
class RingCandidateTrack:
  track_id: int
  detections: list[JointDetection] = field(default_factory=list)
  frame_indices: list[int] = field(default_factory=list)
  start_frame: int = 0
  last_frame: int = 0
  interface_valid: bool = False
  counted: bool = False
  joint_prior_frames: list[int] = field(default_factory=list)

  @property
  def latest(self) -> JointDetection | None:
    return self.detections[-1] if self.detections else None

  @property
  def hit_count(self) -> int:
    return len(self.detections)

  @property
  def mean_confidence(self) -> float:
    if not self.detections:
      return 0.0
    return float(np.mean([d.confidence for d in self.detections]))

  @property
  def consecutive_hits(self) -> int:
    return len(self.frame_indices)


class InterfaceRingTracker:
  """跟踪环状候选：连续≥2次绿环（中断≤30帧）且同段内出现≥4环则判定接口。"""

  def __init__(
      self,
      min_consecutive_hits: int = 2,
      min_rings_at_joint: int = 4,
      max_interruption_frames: int = 30,
      interface_segment_window: int = 120,
      max_center_drift_x: float = 0.15,
      max_center_drift_y: float = 0.25,
      max_missing_updates: int = 3,
      frame_interval: int = 10,
  ) -> None:
    self.min_consecutive_hits = max(2, min_consecutive_hits)
    self.min_rings_at_joint = min_rings_at_joint
    self.max_interruption_frames = max_interruption_frames
    self.interface_segment_window = interface_segment_window
    self.max_center_drift_x = max_center_drift_x
    self.max_center_drift_y = max_center_drift_y
    self.max_missing_updates = max_missing_updates
    self.frame_interval = max(1, frame_interval)
    self.tracks: list[RingCandidateTrack] = []
    self._next_id = 1
    self._missing: dict[int, int] = {}
    self._frame_ring_counts: dict[int, int] = {}

  def reset(self) -> None:
    self.tracks.clear()
    self._next_id = 1
    self._missing.clear()
    self._frame_ring_counts.clear()

  def frame_ring_count(self, frame_idx: int) -> int:
    return self._frame_ring_counts.get(frame_idx, 0)

  def is_joint_frame(self, frame_idx: int) -> bool:
    return self.frame_ring_count(frame_idx) >= self.min_rings_at_joint

  def update(
      self,
      candidates: list[JointDetection],
      frame_idx: int,
      frame_shape: tuple,
  ) -> None:
    h, w = frame_shape[:2]
    ring_count = len(candidates)
    self._frame_ring_counts[frame_idx] = ring_count
    at_joint = ring_count >= self.min_rings_at_joint
    max_dist = min(h, w) * 0.20
    matched: set[int] = set()

    for track in self.tracks:
      if track.latest is None:
        continue
      if frame_idx - track.last_frame > self.max_interruption_frames:
        continue
      best_idx = -1
      best_score = -1.0
      for idx, det in enumerate(candidates):
        if idx in matched:
          continue
        score = self._match_score(track, det, max_dist)
        if score > best_score:
          best_score = score
          best_idx = idx
      if best_idx >= 0 and best_score > 0.25:
        det = candidates[best_idx]
        track.detections.append(det)
        track.frame_indices.append(frame_idx)
        track.last_frame = frame_idx
        if at_joint:
          track.joint_prior_frames.append(frame_idx)
        matched.add(best_idx)
        self._missing[track.track_id] = 0
        track.interface_valid = self._validate_interface(track, w, h)

    for idx, det in enumerate(candidates):
      if idx in matched:
        continue
      track = RingCandidateTrack(
          track_id=self._next_id,
          detections=[det],
          frame_indices=[frame_idx],
          start_frame=frame_idx,
          last_frame=frame_idx,
          joint_prior_frames=[frame_idx] if at_joint else [],
      )
      self._next_id += 1
      self.tracks.append(track)
      self._missing[track.track_id] = 0

    active: list[RingCandidateTrack] = []
    updated_ids = {t.track_id for t in self.tracks if t.last_frame == frame_idx}
    for track in self.tracks:
      if track.track_id not in updated_ids and track.last_frame < frame_idx:
        self._missing[track.track_id] = self._missing.get(track.track_id, 0) + 1
      if self._missing.get(track.track_id, 0) <= self.max_missing_updates:
        active.append(track)
      else:
        self._missing.pop(track.track_id, None)
    self.tracks = active

  def select_interface_detection(
      self, frame_idx: int, frame_shape: tuple
  ) -> list[JointDetection]:
    """连续≥2次绿环且同段内满足≥4环先验时，选出管节接口环（最多一个）。"""
    h, w = frame_shape[:2]
    best_det: JointDetection | None = None
    best_score = -1.0

    for track in self.tracks:
      if track.last_frame != frame_idx or track.latest is None:
        continue
      if not track.interface_valid:
        continue
      score = self._interface_score(track, w, h)
      if score > best_score:
        best_score = score
        best_det = track.latest

    return [best_det] if best_det is not None else []

  def get_interface_tracks(self) -> list[RingCandidateTrack]:
    return [t for t in self.tracks if t.interface_valid]

  def segment_has_joint_rings(self, end_frame: int) -> bool:
    start = max(0, end_frame - self.interface_segment_window)
    for frame_idx in range(start, end_frame + 1):
      if self.is_joint_frame(frame_idx):
        return True
    return False

  def _validate_interface(self, track: RingCandidateTrack, w: int, h: int) -> bool:
    if track.hit_count < self.min_consecutive_hits:
      return False
    if self._longest_consecutive_run(track.frame_indices) < self.min_consecutive_hits:
      return False
    return self._both_conditions_in_window(track)

  def _both_conditions_in_window(self, track: RingCandidateTrack) -> bool:
    segment_end = track.last_frame
    segment_start = max(0, segment_end - self.interface_segment_window)

    hits = [f for f in track.frame_indices if segment_start <= f <= segment_end]
    if self._longest_consecutive_run(hits) < self.min_consecutive_hits:
      return False

    for frame_idx in range(segment_start, segment_end + 1):
      if self.is_joint_frame(frame_idx):
        return True
    return False

  def _longest_consecutive_run(self, frame_indices: list[int]) -> int:
    if not frame_indices:
      return 0
    sorted_frames = sorted(set(frame_indices))
    if len(sorted_frames) == 1:
      return 1
    max_run = 1
    run = 1
    for i in range(1, len(sorted_frames)):
      gap = sorted_frames[i] - sorted_frames[i - 1]
      if gap <= self.max_interruption_frames:
        run += 1
        max_run = max(max_run, run)
      else:
        run = 1
    return max_run

  def _interface_score(self, track: RingCandidateTrack, w: int, h: int) -> float:
    conf = float(np.mean([d.confidence for d in track.detections]))
    run = self._longest_consecutive_run(track.frame_indices)
    consecutive_score = min(1.0, run / max(self.min_consecutive_hits, 1))
    joint_score = 1.0 if self.segment_has_joint_rings(track.last_frame) else 0.0
    return 0.45 * conf + 0.35 * consecutive_score + 0.20 * joint_score

  def _match_score(
      self, track: RingCandidateTrack, det: JointDetection, max_dist: float
  ) -> float:
    last = track.latest
    if last is None:
      return 0.0
    dist = np.hypot(det.x - last.x, det.y - last.y)
    dist_score = max(0.0, 1.0 - dist / max_dist)
    radius_growth = (det.radius - last.radius) / max(last.radius, 1.0)
    growth_score = min(1.0, max(0.0, radius_growth * 4.0 + 0.5))
    return 0.65 * dist_score + 0.35 * growth_score


JointTrack = RingCandidateTrack
JointTracker = InterfaceRingTracker
