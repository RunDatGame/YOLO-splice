"""等间隔行走帧先验：过滤误检环、约束管节数。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class JointEvent:
  frame_idx: int
  confidence: float
  track_id: int
  walking_frames_since_prev: int = 0


def filter_by_walking_spacing(
    events: list[JointEvent],
    gap_low_ratio: float = 0.22,
    gap_high_ratio: float = 3.8,
) -> list[JointEvent]:
  """仅保留接口间行走帧数与典型间距一致的管节事件。"""
  if len(events) <= 1:
    return list(events)

  ordered = sorted(events, key=lambda e: e.frame_idx)
  gaps = [e.walking_frames_since_prev for e in ordered[1:] if e.walking_frames_since_prev > 0]
  if not gaps:
    return [ordered[0]]

  median_gap = float(np.median(gaps))
  if median_gap <= 0:
    return ordered

  kept: list[JointEvent] = [ordered[0]]
  for event in ordered[1:]:
    gap = event.walking_frames_since_prev
    if gap <= 0:
      continue
    if gap_low_ratio * median_gap <= gap <= gap_high_ratio * median_gap:
      kept.append(event)
      recent = [e.walking_frames_since_prev for e in kept[1:] if e.walking_frames_since_prev > 0]
      if recent:
        median_gap = float(np.median(recent))

  return kept


def select_best_spacing_chain(
    events: list[JointEvent],
    max_count: int,
    tolerance: float = 0.55,
) -> list[JointEvent]:
  """从过多候选中选取最符合等间隔行走帧的管节序列。"""
  ordered = sorted(events, key=lambda e: e.frame_idx)
  if len(ordered) <= max_count:
    return ordered

  gaps = [e.walking_frames_since_prev for e in ordered[1:] if e.walking_frames_since_prev > 0]
  median_gap = float(np.median(gaps)) if gaps else 1.0

  best_chain: list[JointEvent] = []
  for start in ordered[: min(5, len(ordered))]:
    chain = [start]
    target = median_gap
    for event in ordered:
      if event.frame_idx <= start.frame_idx:
        continue
      gap = event.walking_frames_since_prev
      if gap <= 0:
        continue
      rel_err = abs(gap - target) / max(target, 1.0)
      if rel_err <= tolerance:
        chain.append(event)
        target = float(np.median([e.walking_frames_since_prev for e in chain[1:]]))

    if len(chain) > len(best_chain):
      best_chain = chain
    if len(best_chain) >= max_count:
      break

  if len(best_chain) > max_count:
    best_chain = sorted(best_chain, key=lambda e: e.confidence, reverse=True)[:max_count]
    best_chain.sort(key=lambda e: e.frame_idx)

  return best_chain


def enforce_joint_count_range(
    events: list[JointEvent],
    min_count: int = 6,
    max_count: int = 50,
) -> list[JointEvent]:
  """结合 6~50 管节先验，输出最终管节事件列表。"""
  if not events:
    return []

  ordered = sorted(events, key=lambda e: e.frame_idx)
  n = len(ordered)

  if n > max_count:
    filtered = filter_by_walking_spacing(ordered)
    return select_best_spacing_chain(filtered if filtered else ordered, max_count)

  if n >= min_count:
    return ordered

  relaxed = filter_by_walking_spacing(ordered, gap_low_ratio=0.12, gap_high_ratio=6.0)
  if len(relaxed) >= n:
    return relaxed
  return ordered


def estimate_joint_count_from_events(events: list[JointEvent], min_count: int = 6, max_count: int = 50) -> int:
  return len(enforce_joint_count_range(events, min_count, max_count))
