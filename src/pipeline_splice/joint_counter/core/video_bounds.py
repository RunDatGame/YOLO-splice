"""视频有效处理范围：排除终点复杂场景对应的末尾帧。"""

from __future__ import annotations

DEFAULT_TAIL_TRIM_FRAMES = 500


def effective_frame_limit(total_frames: int, tail_trim_frames: int = DEFAULT_TAIL_TRIM_FRAMES) -> int:
  """
  返回可用于处理的帧上界（不含）：frame_idx < 返回值 的帧才会参与标定与计数。
  当总帧数不超过 tail_trim_frames 时，不裁剪，返回 total_frames。
  """
  if total_frames <= 0:
    return 0
  if tail_trim_frames <= 0 or total_frames <= tail_trim_frames:
    return total_frames
  return total_frames - tail_trim_frames


def tail_trim_summary(total_frames: int, tail_trim_frames: int = DEFAULT_TAIL_TRIM_FRAMES) -> str:
  """生成裁剪说明，供日志输出。"""
  limit = effective_frame_limit(total_frames, tail_trim_frames)
  if total_frames <= 0:
    return "无法读取视频总帧数，未做末尾裁剪"
  if limit >= total_frames:
    return (
        f"视频总帧数 {total_frames} 不超过末尾裁剪阈值 {tail_trim_frames}，"
        f"将全部 {total_frames} 帧用于处理"
    )
  trimmed = total_frames - limit
  return (
      f"已排除视频末尾 {trimmed} 帧（总帧数 {total_frames}，"
      f"有效处理前 {limit} 帧，frame 0~{limit - 1}）"
  )
