"""基于移动帧圆环数量曲线的接口检测（包络线 + 最小宽度峰 + 周期 NMS）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks, peak_widths

from .spacing import JointEvent, enforce_joint_count_range

# 管节峰在 X 轴（采样点）上的最小跨度，过滤窄刺
MIN_JOINT_PEAK_WIDTH = 6

# find_peaks prominence 系数（越低越不易漏检；见 _envelope_prominence / find_peaks_on_envelope）
PROMINENCE_RANGE_SCALE = 0.008
PROMINENCE_STD_SCALE = 0.045
PROMINENCE_FLOOR = 0.012
# 逐级降低 prominence 重试（完全动态输出，不预设目标峰数）
PROMINENCE_RETRY_SCALES = (1.0, 0.5, 0.28, 0.15, 0.08, 0.04, 0.02, 0.01)
# 周期 NMS 抑制半径 = period * ratio（越小越不易合并掉相邻真峰）
NMS_SUPPRESS_RATIO = 0.40
# 半高宽过滤系数（相对 min_width，略放宽以减少漏检）
PEAK_WIDTH_KEEP_RATIO = 0.58


@dataclass
class RingCountPoint:
  frame_idx: int
  ring_count: int
  is_moving: bool = True


@dataclass
class CurveAnalysisResult:
  moving_frames: list[int]
  moving_counts: list[int]
  smoothed_counts: list[float]
  detrended_counts: list[float]
  estimated_period_samples: int
  major_peak_indices: list[int]
  major_peak_frames: list[int]
  peak_prominences: list[float]


@dataclass
class PeakDetectionResult:
  joint_frames: list[int]
  joint_events: list[JointEvent]
  moving_frames: list[int] = field(default_factory=list)
  moving_counts: list[int] = field(default_factory=list)
  smoothed_counts: list[float] = field(default_factory=list)
  detrended_counts: list[float] = field(default_factory=list)
  estimated_period_samples: int = 0
  peak_indices: list[int] = field(default_factory=list)
  peak_prominences: list[float] = field(default_factory=list)


class MovingFrameFilter:
  def __init__(self, diff_threshold: float = 0.45) -> None:
    self.diff_threshold = diff_threshold
    self._prev_gray: np.ndarray | None = None
    self._last_is_moving: bool = True

  def reset(self) -> None:
    self._prev_gray = None
    self._last_is_moving = True

  def observe(self, frame_bgr: np.ndarray) -> bool:
    """每帧调用一次，更新运动状态。"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if self._prev_gray is None or self._prev_gray.shape != gray.shape:
      self._prev_gray = gray
      self._last_is_moving = True
      return True
    diff = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
    self._prev_gray = gray
    self._last_is_moving = diff >= self.diff_threshold
    return self._last_is_moving

  def is_moving(self) -> bool:
    return self._last_is_moving


def flatten_edge_margins(y: np.ndarray, margin_ratio: float = 0.05) -> np.ndarray:
  """首尾拉平到边界值，压制端部异常而不制造假谷。"""
  n = len(y)
  out = y.astype(np.float64).copy()
  if n < 10:
    return out
  margin = max(1, int(round(n * margin_ratio)))
  out[:margin] = out[margin]
  out[n - margin :] = out[n - margin - 1]
  return out


def preprocess_wave_signal(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """自动去趋势 + 多尺度平滑（无固定窗口参数）。"""
  y = np.asarray(y, dtype=np.float64)
  n = len(y)
  if n < 2:
    return y.copy(), np.zeros_like(y)

  trend_sigma = max(1.0, n // 50 + 1)
  trend = gaussian_filter1d(y, sigma=trend_sigma)
  y_detrend = y - trend
  sig_b = max(1.0, min(3.0, n / 10.0))
  y_smooth = (
      gaussian_filter1d(y_detrend, sigma=1.0)
      + gaussian_filter1d(y_detrend, sigma=sig_b)
  ) / 2.0
  return y_smooth, trend


def estimate_period_fft(y: np.ndarray, fallback: int = 40) -> int:
  """FFT 主频估计周期。"""
  n = len(y)
  if n < 4:
    return max(2, min(fallback, max(2, n // 2)))

  yf = np.abs(np.fft.fft(y.astype(np.float64)))
  yf[0] = 0.0
  half = len(yf) // 2
  if half <= 1:
    return max(2, min(fallback, max(2, n // 2)))

  peak = int(np.argmax(yf[:half]))
  if peak <= 0:
    return max(2, min(fallback, max(2, n // 2)))

  period = int(round(n / (peak + 1)))
  return max(2, min(period, max(2, n // 2)))


def estimate_period_acf(y: np.ndarray) -> int | None:
  """自相关估计周期。"""
  n = len(y)
  if n < 6:
    return None

  y_norm = (y.astype(np.float64) - np.mean(y)) / (np.std(y) + 1e-8)
  ac = correlate(y_norm, y_norm, mode="full")
  ac = ac[len(ac) // 2 :]
  peaks, _ = find_peaks(ac)
  if peaks.size == 0:
    return None

  best = int(peaks[int(np.argmax(ac[peaks]))])
  return max(2, min(best, max(2, n // 2)))


def estimate_period_fused(y: np.ndarray, fallback: int = 40) -> int:
  """FFT + 自相关融合周期。"""
  p_fft = estimate_period_fft(y, fallback)
  p_ac = estimate_period_acf(y)
  if p_ac is None:
    return p_fft
  return max(2, int(round((p_fft + p_ac) / 2)))


def _edge_margin(n: int, ratio: float) -> int:
  return max(1, int(round(n * ratio))) if n >= 10 else 0


def filter_edge_peaks(
    peaks: list[int],
    n: int,
    filter_margin_ratio: float = 0.03,
    min_samples: int = 20,
) -> list[int]:
  """丢弃首尾边界区内的波包中心。"""
  if not peaks or n < min_samples:
    return peaks
  margin = _edge_margin(n, filter_margin_ratio)
  if margin * 2 >= n:
    return peaks
  return sorted(p for p in peaks if margin < p < n - margin)


def build_envelope(y: np.ndarray, sigma: float | None = None) -> np.ndarray:
  """构造包络线：去趋势 + 多尺度平滑 + 高斯包络。"""
  y_smooth, _trend = preprocess_wave_signal(y)
  n = len(y_smooth)
  if n < 2:
    return y_smooth
  sig = sigma if sigma is not None else max(3.0, min(6.0, n / 15.0))
  sig = max(2.0, min(float(sig), n / 4.0))
  return gaussian_filter1d(y_smooth, sigma=sig)


def _min_peak_width(n: int, requested: int = MIN_JOINT_PEAK_WIDTH) -> int:
  return max(3, min(requested, max(3, n // 3)))


def _envelope_prominence(envelope: np.ndarray) -> float:
  """包络线 find_peaks 的 prominence 阈值（相对凸起度，非绝对高度）。"""
  sig_range = float(np.max(envelope) - np.min(envelope))
  sig_std = float(np.std(envelope))
  return max(
      PROMINENCE_FLOOR,
      sig_range * PROMINENCE_RANGE_SCALE,
      sig_std * PROMINENCE_STD_SCALE,
  )


def _filter_by_peak_width(
    envelope: np.ndarray,
    idx: np.ndarray,
    min_width: int,
) -> np.ndarray:
  if idx.size == 0:
    return idx
  widths = peak_widths(envelope, idx, rel_height=0.5)[0]
  keep = widths >= max(2.0, min_width * PEAK_WIDTH_KEEP_RATIO)
  return idx[keep]


def find_peaks_on_envelope(
    envelope: np.ndarray,
    period: int,
    min_width: int = MIN_JOINT_PEAK_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
  """
  在包络线上找宽峰：多档 prominence 并集 + width + distance≈0.4T。
  峰个数完全由波形决定，不预设目标数量。
  """
  n = len(envelope)
  if n < 3:
    return np.array([], dtype=int), np.array([])

  width = _min_peak_width(n, min_width)
  base_prom = _envelope_prominence(envelope)
  base_distance = max(width, int(round(max(2, period) * 0.40)))

  collected: list[int] = []

  for scale in PROMINENCE_RETRY_SCALES:
    prom = max(PROMINENCE_FLOOR * max(scale, 0.5), base_prom * scale)
    if scale <= 0.15:
      prom = max(0.008, base_prom * scale)
    distance = base_distance if scale >= 0.45 else max(2, base_distance // 2)
    if scale <= 0.15:
      distance = max(2, width)
    w = width if scale >= 0.28 else max(3, width - 2)
    if scale <= 0.15:
      w = max(3, min(width, 5))
    if scale <= 0.04:
      w = 3

    idx, _props = find_peaks(
        envelope,
        prominence=prom,
        width=w,
        distance=distance,
    )
    if scale <= 0.04:
      idx = _filter_by_peak_width(envelope, idx, max(3, w - 1))
    else:
      idx = _filter_by_peak_width(envelope, idx, w)
    if idx.size:
      collected.extend(int(i) for i in idx)

  if not collected:
    return np.array([], dtype=int), np.array([])

  best_idx = np.unique(np.asarray(collected, dtype=int))
  prominences = np.asarray([float(envelope[i]) for i in best_idx], dtype=np.float64)
  return best_idx, prominences


def nms_peaks_on_envelope(
    envelope: np.ndarray,
    candidates: np.ndarray,
    period: int,
    suppress_ratio: float = NMS_SUPPRESS_RATIO,
) -> list[int]:
  """周期 NMS：包络幅度从大到小，抑制 T*ratio 内的重复峰。"""
  if candidates.size == 0:
    return []

  radius = max(MIN_JOINT_PEAK_WIDTH, int(round(max(2, period) * suppress_ratio)))
  cand = np.asarray(candidates, dtype=int)
  order = np.argsort(envelope[cand])[::-1]
  kept: list[int] = []
  suppressed = np.zeros(len(cand), dtype=bool)

  for i in order:
    if suppressed[i]:
      continue
    pos = int(cand[i])
    kept.append(pos)
    suppressed |= np.abs(cand - pos) < radius

  return sorted(kept)


def detect_envelope_peaks(
    raw: np.ndarray,
    margin_ratio: float = 0.05,
    period_fallback: int = 40,
    min_width: int = MIN_JOINT_PEAK_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[int], list[float]]:
  """
  包络线管节检测（主入口）：
  预处理 → 包络 → find_peaks(width>=7) → 周期 NMS → 边界清洗。
  """
  n = len(raw)
  if n < 2:
    empty: list[int] = []
    z = raw.copy() if n else np.array([], dtype=np.float64)
    return z, z, np.zeros_like(z), 0, empty, []

  envelope = build_envelope(raw)
  y_work = flatten_edge_margins(envelope, margin_ratio)
  _, trend = preprocess_wave_signal(raw)
  period = estimate_period_fused(y_work, fallback=period_fallback)

  idx, prom_arr = find_peaks_on_envelope(y_work, period, min_width=min_width)
  if idx.size == 0:
    return y_work, y_work, trend, period, [], []

  peak_indices = nms_peaks_on_envelope(y_work, idx, period)
  peak_indices = filter_edge_peaks(peak_indices, n, margin_ratio)

  prom_map = {int(i): float(p) for i, p in zip(idx, prom_arr)}
  prominences = [prom_map.get(i, float(y_work[i])) for i in peak_indices]
  return y_work, y_work, trend, period, peak_indices, prominences


def detect_wave_packets(
    raw: np.ndarray,
    margin_ratio: float = 0.05,
    period_fallback: int = 40,
    min_width: int = MIN_JOINT_PEAK_WIDTH,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[int], list[float]]:
  """兼容别名，实际走包络峰检测。"""
  kwargs.pop("envelope_sigma", None)
  kwargs.pop("score_std_factor", None)
  kwargs.pop("min_period_ratio", None)
  kwargs.pop("suppress_ratio", None)
  return detect_envelope_peaks(
      raw,
      margin_ratio=margin_ratio,
      period_fallback=period_fallback,
      min_width=min_width,
  )


def detect_ridge_segments(raw: np.ndarray, **kwargs) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, list[int], list[float]
]:
  """兼容旧接口，转发至 detect_wave_packets。"""
  mapped = dict(kwargs)
  mapped.pop("envelope_sigma", None)
  mapped.pop("score_std_factor", None)
  mapped.pop("min_period_ratio", None)
  mapped.pop("suppress_ratio", None)
  return detect_wave_packets(raw, **mapped)


def detect_wide_peaks_robust(raw: np.ndarray, **kwargs) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, list[int], list[float]
]:
  return detect_wave_packets(raw, **kwargs)


def robust_periodic_peak_detection(raw: np.ndarray, **kwargs) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, int, list[int], list[float]
]:
  return detect_wave_packets(raw, **kwargs)


class RingPeakJointCounter:
  """
  包络线管节检测：build_envelope → find_peaks(width>=7) → 周期 NMS。
  核心实现在 detect_envelope_peaks() / RingPeakJointCounter._analyze_curve()。
  """

  def __init__(
      self,
      expected_min: int = 6,
      expected_max: int = 50,
      spacing_low_ratio: float = 0.22,
      spacing_high_ratio: float = 3.5,
      frame_interval: int = 10,
  ) -> None:
    self.expected_min = expected_min
    self.expected_max = expected_max
    self.spacing_low_ratio = spacing_low_ratio
    self.spacing_high_ratio = spacing_high_ratio
    self.frame_interval = max(1, frame_interval)

    self.motion_filter = MovingFrameFilter()
    self._moving_frames: list[int] = []
    self._moving_counts: list[int] = []
    self._smoothed_counts: list[float] = []
    self._detrended_counts: list[float] = []
    self._period_samples = 0
    self._peak_indices: list[int] = []
    self._peak_prominences: list[float] = []
    self._joint_frames: list[int] = []
    self._curve_finalized = False

  def reset(self) -> None:
    self.motion_filter.reset()
    self._moving_frames.clear()
    self._moving_counts.clear()
    self._smoothed_counts.clear()
    self._detrended_counts.clear()
    self._period_samples = 0
    self._peak_indices.clear()
    self._peak_prominences.clear()
    self._joint_frames.clear()
    self._curve_finalized = False

  @property
  def count(self) -> int:
    return len(self._joint_frames)

  @property
  def moving_frames(self) -> list[int]:
    return list(self._moving_frames)

  @property
  def moving_counts(self) -> list[int]:
    return list(self._moving_counts)

  @property
  def smoothed_counts(self) -> list[float]:
    return list(self._smoothed_counts)

  @property
  def detrended_counts(self) -> list[float]:
    return list(self._detrended_counts)

  @property
  def joint_frames(self) -> list[int]:
    return list(self._joint_frames)

  @property
  def peak_indices(self) -> list[int]:
    return list(self._peak_indices)

  def is_moving_frame(self, frame_bgr: np.ndarray) -> bool:
    return self.motion_filter.observe(frame_bgr)

  def observe_motion(self, frame_bgr: np.ndarray) -> bool:
    return self.motion_filter.observe(frame_bgr)

  def record_sample(self, frame_idx: int, ring_count: int) -> int:
    self._moving_frames.append(frame_idx)
    self._moving_counts.append(ring_count)
    if not self._curve_finalized:
      self._apply_preview_detection()
    return self.count

  def update(self, frame_idx: int, ring_count: int, frame_bgr: np.ndarray | None = None) -> int:
    if frame_bgr is not None and not self.observe_motion(frame_bgr):
      return self.count
    return self.record_sample(frame_idx, ring_count)

  def finalize(self) -> PeakDetectionResult:
    analysis = self._analyze_curve(self._moving_frames, self._moving_counts)
    self._apply_analysis(analysis)
    self._joint_frames = self._finalize_joint_frames(analysis.major_peak_frames)
    self._curve_finalized = True

    events = [
        JointEvent(
            frame_idx=frame,
            confidence=min(
                1.0,
                0.45 + prom / max(max(self._peak_prominences) if self._peak_prominences else 1.0, 1.0),
            ),
            track_id=i + 1,
            walking_frames_since_prev=(
                frame - self._joint_frames[i - 1] if i > 0 else 0
            ),
        )
        for i, (frame, prom) in enumerate(
            zip(self._joint_frames, self._prominences_for_frames(self._joint_frames))
        )
    ]
    filtered = enforce_joint_count_range(
        events, self.expected_min, self.expected_max
    )
    self._joint_frames = [e.frame_idx for e in filtered]

    return PeakDetectionResult(
        joint_frames=list(self._joint_frames),
        joint_events=filtered,
        moving_frames=list(self._moving_frames),
        moving_counts=list(self._moving_counts),
        smoothed_counts=list(self._smoothed_counts),
        detrended_counts=list(self._detrended_counts),
        estimated_period_samples=self._period_samples,
        peak_indices=list(self._peak_indices),
        peak_prominences=list(self._peak_prominences),
    )

  def _apply_preview_detection(self) -> None:
    analysis = self._analyze_curve(self._moving_frames, self._moving_counts)
    self._apply_analysis(analysis)
    # 预览阶段直接使用 NMS 结果，避免间距过滤把已识别尖峰撤销
    self._joint_frames = list(analysis.major_peak_frames)

  def _apply_analysis(self, analysis: CurveAnalysisResult) -> None:
    self._smoothed_counts = analysis.smoothed_counts
    self._detrended_counts = analysis.detrended_counts
    self._period_samples = analysis.estimated_period_samples
    self._peak_indices = analysis.major_peak_indices
    self._peak_prominences = analysis.peak_prominences

  def _analyze_curve(
      self, frames: list[int], counts: list[int]
  ) -> CurveAnalysisResult:
    empty = CurveAnalysisResult(
        moving_frames=list(frames),
        moving_counts=list(counts),
        smoothed_counts=[float(c) for c in counts],
        detrended_counts=[float(c) for c in counts],
        estimated_period_samples=0,
        major_peak_indices=[],
        major_peak_frames=[],
        peak_prominences=[],
    )
    if len(counts) < 2:
      return empty

    raw = np.array(counts, dtype=np.float64)
    smoothed, detrended, _baseline, period, merged_idx, prominences = detect_envelope_peaks(
        raw,
        margin_ratio=0.03,
        period_fallback=40,
        min_width=MIN_JOINT_PEAK_WIDTH,
    )
    major_frames = [frames[i] for i in merged_idx]

    return CurveAnalysisResult(
        moving_frames=list(frames),
        moving_counts=list(counts),
        smoothed_counts=[float(v) for v in smoothed],
        detrended_counts=[float(v) for v in detrended],
        estimated_period_samples=period,
        major_peak_indices=merged_idx,
        major_peak_frames=major_frames,
        peak_prominences=prominences,
    )

  def _prominences_for_frames(self, frames: list[int]) -> list[float]:
    idx_map = {
        self._moving_frames[i]: self._peak_prominences[j]
        for j, i in enumerate(self._peak_indices)
        if j < len(self._peak_prominences) and i < len(self._moving_frames)
    }
    return [idx_map.get(f, 1.0) for f in frames]

  def _finalize_joint_frames(self, peak_frames: list[int]) -> list[int]:
    """最终计数：数量在合理范围内时保留全部 NMS 尖峰，仅超上限时才做间距裁剪。"""
    ordered = sorted(set(peak_frames))
    if not ordered:
      return []
    if len(ordered) <= self.expected_max:
      return ordered
    return self._trim_excess_peaks(ordered)

  def _trim_excess_peaks(self, peak_frames: list[int]) -> list[int]:
    """仅在候选过多时，按间距一致性裁减到 expected_max。"""
    ordered = sorted(peak_frames)
    if len(ordered) <= self.expected_max:
      return ordered

    gaps = [ordered[i] - ordered[i - 1] for i in range(1, len(ordered))]
    median_gap = float(np.median(gaps))
    if median_gap <= 0:
      return ordered[: self.expected_max]

    kept = [ordered[0]]
    for i in range(1, len(ordered)):
      gap = ordered[i] - kept[-1]
      if self.spacing_low_ratio * median_gap <= gap <= self.spacing_high_ratio * median_gap:
        kept.append(ordered[i])
        if len(kept) >= self.expected_max:
          break
        recent = [kept[j] - kept[j - 1] for j in range(1, len(kept))]
        median_gap = float(np.median(recent))

    if len(kept) >= min(self.expected_min, len(ordered) // 2):
      return kept[: self.expected_max]
    return ordered[: self.expected_max]
