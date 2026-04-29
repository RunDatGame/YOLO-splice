import os
import re
from pathlib import Path

import pandas as pd
from utils.general import LOGGER


def get_video_start_time(video_path):
    video_name = Path(video_path).stem
    match = re.search(r"(\d{4}[-/]?\d{2}[-/]?\d{2})[-_](\d{6})", video_name)
    if match:
        ts_str = f"{match.group(1).replace('/', '-').replace('_', '-')} {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
        try:
            return pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        except ValueError:
            return None
    return None


def load_csv_mileage_map(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python")
        if df.shape[1] < 6:
            return None
        base_time = pd.to_datetime(df[0], errors="coerce")
        seconds = pd.to_numeric(df[3], errors="coerce").fillna(0)
        millis = pd.to_numeric(df[4], errors="coerce").fillna(0)
        full_timestamps = base_time + pd.to_timedelta(seconds, unit="s") + pd.to_timedelta(millis, unit="ms")
        mil = pd.to_numeric(df[5], errors="coerce")
        return mil[full_timestamps.notna() & mil.notna()].set_axis(
            full_timestamps[full_timestamps.notna() & mil.notna()]
        )
    except Exception as exc:
        LOGGER.warning(f"里程 CSV 解析失败 ({csv_path}): {exc}")
        return None


def get_frame_mileage(video_path, f_num, start_t, fps, m_map):
    if start_t is None or fps is None or m_map is None:
        return None
    t_target = start_t + pd.to_timedelta(f_num / fps, unit="s")
    try:
        m_map = m_map.sort_index()
        loc = m_map.index.searchsorted(t_target)
        if loc == 0:
            return m_map.iloc[0]
        elif loc >= len(m_map):
            return m_map.iloc[-1]
        t0, t1 = m_map.index[loc - 1], m_map.index[loc]
        m0, m1 = m_map.iloc[loc - 1], m_map.iloc[loc]
        dt_total = (t1 - t0).value
        dt_frame = (t_target - t0).value
        return round(m0 + (m1 - m0) * (dt_frame / dt_total) if dt_total > 0 else m0, 4)
    except (TypeError, ValueError, IndexError) as exc:
        LOGGER.warning(f"里程插值失败 (frame {f_num}): {exc}")
        return None
