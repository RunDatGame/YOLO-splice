import os
import re
from pathlib import Path

import pandas as pd
from utils.general import LOGGER

THREE_CHANNEL_COLUMN_COUNT = 14
LEGACY_SINGLE_CHANNEL_MIN_COLUMNS = 6


def _extract_time_from_csv_first_row(csv_path):
    """从 CSV 第一行提取时间戳（支持单通道/三通道两种格式）。"""
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python", nrows=1)
        if df.empty or df.shape[1] < 2:
            return None
        base_time = pd.to_datetime(df.iloc[0, 0], errors="coerce")
        if pd.isna(base_time):
            return None
        # 根据列数判断格式并提取毫秒
        if df.shape[1] < 15:
            millis = pd.to_numeric(df.iloc[0, 1], errors="coerce")
        else:
            millis = pd.to_numeric(df.iloc[0, 4], errors="coerce")
        if pd.notna(millis):
            return base_time + pd.to_timedelta(millis, unit="ms")
        return base_time
    except Exception:
        return None


def get_video_start_time(video_path, csv_path=None):
    video_name = Path(video_path).stem
    match = re.search(r"(\d{4}[-/]?\d{2}[-/]?\d{2})[-_](\d{6})", video_name)
    if match:
        ts_str = f"{match.group(1).replace('/', '-').replace('_', '-')} {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
        try:
            return pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        except ValueError:
            pass
    # Fallback：从 CSV 第一行提取时间
    if csv_path and os.path.exists(csv_path):
        fallback = _extract_time_from_csv_first_row(csv_path)
        if fallback is not None:
            LOGGER.info(f"视频名称未包含时间，已从 CSV 提取起始时间: {fallback}")
            return fallback
    return None


def _build_mileage_series(timestamps, mileage):
    valid = timestamps.notna() & mileage.notna()
    if not valid.any():
        return None
    return mileage[valid].set_axis(timestamps[valid])


def _parse_timestamp_column(column):
    return pd.to_datetime(column, errors="coerce", format="mixed")


def _load_three_channel_mileage_map(df):
    # 三通道雷达 CSV:
    # 0=时间, 1=毫秒, 2=里程, 3=道数, 后续为姿态/加速度数据。
    base_time = _parse_timestamp_column(df[0])
    millis = pd.to_numeric(df[1], errors="coerce").fillna(0)
    timestamps = base_time + pd.to_timedelta(millis, unit="ms")
    mileage = pd.to_numeric(df[2], errors="coerce")
    return _build_mileage_series(timestamps, mileage)


def _load_single_channel_mileage_map(df):
    # 旧单通道 CSV:
    # 0=完整时间, 4=毫秒, 5=里程。
    # 注：col0 已含完整时分秒，不再重复叠加 col3 的秒数。
    base_time = _parse_timestamp_column(df[0])
    millis = pd.to_numeric(df[4], errors="coerce").fillna(0)
    timestamps = base_time + pd.to_timedelta(millis, unit="ms")
    mileage = pd.to_numeric(df[5], errors="coerce")
    return _build_mileage_series(timestamps, mileage)


def load_csv_mileage_map(csv_path):
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python")
        column_count = df.shape[1]
        if column_count == THREE_CHANNEL_COLUMN_COUNT:
            mileage_map = _load_three_channel_mileage_map(df)
            radar_type = "三通道"
        elif column_count >= LEGACY_SINGLE_CHANNEL_MIN_COLUMNS:
            mileage_map = _load_single_channel_mileage_map(df)
            radar_type = "单通道"
        else:
            LOGGER.warning(f"里程 CSV 列数不足 ({column_count}): {csv_path}")
            return None

        if mileage_map is None:
            LOGGER.warning(f"{radar_type}里程 CSV 没有有效时间/里程数据: {csv_path}")
            return None
        LOGGER.info(f"已按{radar_type}格式解析里程 CSV: {csv_path} ({column_count}列)")
        return mileage_map
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
