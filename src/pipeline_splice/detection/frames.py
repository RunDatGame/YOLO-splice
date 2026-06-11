import glob
import json
import os
import sys
from functools import lru_cache

import cv2


def save_frames(video_path, output_folder, interval=None):
    print(f"\n[步骤 1] 视频抽帧处理")
    os.makedirs(output_folder, exist_ok=True)
    if interval is None:
        interval = 1000

    video_path = os.path.abspath(video_path)
    info_file = os.path.join(output_folder, "extraction_info.txt")
    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                cache_info = json.load(f)
            video_stat = os.stat(video_path)
            if (
                cache_info.get("video_path") == video_path
                and cache_info.get("interval") == interval
                and cache_info.get("video_mtime") == video_stat.st_mtime
                and cache_info.get("video_size") == video_stat.st_size
            ):
                files = glob.glob(os.path.join(output_folder, "frame_*.png"))
                if files:
                    cap = cv2.VideoCapture(video_path)
                    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
                    cap.release()
                    print(f"[INFO] 命中缓存，跳过抽帧。")
                    return len(files), w, h, fps
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    for f in glob.glob(os.path.join(output_folder, "frame_*.png")):
        os.remove(f)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, None, None, None
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    cnt, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{cnt}.png"), frame)
            saved += 1
            if saved % 20 == 0:
                sys.stdout.write(f"\r  -> 已提取: {saved} 帧...")
                sys.stdout.flush()
        cnt += 1
    cap.release()
    print("\n[INFO] 抽帧完成。")
    try:
        video_stat = os.stat(video_path)
        cache_info = {
            "video_path": video_path,
            "interval": interval,
            "video_mtime": video_stat.st_mtime,
            "video_size": video_stat.st_size,
        }
    except OSError:
        cache_info = {"video_path": video_path, "interval": interval}
    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(cache_info, f)
    return saved, w, h, fps


def filter_valid_frames(video_path, frame_dir, csv_path, fps):
    from .mileage import (
        get_frame_mileage,
        get_frame_rotation,
        get_video_start_time,
        load_csv_mileage_map,
        load_csv_rotation_map,
    )

    print(f"\n[步骤 2] 里程同步处理")
    start_time = get_video_start_time(video_path, csv_path)
    if start_time is None:
        return []
    m_map = load_csv_mileage_map(csv_path)
    if m_map is None:
        return []
    rotation_y_map = load_csv_rotation_map(csv_path, axis="y")
    start_mileage_abs = get_frame_mileage(video_path, 1, start_time, fps, m_map)
    files = sorted(
        glob.glob(os.path.join(frame_dir, "frame_*.png")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    valid = []
    for p in files:
        try:
            n = int(os.path.basename(p).split("_")[-1].split(".")[0])
            m_abs = get_frame_mileage(video_path, n, start_time, fps, m_map)
            if m_abs is not None:
                rotation_y = get_frame_rotation(video_path, n, start_time, fps, rotation_y_map)
                valid.append((p, m_abs, m_abs - start_mileage_abs, rotation_y))
        except (ValueError, IndexError):
            continue
    print(f"[INFO] 有效同步帧数: {len(valid)}")
    return valid


@lru_cache(maxsize=256)
def _load_frame_cached(frame_path: str):
    """LRU 缓存的帧加载器，防止处理长视频时内存无限增长。"""
    return cv2.imread(frame_path)
