from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "cylindrical_texture.txt"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline_core.config import parse_config_file

@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    mileage_abs: float
    mileage_rel: float
    image: np.ndarray

def load_defaults(work_dir: Path, config_path: Path | None = None) -> dict[str, str | int | float]:
    if config_path is not None and config_path.exists():
        return parse_config_file(config_path)
    candidates = [
        work_dir / "config.txt",
        SCRIPT_DIR / "config.txt",
        work_dir / "YOLO" / "config.txt",
    ]
    for config_path in candidates:
        if config_path.exists():
            return parse_config_file(config_path)
    return {}

def resolve_input_path(work_dir: Path, raw_value: str | Path | None) -> Path | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return (work_dir / path).resolve()

def get_video_start_time(video_path: Path) -> pd.Timestamp | None:
    match = re.search(r"(\d{4}[-/]?\d{2}[-/]?\d{2})[-_](\d{6})", video_path.stem)
    if not match:
        return None
    ts_str = (
        f"{match.group(1).replace('/', '-').replace('_', '-')} "
        f"{match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
    )
    try:
        return pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    except ValueError:
        return None

def load_csv_mileage_map(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python")
    except Exception:
        return None
    if df.shape[1] < 6:
        return None

    base_time = pd.to_datetime(df[0], errors="coerce")
    seconds = pd.to_numeric(df[3], errors="coerce").fillna(0)
    millis = pd.to_numeric(df[4], errors="coerce").fillna(0)
    full_timestamps = base_time + pd.to_timedelta(seconds, unit="s") + pd.to_timedelta(millis, unit="ms")
    mileage = pd.to_numeric(df[5], errors="coerce")

    valid = full_timestamps.notna() & mileage.notna()
    if not valid.any():
        return None
    return mileage[valid].set_axis(full_timestamps[valid]).sort_index()

def get_frame_mileage(start_t: pd.Timestamp, fps: float, frame_index: int, mileage_map: pd.Series) -> float | None:
    if start_t is None or fps <= 0 or mileage_map is None or mileage_map.empty:
        return None

    target = start_t + pd.to_timedelta(frame_index / fps, unit="s")
    loc = mileage_map.index.searchsorted(target)
    if loc <= 0:
        return float(mileage_map.iloc[0])
    if loc >= len(mileage_map):
        return float(mileage_map.iloc[-1])

    t0, t1 = mileage_map.index[loc - 1], mileage_map.index[loc]
    m0, m1 = float(mileage_map.iloc[loc - 1]), float(mileage_map.iloc[loc])
    dt_total = (t1 - t0).value
    if dt_total <= 0:
        return m0
    dt_frame = (target - t0).value
    return float(m0 + (m1 - m0) * (dt_frame / dt_total))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def load_samples_from_frames(
    frame_dir: Path,
    start_t: pd.Timestamp,
    fps: float,
    mileage_map: pd.Series,
) -> list[FrameSample]:
    samples: list[FrameSample] = []
    for frame_path in sorted(frame_dir.glob("frame_*.*")):
        match = re.search(r"frame_(\d+)", frame_path.stem)
        if not match:
            continue
        frame_index = int(match.group(1))
        mileage_abs = get_frame_mileage(start_t, fps, frame_index, mileage_map)
        if mileage_abs is None:
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        samples.append(FrameSample(frame_index, mileage_abs, 0.0, image))

    if not samples:
        return []

    start_mileage = samples[0].mileage_abs
    return [
        FrameSample(s.frame_index, s.mileage_abs, s.mileage_abs - start_mileage, s.image)
        for s in samples
    ]

def extract_samples_from_video(
    video_path: Path,
    interval: int,
    start_t: pd.Timestamp,
    mileage_map: pd.Series,
    save_dir: Path | None = None,
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"视频无法读取: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"无法获取视频 FPS: {video_path}")

    if save_dir is not None:
        ensure_dir(save_dir)

    samples: list[FrameSample] = []
    frame_index = 0
    start_mileage: float | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % interval != 0:
            frame_index += 1
            continue

        mileage_abs = get_frame_mileage(start_t, fps, frame_index, mileage_map)
        if mileage_abs is None:
            frame_index += 1
            continue
        if start_mileage is None:
            start_mileage = mileage_abs

        if save_dir is not None:
            cv2.imwrite(str(save_dir / f"frame_{frame_index}.png"), frame)

        samples.append(FrameSample(frame_index, mileage_abs, mileage_abs - start_mileage, frame))
        frame_index += 1

    cap.release()
    return samples

def build_unwrapped_polar(
    image: np.ndarray,
    angle_samples: int,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
) -> np.ndarray:
    height, width = image.shape[:2]
    center = (width * (0.5 + center_offset_x), height * (0.5 + center_offset_y))
    max_radius = min(width, height) * 0.48
    radial_samples = max(512, int(max_radius))

    polar = cv2.warpPolar(
        image,
        (radial_samples, angle_samples), 
        center,
        max_radius,
        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR,
    )
    
    unwrapped = np.transpose(polar, (1, 0, 2))
    return np.ascontiguousarray(unwrapped)

def render_texture(
    samples: list[FrameSample],
    total_length: float,
    texture_width: int,
    pixels_per_meter: int,
    sample_radius_ratio: float,
    center_offset_x: float,
    center_offset_y: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if not samples:
        raise RuntimeError("没有可用帧样本，无法生成纹理")

    total_length = max(total_length, 0.1)
    texture_height = max(512, int(math.ceil(total_length * pixels_per_meter)))
    canvas = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
    debug_rows: list[dict[str, float]] = []

    unwrapped_frames: list[np.ndarray] = []
    for sample in samples:
        unwrapped = build_unwrapped_polar(
            sample.image,
            angle_samples=texture_width,
            center_offset_x=center_offset_x,
            center_offset_y=center_offset_y,
        )
        unwrapped_frames.append(unwrapped)

    radial_samples = unwrapped_frames[0].shape[0]

    shifts = [0.0]
    accumulated_shift = 0.0
    ref_start = int(radial_samples * 0.70)
    ref_end = int(radial_samples * 0.95)

    for i in range(1, len(unwrapped_frames)):
        prev_gray = cv2.cvtColor(unwrapped_frames[i-1][ref_start:ref_end, :], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(unwrapped_frames[i][ref_start:ref_end, :], cv2.COLOR_BGR2GRAY)
        
        shift_tuple, _ = cv2.phaseCorrelate(np.float32(prev_gray), np.float32(curr_gray))
        dx = shift_tuple[0]
        
        if abs(dx) > texture_width * 0.15:
            dx = 0.0
            
        accumulated_shift -= dx
        shifts.append(accumulated_shift)

    sample_y_center = int(radial_samples * sample_radius_ratio)

    for idx, sample in enumerate(samples):
        y0 = int(round((sample.mileage_rel / total_length) * (texture_height - 1)))
        
        if idx + 1 < len(samples):
            next_rel = samples[idx + 1].mileage_rel
            y1 = int(round((next_rel / total_length) * (texture_height - 1)))
        else:
            y1 = y0 + max(1, int(texture_height / len(samples))) 

        y0 = max(0, min(texture_height - 1, y0))
        y1 = max(y0 + 1, min(texture_height, y1))
        actual_h = y1 - y0
        
        if actual_h <= 0:
            continue

        row_start = sample_y_center - actual_h // 2
        row_end = row_start + actual_h

        if row_start < 0 or row_end > radial_samples:
            row_start = max(0, row_start)
            row_end = min(radial_samples, row_end)
            raw_patch = unwrapped_frames[idx][row_start:row_end, :]
            if raw_patch.shape[0] == 0: continue
            patch = cv2.resize(raw_patch, (texture_width, actual_h), interpolation=cv2.INTER_LINEAR)
        else:
            patch = unwrapped_frames[idx][row_start:row_end, :].copy()

        # 修复Y轴倒置
        patch = patch[::-1, :, :]

        # 周向旋转对齐
        shift_px = int(round(shifts[idx]))
        if shift_px != 0:
            patch = np.roll(patch, shift_px, axis=1)

        # ====== V6 核心：无损曝光补偿 (彻底丢弃 Alpha 重叠) ======
        # 如果不是第一帧，并且切片高度足够，我们通过调光来消除接缝亮度差
        if idx > 0 and actual_h > 1 and y0 > 0:
            # 获取画布上上一帧最后一行的颜色
            prev_row = canvas[y0 - 1, :, :].astype(np.float32)
            # 获取当前帧第一行的颜色
            curr_row = patch[0, :, :].astype(np.float32)

            # 计算平均亮度
            mean_prev = np.mean(prev_row)
            mean_curr = np.mean(curr_row)

            # 防止黑图导致除以0
            if mean_curr > 5.0 and mean_prev > 5.0:
                # 计算亮度倍率 (例如：上一帧比当前帧亮20%，gain就是1.2)
                gain = mean_prev / mean_curr
                # 限制最大最小倍率，防止过曝成纯白或死黑
                gain = max(0.6, min(1.6, gain))

                # 构建一个渐变倍率矩阵：
                # 第一行强行等于上一帧的亮度，最后一行恢复当前帧1.0的自然亮度
                gain_gradient = np.linspace(gain, 1.0, actual_h).reshape(-1, 1, 1)
                
                # 直接将光照补丁乘到当前切片上
                patch = np.clip(patch.astype(np.float32) * gain_gradient, 0, 255).astype(np.uint8)
        # =========================================================

        # 硬切硬贴：一个像素都不偏移，绝对保护几何病害位置
        canvas[y0:y1, :, :] = patch

        debug_rows.append({
            "frame_index": sample.frame_index,
            "mileage_rel": round(sample.mileage_rel, 6),
            "y_start": y0, "y_end": y1, "shift_px": shift_px
        })

    return canvas, debug_rows

def write_obj_with_texture(
    output_dir: Path, stem: str, radius: float, length: float, radial_segments: int, length_segments: int
) -> tuple[Path, Path]:
    obj_path = output_dir / f"{stem}.obj"
    mtl_path = output_dir / f"{stem}.mtl"
    texture_name = f"{stem}.png"

    vertices, uvs, faces = [], [], []

    for i in range(length_segments + 1):
        z = length * (i / length_segments)
        v = i / length_segments
        for j in range(radial_segments + 1):
            angle = 2.0 * math.pi * (j / radial_segments)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            u = j / radial_segments
            vertices.append((x, y, z))
            uvs.append((u, 1.0 - v))

    ring = radial_segments + 1
    for i in range(length_segments):
        for j in range(radial_segments):
            a = i * ring + j + 1
            b = a + 1
            c = a + ring + 1
            d = a + ring
            faces.append((a, b, c, d))

    with mtl_path.open("w", encoding="utf-8") as fh:
        fh.write("newmtl PipeTexture\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n")
        fh.write(f"map_Kd {texture_name}\n")

    with obj_path.open("w", encoding="utf-8") as fh:
        fh.write(f"mtllib {mtl_path.name}\no CylindricalTexturePipe\n")
        for x, y, z in vertices: fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs: fh.write(f"vt {u:.6f} {v:.6f}\n")
        fh.write("usemtl PipeTexture\n")
        for a, b, c, d in faces: fh.write(f"f {a}/{a} {b}/{b} {c}/{c} {d}/{d}\n")

    return obj_path, mtl_path

def main() -> None:
    parser = argparse.ArgumentParser(description="基于里程的圆柱纹理重建原型 V6 (几何高精+无损曝光补偿)")
    parser.add_argument("--video", help="视频路径，默认读取 config.txt")
    parser.add_argument("--csv", help="里程 CSV 路径，默认读取 config.txt 的 raw")
    parser.add_argument("--work-dir", default=".", help="项目工作目录")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--frames-dir", help="已存在的切帧目录，优先使用")
    parser.add_argument("--save-frames", action="store_true", help="保存采样帧")
    parser.add_argument("--interval", type=int, help="抽帧间隔")
    parser.add_argument("--texture-width", type=int, default=2048, help="圆柱纹理宽度")
    parser.add_argument("--pixels-per-meter", type=int, default=512, help="轴向每米像素")
    parser.add_argument("--sample-radius-ratio", type=float, default=0.85, help="取样半径比例(建议0.8-0.95)")
    parser.add_argument("--center-offset-x", type=float, default=0.0, help="图像中心 X 偏移")
    parser.add_argument("--center-offset-y", type=float, default=0.0, help="图像中心 Y 偏移")
    parser.add_argument("--radius-source", choices=["inner", "outer"], default="inner", help="半径使用内/外径")
    parser.add_argument("--radial-segments", type=int, default=256, help="输出周向分段数")
    parser.add_argument("--length-segments", type=int, default=512, help="输出轴向分段数")
    parser.add_argument("--output-name", default="cylindrical_texture_pipe", help="输出前缀")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    config_path = Path(args.config).resolve() if args.config else None
    defaults = load_defaults(work_dir, config_path)
    video_path = resolve_input_path(work_dir, args.video if args.video else defaults.get("video"))
    csv_path = resolve_input_path(work_dir, args.csv if args.csv else defaults.get("raw"))
    interval = args.interval or int(defaults.get("interval", 30))
    inner = float(defaults.get("inner", 0.54))
    outer = float(defaults.get("outer", 0.60))

    if not video_path or not csv_path or not video_path.exists() or not csv_path.exists():
        raise FileNotFoundError("文件路径错误，请检查 video 或 csv 配置。")

    start_t = get_video_start_time(video_path)
    mileage_map = load_csv_mileage_map(csv_path)

    output_dir = work_dir / "outputs" / "cylindrical_texture" / video_path.stem
    ensure_dir(output_dir)

    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else work_dir / "runs" / "detect" / video_path.stem / "frames"
    saved_frames_dir = output_dir / "sampled_frames" if args.save_frames else None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frames_dir.exists() and any(frames_dir.glob("frame_*.*")):
        samples = load_samples_from_frames(frames_dir, start_t, fps, mileage_map)
    else:
        samples = extract_samples_from_video(video_path, interval, start_t, mileage_map, save_dir=saved_frames_dir)

    total_length = max(samples[-1].mileage_rel, 0.1)
    radius = (inner if args.radius_source == "inner" else outer) / 2.0

    texture, debug_rows = render_texture(
        samples=samples,
        total_length=total_length,
        texture_width=args.texture_width,
        pixels_per_meter=args.pixels_per_meter,
        sample_radius_ratio=args.sample_radius_ratio,
        center_offset_x=args.center_offset_x,
        center_offset_y=args.center_offset_y,
    )

    texture_path = output_dir / f"{args.output_name}.png"
    Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)).save(texture_path)
    obj_path, mtl_path = write_obj_with_texture(
        output_dir, args.output_name, radius, total_length, args.radial_segments, args.length_segments
    )

    print(f"长度: {total_length:.3f} m, 纹理输出: {texture_path}")

if __name__ == "__main__":
    main()
