from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline_splice.core.config import parse_config_file
from pipeline_splice.detection.mileage import get_frame_mileage, get_video_start_time, load_csv_mileage_map


@dataclass(frozen=True)
class FrameSample:
    frame_index: int
    mileage_abs: float
    mileage_rel: float
    image: np.ndarray


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_samples_from_video(
    video_path: Path,
    interval: int,
    start_t,
    mileage_map,
) -> list[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"视频无法读取: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError(f"无法获取视频 FPS: {video_path}")

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

        mileage_abs = get_frame_mileage(video_path.stem, frame_index, start_t, fps, mileage_map)
        if mileage_abs is None:
            frame_index += 1
            continue
        if start_mileage is None:
            start_mileage = mileage_abs

        samples.append(FrameSample(frame_index, mileage_abs, mileage_abs - start_mileage, frame))
        frame_index += 1

    cap.release()
    return samples


def _load_samples_from_frames(
    frame_dir: Path,
    start_t,
    fps: float,
    mileage_map,
) -> list[FrameSample]:
    import re

    samples: list[FrameSample] = []
    for frame_path in sorted(frame_dir.glob("frame_*.*")):
        match = re.search(r"frame_(\d+)", frame_path.stem)
        if not match:
            continue
        frame_index = int(match.group(1))
        mileage_abs = get_frame_mileage(frame_path.stem, frame_index, start_t, fps, mileage_map)
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


def _build_unwrapped_polar(
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


def _render_texture(
    samples: list[FrameSample],
    total_length: float,
    texture_width: int,
    pixels_per_meter: int,
    sample_radius_ratio: float,
    center_offset_x: float,
    center_offset_y: float,
) -> tuple[np.ndarray, list[dict]]:
    if not samples:
        raise RuntimeError("没有可用帧样本，无法生成纹理")

    total_length = max(total_length, 0.1)
    texture_height = max(512, int(math.ceil(total_length * pixels_per_meter)))
    canvas = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
    debug_rows: list[dict] = []

    unwrapped_frames: list[np.ndarray] = []
    for sample in samples:
        unwrapped = _build_unwrapped_polar(
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
        prev_gray = cv2.cvtColor(unwrapped_frames[i - 1][ref_start:ref_end, :], cv2.COLOR_BGR2GRAY)
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
            if raw_patch.shape[0] == 0:
                continue
            patch = cv2.resize(raw_patch, (texture_width, actual_h), interpolation=cv2.INTER_LINEAR)
        else:
            patch = unwrapped_frames[idx][row_start:row_end, :].copy()

        # 修复Y轴倒置
        patch = patch[::-1, :, :]

        # 周向旋转对齐
        shift_px = int(round(shifts[idx]))
        if shift_px != 0:
            patch = np.roll(patch, shift_px, axis=1)

        # 曝光补偿
        if idx > 0 and actual_h > 1 and y0 > 0:
            prev_row = canvas[y0 - 1, :, :].astype(np.float32)
            curr_row = patch[0, :, :].astype(np.float32)
            mean_prev = np.mean(prev_row)
            mean_curr = np.mean(curr_row)
            if mean_curr > 5.0 and mean_prev > 5.0:
                gain = mean_prev / mean_curr
                gain = max(0.6, min(1.6, gain))
                gain_gradient = np.linspace(gain, 1.0, actual_h).reshape(-1, 1, 1)
                patch = np.clip(patch.astype(np.float32) * gain_gradient, 0, 255).astype(np.uint8)

        canvas[y0:y1, :, :] = patch

        debug_rows.append({
            "frame_index": sample.frame_index,
            "mileage_rel": round(sample.mileage_rel, 6),
            "y_start": y0,
            "y_end": y1,
            "shift_px": shift_px,
        })

    return canvas, debug_rows


def _write_obj_with_texture(
    output_dir: Path,
    stem: str,
    radius: float,
    length: float,
    radial_segments: int,
    length_segments: int,
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
        for x, y, z in vertices:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in uvs:
            fh.write(f"vt {u:.6f} {v:.6f}\n")
        fh.write("usemtl PipeTexture\n")
        for a, b, c, d in faces:
            fh.write(f"f {a}/{a} {b}/{b} {c}/{c} {d}/{d}\n")

    return obj_path, mtl_path


def build_uv_pipeline(
    video_path: Path,
    csv_path: Path,
    work_dir: Path,
    config_path: Path | None = None,
    frames_dir: Path | None = None,
    interval: int | None = None,
    texture_width: int = 2048,
    pixels_per_meter: int = 512,
    sample_radius_ratio: float = 0.85,
    center_offset_x: float = 0.0,
    center_offset_y: float = 0.0,
    radius_source: str = "inner",
    radial_segments: int = 256,
    length_segments: int = 512,
    output_name: str = "cylindrical_texture_pipe",
) -> dict:
    """UV 圆柱纹理重建主入口。"""
    _ensure_dir(work_dir)

    if config_path is not None and config_path.exists():
        defaults = parse_config_file(config_path)
    else:
        defaults = {}

    interval = interval or int(defaults.get("interval", 30))
    inner = float(defaults.get("inner", 0.54))
    outer = float(defaults.get("outer", 0.60))

    if not video_path.exists() or not csv_path.exists():
        raise FileNotFoundError("视频或 CSV 路径无效")

    start_t = get_video_start_time(str(video_path))
    mileage_map = load_csv_mileage_map(str(csv_path))

    output_dir = work_dir / "outputs" / "uv_texture" / video_path.stem
    _ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frames_dir is not None and frames_dir.exists() and any(frames_dir.glob("frame_*.*")):
        samples = _load_samples_from_frames(frames_dir, start_t, fps, mileage_map)
    else:
        samples = _extract_samples_from_video(video_path, interval, start_t, mileage_map)

    if not samples:
        raise RuntimeError("未能提取任何有效帧样本")

    total_length = max(samples[-1].mileage_rel, 0.1)
    radius = (inner if radius_source == "inner" else outer) / 2.0

    texture, debug_rows = _render_texture(
        samples=samples,
        total_length=total_length,
        texture_width=texture_width,
        pixels_per_meter=pixels_per_meter,
        sample_radius_ratio=sample_radius_ratio,
        center_offset_x=center_offset_x,
        center_offset_y=center_offset_y,
    )

    texture_path = output_dir / f"{output_name}.png"
    Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)).save(texture_path)

    obj_path, mtl_path = _write_obj_with_texture(
        output_dir, output_name, radius, total_length, radial_segments, length_segments
    )

    return {
        "total_length": total_length,
        "radius": radius,
        "texture_path": texture_path,
        "obj_path": obj_path,
        "mtl_path": mtl_path,
        "output_dir": output_dir,
        "sample_count": len(samples),
        "debug_rows": debug_rows,
    }
