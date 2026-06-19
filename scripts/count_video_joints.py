#!/usr/bin/env python3
"""Experimental pipe joint counting utility.

This script is intentionally kept independent from the main watcher flow.
It reuses the validated temporal-clustering approach and can be packaged
alongside PipelineWatcher for later integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


@dataclass
class FrameDetection:
    frame_idx: int
    sample_idx: int
    time_s: float
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x: float
    center_y: float
    width: float
    height: float
    area_ratio: float
    center_dist_norm: float
    center_hit: int
    score: float


@dataclass
class JointEvent:
    event_id: int
    start_frame: int
    end_frame: int
    hit_count: int
    duration_frames: int
    best_frame: int
    best_time_s: float
    best_confidence: float
    best_area_ratio: float
    best_center_x: float
    best_center_y: float
    best_center_dist_norm: float
    mean_confidence: float
    area_threshold: float
    best_mileage: float | None
    delta_prev_mileage: float | None
    keep: int
    reject_reason: str


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="")
    parser.add_argument(
        "--weights",
        default=str(project_root / "weights" / "pipe_joint_best.pt"),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--sample-rate", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--max-det", type=int, default=3)
    parser.add_argument("--dark-threshold", type=float, default=55.0)
    parser.add_argument("--center-gate-x", type=float, default=0.22)
    parser.add_argument("--center-gate-y", type=float, default=0.22)
    parser.add_argument("--max-missed-samples", type=int, default=1)
    parser.add_argument("--merge-gap-samples", type=int, default=8)
    parser.add_argument("--min-cluster-hits", type=int, default=3)
    parser.add_argument("--strong-center-conf", type=float, default=0.35)
    parser.add_argument("--area-quantile", type=float, default=0.85)
    parser.add_argument("--min-area-threshold", type=float, default=0.55)
    parser.add_argument("--min-peak-area", type=float, default=0.65)
    parser.add_argument("--mileage-duplicate-ratio", type=float, default=0.45)
    parser.add_argument("--pipe-end-margin", type=float, default=0.35)
    parser.add_argument("--pipe-end-plateau-tol", type=float, default=0.08)
    parser.add_argument("--save-all-frames", action="store_true")
    return parser.parse_args()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "frames": output_dir / "frames",
        "events": output_dir / "events",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def compute_center_metrics(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
    center_gate_x: float,
    center_gate_y: float,
) -> tuple[float, float, float, float, float, int]:
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    dx = abs(center_x - frame_width / 2.0) / max(frame_width / 2.0, 1.0)
    dy = abs(center_y - frame_height / 2.0) / max(frame_height / 2.0, 1.0)
    center_dist_norm = math.sqrt(dx * dx + dy * dy)
    center_hit = int(dx <= center_gate_x and dy <= center_gate_y)
    return center_x, center_y, dx, dy, center_dist_norm, center_hit


def choose_best_detection(
    boxes,
    frame_width: int,
    frame_height: int,
    center_gate_x: float,
    center_gate_y: float,
) -> dict[str, float | int] | None:
    if boxes is None or len(boxes) == 0:
        return None

    best_payload = None
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[idx].tolist()]
        conf = float(boxes.conf[idx])
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area_ratio = (width * height) / max(frame_width * frame_height, 1.0)
        center_x, center_y, _, _, center_dist_norm, center_hit = compute_center_metrics(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
            center_gate_x,
            center_gate_y,
        )
        score = conf - 0.28 * center_dist_norm + 0.12 * min(area_ratio / 0.08, 1.0)
        if center_hit:
            score += 0.08
        payload = {
            "confidence": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
            "area_ratio": area_ratio,
            "center_dist_norm": center_dist_norm,
            "center_hit": center_hit,
            "score": score,
        }
        if best_payload is None or payload["score"] > best_payload["score"]:
            best_payload = payload
    return best_payload


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def parse_timestamp_column(column: pd.Series) -> pd.Series:
    return pd.to_datetime(column, errors="coerce", format="mixed")


def extract_time_from_csv_first_row(csv_path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python", nrows=1)
        if df.empty or df.shape[1] < 2:
            return None
        base_time = pd.to_datetime(df.iloc[0, 0], errors="coerce")
        if pd.isna(base_time):
            return None
        millis_idx = 1 if df.shape[1] == 14 else 4
        millis = pd.to_numeric(df.iloc[0, millis_idx], errors="coerce")
        if pd.notna(millis):
            return base_time + pd.to_timedelta(millis, unit="ms")
        return base_time
    except Exception:
        return None


def get_video_start_time(video_path: Path, csv_path: Path | None) -> pd.Timestamp | None:
    import re

    match = re.search(r"(\d{4}[-/]?\d{2}[-/]?\d{2})[-_](\d{6})", video_path.stem)
    if match:
        ts_str = (
            f"{match.group(1).replace('/', '-').replace('_', '-')}"
            f" {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
        )
        parsed = pd.to_datetime(ts_str, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if pd.notna(parsed):
            return parsed
    if csv_path and csv_path.exists():
        return extract_time_from_csv_first_row(csv_path)
    return None


def build_mileage_series(csv_path: Path) -> pd.Series | None:
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, header=None, sep=r"\s{2,}|,", engine="python")
        if df.shape[1] == 14:
            base_time = parse_timestamp_column(df[0])
            millis = pd.to_numeric(df[1], errors="coerce").fillna(0)
            timestamps = base_time + pd.to_timedelta(millis, unit="ms")
            mileage = pd.to_numeric(df[2], errors="coerce")
        elif df.shape[1] >= 6:
            base_time = parse_timestamp_column(df[0])
            millis = pd.to_numeric(df[4], errors="coerce").fillna(0)
            timestamps = base_time + pd.to_timedelta(millis, unit="ms")
            mileage = pd.to_numeric(df[5], errors="coerce")
        else:
            return None

        valid = timestamps.notna() & mileage.notna()
        if not valid.any():
            return None
        return mileage[valid].set_axis(timestamps[valid]).sort_index()
    except Exception:
        return None


def get_frame_mileage(
    frame_idx: int,
    start_time: pd.Timestamp | None,
    fps: float,
    mileage_map: pd.Series | None,
) -> float | None:
    if start_time is None or not fps or mileage_map is None or mileage_map.empty:
        return None
    target_time = start_time + pd.to_timedelta(frame_idx / fps, unit="s")
    try:
        loc = mileage_map.index.searchsorted(target_time)
        if loc == 0:
            return round(float(mileage_map.iloc[0]), 4)
        if loc >= len(mileage_map):
            return round(float(mileage_map.iloc[-1]), 4)
        t0, t1 = mileage_map.index[loc - 1], mileage_map.index[loc]
        m0, m1 = float(mileage_map.iloc[loc - 1]), float(mileage_map.iloc[loc])
        dt_total = (t1 - t0).value
        dt_frame = (target_time - t0).value
        value = m0 + (m1 - m0) * (dt_frame / dt_total) if dt_total > 0 else m0
        return round(value, 4)
    except Exception:
        return None


def cluster_detections(
    detections: list[FrameDetection],
    area_threshold: float,
    max_missed_samples: int,
    merge_gap_samples: int,
    min_cluster_hits: int,
    strong_center_conf: float,
    min_peak_area: float,
) -> list[JointEvent]:
    if not detections:
        return []

    filtered = [det for det in detections if det.center_hit and det.area_ratio >= area_threshold]
    if not filtered:
        return []

    clusters: list[list[FrameDetection]] = [[filtered[0]]]
    for det in filtered[1:]:
        prev = clusters[-1][-1]
        if det.sample_idx - prev.sample_idx <= max_missed_samples + 1:
            clusters[-1].append(det)
        else:
            clusters.append([det])

    merged_clusters: list[list[FrameDetection]] = []
    for cluster in clusters:
        if not merged_clusters:
            merged_clusters.append(cluster)
            continue
        previous = merged_clusters[-1]
        gap = cluster[0].sample_idx - previous[-1].sample_idx
        if gap <= merge_gap_samples:
            previous.extend(cluster)
        else:
            merged_clusters.append(cluster)

    events: list[JointEvent] = []
    for event_id, cluster in enumerate(merged_clusters, start=1):
        best = max(cluster, key=lambda item: (item.area_ratio, item.score))
        hit_count = len(cluster)
        mean_confidence = sum(item.confidence for item in cluster) / hit_count

        keep = 1
        reject_reason = ""
        if best.area_ratio < min_peak_area:
            keep = 0
            reject_reason = "peak_area_too_small"
        elif hit_count < min_cluster_hits and not (
            best.center_hit and best.confidence >= strong_center_conf and best.area_ratio >= 0.85
        ):
            keep = 0
            reject_reason = "too_few_hits"

        events.append(
            JointEvent(
                event_id=event_id,
                start_frame=cluster[0].frame_idx,
                end_frame=cluster[-1].frame_idx,
                hit_count=hit_count,
                duration_frames=cluster[-1].frame_idx - cluster[0].frame_idx,
                best_frame=best.frame_idx,
                best_time_s=best.time_s,
                best_confidence=best.confidence,
                best_area_ratio=best.area_ratio,
                best_center_x=best.center_x,
                best_center_y=best.center_y,
                best_center_dist_norm=best.center_dist_norm,
                mean_confidence=round(mean_confidence, 4),
                area_threshold=round(area_threshold, 4),
                best_mileage=None,
                delta_prev_mileage=None,
                keep=keep,
                reject_reason=reject_reason,
            )
        )
    return events


def enrich_events_with_mileage(
    events: list[JointEvent],
    csv_path: Path | None,
    video_path: Path,
    fps: float,
    duplicate_ratio: float,
) -> tuple[list[JointEvent], float | None, float | None]:
    if not csv_path or not csv_path.exists():
        return events, None, None

    mileage_map = build_mileage_series(csv_path)
    start_time = get_video_start_time(video_path, csv_path)
    if mileage_map is None or start_time is None:
        return events, None, None

    max_csv_mileage = round(float(mileage_map.max()), 4)

    enriched: list[JointEvent] = []
    prev_kept_mileage: float | None = None
    for event in events:
        mileage = get_frame_mileage(event.best_frame, start_time, fps, mileage_map)
        delta_prev = None
        if event.keep and mileage is not None and prev_kept_mileage is not None:
            delta_prev = round(mileage - prev_kept_mileage, 4)
        enriched.append(
            JointEvent(
                **{
                    **asdict(event),
                    "best_mileage": mileage,
                    "delta_prev_mileage": delta_prev,
                }
            )
        )
        if event.keep and mileage is not None:
            prev_kept_mileage = mileage

    positive_gaps = [
        event.delta_prev_mileage
        for event in enriched
        if event.keep and event.delta_prev_mileage is not None and event.delta_prev_mileage > 0
    ]
    if len(positive_gaps) < 2:
        return enriched, None, max_csv_mileage

    median_gap = round(float(pd.Series(positive_gaps).median()), 4)
    duplicate_gap_threshold = median_gap * duplicate_ratio

    adjusted = enriched[:]
    for idx in range(1, len(adjusted)):
        prev = adjusted[idx - 1]
        curr = adjusted[idx]
        if not prev.keep or not curr.keep:
            continue
        if curr.best_mileage is None or prev.best_mileage is None:
            continue
        gap = curr.best_mileage - prev.best_mileage
        if gap <= duplicate_gap_threshold:
            drop_prev = (
                prev.best_area_ratio < curr.best_area_ratio
                or (prev.best_area_ratio == curr.best_area_ratio and prev.best_confidence <= curr.best_confidence)
            )
            if drop_prev:
                adjusted[idx - 1] = JointEvent(
                    **{
                        **asdict(prev),
                        "keep": 0,
                        "reject_reason": f"duplicate_by_mileage<{duplicate_gap_threshold:.3f}",
                    }
                )
            else:
                adjusted[idx] = JointEvent(
                    **{
                        **asdict(curr),
                        "keep": 0,
                        "reject_reason": f"duplicate_by_mileage<{duplicate_gap_threshold:.3f}",
                    }
                )

    prev_kept_mileage = None
    final_events: list[JointEvent] = []
    for event in adjusted:
        delta_prev = None
        if event.keep and event.best_mileage is not None and prev_kept_mileage is not None:
            delta_prev = round(event.best_mileage - prev_kept_mileage, 4)
        final_event = JointEvent(
            **{
                **asdict(event),
                "delta_prev_mileage": delta_prev,
            }
        )
        final_events.append(final_event)
        if final_event.keep and final_event.best_mileage is not None:
            prev_kept_mileage = final_event.best_mileage

    return final_events, median_gap, max_csv_mileage


def prune_events_beyond_pipe_end(
    events: list[JointEvent],
    max_csv_mileage: float | None,
    pipe_end_margin: float,
    pipe_end_plateau_tol: float,
) -> list[JointEvent]:
    if max_csv_mileage is None:
        return events

    end_zone_start = max_csv_mileage - pipe_end_margin
    seen_terminal_event = False
    terminal_mileage: float | None = None
    adjusted: list[JointEvent] = []

    for event in events:
        if not event.keep or event.best_mileage is None:
            adjusted.append(event)
            continue

        if not seen_terminal_event:
            adjusted.append(event)
            if event.best_mileage >= end_zone_start:
                seen_terminal_event = True
                terminal_mileage = event.best_mileage
            continue

        same_plateau = terminal_mileage is not None and event.best_mileage >= terminal_mileage - pipe_end_plateau_tol
        in_end_zone = event.best_mileage >= end_zone_start
        if same_plateau or in_end_zone:
            adjusted.append(
                JointEvent(
                    **{
                        **asdict(event),
                        "keep": 0,
                        "reject_reason": f"beyond_pipe_end>={end_zone_start:.3f}",
                    }
                )
            )
        else:
            adjusted.append(event)

    prev_kept_mileage: float | None = None
    final_events: list[JointEvent] = []
    for event in adjusted:
        delta_prev = None
        if event.keep and event.best_mileage is not None and prev_kept_mileage is not None:
            delta_prev = round(event.best_mileage - prev_kept_mileage, 4)
        final_event = JointEvent(
            **{
                **asdict(event),
                "delta_prev_mileage": delta_prev,
            }
        )
        final_events.append(final_event)
        if final_event.keep and final_event.best_mileage is not None:
            prev_kept_mileage = final_event.best_mileage
    return final_events


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_csv_path(csv_arg: str, video_path: Path) -> Path | None:
    if csv_arg:
        csv_path = Path(csv_arg)
        return csv_path if csv_path.exists() else None

    nearby_candidates = sorted(video_path.parent.glob("*.csv"))
    if len(nearby_candidates) == 1:
        return nearby_candidates[0]
    for candidate in nearby_candidates:
        if candidate.stem == video_path.stem:
            return candidate
    return None


def main() -> None:
    args = parse_args()
    if not args.video:
        raise ValueError("请通过 --video 指定输入视频")

    video_path = Path(args.video)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir) if args.output_dir else (video_path.parent / "output_joint_count")
    csv_path = resolve_csv_path(args.csv, video_path)
    paths = ensure_dirs(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path}")
    print(f"Frames: {total_frames}, FPS: {fps:.3f}, Size: {frame_width}x{frame_height}")

    out_fps = min(max(fps / max(args.sample_rate, 1), 1.0), 8.0) if fps > 0 else 4.0
    writer = cv2.VideoWriter(
        str(output_dir / "joint_detection_sampled.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (frame_width, frame_height),
    )

    detections: list[FrameDetection] = []
    sampled_frames = 0
    dark_skipped = 0
    frame_idx = 0
    sample_idx = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.sample_rate != 0:
            frame_idx += 1
            continue

        sampled_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_gray = float(gray.mean())
        annotated = frame.copy()

        cv2.putText(
            annotated,
            f"frame={frame_idx} sample={sample_idx}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        best_detection = None
        if mean_gray < args.dark_threshold:
            dark_skipped += 1
            cv2.putText(
                annotated,
                f"dark skip mean={mean_gray:.1f}",
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        else:
            results = model(frame, conf=args.conf, verbose=False, max_det=args.max_det)
            boxes = results[0].boxes
            best_payload = choose_best_detection(
                boxes,
                frame_width,
                frame_height,
                args.center_gate_x,
                args.center_gate_y,
            )
            if best_payload is not None:
                best_detection = FrameDetection(
                    frame_idx=frame_idx,
                    sample_idx=sample_idx,
                    time_s=(frame_idx / fps) if fps > 0 else 0.0,
                    confidence=round(float(best_payload["confidence"]), 4),
                    x1=round(float(best_payload["x1"]), 2),
                    y1=round(float(best_payload["y1"]), 2),
                    x2=round(float(best_payload["x2"]), 2),
                    y2=round(float(best_payload["y2"]), 2),
                    center_x=round(float(best_payload["center_x"]), 2),
                    center_y=round(float(best_payload["center_y"]), 2),
                    width=round(float(best_payload["width"]), 2),
                    height=round(float(best_payload["height"]), 2),
                    area_ratio=round(float(best_payload["area_ratio"]), 6),
                    center_dist_norm=round(float(best_payload["center_dist_norm"]), 4),
                    center_hit=int(best_payload["center_hit"]),
                    score=round(float(best_payload["score"]), 4),
                )
                detections.append(best_detection)
                x1 = int(best_detection.x1)
                y1 = int(best_detection.y1)
                x2 = int(best_detection.x2)
                y2 = int(best_detection.y2)
                color = (0, 255, 0) if best_detection.center_hit else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"joint conf={best_detection.confidence:.2f} dist={best_detection.center_dist_norm:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
            else:
                cv2.putText(
                    annotated,
                    "no joint",
                    (20, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

        half_w = frame_width // 2
        half_h = frame_height // 2
        gate_x = int(half_w * args.center_gate_x)
        gate_y = int(half_h * args.center_gate_y)
        cv2.rectangle(
            annotated,
            (half_w - gate_x, half_h - gate_y),
            (half_w + gate_x, half_h + gate_y),
            (255, 0, 0),
            2,
        )

        if args.save_all_frames:
            cv2.imwrite(str(paths["frames"] / f"frame_{frame_idx:06d}.jpg"), annotated)
        writer.write(annotated)

        if sampled_frames % 50 == 0:
            elapsed = time.time() - started
            rate = sampled_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"[{sampled_frames}] frame={frame_idx} dets={len(detections)} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}fps"
            )

        sample_idx += 1
        frame_idx += 1

    cap.release()
    writer.release()

    area_values = [det.area_ratio for det in detections if det.center_hit]
    area_threshold = max(args.min_area_threshold, quantile(area_values, args.area_quantile))

    events = cluster_detections(
        detections,
        area_threshold=area_threshold,
        max_missed_samples=args.max_missed_samples,
        merge_gap_samples=args.merge_gap_samples,
        min_cluster_hits=args.min_cluster_hits,
        strong_center_conf=args.strong_center_conf,
        min_peak_area=args.min_peak_area,
    )
    events, median_mileage_gap, max_csv_mileage = enrich_events_with_mileage(
        events,
        csv_path=csv_path,
        video_path=video_path,
        fps=fps,
        duplicate_ratio=args.mileage_duplicate_ratio,
    )
    events = prune_events_beyond_pipe_end(
        events,
        max_csv_mileage=max_csv_mileage,
        pipe_end_margin=args.pipe_end_margin,
        pipe_end_plateau_tol=args.pipe_end_plateau_tol,
    )
    kept_events = [event for event in events if event.keep]
    keep_map = {event.best_frame: event for event in kept_events}

    cap = cv2.VideoCapture(str(video_path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        event = keep_map.get(current)
        if event is None:
            continue
        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"event={event.event_id} frame={event.best_frame} conf={event.best_confidence:.2f}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(
            str(paths["events"] / f"event_{event.event_id:03d}_frame_{event.best_frame:06d}.jpg"),
            annotated,
        )
    cap.release()

    raw_rows = [asdict(det) for det in detections]
    event_rows = [asdict(event) for event in events]
    if raw_rows:
        write_csv(output_dir / "joint_raw_detections.csv", raw_rows, list(raw_rows[0].keys()))
    if event_rows:
        write_csv(output_dir / "joint_events.csv", event_rows, list(event_rows[0].keys()))

    summary = {
        "video": str(video_path),
        "weights": str(weights_path),
        "csv": str(csv_path) if csv_path else "",
        "sample_rate": args.sample_rate,
        "conf": args.conf,
        "max_det": args.max_det,
        "dark_threshold": args.dark_threshold,
        "center_gate_x": args.center_gate_x,
        "center_gate_y": args.center_gate_y,
        "max_missed_samples": args.max_missed_samples,
        "merge_gap_samples": args.merge_gap_samples,
        "min_cluster_hits": args.min_cluster_hits,
        "strong_center_conf": args.strong_center_conf,
        "area_quantile": args.area_quantile,
        "min_area_threshold": args.min_area_threshold,
        "computed_area_threshold": round(area_threshold, 4),
        "min_peak_area": args.min_peak_area,
        "mileage_duplicate_ratio": args.mileage_duplicate_ratio,
        "median_mileage_gap": median_mileage_gap,
        "max_csv_mileage": max_csv_mileage,
        "pipe_end_margin": args.pipe_end_margin,
        "pipe_end_plateau_tol": args.pipe_end_plateau_tol,
        "pipe_end_zone_start": round(max_csv_mileage - args.pipe_end_margin, 4) if max_csv_mileage is not None else None,
        "total_frames": total_frames,
        "fps": fps,
        "sampled_frames": sampled_frames,
        "dark_skipped_frames": dark_skipped,
        "raw_detection_count": len(detections),
        "joint_event_count": len(kept_events),
        "rejected_event_count": len(events) - len(kept_events),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (output_dir / "joint_count_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("JOINT COUNT SUMMARY")
    print("=" * 60)
    print(f"Sampled frames: {sampled_frames}")
    print(f"Dark skipped: {dark_skipped}")
    print(f"Raw detections: {len(detections)}")
    print(f"Computed area threshold: {area_threshold:.4f}")
    if csv_path:
        print(f"CSV mileage: {csv_path}")
        print(f"Median mileage gap: {median_mileage_gap if median_mileage_gap is not None else 'N/A'}")
        print(f"Max CSV mileage: {max_csv_mileage if max_csv_mileage is not None else 'N/A'}")
        if max_csv_mileage is not None:
            print(f"Pipe end zone start: {max_csv_mileage - args.pipe_end_margin:.4f}")
    print(f"Joint events kept: {len(kept_events)}")
    print(f"Joint events rejected: {len(events) - len(kept_events)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
