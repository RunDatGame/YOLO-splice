import os
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from ._common import ROOT

_defect_id_counter = 0


def save_defect_screenshot(img, bbox, defect_info, output_dir):
    full_dir = os.path.join(ROOT, "runs", "detect", output_dir)
    os.makedirs(full_dir, exist_ok=True)
    if img is None:
        return None
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = img.shape[:2]
    pad_x, pad_y = max(30, int((x2 - x1) * 0.2)), max(30, int((y2 - y1) * 0.2))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    fname = f"{defect_info['编号']}_{defect_info['模型类型']}_管节{defect_info['管节序号']}.png"
    path = os.path.join(full_dir, fname)
    if cv2.imwrite(path, roi):
        return path
    return None


def generate_defect_id() -> str:
    global _defect_id_counter
    _defect_id_counter += 1
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    return f"InerDis{timestamp}{_defect_id_counter:03d}"


def process_best_defects(best_defects, result_dir, pipe_params, device="cpu", output_dir=None, use_depth=True, wall_thickness=0.1, rebar_spacing=0.0):
    from .depth import calculate_along_pipe_distance, get_frame_depth_map
    from .frames import _load_frame_cached

    print(f"\n[步骤 4] 分析与导出")
    if not best_defects:
        print("未发现病害")
        return
    pipe_inner, pipe_outer, seg_len, start_seg = pipe_params
    final_res = []

    shot_dir = "defect_screenshots"

    for seg_idx, defects in sorted(best_defects.items()):
        frame_groups = {}
        for m_type, d in defects.items():
            frame_groups.setdefault(d["frame_path"], []).append((m_type, d))

        for frame_path, items in frame_groups.items():
            curr_img = _load_frame_cached(frame_path)
            if curr_img is None:
                continue

            depth_map = get_frame_depth_map(curr_img) if use_depth else None

            for m_type, d in items:
                if use_depth and depth_map is not None:
                    x, y = int(d["bbox_center"][0]), int(d["bbox_center"][1])
                    dh, dw = depth_map.shape
                    depth = float(depth_map[min(y, dh - 1), min(x, dw - 1)])
                else:
                    depth = None

                segment_relative_mileage = d["absolute_mileage"] % seg_len
                final_segment_mileage = segment_relative_mileage + calculate_along_pipe_distance(
                    depth, pipe_inner / 2
                )
                final_segment_mileage = min(final_segment_mileage, seg_len)

                info = {
                    "编号": generate_defect_id(),
                    "模型类型": d["model_type"],
                    "管节序号": seg_idx,
                    "管节内径": pipe_inner,
                    "管节外径": pipe_outer,
                    "管节长度": seg_len,
                    "管道壁厚": wall_thickness,
                    "钢筋间距": rebar_spacing,
                    "节内里程": round(final_segment_mileage, 3),
                    "偏移距": d.get("offset", ""),
                    "轴线偏角": d.get("axis_angle", ""),
                    "病害长": d["length"],
                    "病害宽": d["width"],
                    "病害高": d.get("height", ""),
                    "严重等级": str(d["severity_val"]),
                    "模型路径": "",
                    "数据截图": "",
                }
                s_path = save_defect_screenshot(curr_img, d["bbox"], info, shot_dir)
                if s_path:
                    info["数据截图"] = s_path
                final_res.append(info)

    if final_res:
        # 每管节只保留严重等级最高的2个病害
        seg_groups = {}
        for d in final_res:
            sid = d.get("管节序号", 0)
            seg_groups.setdefault(sid, []).append(d)
        filtered = []
        for sid, items in seg_groups.items():
            items.sort(key=lambda x: (int(str(x.get("严重等级", "1"))), float(str(x.get("病害长", 0))) * float(str(x.get("病害宽", 0)))), reverse=True)
            filtered.extend(items[:2])
        final_res = filtered

        csv_output_dir = Path(output_dir) if output_dir else ROOT
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(final_res).to_csv(
            csv_output_dir / "defect_results_full.csv", index=False, encoding="utf-8-sig"
        )

        print(f"导出完成: {csv_output_dir / 'defect_results_full.csv'}")
