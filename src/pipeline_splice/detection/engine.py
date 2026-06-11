# src/pipeline_splice/detection/engine.py
"""检测引擎主入口，向后兼容地重新导出各子模块的公共 API。"""

import argparse
import os
from pathlib import Path

from utils.general import LOGGER

from ._common import ROOT, device, get_resource_path
from .depth import calculate_along_pipe_distance, get_frame_depth_map, init_depth_model
from .detector import (
    DEFECT_TYPE_MAPPING,
    LetterBox,
    calculate_defect_dimensions,
    filter_best_defects,
    get_angle_range,
    suppress_stdout,
)
from .export import process_best_defects, save_defect_screenshot
from .frames import _load_frame_cached, filter_valid_frames, save_frames
from .mileage import get_frame_mileage, get_frame_roll, get_video_start_time, load_csv_mileage_map, load_csv_roll_map


def read_config(path):
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line and not line.startswith("#"):
                k, v = [x.strip() for x in line.split(":", 1)]
                if k in ["inner", "outer", "length"]:
                    params[k] = float(v)
                elif k in ["segment", "interval"]:
                    params[k] = int(v)
                else:
                    params[k] = os.path.normpath(v)
    return params


def run(config_path, visual_callback=None):
    try:
        conf = read_config(config_path)
    except Exception as e:
        LOGGER.error(f"Config Error: {e}")
        return
    video_path = conf["video"]
    result_dir = Path(video_path).stem
    frame_dir = str(Path(ROOT / "runs/detect" / result_dir / "frames"))

    cnt, w, h, fps = save_frames(video_path, frame_dir, conf["interval"])
    if not cnt:
        return
    valid = filter_valid_frames(video_path, frame_dir, conf["raw"], fps)
    if not valid:
        return

    pipe_params = (conf["inner"], conf["outer"], conf["length"], conf["segment"])
    best = filter_best_defects(valid, w, h, pipe_params, device=device, visual_callback=visual_callback)
    process_best_defects(best, result_dir, pipe_params, output_dir=ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.txt")
    opt = parser.parse_args()
    run(opt.config)
