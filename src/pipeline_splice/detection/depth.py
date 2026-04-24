import math
import os
import warnings

import numpy as np
import torch
from utils.general import LOGGER

from ._common import device, get_resource_path

model_configs = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}}
_depth_model = None


def init_depth_model():
    global _depth_model
    if _depth_model is None:
        weight_file = get_resource_path("checkpoints/depth_anything_v2_metric_hypersim_vits.pth")
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"深度权重文件不存在: {weight_file}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from depth_anything_v2.dpt import DepthAnythingV2

            _depth_model = DepthAnythingV2(**model_configs["vits"])
            _depth_model.load_state_dict(torch.load(weight_file, map_location=device))
            _depth_model.to(device).eval()
    return _depth_model


def get_frame_depth_map(img):
    """对整帧进行深度推理并返回深度图，同帧多个缺陷时复用。"""
    if img is None:
        return None
    try:
        model = init_depth_model()
    except (FileNotFoundError, RuntimeError) as exc:
        LOGGER.warning(f"深度模型加载失败: {exc}")
        return None
    try:
        with torch.no_grad():
            depth_map = model.infer_image(img)
        return depth_map
    except (RuntimeError, ValueError, IndexError, AttributeError) as exc:
        LOGGER.warning(f"深度估计失败: {exc}")
        return None


def calculate_along_pipe_distance(hypotenuse, pipe_radius):
    if hypotenuse is None or hypotenuse <= pipe_radius:
        return 0.0
    return round(math.sqrt(hypotenuse ** 2 - pipe_radius ** 2), 3)
