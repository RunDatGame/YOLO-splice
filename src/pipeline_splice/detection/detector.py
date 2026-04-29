import math
import os
import sys
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

from ._common import get_resource_path
from .frames import _load_frame_cached

DEFECT_TYPE_MAPPING = {
    "CKS": "错口",
    "CKM": "错口",
    "CKL": "错口",
    "FSS": "腐蚀",
    "FSM": "腐蚀",
    "FSL": "腐蚀",
    "PLS": "破裂",
    "PLM": "破裂",
    "PLL": "破裂",
}


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class LetterBox:
    def __init__(self, size=(640, 640), auto=False, stride=32):
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto
        self.stride = stride

    def __call__(self, im):
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)
        h, w = round(imh * r), round(imw * r)
        hs, ws = (
            tuple(math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        )
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


def calculate_defect_dimensions(bbox, pixel_per_meter=1000.0):
    return round((bbox[2] - bbox[0]) / pixel_per_meter, 3), round((bbox[3] - bbox[1]) / pixel_per_meter, 3)


def get_angle_range(bbox, cx, cy):
    def calc(x, y):
        d = math.degrees(math.atan2(cy - y, x - cx))
        return d + 360 if d < 0 else d

    angles = [calc(bbox[0], bbox[1]), calc(bbox[2], bbox[1]), calc(bbox[0], bbox[3]), calc(bbox[2], bbox[3])]
    mn, mx = min(angles), max(angles)
    span = mx - mn
    return (
        (round(mx, 2), round(mn, 2), round(360 - span, 2), True)
        if span > 180
        else (round(mn, 2), round(mx, 2), round(span, 2), False)
    )


@smart_inference_mode()
def filter_best_defects(
    valid_frames, w, h, pipe_params, device="cpu", visual_callback=None, pixel_per_meter=1000.0
):
    print(f"\n[步骤 3] AI 病害检测")
    yolo_weights = str(get_resource_path("weights/best.pt"))
    if not os.path.exists(yolo_weights):
        raise FileNotFoundError(f"YOLO 权重文件不存在: {yolo_weights}")

    with suppress_stdout():
        model = DetectMultiBackend(
            yolo_weights, device=select_device(device), dnn=False, data=str(get_resource_path("data/coco.yaml")), fp16=False
        )
        model.warmup(imgsz=(1, 3, 640, 640))

    _, _, seg_len, start_seg = pipe_params
    cx, cy = w / 2, h / 2
    top_defects = {}

    print(f"正在扫描 {len(valid_frames)} 帧...")
    for idx, (path, mil_abs, mil_rel) in enumerate(valid_frames):
        if idx % 5 == 0:
            sys.stdout.write(f"\r  -> 进度: {int((idx / len(valid_frames)) * 100)}%")
            sys.stdout.flush()

        seg_idx = start_seg + int(mil_rel // seg_len)
        im0s = cv2.imread(path)
        if im0s is None:
            continue

        im = LetterBox((640, 640), stride=model.stride, auto=model.pt)(im0s)
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(model.device).float() / 255.0
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.05, 0.45, max_det=1000)

        annotator = Annotator(im0s, line_width=3, example=str(model.names))

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    lbl = model.names[int(cls)]
                    if lbl in DEFECT_TYPE_MAPPING:
                        annotator.box_label(xyxy, f"{lbl} {conf:.2f}", color=colors(int(cls), True))
                        bbox = [c.item() for c in xyxy]
                        d_len, d_wid = calculate_defect_dimensions(bbox, pixel_per_meter)
                        info = {
                            "frame_path": path,
                            "bbox": bbox,
                            "bbox_center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                            "segment_index": seg_idx,
                            "absolute_mileage": mil_abs,
                            "model_type": DEFECT_TYPE_MAPPING[lbl],
                            "main_type": lbl[:2],
                            "sub_type": lbl,
                            "confidence": conf.item(),
                            "severity_val": 3 if lbl[-1] == "L" else 2 if lbl[-1] == "M" else 1,
                            "offset": round(
                                math.hypot(
                                    (bbox[0] + bbox[2]) / 2 - cx, (bbox[1] + bbox[3]) / 2 - cy
                                )
                                / pixel_per_meter,
                                3,
                            ),
                            "axis_angle": get_angle_range(bbox, cx, cy)[0],
                            "length": d_len,
                            "width": d_wid,
                            "height": round(d_len * d_wid * 0.1, 3),
                        }
                        key = (seg_idx, info["main_type"])
                        if key not in top_defects or (info["severity_val"] > top_defects[key]["severity_val"]):
                            top_defects[key] = info

        if visual_callback:
            visual_callback(annotator.result())

    sys.stdout.write(f"\r  -> 进度: 100%\n")
    sys.stdout.flush()
    seg_best = {}
    for (s, t), d in top_defects.items():
        if s not in seg_best:
            seg_best[s] = {}
        seg_best[s][t] = d
    return seg_best
