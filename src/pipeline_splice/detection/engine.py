# src/pipeline_splice/detection/engine.py
import json
import sys
import os
import time
import math
import glob
import re
import logging
import warnings
from functools import lru_cache
from pathlib import Path
from contextlib import contextmanager

import cv2
import torch
import pandas as pd
import numpy as np
import argparse
import shutil


# --- YOLO 及本地模块 ---
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors

# --- 全局常量 ---
PIPE_WALL_THICKNESS = 0.01
REINFORCEMENT_SPACING = 0
PIXEL_PER_METER = 1000
DEFECT_TYPE_MAPPING = {
    'CKS': '错口', 'CKM': '错口', 'CKL': '错口',
    'FSS': '腐蚀', 'FSM': '腐蚀', 'FSL': '腐蚀',
    'PLS': '破裂', 'PLM': '破裂', 'PLL': '破裂'
}
model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
depth_model = None
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ==========================================
# 工具函数 (LetterBox, suppress_stdout 等)
# ==========================================

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
        hs, ws = tuple(math.ceil(x / self.stride) * self.stride for x in (h, w)) if self.auto else (self.h, self.w)
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top:top + h, left:left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im_out


# 项目根目录（YOLO-splice 根目录）
if getattr(sys, 'frozen', False):
    ROOT = Path(sys.executable).parent
else:
    # engine.py 位于 src/pipeline_splice/detection/engine.py
    ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'): return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath(str(ROOT)), relative_path)


def init_depth_model():
    global depth_model
    if depth_model is None:
        weight_file = get_resource_path('checkpoints/depth_anything_v2_metric_hypersim_vits.pth')
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"深度权重文件不存在: {weight_file}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from depth_anything_v2.dpt import DepthAnythingV2
            depth_model = DepthAnythingV2(**model_configs['vits'])
            depth_model.load_state_dict(torch.load(weight_file, map_location=device))
            depth_model.to(device).eval()
    return depth_model


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
    if hypotenuse is None or hypotenuse <= pipe_radius: return 0.0
    return round(math.sqrt(hypotenuse ** 2 - pipe_radius ** 2), 3)


def calculate_defect_dimensions(bbox):
    return round((bbox[2] - bbox[0]) / PIXEL_PER_METER, 3), round((bbox[3] - bbox[1]) / PIXEL_PER_METER, 3)


def get_angle_range(bbox, cx, cy):
    def calc(x, y):
        d = math.degrees(math.atan2(cy - y, x - cx))
        return d + 360 if d < 0 else d

    angles = [calc(bbox[0], bbox[1]), calc(bbox[2], bbox[1]), calc(bbox[0], bbox[3]), calc(bbox[2], bbox[3])]
    mn, mx = min(angles), max(angles)
    span = mx - mn
    return (round(mx, 2), round(mn, 2), round(360 - span, 2), True) if span > 180 else \
        (round(mn, 2), round(mx, 2), round(span, 2), False)


def get_video_start_time(video_path):
    video_name = Path(video_path).stem
    match = re.search(r'(\d{4}[-/]?\d{2}[-/]?\d{2})[-_](\d{6})', video_name)
    if match:
        ts_str = f"{match.group(1).replace('/', '-').replace('_', '-')} {match.group(2)[:2]}:{match.group(2)[2:4]}:{match.group(2)[4:]}"
        try:
            return pd.to_datetime(ts_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
        except ValueError:
            return None
    return None


def load_csv_mileage_map(csv_path):
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path, header=None, sep=r'\s{2,}|,', engine='python')
        if df.shape[1] < 6: return None
        base_time = pd.to_datetime(df[0], errors='coerce')
        seconds = pd.to_numeric(df[3], errors='coerce').fillna(0)
        millis = pd.to_numeric(df[4], errors='coerce').fillna(0)
        full_timestamps = base_time + pd.to_timedelta(seconds, unit='s') + pd.to_timedelta(millis, unit='ms')
        mil = pd.to_numeric(df[5], errors='coerce')
        return mil[full_timestamps.notna() & mil.notna()].set_axis(
            full_timestamps[full_timestamps.notna() & mil.notna()])
    except Exception as exc:
        LOGGER.warning(f"里程 CSV 解析失败 ({csv_path}): {exc}")
        return None


def get_frame_mileage(video_path, f_num, start_t, fps, m_map):
    if start_t is None or fps is None or m_map is None: return None
    t_target = start_t + pd.to_timedelta((f_num - 1) / fps, unit='s')
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


def save_frames(video_path, output_folder, interval=None):
    print(f"\n[步骤 1] 视频抽帧处理")
    os.makedirs(output_folder, exist_ok=True)
    if interval is None:
        interval = 1000

    video_path = os.path.abspath(video_path)
    info_file = os.path.join(output_folder, 'extraction_info.txt')
    if os.path.exists(info_file):
        try:
            with open(info_file, 'r', encoding='utf-8') as f:
                cache_info = json.load(f)
            video_stat = os.stat(video_path)
            if (
                cache_info.get('video_path') == video_path
                and cache_info.get('interval') == interval
                and cache_info.get('video_mtime') == video_stat.st_mtime
                and cache_info.get('video_size') == video_stat.st_size
            ):
                files = glob.glob(os.path.join(output_folder, 'frame_*.png'))
                if files:
                    cap = cv2.VideoCapture(video_path)
                    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
                    cap.release()
                    print(f"[INFO] 命中缓存，跳过抽帧。")
                    return len(files), w, h, fps
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    for f in glob.glob(os.path.join(output_folder, 'frame_*.png')): os.remove(f)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0, None, None, None
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    cnt, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if cnt % interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{cnt}.png"), frame)
            saved += 1
            if saved % 20 == 0: sys.stdout.write(f"\r  -> 已提取: {saved} 帧..."); sys.stdout.flush()
        cnt += 1
    cap.release()
    print("\n[INFO] 抽帧完成。")
    try:
        video_stat = os.stat(video_path)
        cache_info = {
            'video_path': video_path,
            'interval': interval,
            'video_mtime': video_stat.st_mtime,
            'video_size': video_stat.st_size,
        }
    except OSError:
        cache_info = {'video_path': video_path, 'interval': interval}
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(cache_info, f)
    return saved, w, h, fps


def filter_valid_frames(video_path, frame_dir, csv_path, fps):
    print(f"\n[步骤 2] 里程同步处理")
    start_time = get_video_start_time(video_path)
    if start_time is None: return []
    m_map = load_csv_mileage_map(csv_path)
    if m_map is None: return []
    start_mileage_abs = get_frame_mileage(video_path, 1, start_time, fps, m_map)
    files = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')),
                   key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    valid = []
    for p in files:
        try:
            n = int(os.path.basename(p).split('_')[-1].split('.')[0])
            m_abs = get_frame_mileage(video_path, n, start_time, fps, m_map)
            if m_abs is not None:
                valid.append((p, m_abs, m_abs - start_mileage_abs))
        except (ValueError, IndexError):
            continue
    print(f"[INFO] 有效同步帧数: {len(valid)}")
    return valid


@smart_inference_mode()
def filter_best_defects(valid_frames, w, h, pipe_params, device='cpu', visual_callback=None):
    print(f"\n[步骤 3] AI 病害检测")
    yolo_weights = str(get_resource_path('weights/best.pt'))
    if not os.path.exists(yolo_weights): return {}

    with suppress_stdout():
        model = DetectMultiBackend(yolo_weights, device=select_device(device), dnn=False,
                                   data=str(ROOT / 'data/coco.yaml'), fp16=False)
        model.warmup(imgsz=(1, 3, 640, 640))

    _, _, seg_len, start_seg = pipe_params
    cx, cy = w / 2, h / 2
    top_defects = {}

    print(f"正在扫描 {len(valid_frames)} 帧...")
    for idx, (path, mil_abs, mil_rel) in enumerate(valid_frames):
        if idx % 5 == 0:
            sys.stdout.write(f"\r  -> 进度: {int((idx / len(valid_frames)) * 100)}%");
            sys.stdout.flush()

        seg_idx = start_seg + int(mil_rel // seg_len)
        im0s = cv2.imread(path)
        if im0s is None: continue

        im = LetterBox((640, 640), stride=model.stride, auto=model.pt)(im0s)
        im = torch.from_numpy(im.transpose((2, 0, 1))).to(model.device).float() / 255.0
        if len(im.shape) == 3: im = im[None]

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, 0.05, 0.45, max_det=1000)

        annotator = Annotator(im0s, line_width=3, example=str(model.names))

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    lbl = model.names[int(cls)]
                    if lbl in DEFECT_TYPE_MAPPING:
                        annotator.box_label(xyxy, f'{lbl} {conf:.2f}', color=colors(int(cls), True))
                        bbox = [c.item() for c in xyxy]
                        d_len, d_wid = calculate_defect_dimensions(bbox)
                        info = {
                            'frame_path': path, 'bbox': bbox,
                            'bbox_center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                            'segment_index': seg_idx, 'absolute_mileage': mil_abs,
                            'model_type': DEFECT_TYPE_MAPPING[lbl], 'main_type': lbl[:2], 'sub_type': lbl,
                            'confidence': conf.item(),
                            'severity_val': 3 if lbl[-1] == 'L' else 2 if lbl[-1] == 'M' else 1,
                            'offset': round(math.hypot((bbox[0] + bbox[2]) / 2 - cx,
                                                       (bbox[1] + bbox[3]) / 2 - cy) / PIXEL_PER_METER, 3),
                            'axis_angle': get_angle_range(bbox, cx, cy)[0], 'length': d_len, 'width': d_wid,
                            'height': round(d_len * d_wid * 0.1, 3)
                        }
                        key = (seg_idx, info['main_type'])
                        if key not in top_defects or (info['severity_val'] > top_defects[key]['severity_val']):
                            top_defects[key] = info

        if visual_callback:
            visual_callback(annotator.result())

    sys.stdout.write(f"\r  -> 进度: 100%\n");
    sys.stdout.flush()
    seg_best = {}
    for (s, t), d in top_defects.items():
        if s not in seg_best: seg_best[s] = {}
        seg_best[s][t] = d
    return seg_best


def save_defect_screenshot(img, bbox, defect_info, output_dir):
    full_dir = os.path.join(ROOT, 'runs', 'detect', output_dir)
    os.makedirs(full_dir, exist_ok=True)
    if img is None: return None
    x1, y1, x2, y2 = [int(c) for c in bbox]
    h, w = img.shape[:2]
    pad_x, pad_y = max(30, int((x2 - x1) * 0.2)), max(30, int((y2 - y1) * 0.2))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return None
    fname = f"{defect_info['编号']}_{defect_info['模型类型']}_管节{defect_info['管节序号']}.png"
    path = os.path.join(full_dir, fname)
    if cv2.imwrite(path, roi): return path
    return None


@lru_cache(maxsize=64)
def _load_frame_cached(frame_path: str):
    """LRU 缓存的帧加载器，防止处理长视频时内存无限增长。"""
    return cv2.imread(frame_path)


def process_best_defects(best_defects, result_dir, pipe_params, device='cpu', output_dir=None, use_depth=True):
    print(f"\n[步骤 4] 分析与导出")
    if not best_defects: print("未发现病害"); return
    pipe_inner, pipe_outer, seg_len, start_seg = pipe_params
    final_res = []
    rec_id = 1

    shot_dir = "defect_screenshots"

    for seg_idx, defects in sorted(best_defects.items()):
        # 按帧路径分组，同一帧只做一次深度推理
        frame_groups = {}
        for m_type, d in defects.items():
            frame_groups.setdefault(d['frame_path'], []).append((m_type, d))

        for frame_path, items in frame_groups.items():
            curr_img = _load_frame_cached(frame_path)
            if curr_img is None:
                continue

            depth_map = get_frame_depth_map(curr_img) if use_depth else None

            for m_type, d in items:
                if use_depth and depth_map is not None:
                    x, y = int(d['bbox_center'][0]), int(d['bbox_center'][1])
                    dh, dw = depth_map.shape
                    depth = float(depth_map[min(y, dh - 1), min(x, dw - 1)])
                else:
                    depth = None

                # 计算节内里程（管节内的相对位置）
                segment_relative_mileage = d['absolute_mileage'] % seg_len
                final_segment_mileage = segment_relative_mileage + calculate_along_pipe_distance(depth, pipe_inner / 2)
                final_segment_mileage = min(final_segment_mileage, seg_len)

                info = {
                    '编号': rec_id, '模型类型': d['model_type'], '严重等级': str(d['severity_val']),
                    '管节序号': seg_idx, '管节内径': pipe_inner, '管节外径': pipe_outer,
                    '管节长度': seg_len, '节内里程': round(final_segment_mileage, 3),
                    '病害长': d['length'], '病害宽': d['width']
                }
                s_path = save_defect_screenshot(curr_img, d['bbox'], info, shot_dir)
                if s_path:
                    info['数据截图'] = s_path
                    final_res.append(info)
                    rec_id += 1

    if final_res:
        csv_output_dir = Path(output_dir) if output_dir else ROOT
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(final_res).to_csv(csv_output_dir / 'defect_results_full.csv', index=False, encoding='utf-8-sig')

        print(f"导出完成: {csv_output_dir / 'defect_results_full.csv'}")


def read_config(path):
    params = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line and not line.startswith('#'):
                k, v = [x.strip() for x in line.split(':', 1)]
                if k in ['inner', 'outer', 'length']:
                    params[k] = float(v)
                elif k in ['segment', 'interval']:
                    params[k] = int(v)
                else:
                    params[k] = os.path.normpath(v)
    return params


def run(config_path, visual_callback=None):
    try:
        conf = read_config(config_path)
    except Exception as e:
        LOGGER.error(f"Config Error: {e}"); return
    video_path = conf['video']
    result_dir = Path(video_path).stem
    frame_dir = str(Path(ROOT / 'runs/detect' / result_dir / 'frames'))

    cnt, w, h, fps = save_frames(video_path, frame_dir, conf['interval'])
    if not cnt: return
    valid = filter_valid_frames(video_path, frame_dir, conf['raw'], fps)
    if not valid: return

    pipe_params = (conf['inner'], conf['outer'], conf['length'], conf['segment'])
    best = filter_best_defects(valid, w, h, pipe_params, device=device, visual_callback=visual_callback)
    process_best_defects(best, result_dir, pipe_params, device=device, output_dir=ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.txt')
    opt = parser.parse_args()
    run(opt.config)
