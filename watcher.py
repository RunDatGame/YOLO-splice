import os
import sys
import time
import logging
import threading
import csv
import json
import struct
import subprocess
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ==========================================
# 打包后路径兼容
# ==========================================
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
    if hasattr(sys, "_MEIPASS"):
        RESOURCE_DIR = Path(sys._MEIPASS)
    else:
        RESOURCE_DIR = BASE_DIR
else:
    BASE_DIR = Path(__file__).resolve().parent
    RESOURCE_DIR = BASE_DIR

SRC_DIR = RESOURCE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_splice.core.runner import process_task

# ==========================================
# 配置
# ==========================================
WATCH_FILE = BASE_DIR / "task_list.txt"
LOG_FILE = BASE_DIR / "pipeline_watcher.log"
DEFAULT_CONFIG = BASE_DIR / "config.txt"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

processing_lock = threading.Lock()
file_lock = threading.Lock()


class TaskFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if Path(event.src_path).resolve() == WATCH_FILE.resolve():
            logging.info(f"检测到文件修改: {event.src_path}")
            time.sleep(0.5)
            with processing_lock:
                process_task_list(WATCH_FILE)


def read_first_line(txt_path: Path) -> str | None:
    with file_lock:
        if not txt_path.exists():
            return None
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
            return lines[0].strip() if lines else None
        except Exception as exc:
            logging.error(f"读取文件失败: {exc}")
            return None


def remove_first_line(txt_path: Path) -> bool:
    with file_lock:
        if not txt_path.exists():
            return False
        try:
            lines = txt_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                return False
            txt_path.write_text("\n".join(lines[1:]) + ("\n" if len(lines) > 1 else ""), encoding="utf-8")
            return True
        except Exception as exc:
            logging.error(f"删除行失败: {exc}")
            return False


def handle_task(csv_path: str, video_path: str) -> bool:
    csv_path = csv_path.strip().strip('"').strip("'")
    video_path = video_path.strip().strip('"').strip("'")

    video_dir = Path(video_path).parent.resolve()
    print(f"\n[Watcher] New task:")
    print(f"  CSV: {csv_path}")
    print(f"  MP4: {video_path}")
    print(f"  OutputDir: {video_dir}")

    config_path = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None

    try:
        result = process_task(
            csv_path=csv_path,
            video_path=video_path,
            work_dir=video_dir,
            config_path=config_path,
        )
        _repair_patch_glb_if_needed(result)
        logging.info("任务执行完成")
        print("[OK] 任务完成")
        return True
    except Exception as exc:
        logging.error(f"任务执行出错: {exc}")
        import traceback
        traceback.print_exc()
        return False


def _load_defect_ids_from_csv(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            defect_id = str(row.get("编号", "")).strip()
            if defect_id:
                ids.add(defect_id)
    return ids


def _load_node_names_from_glb(glb_path: Path) -> set[str]:
    if not glb_path.exists():
        return set()
    data = glb_path.read_bytes()
    if len(data) < 12:
        return set()
    _, _, total_length = struct.unpack_from("<III", data, 0)
    offset = 12
    while offset + 8 <= min(total_length, len(data)):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type != 0x4E4F534A:
            continue
        doc = json.loads(chunk.decode("utf-8"))
        return {
            str(node.get("name", "")).strip()
            for node in doc.get("nodes", [])
            if str(node.get("name", "")).strip()
        }
    return set()


def _infer_total_segments_from_csv(csv_path: Path) -> int:
    max_seg = -1
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                max_seg = max(max_seg, int(float(str(row.get("管节序号", "")).strip())))
            except (TypeError, ValueError):
                continue
    return max_seg + 1 if max_seg >= 0 else 0


def _repair_patch_glb_if_needed(result) -> None:
    try:
        config = result.config
        paths = result.paths
        if config.model_mode != "library" or config.blender_path is None:
            return
        csv_ids = _load_defect_ids_from_csv(paths.matched_csv_output)
        if not csv_ids:
            return
        glb_names = _load_node_names_from_glb(paths.patch_glb)
        if csv_ids == glb_names:
            return

        blender_script = RESOURCE_DIR / "scripts" / "blender_splice.py"
        total_segments = _infer_total_segments_from_csv(paths.matched_csv_output)
        command = [
            str(config.blender_path),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--csv",
            str(paths.matched_csv_output),
            "--output",
            str(paths.final_glb),
            "--manhole",
            str(config.manhole_path),
            "--inner",
            str(config.inner),
            "--outer",
            str(config.outer),
            "--wall-thickness",
            str(config.wall_thickness),
            "--global-x-offset",
            str(config.global_x_offset),
            "--manhole-half-length",
            str(config.manhole_half_length),
            "--dataset",
            str(config.dataset_path) if config.dataset_path else "",
            "--default-model",
            str(config.default_model) if config.default_model else "QKG",
            "--default-defect",
            "FS1,PL1",
            "--patches-output",
            str(paths.patch_glb),
            "--total-segments",
            str(total_segments),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as exc:
        logging.warning(f"病害贴片导出修复失败: {exc}")


def process_task_list(txt_path: Path):
    try:
        first_line = read_first_line(txt_path)
        if not first_line:
            return

        parts = first_line.split()
        if len(parts) < 2:
            logging.error(f"格式错误跳过: {first_line}")
            remove_first_line(txt_path)
            return

        csv_path, video_path = parts[0], parts[1]
        success = handle_task(csv_path, video_path)

        if remove_first_line(txt_path):
            if success:
                logging.info(f"成功并移除: {first_line}")
            else:
                logging.warning(f"失败但移除: {first_line}")
            time.sleep(0.1)
            process_task_list(txt_path)
    except Exception as exc:
        logging.error(f"循环异常: {exc}")


def main():
    if not WATCH_FILE.exists():
        WATCH_FILE.write_text("", encoding="utf-8")
        print(f"Created watch file: {WATCH_FILE}")

    observer = Observer()
    handler = TaskFileHandler()
    observer.schedule(handler, str(BASE_DIR), recursive=False)
    observer.start()

    print(f"PipelineWatcher (UV mode) watching: {WATCH_FILE}")
    print("Enter in file: ./xxx.csv ./xxx.mp4")

    try:
        if WATCH_FILE.stat().st_size > 0:
            with processing_lock:
                process_task_list(WATCH_FILE)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
