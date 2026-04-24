import logging
import os
import shlex
import sys
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline_core.runner import process_task


def run_watcher(watch_filename: str, fixed_mode: str, mode_label: str) -> None:
    if getattr(sys, "frozen", False):
        current_dir = str(PROJECT_DIR)
    else:
        current_dir = str(PROJECT_DIR)

    watch_root = CURRENT_DIR
    watch_file = os.path.join(watch_root, watch_filename)
    log_file = os.path.join(watch_root, f"path_watcher_{fixed_mode}.log")
    config_file = PROJECT_DIR / "configs" / ("reconstruction.txt" if fixed_mode == "reconstruction" else "splice.txt")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    processing_lock = threading.Lock()
    file_lock = threading.Lock()

    def remove_first_line(txt_path):
        with file_lock:
            try:
                if not os.path.exists(txt_path):
                    return False
                with open(txt_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                if not lines:
                    return False
                with open(txt_path, "w", encoding="utf-8") as file:
                    file.writelines(lines[1:])
                return True
            except Exception as exc:
                logging.error(f"删除行失败: {exc}")
                return False

    def read_first_line(txt_path):
        with file_lock:
            try:
                if not os.path.exists(txt_path):
                    return None
                with open(txt_path, "r", encoding="utf-8") as file:
                    lines = file.readlines()
                if not lines:
                    return None
                return lines[0].strip()
            except Exception as exc:
                logging.error(f"读取文件失败: {exc}")
                return None

    def process_line(csv_path, mp4_path):
        csv_path = csv_path.strip().strip('"').strip("'")
        mp4_path = mp4_path.strip().strip('"').strip("'")

        print(f"\n[PathWatcher] 新任务启动:")
        print(f"  CSV: {csv_path}")
        print(f"  MP4: {mp4_path}")
        print(f"  Mode: {mode_label}")
        print(f"  WorkDir: {current_dir}")

        try:
            process_task(
                csv_path=csv_path,
                video_path=mp4_path,
                work_dir=current_dir,
                model_mode=fixed_mode,
                config_path=config_file,
            )
            logging.info("任务执行完成")
            return True
        except Exception as exc:
            logging.error(f"任务执行出错: {exc}")
            import traceback

            traceback.print_exc()
            return False

    def process_txt_file(txt_path):
        try:
            first_line = read_first_line(txt_path)
            if not first_line:
                return

            parts = shlex.split(first_line, posix=False)
            if len(parts) < 2:
                logging.error(f"格式错误跳过: {first_line}")
                remove_first_line(txt_path)
                return

            csv_path, mp4_path = parts[0], parts[1]
            success = process_line(csv_path, mp4_path)

            if remove_first_line(txt_path):
                if success:
                    logging.info(f"成功并移除: {first_line}")
                else:
                    logging.warning(f"失败但移除: {first_line}")
                time.sleep(0.1)
                process_txt_file(txt_path)
        except Exception as exc:
            logging.error(f"循环异常: {exc}")

    class TxtFileHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if os.path.abspath(event.src_path) == os.path.abspath(watch_file):
                logging.info(f"检测到文件修改: {event.src_path}")
                time.sleep(0.5)
                with processing_lock:
                    process_txt_file(watch_file)

    if not os.path.exists(watch_file):
        with open(watch_file, "w", encoding="utf-8"):
            pass
        print(f"已创建监听文件: {watch_file}")

    observer = Observer()
    observer.schedule(TxtFileHandler(), str(watch_root), recursive=False)
    observer.start()

    print(f"{mode_label} PathWatcher 正在监听: {watch_file}")
    print("请在文件中输入: ./xxx.csv ./xxx.mp4")

    try:
        if os.path.getsize(watch_file) > 0:
            with processing_lock:
                process_txt_file(watch_file)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
