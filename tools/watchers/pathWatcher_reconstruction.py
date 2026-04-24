from watcher_common import run_watcher


if __name__ == "__main__":
    run_watcher(
        "task_list_reconstruction.txt",
        fixed_mode="reconstruction",
        mode_label="三维重建算法",
    )
