from __future__ import annotations

import sys
from pathlib import Path

from .contracts import PipelineConfig, TaskInput, TaskPaths

FLOAT_KEYS = {"inner", "outer", "length", "meshroom_default_fov", "wall_thickness", "rebar_spacing"}
INT_KEYS = {"segment", "interval", "meshroom_depth_downscale"}
MODEL_MODE_ALIASES = {
    "library": "library",
    "splice": "library",
    "model": "library",
    "拼接": "library",
    "拼接模型": "library",
    "reconstruction": "reconstruction",
    "meshroom": "reconstruction",
    "algo": "reconstruction",
    "algorithm": "reconstruction",
    "算法": "reconstruction",
    "重建": "reconstruction",
    "三维重建": "reconstruction",
}


def get_runtime_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent.parent.parent


def resolve_path(base_dir: Path, relative_path: str | Path | None) -> Path | None:
    if relative_path is None:
        return None
    path_str = str(relative_path).strip()
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalize_meshroom_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_dir():
        batch_exe = path / "meshroom_batch.exe"
        gui_exe = path / "Meshroom.exe"
        if batch_exe.exists():
            return batch_exe
        if gui_exe.exists():
            return gui_exe
        return path

    lower_name = path.name.lower()
    if lower_name == "meshroom.exe":
        batch_exe = path.with_name("meshroom_batch.exe")
        if batch_exe.exists():
            return batch_exe
    return path


def resolve_input_path(base_dir: Path, raw_path: str | Path | None) -> Path | None:
    path = resolve_path(base_dir, raw_path)
    if path is None:
        return None
    if path.exists():
        return path

    filename = path.name
    fallback_candidates = [
        (base_dir / filename).resolve(),
        (base_dir.parent / filename).resolve(),
        (base_dir / "samples" / filename).resolve(),
        (base_dir.parent / "samples" / filename).resolve(),
    ]

    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate
    return path


def normalize_model_mode(raw_mode: str | None) -> str | None:
    if raw_mode is None:
        return None
    mode = str(raw_mode).strip().lower()
    if not mode:
        return None
    normalized = MODEL_MODE_ALIASES.get(mode)
    if normalized is None:
        raise ValueError(f"不支持的建模方式: {raw_mode}")
    return normalized


def parse_config_file(config_path: Path) -> dict[str, str | int | float]:
    config: dict[str, str | int | float] = {}
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if key in FLOAT_KEYS:
                config[key] = float(value)
            elif key in INT_KEYS:
                config[key] = int(value)
            else:
                config[key] = value
    return config


def update_task_inputs(config_path: Path, video_path: Path, csv_path: Path) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在: {config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    has_video = False
    has_raw = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("video:"):
            updated_lines.append(f"video: {video_path}")
            has_video = True
        elif stripped.startswith("raw:"):
            updated_lines.append(f"raw: {csv_path}")
            has_raw = True
        else:
            updated_lines.append(line)

    if not has_video:
        updated_lines.append(f"video: {video_path}")
    if not has_raw:
        updated_lines.append(f"raw: {csv_path}")

    config_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")


def build_task_input(csv_path: str | Path, video_path: str | Path, work_dir: str | Path = ".") -> TaskInput:
    work_dir_path = Path(work_dir).resolve()
    csv_full_path = resolve_input_path(work_dir_path, csv_path)
    video_full_path = resolve_input_path(work_dir_path, video_path)
    if csv_full_path is None or video_full_path is None:
        raise ValueError("CSV 和视频路径不能为空")

    output_dir = video_full_path.parent if video_full_path.parent.exists() else work_dir_path
    return TaskInput(
        csv_path=csv_full_path,
        video_path=video_full_path,
        work_dir=work_dir_path,
        output_dir=output_dir,
    )


def build_task_paths(task: TaskInput, model_mode: str) -> TaskPaths:
    mode_tag = "splice" if model_mode == "library" else "reconstruction"
    video_parent = task.video_path.parent
    return TaskPaths(
        detect_csv_local=task.work_dir / "runs" / "detect" / "defect_results_full.csv",
        detect_csv_output=video_parent / "defect_results_full.csv",
        matched_csv_output=video_parent / f"detect_results_matched_{mode_tag}.csv",
        reconstruction_dir=video_parent / "meshroom_reconstruction",
        final_glb=video_parent / "pipeline_In.glb",
        info_txt=video_parent / f"video_Config_{mode_tag}.txt",
        mode_tag=mode_tag,
    )


def load_pipeline_config(config_path: Path | None, task: TaskInput, model_mode_override: str | None = None) -> PipelineConfig:
    if config_path is None or not config_path.exists():
        candidates = [
            task.work_dir / "config.txt",
            task.work_dir / "config" / "splice.txt",
            get_runtime_dir() / "config" / "splice.txt",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError(f"找不到配置文件，已尝试: {[str(c) for c in candidates]}")
    raw_config = parse_config_file(config_path)
    work_dir = task.work_dir

    blender_path = resolve_path(work_dir, raw_config.get("blender_path"))
    dataset_path = resolve_path(work_dir, raw_config.get("dataset_path"))
    meshroom_path = normalize_meshroom_path(resolve_path(work_dir, raw_config.get("meshroom_path")))
    manhole_path = resolve_path(work_dir, raw_config.get("manhole_path"))
    # 权重路径默认指向打包在 _internal 内的资源（通过 get_runtime_dir 解析）
    runtime_dir = get_runtime_dir()
    yolo_weights_raw = raw_config.get("yolo_weights")
    depth_weights_raw = raw_config.get("depth_weights")
    yolo_weights = resolve_path(runtime_dir, yolo_weights_raw) if yolo_weights_raw else (runtime_dir / "weights" / "best.pt")
    depth_weights = resolve_path(runtime_dir, depth_weights_raw) if depth_weights_raw else (runtime_dir / "checkpoints" / "depth_anything_v2_metric_hypersim_vits.pth")

    video_path = resolve_path(work_dir, raw_config.get("video")) or task.video_path
    raw_csv_path = resolve_path(work_dir, raw_config.get("raw")) or task.csv_path
    configured_model_mode = normalize_model_mode(str(raw_config.get("model_mode", "library")))
    model_mode = normalize_model_mode(model_mode_override) or configured_model_mode or "library"
    reconstruction_command = str(raw_config.get("reconstruction_command", "")).strip()
    reconstruction_mesh = str(raw_config.get("reconstruction_mesh", "")).strip()
    reconstruction_scale_target = str(raw_config.get("reconstruction_scale_target", "inner")).strip().lower() or "inner"
    meshroom_describer_types = str(raw_config.get("meshroom_describer_types", "sift,akaze")).strip() or "sift,akaze"
    meshroom_describer_preset = str(raw_config.get("meshroom_describer_preset", "normal")).strip() or "normal"
    meshroom_match_method = (
        str(raw_config.get("meshroom_match_method", "SequentialAndVocabularyTree")).strip()
        or "SequentialAndVocabularyTree"
    )
    meshroom_default_fov = float(raw_config.get("meshroom_default_fov", 45.0))
    meshroom_depth_downscale = int(raw_config.get("meshroom_depth_downscale", 2))
    use_depth = str(raw_config.get("use_depth", "true")).strip().lower() not in ("false", "0", "no", "off")
    default_model = str(raw_config.get("default_model", "QKG")).strip() or "QKG"
    skip_ck = str(raw_config.get("skip_ck", "true")).strip().lower() not in ("false", "0", "no", "off")
    one_model_per_segment = str(raw_config.get("one_model_per_segment", "true")).strip().lower() not in ("false", "0", "no", "off")
    global_x_offset = float(raw_config.get("global_x_offset", -2.0))
    manhole_half_length = float(raw_config.get("manhole_half_length", 2.0))
    pixel_per_meter = float(raw_config.get("pixel_per_meter", 1000.0))

    return PipelineConfig(
        config_path=config_path,
        blender_path=blender_path,
        dataset_path=dataset_path,
        meshroom_path=meshroom_path,
        video_path=video_path,
        raw_csv_path=raw_csv_path,
        model_mode=model_mode,
        reconstruction_command=reconstruction_command,
        reconstruction_mesh=reconstruction_mesh,
        reconstruction_scale_target=reconstruction_scale_target,
        meshroom_describer_types=meshroom_describer_types,
        meshroom_describer_preset=meshroom_describer_preset,
        meshroom_match_method=meshroom_match_method,
        meshroom_default_fov=meshroom_default_fov,
        meshroom_depth_downscale=meshroom_depth_downscale,
        manhole_path=manhole_path,
        yolo_weights=yolo_weights,
        depth_weights=depth_weights,
        inner=float(raw_config["inner"]),
        outer=float(raw_config["outer"]),
        length=float(raw_config["length"]),
        segment=int(raw_config["segment"]),
        interval=int(raw_config["interval"]),
        wall_thickness=float(raw_config.get("wall_thickness", 0.1)),
        rebar_spacing=float(raw_config.get("rebar_spacing", 0.0)),
        use_depth=use_depth,
        default_model=default_model,
        skip_ck=skip_ck,
        one_model_per_segment=one_model_per_segment,
        global_x_offset=global_x_offset,
        manhole_half_length=manhole_half_length,
        pixel_per_meter=pixel_per_meter,
    )


def validate_runtime_config(config: PipelineConfig, task: TaskInput) -> None:
    common_required = {
        "视频文件": task.video_path,
        "里程 CSV": task.csv_path,
        "Blender": config.blender_path,
        "YOLO 权重": config.yolo_weights,
    }
    if config.use_depth:
        common_required["深度权重"] = config.depth_weights

    if config.model_mode == "library":
        common_required["缺陷模型库"] = config.dataset_path
        common_required["井室模型"] = config.manhole_path
    elif config.model_mode == "reconstruction":
        if not config.reconstruction_command and not config.reconstruction_mesh and config.meshroom_path is None:
            raise FileNotFoundError(
                "重建模式缺少配置: reconstruction_command、reconstruction_mesh、meshroom_path 至少需要提供一个"
            )
        if not config.reconstruction_command and not config.reconstruction_mesh and config.meshroom_path is not None:
            common_required["Meshroom"] = config.meshroom_path
    else:
        raise ValueError(f"不支持的 model_mode: {config.model_mode}")

    missing = [f"{name}: {path}" for name, path in common_required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError("运行所需文件缺失:\n" + "\n".join(missing))


def validate_detection_config(config: PipelineConfig, task: TaskInput) -> None:
    required = {
        "视频文件": task.video_path,
        "里程 CSV": task.csv_path,
        "YOLO 权重": config.yolo_weights,
    }
    if config.use_depth:
        required["深度权重"] = config.depth_weights
    missing = [f"{name}: {path}" for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError("检测阶段所需文件缺失:\n" + "\n".join(missing))
