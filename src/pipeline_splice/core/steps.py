from __future__ import annotations

import csv
import json
import os
import shutil
import shlex
import struct
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
from pipeline_splice.detection import engine as detect_engine

from pipeline_splice.modeling.matcher import process_csv

from .contracts import DetectArtifacts, PipelineConfig, StepResult, TaskInput, TaskPaths
from .detect_stage import run_detection_stage

PIPE_JOINT_OVERLAP = 0.04
RECONSTRUCTION_MESH_NAMES = {
    "texturedmesh.obj",
    "texturedmesh.glb",
    "texturedmesh.gltf",
    "texturedmesh.ply",
    "texturedmesh.stl",
    "mesh.obj",
    "mesh.ply",
    "model.obj",
    "model.ply",
}


def _ascii_safe_name(value: str) -> str:
    safe = "".join(ch if ch.isascii() and (ch.isalnum() or ch in "-_.") else "_" for ch in value)
    return safe.strip("._") or "meshroom_job"


def prepare_meshroom_workspace(task: TaskInput, frame_dir: Path) -> tuple[Path, Path, Path, Path]:
    temp_root = Path(tempfile.gettempdir()) / "yolo_meshroom"
    job_name = _ascii_safe_name(task.video_path.stem)
    workspace_dir = temp_root / f"{job_name}_{uuid.uuid4().hex[:8]}"
    images_dir = workspace_dir / "images"
    reconstruction_dir = workspace_dir / "reconstruction"
    cache_dir = workspace_dir / "cache"

    images_dir.mkdir(parents=True, exist_ok=True)
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_images = sorted(path for path in frame_dir.glob("*") if path.is_file())
    for index, source in enumerate(source_images, start=1):
        target = images_dir / f"frame_{index:06d}.jpg"
        image = cv2.imread(str(source))
        if image is None:
            shutil.copy2(source, target.with_suffix(source.suffix))
            continue
        success = cv2.imwrite(str(target), image)
        if not success:
            raise RuntimeError(f"无法写入 Meshroom 临时图像: {target}")

    return workspace_dir, images_dir, reconstruction_dir, cache_dir


def get_script_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    # steps.py is at src/pipeline_splice/core/steps.py; project root is 3 levels up
    return Path(__file__).resolve().parent.parent.parent.parent


def copy_if_needed(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def get_frame_dir(task: TaskInput) -> Path:
    return task.work_dir / "runs" / "detect" / task.video_path.stem / "frames"


def estimate_total_segments(config: PipelineConfig) -> int:
    try:
        from pipeline_splice.detection.mileage import load_csv_mileage_map

        m_map = load_csv_mileage_map(str(config.raw_csv_path))
        if m_map is None:
            return 0
        max_mileage = float(m_map.max())
        segment_pitch = max(config.length - PIPE_JOINT_OVERLAP, 0.001)
        return int(max_mileage / segment_pitch) + 1
    except Exception:
        return 0


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


def patch_names_match_csv(csv_path: Path, patch_glb_path: Path) -> bool:
    csv_ids = _load_defect_ids_from_csv(csv_path)
    if not csv_ids:
        return True
    glb_names = _load_node_names_from_glb(patch_glb_path)
    return csv_ids == glb_names


def rewrite_result_csv_model_paths(csv_path: Path, final_glb: Path, patch_glb: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    if not headers:
        return
    patch_glb_str = str(patch_glb)
    for row in rows:
        row["模型路径"] = patch_glb_str

    headers = [header for header in headers if header != "病害模型库路径"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key in headers})


def extract_video_frames(task: TaskInput, config: PipelineConfig, force: bool = False) -> StepResult:
    frame_dir = get_frame_dir(task)
    frame_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(frame_dir.glob("frame_*.*"))
    if existing_frames and not force:
        return StepResult("extract", True, f"复用已有切帧: {len(existing_frames)} 张", frame_dir)

    if force:
        for existing in frame_dir.glob("*"):
            if existing.is_file():
                existing.unlink()

    frame_count, _, _, _ = detect_engine.save_frames(
        str(task.video_path), str(frame_dir), config.interval
    )

    if not frame_count:
        return StepResult("extract", False, f"视频抽帧失败: {task.video_path}")

    return StepResult("extract", True, f"抽帧完成: {frame_count} 张", frame_dir)


class _TemplateDict(dict):
    def __missing__(self, key):  # pragma: no cover - defensive fallback
        raise KeyError(f"缺少模板变量: {key}")


def expand_runtime_value(value: str, task: TaskInput, config: PipelineConfig, paths: TaskPaths) -> str:
    context = _TemplateDict(
        work_dir=str(task.work_dir),
        output_dir=str(task.output_dir),
        video_path=str(task.video_path),
        video_stem=task.video_path.stem,
        csv_path=str(task.csv_path),
        detect_csv=str(paths.detect_csv_output),
        matched_csv=str(paths.matched_csv_output),
        frames_dir=str(get_frame_dir(task)),
        reconstruction_dir=str(paths.reconstruction_dir),
        final_glb=str(paths.final_glb),
        inner=config.inner,
        outer=config.outer,
        segment_length=config.length,
        segment_start=config.segment,
    )
    try:
        return value.format_map(context)
    except KeyError as exc:
        raise ValueError(f"配置值包含未知模板变量: {exc}") from exc


def resolve_runtime_path(raw_value: str, task: TaskInput, config: PipelineConfig, paths: TaskPaths) -> Path:
    expanded = expand_runtime_value(raw_value, task, config, paths).strip()
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (task.work_dir / path).resolve()


def prepare_reconstruction_csv(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise RuntimeError("检测 CSV 没有表头，无法为重建模式准备结果文件")

        fieldnames = list(reader.fieldnames)
        if "模型路径" not in fieldnames:
            fieldnames.append("模型路径")

        with target.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                row["模型路径"] = ""
                writer.writerow(row)


def find_reconstruction_mesh(reconstruction_dir: Path) -> Path | None:
    preferred = [
        reconstruction_dir / "texturedMesh.obj",
        reconstruction_dir / "texturedMesh.glb",
        reconstruction_dir / "texturedMesh.gltf",
        reconstruction_dir / "texturedMesh.ply",
        reconstruction_dir / "MeshroomCache" / "texturing" / "texturedMesh.obj",
    ]

    for candidate in preferred:
        if candidate.exists():
            return candidate

    for root, _, files in os.walk(reconstruction_dir):
        for file_name in files:
            if file_name.lower() in RECONSTRUCTION_MESH_NAMES:
                return Path(root) / file_name
    return None


def run_meshroom_reconstruction(task: TaskInput, config: PipelineConfig, paths: TaskPaths) -> StepResult:
    if config.meshroom_path is None:
        return StepResult("reconstruct", False, "未配置 meshroom_path，无法执行内建 Meshroom 重建")

    frame_dir = get_frame_dir(task)
    if not frame_dir.exists():
        return StepResult("reconstruct", False, f"抽帧目录不存在，无法执行重建: {frame_dir}")

    workspace_dir, images_dir, reconstruction_dir, cache_dir = prepare_meshroom_workspace(task, frame_dir)
    log_path = workspace_dir / "meshroom.log"

    try:
        command = [
            str(config.meshroom_path),
            "-i",
            str(images_dir),
            "-o",
            str(reconstruction_dir),
            "--cache",
            str(cache_dir),
            "-p",
            "photogrammetry",
            "--paramOverrides",
            f"FeatureExtraction:describerTypes={config.meshroom_describer_types}",
            "--paramOverrides",
            f"FeatureExtraction:describerPreset={config.meshroom_describer_preset}",
            "--paramOverrides",
            f"ImageMatching:method={config.meshroom_match_method}",
            "--paramOverrides",
            f"CameraInit:defaultFieldOfView={config.meshroom_default_fov}",
            "-v",
            "info",
        ]

        if config.meshroom_depth_downscale > 1:
            command.extend(
                [
                    "--paramOverrides",
                    f"DepthMap:downscale={config.meshroom_depth_downscale}",
                ]
            )

        with log_path.open("w", encoding="utf-8") as log_file:
            subprocess.run(
                command,
                check=True,
                cwd=workspace_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )

        paths.reconstruction_dir.mkdir(parents=True, exist_ok=True)
        if paths.reconstruction_dir.exists():
            shutil.rmtree(paths.reconstruction_dir, ignore_errors=True)
        shutil.copytree(reconstruction_dir, paths.reconstruction_dir, dirs_exist_ok=True)

        mesh_path = find_reconstruction_mesh(paths.reconstruction_dir)
        if mesh_path is None:
            details = [
                f"Meshroom 已执行，但未找到重建 mesh: {paths.reconstruction_dir}",
                f"临时重建目录: {reconstruction_dir}",
            ]
            if log_path.exists():
                try:
                    log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
                    if log_tail.strip():
                        details.append("Meshroom 日志尾部:")
                        details.append(log_tail)
                except OSError:
                    pass
            return StepResult("reconstruct", False, "\n".join(details))

        return StepResult("reconstruct", True, "Meshroom 重建完成", mesh_path)

    except subprocess.CalledProcessError as error:
        details: list[str] = [
            f"Meshroom 重建失败，返回码: {error.returncode}",
            f"运行目录: {workspace_dir}",
            f"图像目录: {images_dir}",
            "执行命令:",
            " ".join(f'"{part}"' if " " in part else part for part in command),
        ]
        if log_path.exists():
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="ignore")[-4000:]
                if log_tail.strip():
                    details.append("Meshroom 日志尾部:")
                    details.append(log_tail)
            except OSError:
                pass
        return StepResult("reconstruct", False, "\n".join(details))

    finally:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)


def run_detect(task: TaskInput, config: PipelineConfig, paths: TaskPaths, visual_callback=None) -> StepResult:
    artifacts = run_detection_stage(
        config,
        work_dir=task.work_dir,
        visual_callback=visual_callback,
        output_dir=task.work_dir / "runs" / "detect",
    )

    if not isinstance(artifacts, DetectArtifacts) or not artifacts.detect_csv_local.exists():
        return StepResult("detect", False, "未生成检测 CSV")

    details = (
        f"检测完成: 抽帧 {artifacts.frame_count} 张, "
        f"有效帧 {artifacts.valid_frame_count} 张"
    )
    return StepResult("detect", True, details, artifacts.detect_csv_local)


def run_match(config: PipelineConfig, paths: TaskPaths) -> StepResult:
    if config.model_mode == "reconstruction":
        prepare_reconstruction_csv(paths.detect_csv_local, paths.matched_csv_output)
        return StepResult("match", True, "已跳过模型库匹配，保留检测结果供重建模式使用", paths.matched_csv_output)

    if config.dataset_path is None:
        return StepResult("match", False, "缺少 dataset_path，无法执行模型匹配")

    process_csv(
        str(paths.detect_csv_local),
        str(config.dataset_path),
        str(paths.matched_csv_output),
        default_model=config.default_model,
        skip_ck=config.skip_ck,
        one_per_segment=config.one_model_per_segment,
        total_segments=estimate_total_segments(config),
        pipe_inner=config.inner,
        pipe_outer=config.outer,
        segment_length_hint=config.length,
        wall_thickness=config.wall_thickness,
        rebar_spacing=config.rebar_spacing,
    )
    if not paths.matched_csv_output.exists():
        return StepResult("match", False, "未生成匹配 CSV")
    return StepResult("match", True, "模型匹配完成", paths.matched_csv_output)


def run_reconstruction(task: TaskInput, config: PipelineConfig, paths: TaskPaths) -> StepResult:
    if config.model_mode != "reconstruction":
        return StepResult("reconstruct", True, "当前为模型库模式，无需重建")

    paths.reconstruction_dir.mkdir(parents=True, exist_ok=True)
    prepare_reconstruction_csv(paths.detect_csv_local, paths.matched_csv_output)

    if config.reconstruction_command:
        command_text = expand_runtime_value(config.reconstruction_command, task, config, paths)
        command = shlex.split(command_text, posix=False)
        try:
            subprocess.run(command, check=True, cwd=task.work_dir)
        except subprocess.CalledProcessError as error:
            return StepResult("reconstruct", False, f"外部重建命令执行失败: {error}")
    elif not config.reconstruction_mesh and config.meshroom_path is not None:
        return run_meshroom_reconstruction(task, config, paths)

    if config.reconstruction_mesh:
        mesh_path = resolve_runtime_path(config.reconstruction_mesh, task, config, paths)
        if not mesh_path.exists():
            return StepResult("reconstruct", False, f"重建 mesh 不存在: {mesh_path}")
        return StepResult("reconstruct", True, "重建 mesh 已就绪", mesh_path)

    mesh_path = find_reconstruction_mesh(paths.reconstruction_dir)
    if mesh_path is None:
        return StepResult("reconstruct", False, "外部重建已执行，但未定位到重建 mesh；请配置 reconstruction_mesh")

    return StepResult("reconstruct", True, "重建 mesh 已就绪", mesh_path)


def run_export(task: TaskInput, config: PipelineConfig, paths: TaskPaths, source_mesh: Path | None = None) -> StepResult:
    if config.blender_path is None:
        return StepResult("export", False, "缺少 blender_path，无法导出 GLB")
    paths.final_glb.parent.mkdir(parents=True, exist_ok=True)

    if config.model_mode == "reconstruction":
        if source_mesh is None:
            return StepResult("export", False, "重建模式缺少 source_mesh，无法导出 GLB")

        blender_script = get_script_dir() / "scripts" / "blender_reconstruction.py"
        command = [
            str(config.blender_path),
            "--background",
            "--python",
            str(blender_script),
            "--",
            "--mesh",
            str(source_mesh),
            "--output",
            str(paths.final_glb),
            "--inner",
            str(config.inner),
            "--outer",
            str(config.outer),
            "--scale-target",
            str(config.reconstruction_scale_target),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as error:
            stderr_text = error.stderr[-4000:] if error.stderr else ""
            return StepResult("export", False, f"Blender 重建导出失败: {error}\n{stderr_text}")

        if not paths.final_glb.exists():
            return StepResult("export", False, "未生成重建 GLB 文件")
        return StepResult("export", True, "重建 GLB 生成成功", paths.final_glb)

    blender_script = get_script_dir() / "scripts" / "blender_splice.py"
    if config.manhole_path is None:
        return StepResult("export", False, "缺少 manhole_path，无法导出拼接 GLB")

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
    ]

    # 计算总管节数（基于里程 CSV 最大里程）
    total_segments = estimate_total_segments(config)
    command.extend(["--total-segments", str(total_segments)])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as error:
        stderr_text = error.stderr[-4000:] if error.stderr else ""
        return StepResult("export", False, f"Blender 运行出错: {error}\n{stderr_text}")

    if _load_defect_ids_from_csv(paths.matched_csv_output):
        rerun_marker = paths.final_glb.parent / "_patch_export_rerun_marker.txt"
        rerun_marker.write_text("rerun-started", encoding="utf-8")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
            rerun_marker.write_text("rerun-completed", encoding="utf-8")
        except subprocess.CalledProcessError as error:
            stderr_text = error.stderr[-4000:] if error.stderr else ""
            return StepResult("export", False, f"Blender 二次导出出错: {error}\n{stderr_text}")

    if not paths.final_glb.exists():
        return StepResult("export", False, "未生成管道 GLB 文件")
    rewrite_result_csv_model_paths(paths.matched_csv_output, paths.final_glb, paths.patch_glb)

    return StepResult("export", True, "GLB 生成成功", paths.final_glb)


def write_info_file(config: PipelineConfig, paths: TaskPaths, source_mesh: Path | None = None) -> StepResult:
    paths.info_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"管节外径: {config.outer}",
        f"建模模式: {config.model_mode}",
    ]
    if config.model_mode == "reconstruction":
        lines.append(f"尺度对齐: {config.reconstruction_scale_target}")
    if source_mesh is not None:
        lines.append(f"模型来源: {source_mesh}")

    paths.info_txt.write_text("\n".join(lines), encoding="utf-8")
    return StepResult("info", True, "信息文件生成", paths.info_txt)
