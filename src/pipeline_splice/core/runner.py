from __future__ import annotations

from pathlib import Path

from .config import (
    build_task_input,
    build_task_paths,
    load_pipeline_config,
    normalize_model_mode,
    validate_detection_config,
    validate_runtime_config,
)
from .contracts import PipelineResult, StepResult
from .steps import (
    apply_joint_detection_overrides,
    copy_if_needed,
    detect_csv_matches_segment_length,
    extract_video_frames,
    resolve_runtime_path,
    run_detect,
    run_joint_detection,
    run_export,
    run_match,
    run_meshroom_reconstruction,
    run_reconstruction,
    write_info_file,
)


def _build_runtime(
    csv_path: str | Path,
    video_path: str | Path,
    work_dir: str | Path = ".",
    model_mode: str | None = None,
    config_path: str | Path | None = None,
):
    task = build_task_input(csv_path, video_path, work_dir)
    runtime_config_path = Path(config_path).resolve() if config_path else task.work_dir / "config.txt"

    normalized_mode = normalize_model_mode(model_mode)
    config = load_pipeline_config(runtime_config_path, task, model_mode_override=normalized_mode)
    paths = build_task_paths(task, config.model_mode)
    return task, config, paths


def prepare_detection(
    csv_path: str | Path,
    video_path: str | Path,
    work_dir: str | Path = ".",
    visual_callback=None,
    config_path: str | Path | None = None,
) -> PipelineResult:
    task, config, paths = _build_runtime(
        csv_path,
        video_path,
        work_dir,
        model_mode="library",
        config_path=config_path,
    )
    validate_detection_config(config, task)
    result = PipelineResult(task=task, config=config, paths=paths)

    print("=" * 40)
    print("检测阶段启动")
    print(f"视频路径: {task.video_path}")
    print(f"输出目录: {task.output_dir}")
    print("[OK] Config 已更新")
    print(">>> [1/1] 运行 DetectV1...")
    detect_step = run_detect(task, config, paths, visual_callback=visual_callback)
    result.steps.append(detect_step)
    if not detect_step.success:
        print(detect_step.details)
        print("=" * 40)
        return result
    print("[OK] 检测阶段完成")
    print("=" * 40)
    return result


def process_task(
    csv_path: str | Path,
    video_path: str | Path,
    work_dir: str | Path = ".",
    model_mode: str | None = None,
    visual_callback=None,
    reuse_detection: bool = False,
    config_path: str | Path | None = None,
) -> PipelineResult:
    task, config, paths = _build_runtime(
        csv_path,
        video_path,
        work_dir,
        model_mode=model_mode,
        config_path=config_path,
    )
    validate_runtime_config(config, task)
    result = PipelineResult(task=task, config=config, paths=paths)

    print("=" * 40)
    print("任务启动")
    print(f"视频路径: {task.video_path}")
    print(f"输出目录: {task.output_dir}")
    print("[OK] Config 已更新")
    print(f"建模方式: {'拼接模型' if config.model_mode == 'library' else '三维重建算法'}")
    total_steps = 4 if config.enable_joint_detection else 3

    if config.enable_joint_detection:
        print(f">>> [1/{total_steps}] 管节计数...")
        joint_step = run_joint_detection(task, config, paths)
        result.steps.append(joint_step)
        if not joint_step.success:
            print(joint_step.details)
            print("=" * 40)
            return result
        print(f"[OK] {joint_step.details}")
        original_length = config.length
        config = apply_joint_detection_overrides(config, paths)
        result.config = config
        if abs(config.length - original_length) > 1e-6:
            print(f"[INFO] 管节长度已按管节计数结果修正: {original_length}m -> {config.length}m")

    if reuse_detection:
        detect_step_index = 2 if config.enable_joint_detection else 1
        print(f">>> [{detect_step_index}/{total_steps}] 复用已有 DetectV1 结果...")
        can_reuse_detection = paths.detect_csv_local.exists() and detect_csv_matches_segment_length(
            paths.detect_csv_local, config.length
        )
        if not can_reuse_detection:
            detect_step = run_detect(task, config, paths, visual_callback=visual_callback)
        else:
            detect_step = StepResult("detect", True, "已复用现有检测结果", paths.detect_csv_local)
        result.steps.append(detect_step)
        if not detect_step.success:
            print(detect_step.details)
            print("=" * 40)
            return result
    else:
        detect_step_index = 2 if config.enable_joint_detection else 1
        print(f">>> [{detect_step_index}/{total_steps}] 运行 DetectV1...")
        detect_step = run_detect(task, config, paths, visual_callback=visual_callback)
        result.steps.append(detect_step)
        if not detect_step.success:
            print(detect_step.details)
            print("=" * 40)
            return result

    source_mesh = None
    if config.model_mode == "reconstruction":
        reconstruction_step_index = 3 if config.enable_joint_detection else 2
        print(f">>> [{reconstruction_step_index}/{total_steps}] 准备重建 mesh...")
        reconstruction_step = run_reconstruction(task, config, paths)
        result.steps.append(reconstruction_step)
        if not reconstruction_step.success:
            print(reconstruction_step.details)
            print("=" * 40)
            return result
        source_mesh = reconstruction_step.output_path
    else:
        match_step_index = 3 if config.enable_joint_detection else 2
        print(f">>> [{match_step_index}/{total_steps}] 匹配模型...")
        match_step = run_match(config, paths)
        result.steps.append(match_step)
        if not match_step.success:
            print(match_step.details)
            print("=" * 40)
            return result

    export_step_index = total_steps
    print(f">>> [{export_step_index}/{total_steps}] 生成 GLB...")
    print(f"目标文件: {paths.final_glb}")
    export_step = run_export(task, config, paths, source_mesh=source_mesh)
    result.steps.append(export_step)
    print("[OK] GLB 生成成功" if export_step.success else f"[FAIL] {export_step.details}")

    print("=" * 40)
    return result


def process_reconstruction_only(
    csv_path: str | Path,
    video_path: str | Path,
    work_dir: str | Path = ".",
    force_extract: bool = False,
    export_glb: bool = True,
    config_path: str | Path | None = None,
) -> PipelineResult:
    task, config, paths = _build_runtime(
        csv_path,
        video_path,
        work_dir,
        model_mode="reconstruction",
        config_path=config_path,
    )
    result = PipelineResult(task=task, config=config, paths=paths)

    print("=" * 40)
    print("重建调试入口启动")
    print(f"视频路径: {task.video_path}")
    print(f"输出目录: {paths.reconstruction_dir.parent}")
    print("说明: 当前流程不执行病害检测，只复用/生成切帧并进行三维重建")

    if not task.video_path.exists():
        result.steps.append(StepResult("prepare", False, f"视频文件不存在: {task.video_path}"))
        print("=" * 40)
        return result

    if not config.reconstruction_mesh and config.meshroom_path is None:
        result.steps.append(
            StepResult("prepare", False, "缺少 meshroom_path 或 reconstruction_mesh，无法执行重建调试流程")
        )
        print("=" * 40)
        return result

    if export_glb and config.blender_path is None:
        result.steps.append(StepResult("prepare", False, f"缺少 blender_path，无法导出 GLB: {config.config_path}"))
        print("=" * 40)
        return result

    source_mesh = None
    if config.reconstruction_mesh:
        mesh_path = resolve_runtime_path(config.reconstruction_mesh, task, config, paths)
        reconstruction_step = StepResult(
            "reconstruct",
            mesh_path.exists(),
            "重建 mesh 已就绪" if mesh_path.exists() else f"重建 mesh 不存在: {mesh_path}",
            mesh_path if mesh_path.exists() else None,
        )
        result.steps.append(reconstruction_step)
        if not reconstruction_step.success:
            print(reconstruction_step.details)
            print("=" * 40)
            return result
        source_mesh = reconstruction_step.output_path
    else:
        print(">>> [1/3] 准备切帧...")
        extract_step = extract_video_frames(task, config, force=force_extract)
        result.steps.append(extract_step)
        if not extract_step.success:
            print(extract_step.details)
            print("=" * 40)
            return result

        print(">>> [2/3] 运行 Meshroom...")
        reconstruction_step = run_meshroom_reconstruction(task, config, paths)
        result.steps.append(reconstruction_step)
        if not reconstruction_step.success:
            print(reconstruction_step.details)
            print("=" * 40)
            return result
        source_mesh = reconstruction_step.output_path

    if export_glb:
        print(">>> [3/3] 导出 Blender GLB...")
        export_step = run_export(task, config, paths, source_mesh=source_mesh)
        result.steps.append(export_step)
        if not export_step.success:
            print(export_step.details)
            print("=" * 40)
            return result

    print("=" * 40)
    return result
