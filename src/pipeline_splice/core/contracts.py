from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TaskInput:
    csv_path: Path
    video_path: Path
    work_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    blender_path: Path | None
    dataset_path: Path | None
    meshroom_path: Path | None
    video_path: Path
    raw_csv_path: Path
    model_mode: str
    reconstruction_command: str
    reconstruction_mesh: str
    reconstruction_scale_target: str
    meshroom_describer_types: str
    meshroom_describer_preset: str
    meshroom_match_method: str
    meshroom_default_fov: float
    meshroom_depth_downscale: int
    manhole_path: Path | None
    inner: float
    outer: float
    length: float
    segment: int
    interval: int
    use_depth: bool = True
    default_model: str = "QKG"
    skip_ck: bool = True
    one_model_per_segment: bool = True
    global_x_offset: float = -2.0
    manhole_half_length: float = 2.0


@dataclass(frozen=True)
class TaskPaths:
    detect_csv_local: Path
    detect_csv_output: Path
    matched_csv_output: Path
    reconstruction_dir: Path
    final_glb: Path
    info_txt: Path
    mode_tag: str


@dataclass(frozen=True)
class DetectArtifacts:
    result_dir_name: str
    frame_dir: Path
    frame_count: int
    valid_frame_count: int
    detect_csv_local: Path


@dataclass
class StepResult:
    name: str
    success: bool
    details: str
    output_path: Path | None = None


@dataclass
class PipelineResult:
    task: TaskInput
    config: PipelineConfig
    paths: TaskPaths
    steps: list[StepResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(step.success for step in self.steps)
