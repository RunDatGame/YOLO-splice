# build.ps1 - PyInstaller packaging script (library/splice mode)
# Run in YOLO conda env: conda activate YOLO; .\build.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$DateStr = Get-Date -Format "yyyyMMdd_HHmmss"
$DistDir = "E:\YOLO-splice-package\dist-$DateStr"
$Index = 1
while (Test-Path $DistDir) {
    $DistDir = "E:\YOLO-splice-package\dist-${DateStr}_$($Index.ToString("000"))"
    $Index++
}

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

# Clean old build cache
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ProjectRoot\build"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ProjectRoot\dist"

Write-Host "Building PipelineWatcher (UV mode)..." -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot" -ForegroundColor Gray

# Build PyInstaller command
$PyInstallerArgs = @(
    "--onedir",
    "--name", "PipelineWatcher",
    "--distpath", "$DistDir",
    "--workpath", "$ProjectRoot\build",
    "--specpath", "$ProjectRoot",
    "--paths", "$ProjectRoot\src",
    "--paths", "$ProjectRoot",
    "--add-data", "data;data",
    "--add-data", "config;config",
    "--add-data", "weights;weights",
    "--add-data", "checkpoints;checkpoints",
    "--add-data", "scripts;scripts",
    "--add-data", "utils;utils",
    "--hidden-import", "pipeline_splice.core.runner",
    "--hidden-import", "pipeline_splice.core.steps",
    "--hidden-import", "pipeline_splice.core.detect_stage",
    "--hidden-import", "pipeline_splice.core.config",
    "--hidden-import", "pipeline_splice.core.contracts",
    "--collect-submodules", "depth_anything_v2",
    "--hidden-import", "pipeline_splice.detection.engine",
    "--hidden-import", "pipeline_splice.detection.mileage",
    "--hidden-import", "pipeline_splice.detection.frames",
    "--hidden-import", "pipeline_splice.detection.detector",
    "--hidden-import", "pipeline_splice.detection._common",
    "--hidden-import", "pipeline_splice.detection.export",
    "--hidden-import", "pipeline_splice.detection.depth",
    "--hidden-import", "pandas",
    "--hidden-import", "numpy",
    "--hidden-import", "cv2",
    "--hidden-import", "PIL",
    "--hidden-import", "watchdog.observers",
    "--hidden-import", "watchdog.events",
    "--noconfirm",
    "$ProjectRoot\watcher.py"
)

$PythonCmd = $null

# 优先尝试使用 conda 的 YOLO 环境（无需手动激活）
try {
    $condaPath = (Get-Command conda -ErrorAction SilentlyContinue).Source
    if ($condaPath) {
        $PythonCmd = "conda run -n YOLO python"
    }
} catch {
    $PythonCmd = $null
}

# 回退到当前环境中的 python
if (-not $PythonCmd) {
    $PythonCmd = "python"
}

Write-Host "Using Python command: $PythonCmd" -ForegroundColor Gray

Invoke-Expression "$PythonCmd -m PyInstaller @PyInstallerArgs"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed, exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete. Output: $DistDir\PipelineWatcher" -ForegroundColor Green

# Generate config.txt template (user edits this before running)
$configContent = @"
# ================= 环境路径配置 =================
blender_path: ./blender/blender.exe
dataset_path: ./排水管道内缺陷模型库1.0
manhole_path: ./排水管道井室模板库1.0/标准圆形井室0.6.glb
model_mode: library
use_depth: true
inner: 0.54
outer: 0.60
length: 3.0
segment: 1
interval: 100
pixel_per_meter: 1000
default_model: QKG
skip_ck: true
one_model_per_segment: true
global_x_offset: -2.0
manhole_half_length: 2.0
wall_thickness: 0.1
rebar_spacing: 0.0
"@
$configContent | Out-File -FilePath "$DistDir\PipelineWatcher\config.txt" -Encoding UTF8 -NoNewline

# Create video_Config.txt initial template
$videoConfigContent = "管节外径: 0.6`n建模模式: splice`n"
$videoConfigContent | Out-File -FilePath "$DistDir\PipelineWatcher\video_Config.txt" -Encoding UTF8 -NoNewline

# Copy task_list.txt template to output dir
Copy-Item -Force "$ProjectRoot\task_list.txt" "$DistDir\PipelineWatcher\task_list.txt" -ErrorAction SilentlyContinue
if (-not (Test-Path "$DistDir\PipelineWatcher\task_list.txt")) {
    New-Item -ItemType File -Path "$DistDir\PipelineWatcher\task_list.txt" | Out-Null
}

Write-Host "Config ready." -ForegroundColor Green
Write-Host "Package ready." -ForegroundColor Green
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  1. Edit config.txt (blender_path, dataset_path, manhole_path)" -ForegroundColor Gray
Write-Host "  2. Double-click PipelineWatcher.exe to start daemon" -ForegroundColor Gray
Write-Host "  3. Write task to task_list.txt: ./xxx.csv ./xxx.mp4" -ForegroundColor Gray
Write-Host "  4. GLB/CSV output next to input video" -ForegroundColor Gray
