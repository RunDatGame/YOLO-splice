# build.ps1 - PyInstaller packaging script (UV mode)
# Run in YOLO conda env: conda activate YOLO; .\build.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$DistDir = "E:\YOLO-splice-package\dist-uv"

# Ensure output dir exists
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
    "--hidden-import", "pipeline_splice.uv.runner",
    "--hidden-import", "pipeline_splice.uv.texture_builder",
    "--hidden-import", "pipeline_splice.core.config",
    "--hidden-import", "pipeline_splice.core.detect_stage",
    "--hidden-import", "pipeline_splice.core.contracts",
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

& "C:\Users\Administrator\.conda\envs\YOLO\Scripts\pyinstaller.exe" @PyInstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed, exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "Build complete. Output: $DistDir\PipelineWatcher" -ForegroundColor Green

# Copy config/ to output dir (exe-level, for program to read)
Copy-Item -Recurse -Force "$ProjectRoot\config" "$DistDir\PipelineWatcher\config" -ErrorAction SilentlyContinue

# Copy task_list.txt template to output dir
Copy-Item -Force "$ProjectRoot\task_list.txt" "$DistDir\PipelineWatcher\task_list.txt" -ErrorAction SilentlyContinue
if (-not (Test-Path "$DistDir\PipelineWatcher\task_list.txt")) {
    New-Item -ItemType File -Path "$DistDir\PipelineWatcher\task_list.txt" | Out-Null
}

# Create video_Config.txt template (for external use)
$defaultOuter = "0.6"
$templateContent = "管节外径: $defaultOuter`n建模模式: uv_texture`n"
$templateContent | Out-File -FilePath "$DistDir\PipelineWatcher\video_Config.txt" -Encoding UTF8 -NoNewline

Write-Host "Config ready." -ForegroundColor Green
Write-Host "Package ready." -ForegroundColor Green
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  1. Double-click PipelineWatcher.exe to start daemon" -ForegroundColor Gray
Write-Host "  2. Write task to task_list.txt: ./xxx.csv ./xxx.mp4" -ForegroundColor Gray
Write-Host "  3. Output in outputs/uv_texture/ (next to input video)" -ForegroundColor Gray
Write-Host "  4. video_Config.txt has pipe outer diameter and mode (for external use)" -ForegroundColor Gray
