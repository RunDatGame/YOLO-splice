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

function Get-PreviousPackageCandidates {
    param(
        [string]$PackageBaseDir,
        [string]$CurrentDistDir
    )

    $currentName = Split-Path $CurrentDistDir -Leaf
    $candidates = Get-ChildItem -Path $PackageBaseDir -Directory -Filter 'dist-*' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne $currentName } |
        Sort-Object LastWriteTime -Descending

    return @(
        foreach ($candidate in $candidates) {
            $packageRoot = Join-Path $candidate.FullName 'PipelineWatcher'
            if (Test-Path $packageRoot) {
                $packageRoot
            }
        }
    )
}

function Copy-PackagedResourceDirs {
    param(
        [object[]]$SourcePackageRoots,
        [string]$TargetPackageRoot
    )

    if (-not $SourcePackageRoots -or $SourcePackageRoots.Count -eq 0) {
        Write-Host 'No previous package roots found for resource migration.' -ForegroundColor Yellow
        return
    }

    $resourceDirs = @(
        'blender',
        '排水管道内缺陷模型库1.0',
        '排水管道井室模板库1.0'
    )

    foreach ($dirName in $resourceDirs) {
        $targetDir = Join-Path $TargetPackageRoot $dirName

        if (Test-Path $targetDir) {
            Write-Host "Resource already present, skip: $targetDir" -ForegroundColor Gray
            continue
        }

        $sourceDir = $null
        foreach ($packageRoot in $SourcePackageRoots) {
            $candidateSource = Join-Path $packageRoot $dirName
            if (Test-Path $candidateSource) {
                $sourceDir = $candidateSource
                break
            }
        }
        if (-not $sourceDir) {
            Write-Host "Resource missing in all previous packages, skip: $dirName" -ForegroundColor Yellow
            continue
        }

        Write-Host "Copying resource: $sourceDir -> $targetDir" -ForegroundColor Cyan
        Copy-Item -LiteralPath $sourceDir -Destination $targetDir -Recurse
    }
}

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
    "--hidden-import", "pipeline_splice.joint_counter.api",
    "--collect-submodules", "pipeline_splice.joint_counter.core",
    "--collect-submodules", "depth_anything_v2",
    "--hidden-import", "pipeline_splice.detection.engine",
    "--hidden-import", "pipeline_splice.detection.mileage",
    "--hidden-import", "pipeline_splice.detection.frames",
    "--hidden-import", "pipeline_splice.detection.detector",
    "--hidden-import", "pipeline_splice.detection._common",
    "--hidden-import", "pipeline_splice.detection.export",
    "--hidden-import", "pipeline_splice.detection.depth",
    "--collect-submodules", "ultralytics",
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

$PreviousPackageRoots = Get-PreviousPackageCandidates -PackageBaseDir 'E:\YOLO-splice-package' -CurrentDistDir $DistDir
Copy-PackagedResourceDirs -SourcePackageRoots $PreviousPackageRoots -TargetPackageRoot "$DistDir\PipelineWatcher"

# Copy existing config files from project root (avoid auto-generating to prevent encoding issues)
Copy-Item -Force "$ProjectRoot\config.txt" "$DistDir\PipelineWatcher\config.txt" -ErrorAction SilentlyContinue
Copy-Item -Force "$ProjectRoot\config\splice.txt" "$DistDir\PipelineWatcher\config.txt" -ErrorAction SilentlyContinue
Copy-Item -Force "$ProjectRoot\task_list.txt" "$DistDir\PipelineWatcher\task_list.txt" -ErrorAction SilentlyContinue
if (-not (Test-Path "$DistDir\PipelineWatcher\task_list.txt")) {
    New-Item -ItemType File -Path "$DistDir\PipelineWatcher\task_list.txt" | Out-Null
}

Write-Host "Package ready." -ForegroundColor Green
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  1. Edit config.txt (blender_path, dataset_path, manhole_path)" -ForegroundColor Gray
Write-Host "  2. Double-click PipelineWatcher.exe to start daemon" -ForegroundColor Gray
Write-Host "  3. Write task to task_list.txt: ./xxx.csv ./xxx.mp4" -ForegroundColor Gray
Write-Host "  4. GLB/CSV output next to input video" -ForegroundColor Gray
