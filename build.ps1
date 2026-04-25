# build.ps1 - PyInstaller 打包脚本 (UV 模式)
# 在 YOLO conda 环境中运行: conda activate YOLO; .\build.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$DistDir = "E:\YOLO-splice-package\dist"

# 确保输出目录存在
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

# 清理旧的构建缓存
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ProjectRoot\build"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "$ProjectRoot\dist"

Write-Host "开始打包 PipelineWatcher (UV 模式)..." -ForegroundColor Cyan
Write-Host "项目根目录: $ProjectRoot" -ForegroundColor Gray

# 构建 PyInstaller 命令
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
    "--hidden-import", "pipeline_splice.uv.runner",
    "--hidden-import", "pipeline_splice.uv.texture_builder",
    "--hidden-import", "pipeline_splice.core.config",
    "--hidden-import", "pipeline_splice.detection.mileage",
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
    Write-Host "打包失败，返回码: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "打包完成，输出目录: $DistDir\PipelineWatcher" -ForegroundColor Green

# 复制 task_list.txt 模板到输出目录
Copy-Item -Force "$ProjectRoot\task_list.txt" "$DistDir\PipelineWatcher\task_list.txt" -ErrorAction SilentlyContinue
if (-not (Test-Path "$DistDir\PipelineWatcher\task_list.txt")) {
    New-Item -ItemType File -Path "$DistDir\PipelineWatcher\task_list.txt" | Out-Null
}

Write-Host "打包产物已就绪。" -ForegroundColor Green
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  1. 双击 PipelineWatcher.exe 启动守护进程" -ForegroundColor Gray
Write-Host "  2. 向 task_list.txt 写入任务行: ./xxx.csv ./xxx.mp4" -ForegroundColor Gray
Write-Host "  3. 输出结果在 outputs/uv_texture/ 目录下" -ForegroundColor Gray
