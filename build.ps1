# build.ps1 - PyInstaller 打包脚本 (UV 模式)
# 在 YOLO conda 环境中运行: conda activate YOLO; .\build.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$DistDir = "E:\YOLO-splice-package\dist-uv"

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
    Write-Host "打包失败，返回码: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "打包完成，输出目录: $DistDir\PipelineWatcher" -ForegroundColor Green

# 复制 config/ 到输出目录（exe 同级目录，供程序读取）
Copy-Item -Recurse -Force "$ProjectRoot\config" "$DistDir\PipelineWatcher\config" -ErrorAction SilentlyContinue

# 复制 task_list.txt 模板到输出目录
Copy-Item -Force "$ProjectRoot\task_list.txt" "$DistDir\PipelineWatcher\task_list.txt" -ErrorAction SilentlyContinue
if (-not (Test-Path "$DistDir\PipelineWatcher\task_list.txt")) {
    New-Item -ItemType File -Path "$DistDir\PipelineWatcher\task_list.txt" | Out-Null
}

# 创建 video_Config.txt 模板（供外部读取管外径和模式）
$defaultOuter = "0.6"
$templateContent = "管节外径: $defaultOuter`n建模模式: uv_texture`n"
$templateContent | Out-File -FilePath "$DistDir\PipelineWatcher\video_Config.txt" -Encoding UTF8 -NoNewline

Write-Host "配置文件已就绪。" -ForegroundColor Green
Write-Host "打包产物已就绪。" -ForegroundColor Green
Write-Host "使用方法:" -ForegroundColor Yellow
Write-Host "  1. 双击 PipelineWatcher.exe 启动守护进程" -ForegroundColor Gray
Write-Host "  2. 向 task_list.txt 写入任务行: ./xxx.csv ./xxx.mp4" -ForegroundColor Gray
Write-Host "  3. 输出结果在 outputs/uv_texture/ 目录下" -ForegroundColor Gray
Write-Host "  4. video_Config.txt 包含管节外径和模式信息（供外部读取）" -ForegroundColor Gray