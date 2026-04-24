# Pipeline Splice — 排水管道缺陷检测与三维建模

基于 YOLO 的管道缺陷检测 + 模型库拼接三维建模系统。优先支持**缺陷模型库拼接（splice）**流程，同时保留三维重建（reconstruction）兼容能力。

---

## 目录结构

```
YOLO-splice/
├── src/pipeline_splice/        # 核心 Python 包
│   ├── cli.py                  # CLI 入口
│   ├── core/                   # 调度与配置
│   ├── detection/              # 检测引擎（YOLO + Depth）
│   ├── modeling/               # 模型匹配与 Blender 导出封装
│   └── utils/                  # 通用工具
├── scripts/                    # Blender 后台脚本
├── models/                     # YOLO 模型定义
├── utils/                      # YOLO 训练/推理工具
├── weights/                    # YOLO 权重（best.pt）
├── checkpoints/                # Depth Anything V2 权重
├── config/                     # 配置文件
├── methods/                    # 保留的原方法入口（向后兼容）
├── samples/                    # 示例输入
└── assets/                     # 大资源放置说明
```

---

## 环境配置

### 1. 安装依赖

```bash
pip install -e .
```

或 Conda：

```bash
conda env create -f environment.yml
conda activate YOLO
```

### 2. 放置大型资源

- **Blender 程序** → 项目根目录 `blender/`
- **缺陷模型库** → `assets/排水管道内缺陷模型库1.0/`
- **井室模板库** → `assets/排水管道井室模板库1.0/`

详见 `assets/README.md`。

---

## 快速开始

### CLI 运行（推荐）

```bash
pipeline-splice --config config/splice.yaml --video samples/input/2024-07-31_202835_Front.mp4 --csv samples/input/11.csv
```

### Python API

```python
from pipeline_splice.core.runner import process_task

result = process_task(
    csv_path="samples/input/11.csv",
    video_path="samples/input/2024-07-31_202835_Front.mp4",
    work_dir=".",
    model_mode="library",
)
print(result.success)
```

---

## 主要流程

1. **检测阶段**：视频抽帧 → 里程同步 → YOLO 缺陷检测 → Depth 深度估算
2. **匹配阶段**：根据管径、缺陷类型、严重等级匹配预置 GLB 模型
3. **导出阶段**：Blender 后台拼接管节与井室，输出最终 GLB

---

## 保留兼容

- `methods/splice/run.py`、`methods/reconstruction/run.py` 等原入口保留不动
- `main_gui.py`（如需要）可直接复制到项目根目录使用

---

## 版本控制说明

- 小型文件（源码、权重 `.pt`、配置）纳入 Git
- 大型资源（Blender 程序、GLB 模型库、井室模板）被 `.gitignore` 排除，需手动放置
