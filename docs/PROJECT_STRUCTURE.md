# 项目结构说明

## 三种独立方法

- `methods/splice/run.py`
  - 缺陷检测 + 模型库拼接建模
  - 默认配置: `configs/splice.txt`
- `methods/reconstruction/run.py`
  - 三维重建建模
  - 默认配置: `configs/reconstruction.txt`
- `methods/cylindrical_texture/run.py`
  - 圆柱纹理重建
  - 默认配置: `configs/cylindrical_texture.txt`

## 共享代码

- `pipeline_core/`
  - 拼接建模和三维重建共用的调度、配置和步骤封装
- `DetectV1.py`
  - 检测和切帧
- `match_glb_files.py`
  - 病害模型匹配
- `blender_script.py`
  - 拼接建模导出
- `blender_reconstruction_script.py`
  - 重建导出

## 共享资源

- `blender/`
- `weights/`
- `checkpoints/`
- `depth_anything_v2/`
- `models/`
- `utils/`
- `排水管道内缺陷模型库1.0/`
- `排水管道井室模板库1.0/`

## 配置和样例

- `configs/`
  - 三种方法各自的独立配置
- `samples/input/`
  - 示例视频和 CSV

## 辅助工具

- `main_gui.py`
  - 图形界面入口
- `tools/watchers/`
  - 文件监听入口
- `preflight_check.py`
  - 环境检查

## 运行输出

- `outputs/splice/`
- `outputs/reconstruction/`
- `outputs/cylindrical_texture/`

这些目录为运行产物，可按需清理，不属于源码。
