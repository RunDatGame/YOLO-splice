import argparse
import os
import sys

import bpy
import numpy as np


def clean_scene():
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def import_mesh(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    before = {obj.name for obj in bpy.data.objects}

    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=filepath)
        except Exception:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        raise ValueError(f"暂不支持的重建 mesh 格式: {ext}")

    imported = [obj for obj in bpy.data.objects if obj.name not in before and obj.type == "MESH"]
    if not imported:
        raise RuntimeError("导入完成，但没有检测到 Mesh 对象")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]

    if len(imported) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "ReconstructedPipe"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def align_and_scale(obj, inner_diameter, outer_diameter, scale_target):
    mesh = obj.data
    if not mesh.vertices:
        raise RuntimeError("Mesh 不包含顶点，无法处理")

    coords = np.array([list(v.co) for v in mesh.vertices], dtype=np.float64)
    center = coords.mean(axis=0)
    centered = coords - center

    covariance = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1

    aligned = centered @ axes
    mins = aligned.min(axis=0)
    maxs = aligned.max(axis=0)
    extents = maxs - mins

    cross_diameter = float((extents[1] + extents[2]) / 2.0)
    scale_target = (scale_target or "inner").strip().lower()
    if scale_target == "outer":
        target_diameter = outer_diameter
    elif scale_target == "none":
        target_diameter = None
    else:
        target_diameter = inner_diameter

    scale_factor = 1.0
    if target_diameter and cross_diameter > 1e-9:
        scale_factor = float(target_diameter) / cross_diameter
        aligned *= scale_factor
        mins = aligned.min(axis=0)
        maxs = aligned.max(axis=0)
        extents = maxs - mins

    aligned[:, 0] -= mins[0]
    aligned[:, 1] -= (mins[1] + maxs[1]) / 2.0
    aligned[:, 2] -= (mins[2] + maxs[2]) / 2.0

    for vertex, coord in zip(mesh.vertices, aligned):
        vertex.co = coord.tolist()
    mesh.update()

    print(
        "[重建规整] "
        f"length={extents[0]:.4f}m, "
        f"cross={((extents[1] + extents[2]) / 2.0):.4f}m, "
        f"scale={scale_factor:.6f}, "
        f"target={scale_target}"
    )


def export_glb(output_path):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
    )


def main():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="重建 mesh 路径")
    parser.add_argument("--output", required=True, help="输出 GLB 路径")
    parser.add_argument("--inner", required=True, type=float, help="管道内径")
    parser.add_argument("--outer", required=True, type=float, help="管道外径")
    parser.add_argument(
        "--scale-target",
        default="inner",
        choices=["inner", "outer", "none"],
        help="重建 mesh 的尺度对齐基准",
    )
    parsed = parser.parse_args(args)

    if not os.path.exists(parsed.mesh):
        raise FileNotFoundError(f"重建 mesh 不存在: {parsed.mesh}")

    clean_scene()
    obj = import_mesh(parsed.mesh)
    align_and_scale(obj, parsed.inner, parsed.outer, parsed.scale_target)
    export_glb(parsed.output)
    print(f"[成功] 重建 GLB 已导出: {parsed.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[失败] 重建导出脚本异常: {exc}")
        sys.exit(1)
