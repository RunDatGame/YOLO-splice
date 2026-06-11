import bpy
import math
import csv
import os
import sys
import argparse
import hashlib
from datetime import datetime


DEFECT_COLORS = {
    "破裂": (1.0, 0.15, 0.15, 1.0),
    "PL": (1.0, 0.15, 0.15, 1.0),
    "腐蚀": (1.0, 0.45, 0.05, 1.0),
    "FS": (1.0, 0.45, 0.05, 1.0),
    "错口": (1.0, 0.85, 0.10, 1.0),
    "CK": (1.0, 0.85, 0.10, 1.0),
}
DEFAULT_HIGHLIGHT_COLOR = (1.0, 0.0, 1.0, 0.7)
MANHOLE_OUTWARD_OFFSET = 0.4
PATCH_SURFACE_CLEARANCE = 0.012
PIPE_JOINT_OVERLAP = 0.04
RUN_NAME_TOKEN = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
_global_name_counter = 0


def parse_float(value, default=None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def segment_origin(seg_id, segment_length):
    return seg_id * (segment_length - PIPE_JOINT_OVERLAP)


def stable_defect_angle(defect_data):
    key_fields = (
        defect_data.get("_原始编号", defect_data.get("编号", "")),
        defect_data.get("模型类型", ""),
        defect_data.get("管节序号", ""),
        defect_data.get("节内里程", ""),
        defect_data.get("病害长", ""),
        defect_data.get("病害宽", ""),
    )
    key = "|".join(str(v) for v in key_fields)
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 36000) / 100.0


def normalize_defect_id(raw_id, row_index):
    raw = str(raw_id or "").strip()
    if raw:
        return raw
    return f"InerDisRow{row_index:04d}"


def is_placeholder_row(row):
    return False


def next_global_name(base_name, scope_prefix):
    global _global_name_counter
    _global_name_counter += 1
    return f"{base_name}_{scope_prefix}_{RUN_NAME_TOKEN}_{_global_name_counter:04d}"


def classify_imported_object(obj_name, fallback):
    lower_name = str(obj_name).lower()
    if "texture_sleeve" in lower_name or "sleeve" in lower_name:
        return "texture_sleeve"
    if "world" in lower_name:
        return "world"
    if "柱体" in str(obj_name):
        return "柱体"
    if "立方体" in str(obj_name):
        return "立方体"
    return fallback


def rename_imported_objects(objects, scope_prefix, fallback="部件"):
    for obj in objects:
        base_name = classify_imported_object(obj.name, fallback)
        unique_name = next_global_name(base_name, scope_prefix)
        obj.name = unique_name
        if getattr(obj, "data", None) is not None:
            obj.data.name = f"{unique_name}_mesh"


def clean_scene():
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def create_patch_material(name, color):
    mat = bpy.data.materials.new(name=name)
    mat.diffuse_color = color
    mat.use_backface_culling = False
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Emission Color"].default_value = color
    bsdf.inputs["Emission Strength"].default_value = 1.0
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = color[3]
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def create_highlight_patch(defect_data, segment_length, inner_radius, patch_collection):
    """使用绝对累计里程定位贴片。mileage 是从管道起点的绝对距离（米）。"""
    mileage = parse_float(defect_data.get("节内里程"), 0.0)
    axis_angle = parse_float(defect_data.get("轴线偏角"))
    if axis_angle is None:
        offset = parse_float(defect_data.get("偏移距"))
        if offset is not None and inner_radius > 0:
            axis_angle = math.degrees(offset / inner_radius)
        else:
            axis_angle = stable_defect_angle(defect_data)
    axis_angle = axis_angle % 360.0
    patch_len = max(parse_float(defect_data.get("病害长"), 0.1), 0.03)
    patch_wid = max(parse_float(defect_data.get("病害宽"), 0.1), 0.03)
    defect_type = str(defect_data.get("模型类型", ""))
    defect_id = str(defect_data.get("编号") or "")

    # 绝对里程直接作为沿管 X 坐标
    x_center = mileage
    angle_rad = math.radians(axis_angle)
    radius = max(inner_radius - PATCH_SURFACE_CLEARANCE, inner_radius * 0.94)

    import random
    seed_bytes = hashlib.sha1(defect_id.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(seed_bytes[:4], "big"))

    n_radial = 3 + rng.randint(0, 3)
    n_angular = 6 + rng.randint(0, 4)

    x_half = patch_len / 2
    arc_half = (patch_wid / inner_radius) / 2

    verts = []
    for ri in range(n_radial + 1):
        x_frac = ri / max(n_radial, 1)
        x = x_center - x_half + x_frac * patch_len
        for ai in range(n_angular + 1):
            a_frac = ai / max(n_angular, 1)
            a = angle_rad - arc_half + a_frac * 2 * arc_half
            edge_factor = abs(x_frac - 0.5) * 2 + abs(a_frac - 0.5) * 2
            noise_x = rng.uniform(-0.015, 0.015) * edge_factor
            noise_a = rng.uniform(-0.08, 0.08) * edge_factor
            r = radius + rng.uniform(-0.0015, 0.0005)
            a_noisy = a + noise_a
            xx = x + noise_x
            verts.append((xx, r * math.cos(a_noisy), r * math.sin(a_noisy)))

    faces = []
    for ri in range(n_radial):
        for ai in range(n_angular):
            v00 = ri * (n_angular + 1) + ai
            v01 = v00 + 1
            v10 = (ri + 1) * (n_angular + 1) + ai
            v11 = v10 + 1
            # 贴片用于管内观察，法线朝向管内，避免只显示背面时被管壁盖住。
            faces.append((v00, v01, v11, v10))

    mesh = bpy.data.meshes.new(defect_id)
    obj = bpy.data.objects.new(defect_id, mesh)
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    mesh.validate()
    mesh.update()
    patch_collection.objects.link(obj)

    color = DEFECT_COLORS.get(defect_type, DEFAULT_HIGHLIGHT_COLOR)
    mat = create_patch_material(defect_id, color)
    obj.data.materials.append(mat)
    bpy.context.view_layer.update()
    return obj


def import_pipe_segment(model_path, seg_id, seg_offset, segment_length, pipe_collection):
    if not model_path or not os.path.exists(str(model_path)):
        return None

    try:
        bpy.ops.import_scene.gltf(filepath=str(model_path))
    except Exception as e:
        print(f"    [错误] 导入失败 {model_path}: {e}")
        return None

    imported_objects = list(bpy.context.selected_objects)
    segment_objects = []
    for obj in imported_objects:
        if obj.name.lower().startswith("world"):
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            segment_objects.append(obj)

    if not segment_objects:
        print(f"    [警告] {model_path} 无物体")
        return None

    rename_imported_objects(segment_objects, f"seg{seg_id:02d}", fallback="管节部件")

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    container = bpy.context.active_object
    container.name = f"Segment_{seg_id}_Root"

    default_collection = bpy.context.scene.collection
    if container.name in default_collection.objects:
        default_collection.objects.unlink(container)
    if container.name not in pipe_collection.objects:
        pipe_collection.objects.link(container)

    for obj in segment_objects:
        obj.parent = container
        for coll in list(obj.users_collection):
            if coll != pipe_collection:
                coll.objects.unlink(obj)
        if obj.name not in pipe_collection.objects:
            pipe_collection.objects.link(obj)

    # 管节模型库以自身 Z 轴为管长方向；绕 Y 轴旋转 90° 后 Z 轴对齐到 X 轴，
    # 导出 Y-up 后表现为沿管线方向连续排列。
    physical_index = seg_id - seg_offset
    x_pos = segment_origin(physical_index, segment_length) + segment_length / 2
    container.rotation_euler = (0, math.radians(90), 0)
    container.location = (x_pos, 0, 0)
    bpy.context.view_layer.update()

    for obj in segment_objects:
        world_matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world_matrix

    bpy.data.objects.remove(container, do_unlink=True)

    bpy.ops.object.select_all(action='DESELECT')
    return segment_objects


def place_manhole(filepath, name_prefix, location, rotation_z, pipe_collection):
    print(f"  井室 {name_prefix}: X={location[0]:.2f} Y={location[1]:.2f}")
    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except Exception as e:
        print(f"  [错误] 导入井室失败 {filepath}: {e}")
        return

    imported_objects = bpy.context.selected_objects
    if not imported_objects:
        print(f"  [警告] {filepath} 无物体")
        return

    rename_imported_objects(list(imported_objects), name_prefix.lower(), fallback="井室部件")

    # 检查并烘焙因 GLTF unitScale 自动引入的对象 scale，避免纹理视觉拉伸
    for obj in imported_objects:
        if getattr(obj, "data", None) is None:
            continue
        sx, sy, sz = obj.scale
        if abs(sx - 1.0) > 1e-4 or abs(sy - 1.0) > 1e-4 or abs(sz - 1.0) > 1e-4:
            print(f"  [INFO] {obj.name} 存在非单位缩放 ({sx:.4f}, {sy:.4f}, {sz:.4f})，已烘焙到网格")
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(scale=True)

    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    container = bpy.context.active_object
    container.name = next_global_name(name_prefix + "_Root", "manhole_root")

    for obj in imported_objects:
        obj.parent = container
        # 从其他 collection 移除
        for coll in list(obj.users_collection):
            if coll != pipe_collection:
                coll.objects.unlink(obj)
        # 添加到 pipe_collection（子对象也需要在集合中才能被选中导出）
        if obj.name not in pipe_collection.objects:
            pipe_collection.objects.link(obj)

    # 只将容器加入 pipe_collection
    default_collection = bpy.context.scene.collection
    if container.name in default_collection.objects:
        default_collection.objects.unlink(container)
    if container.name not in pipe_collection.objects:
        pipe_collection.objects.link(container)

    # 绕 Z 轴额外旋转 -90°，使井室朝向 Y 轴方向
    container.rotation_euler = (0, 0, rotation_z - math.radians(90))
    container.location = location
    bpy.ops.object.select_all(action='DESELECT')


def find_fallback_models(dataset_path, outer_diameter, default_model):
    """从模型库查找无病害管节的默认模型列表（FS1/DS1 轮换用）。

    Returns:
        模型路径列表 [fs1_path, ds1_path]，过滤掉不存在的。如果都找不到则返回空列表。
    """
    import re
    if not dataset_path or not os.path.isdir(dataset_path):
        return []

    diameter_str = f"{float(outer_diameter):g}m"
    model_prefix = str(default_model).strip().upper()
    # FS1 和 DS1 精确匹配，排除 HUSC
    target_defects = ["FS1", "DS1"]

    all_candidates = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.lower().endswith(".glb"):
                continue
            all_candidates.append(os.path.join(root, f))

    results = []
    for defect in target_defects:
        pattern = re.compile(
            rf"^{re.escape(diameter_str)}{re.escape(model_prefix)}_{re.escape(defect)}\.glb$",
            re.IGNORECASE,
        )
        matches = []
        for path in all_candidates:
            if pattern.match(os.path.basename(path)) and "HUSC" not in os.path.basename(path).upper():
                matches.append(path)
        if matches:
            matches.sort()
            results.append(matches[0])
        else:
            # 模糊匹配 fallback
            for path in all_candidates:
                basename = os.path.basename(path)
                if re.search(re.escape(diameter_str), basename, re.IGNORECASE) and \
                   model_prefix in basename.upper() and \
                   re.search(re.escape(defect), basename, re.IGNORECASE) and \
                   "HUSC" not in basename.upper():
                    results.append(path)
                    break
    return results


def find_default_segment_model(dataset_path, outer_diameter, default_model, default_defects):
    """保留原函数用于兼容，返回单个模型路径。"""
    fallbacks = find_fallback_models(dataset_path, outer_diameter, default_model)
    return fallbacks[0] if fallbacks else ""


def run_pipeline(csv_path, output_path, manhole_path, inner, outer, wall_thickness,
                 global_x_offset=-2.0, manhole_half_length=2.0, dataset_path="",
                 default_model="QKG", default_defects="FS1,PL1", patches_output_path=None,
                 total_segments=0):
    print(f"--- 管道拼接开始 ---")
    print(f"CSV: {csv_path}")
    print(f"输出: {output_path}")
    print(f"管道 内{inner}m / 外{outer}m")

    if not os.path.exists(manhole_path):
        print(f"[致命错误] 井室不存在: {manhole_path}")
        return

    inner_radius = inner / 2.0

    all_rows = []
    segment_length = 3.0
    max_mileage = 0.0

    if not os.path.exists(csv_path):
        print(f"[错误] CSV 不存在: {csv_path}")
        return

    id_counts = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            try:
                segment_length = float(row.get("管节长度", 3.0))
                row_dict = dict(row)
                row_dict['_原始编号'] = row.get("编号", "")
                base_id = normalize_defect_id(row.get("编号"), row_index)
                id_counts[base_id] = id_counts.get(base_id, 0) + 1
                row_dict['编号'] = base_id if id_counts[base_id] == 1 else f"{base_id}_{id_counts[base_id]:02d}"
                all_rows.append(row_dict)
                mil = float(row.get("节内里程", 0))
                if mil > max_mileage:
                    max_mileage = mil
            except (ValueError, TypeError):
                continue

    # 总管节数：由 steps.py 根据 CSV 里程数据计算传入
    if total_segments <= 0:
        # fallback: 根据 CSV 中最大绝对值里程估算
        total_segments = int(max_mileage / segment_length) + 1 if max_mileage > 0 else 1

    # min_seg_id fallback (not used for positioning - just for CSV writing)
    min_seg_id = min((int(float(row.get("管节序号", 0))) for row in all_rows), default=0)

    total_length = total_segments * segment_length - max(total_segments - 1, 0) * PIPE_JOINT_OVERLAP
    print(f"管道总长: {total_length:.2f}m ({total_segments} 节), 行数: {len(all_rows)}, 最大里程: {max_mileage:.2f}m")

    print(f"导出病害贴片行: {len(all_rows)}")

    clean_scene()
    pipe_collection = bpy.data.collections.new("Pipeline")
    bpy.context.scene.collection.children.link(pipe_collection)
    patch_collection = bpy.data.collections.new("DefectPatches")
    bpy.context.scene.collection.children.link(patch_collection)

    manhole_start_x = -0.6 - MANHOLE_OUTWARD_OFFSET
    place_manhole(
        filepath=manhole_path, name_prefix="Manhole_Start",
        location=(manhole_start_x, 0, 0), rotation_z=math.radians(90),
        pipe_collection=pipe_collection,
    )

    seg_model_map = {}
    seg_defects_map = {}
    for row in all_rows:
        mp = row.get("模型路径", "")
        # 根据绝对里程计算管节序号
        try:
            mil = float(row.get("节内里程", 0))
        except (ValueError, TypeError):
            mil = 0
        sid = int(mil / (segment_length - PIPE_JOINT_OVERLAP)) if mil > 0 else 0
        seg_defects_map.setdefault(sid, []).append(row)
        if mp and sid not in seg_model_map:
            seg_model_map[sid] = mp

    # 当某个管节没有病害记录时，从模型库按规则查找默认模型
    default_model_path = next(iter(seg_model_map.values()), "")
    fallback_models = find_fallback_models(dataset_path, outer, default_model)
    if fallback_models:
        print(f"[INFO] 默认模型已选定: {', '.join(os.path.basename(m) for m in fallback_models)}")
    elif dataset_path:
        # 回退到旧逻辑
        fb = find_default_segment_model(dataset_path, outer, default_model, default_defects)
        if fb:
            fallback_models = [fb]
            print(f"[INFO] 默认模型已选定: {os.path.basename(fb)}")
        else:
            print(f"[警告] 模型库中未找到匹配外径 {outer}m 的默认模型，将回退到已有模型")

    # 全管节生成：所有管节都导入模型（有病害的用匹配模型，无病害的用默认模型）
    fallback_idx = 0
    for seg_id in range(total_segments):
        model_path = seg_model_map.get(seg_id, "")
        if not model_path and fallback_models:
            model_path = fallback_models[fallback_idx % len(fallback_models)]
            fallback_idx += 1
        if not model_path and default_model_path:
            model_path = default_model_path
        if model_path:
            import_pipe_segment(model_path, seg_id, 0, segment_length, pipe_collection)
            if seg_id in seg_model_map:
                print(f"    管节 {seg_id}: 导入 {os.path.basename(str(model_path))} (匹配)")
            else:
                fb_type = os.path.basename(str(model_path)).split('_')[-1].replace('.glb', '') if '_' in os.path.basename(str(model_path)) else "默认"
                print(f"    管节 {seg_id}: 导入 {os.path.basename(str(model_path))} ({fb_type})")
        else:
            print(f"    管节 {seg_id}: 无匹配模型")

        for d in seg_defects_map.get(seg_id, []):
            create_highlight_patch(d, segment_length, inner_radius, patch_collection)

    manhole_b_x = total_length + 0.6 + MANHOLE_OUTWARD_OFFSET
    place_manhole(
        filepath=manhole_path, name_prefix="Manhole_End",
        location=(manhole_b_x, 0, 0), rotation_z=math.radians(-90),
        pipe_collection=pipe_collection,
    )

    print(f"--- 拼接完成 管长: {total_length:.2f}m 井室间距: {manhole_b_x - manhole_start_x:.2f}m (沿X轴) ---")
    print(f"导出管道模型: {output_path}")

    # 导出管道模型（管节 + 井室，不含贴片）
    bpy.ops.object.select_all(action='DESELECT')
    for obj in pipe_collection.all_objects:
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=output_path, export_format='GLB',
        use_selection=True, export_yup=True, export_apply=True,
    )
    print("[成功] 管道模型导出完成")

    # 导出病害贴片（如果存在且指定了输出路径）
    patch_objects = list(patch_collection.all_objects)
    if patch_objects and patches_output_path:
        print(f"导出病害贴片: {patches_output_path}")
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action='DESELECT')
        for obj in patch_objects:
            obj.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=patches_output_path, export_format='GLB',
            use_selection=True, export_yup=True, export_apply=True,
        )
        print(f"[成功] 病害贴片导出完成 ({len(patch_objects)} 个贴片)")
    elif not patch_objects:
        print("[信息] 无病害贴片，跳过贴片导出")

    # 将去重后的编号写回 CSV，确保 CSV 与 GLB 贴片编号一致
    try:
        if all_rows:
            try:
                fieldnames = list(all_rows[0].keys())
                if "_原始编号" in fieldnames:
                    fieldnames.remove("_原始编号")
                with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in all_rows:
                        out_row = {k: v for k, v in row.items() if k != "_原始编号"}
                        writer.writerow(out_row)
                print(f"[OK] 已更新 CSV 编号: {csv_path}")
            except Exception as e:
                print(f"[警告] 更新 CSV 编号失败: {e}")
    except Exception as e:
        print(f"[失败] 操作出错: {e}")


if __name__ == "__main__":
    try:
        argv = sys.argv
        if "--" in argv:
            args = argv[argv.index("--") + 1:]
        else:
            args = []

        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', required=True)
        parser.add_argument('--output', required=True)
        parser.add_argument('--manhole', required=True)
        parser.add_argument('--inner', type=float, required=True)
        parser.add_argument('--outer', type=float, required=True)
        parser.add_argument('--wall-thickness', type=float, default=0.1)
        parser.add_argument('--global-x-offset', type=float, default=-2.0)
        parser.add_argument('--manhole-half-length', type=float, default=2.0)
        parser.add_argument('--dataset', default="")
        parser.add_argument('--default-model', default="QKG")
        parser.add_argument('--default-defect', default="FS1,PL1")
        parser.add_argument('--patches-output', default=None)
        parser.add_argument('--total-segments', type=int, default=0)

        args = parser.parse_args(args)

        run_pipeline(
            args.csv, args.output, args.manhole,
            args.inner, args.outer, args.wall_thickness,
            args.global_x_offset, args.manhole_half_length,
            args.dataset,
            args.default_model,
            args.default_defect,
            args.patches_output,
            args.total_segments,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"错误: {e}")
        sys.exit(1)
