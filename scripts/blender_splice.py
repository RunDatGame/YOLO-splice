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


def create_highlight_patch(defect_data, segment_length, seg_offset, inner_radius, main_collection):
    seg_id = int(float(defect_data.get("管节序号", 1)))
    mileage = parse_float(defect_data.get("节内里程"), 0.0)
    mileage = max(0.0, min(mileage, segment_length - 0.02))
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

    physical_index = seg_id - seg_offset
    x_center = segment_origin(physical_index, segment_length) + mileage
    angle_rad = math.radians(axis_angle)
    radius = max(inner_radius - PATCH_SURFACE_CLEARANCE, inner_radius * 0.94)

    import random
    rng = random.Random(hash(defect_id) % (2**31))

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
    main_collection.objects.link(obj)

    color = DEFECT_COLORS.get(defect_type, DEFAULT_HIGHLIGHT_COLOR)
    mat = create_patch_material(defect_id, color)
    obj.data.materials.append(mat)
    return obj


def import_pipe_segment(model_path, seg_id, seg_offset, segment_length, main_collection):
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
    if container.name not in main_collection.objects:
        main_collection.objects.link(container)

    for obj in segment_objects:
        obj.parent = container
        for coll in list(obj.users_collection):
            if coll != main_collection:
                coll.objects.unlink(obj)
        if obj.name not in main_collection.objects:
            main_collection.objects.link(obj)

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


def place_manhole(filepath, name_prefix, location, rotation_z, main_collection):
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
            if coll != main_collection:
                coll.objects.unlink(obj)
        # 添加到 main_collection（子对象也需要在集合中才能被选中导出）
        if obj.name not in main_collection.objects:
            main_collection.objects.link(obj)

    # 只将容器加入 main_collection
    default_collection = bpy.context.scene.collection
    if container.name in default_collection.objects:
        default_collection.objects.unlink(container)
    if container.name not in main_collection.objects:
        main_collection.objects.link(container)

    # 绕 Z 轴额外旋转 -90°，使井室朝向 Y 轴方向
    container.rotation_euler = (0, 0, rotation_z - math.radians(90))
    container.location = location
    bpy.ops.object.select_all(action='DESELECT')


def find_default_segment_model(dataset_path, outer_diameter, default_model, default_defects):
    import re
    if not dataset_path or not os.path.isdir(dataset_path):
        return ""

    diameter_str = f"{float(outer_diameter):g}m"
    model_prefix = str(default_model).strip().upper()
    defects = [d.strip().upper() for d in str(default_defects).split(",") if d.strip()]
    if not defects:
        defects = ["FS1", "PL1"]

    all_candidates = []
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if not f.lower().endswith(".glb"):
                continue
            all_candidates.append(os.path.join(root, f))

    def _is_match(path, defect, allow_husc=False):
        basename = os.path.basename(path)
        if not re.search(re.escape(diameter_str), basename, re.IGNORECASE):
            return False
        if model_prefix not in basename.upper():
            return False
        if not re.search(re.escape(defect), basename, re.IGNORECASE):
            return False
        if not allow_husc and "HUSC" in basename.upper():
            return False
        return True

    # 第一轮：精确匹配 {diameter}{model}_{defect}.glb，排除 HUSC
    for defect in defects:
        exact_matches = []
        pattern = re.compile(
            rf"^{re.escape(diameter_str)}{re.escape(model_prefix)}_{re.escape(defect)}\.glb$",
            re.IGNORECASE,
        )
        for path in all_candidates:
            basename = os.path.basename(path)
            if pattern.match(basename) and "HUSC" not in basename.upper():
                exact_matches.append(path)
        if exact_matches:
            exact_matches.sort()
            return exact_matches[0]

    # 第二轮：模糊匹配，同管径同前缀且含 defect，排除 HUSC
    for defect in defects:
        fuzzy_matches = []
        for path in all_candidates:
            if _is_match(path, defect, allow_husc=False):
                fuzzy_matches.append(path)
        if fuzzy_matches:
            fuzzy_matches.sort()
            return fuzzy_matches[0]

    # 第三轮：允许 HUSC 作为最后 fallback
    for defect in defects:
        husc_matches = []
        for path in all_candidates:
            if _is_match(path, defect, allow_husc=True):
                husc_matches.append(path)
        if husc_matches:
            husc_matches.sort()
            return husc_matches[0]

    return ""


def run_pipeline(csv_path, output_path, manhole_path, inner, outer, wall_thickness,
                 global_x_offset=-2.0, manhole_half_length=2.0, dataset_path="",
                 default_model="QKG", default_defects="FS1,PL1"):
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
    max_seg_id = -1

    if not os.path.exists(csv_path):
        print(f"[错误] CSV 不存在: {csv_path}")
        return

    id_counts = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader, start=1):
            try:
                seg_id = int(float(row.get("管节序号", 0)))
                max_seg_id = max(max_seg_id, seg_id)
                segment_length = float(row.get("管节长度", 3.0))
                row_dict = dict(row)
                row_dict['_原始编号'] = row.get("编号", "")
                base_id = normalize_defect_id(row.get("编号"), row_index)
                id_counts[base_id] = id_counts.get(base_id, 0) + 1
                row_dict['编号'] = base_id if id_counts[base_id] == 1 else f"{base_id}_{id_counts[base_id]:02d}"
                all_rows.append(row_dict)
            except (ValueError, TypeError):
                continue

    min_seg_id = min((int(float(row.get("管节序号", 0))) for row in all_rows), default=0)

    max_global = 0.0
    for row in all_rows:
        sid = int(float(row.get("管节序号", 0)))
        try:
            mil = float(row.get("节内里程", 0))
        except (ValueError, TypeError):
            mil = 0
        physical_index = sid - min_seg_id
        pos = segment_origin(physical_index, segment_length) + mil
        if pos > max_global:
            max_global = pos

    total_segments = max_seg_id - min_seg_id + 1
    total_length = total_segments * segment_length - max(total_segments - 1, 0) * PIPE_JOINT_OVERLAP
    print(f"管道总长: {total_length}m ({total_segments} 节), 原始行: {len(all_rows)}, 最大里程: {max_global:.2f}m")

    valid_rows = []
    for row in all_rows:
        sid = int(float(row.get("管节序号", 0)))
        if min_seg_id <= sid <= max_seg_id:
            valid_rows.append(row)
    all_rows = valid_rows
    print(f"有效病害行: {len(all_rows)}")

    print(f"导出病害贴片行: {len(all_rows)}")

    clean_scene()
    main_collection = bpy.data.collections.new("PipelineAndManholes")
    bpy.context.scene.collection.children.link(main_collection)

    manhole_start_x = -0.6 - MANHOLE_OUTWARD_OFFSET
    place_manhole(
        filepath=manhole_path, name_prefix="Manhole_Start",
        location=(manhole_start_x, 0, 0), rotation_z=math.radians(90),
        main_collection=main_collection,
    )

    seg_model_map = {}
    seg_defects_map = {}
    for row in all_rows:
        sid = int(float(row.get("管节序号", 0)))
        seg_defects_map.setdefault(sid, []).append(row)
        mp = row.get("模型路径", "")
        if mp and sid not in seg_model_map:
            seg_model_map[sid] = mp

    # 当某个管节没有病害记录时，从模型库按规则查找默认模型
    default_model_path = next(iter(seg_model_map.values()), "")
    fallback_model_path = find_default_segment_model(dataset_path, outer, default_model, default_defects)
    if fallback_model_path:
        print(f"[INFO] 默认模型已选定: {os.path.basename(fallback_model_path)}")
    elif dataset_path:
        print(f"[警告] 模型库中未找到匹配外径 {outer}m 的默认模型，将回退到已有模型")

    for seg_id in range(min_seg_id, max_seg_id + 1):
        model_path = seg_model_map.get(seg_id, "")
        if not model_path and fallback_model_path:
            model_path = fallback_model_path
        if not model_path and default_model_path:
            model_path = default_model_path
        if model_path:
            import_pipe_segment(model_path, seg_id, min_seg_id, segment_length, main_collection)
            if seg_id in seg_model_map:
                print(f"    管节 {seg_id}: 导入 {os.path.basename(str(model_path))}")
            else:
                print(f"    管节 {seg_id}: 导入默认模型 {os.path.basename(str(model_path))}")
        else:
            print(f"    管节 {seg_id}: 无匹配模型")

        for d in seg_defects_map.get(seg_id, []):
            create_highlight_patch(d, segment_length, min_seg_id, inner_radius, main_collection)

    manhole_b_x = total_length + 0.6 + MANHOLE_OUTWARD_OFFSET
    place_manhole(
        filepath=manhole_path, name_prefix="Manhole_End",
        location=(manhole_b_x, 0, 0), rotation_z=math.radians(-90),
        main_collection=main_collection,
    )

    print(f"--- 拼接完成 管长: {total_length:.2f}m 井室间距: {manhole_b_x - manhole_start_x:.2f}m (沿X轴) ---")
    print(f"导出: {output_path}")
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.gltf(
            filepath=output_path, export_format='GLB',
            use_selection=True, export_yup=True, export_apply=True,
        )
        print("[成功] 导出完成")

        # 将去重后的编号写回 CSV，确保 CSV 与 GLB 贴片编号一致
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
        print(f"[失败] 导出出错: {e}")


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

        args = parser.parse_args(args)

        run_pipeline(
            args.csv, args.output, args.manhole,
            args.inner, args.outer, args.wall_thickness,
            args.global_x_offset, args.manhole_half_length,
            args.dataset,
            args.default_model,
            args.default_defect,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"错误: {e}")
        sys.exit(1)
