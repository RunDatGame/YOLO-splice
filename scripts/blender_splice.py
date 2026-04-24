import bpy
import math
import csv
import os
import sys
import argparse
import bmesh


# ==========================================
# 辅助函数：清理场景
# ==========================================
def clean_scene():
    """清理默认场景中的所有物体"""
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 清理未使用的无主数据块
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


# ==========================================
# 新增函数：放置井室模型 (已修复集合链接问题)
# ==========================================
def place_manhole(filepath, name_prefix, location, rotation_z, main_collection):
    """
    导入、旋转并放置井室模型。
    - 井室不需要绕 Y 轴旋转 90 度，以保持井盖朝向 Z 轴（向上）。
    - 仅应用 Z 轴旋转 (rotation_z) 来调整井口朝向。
    """
    print(f"放置井室 {name_prefix}: 位置={location}, 旋转Z={math.degrees(rotation_z):.2f}°")

    try:
        bpy.ops.import_scene.gltf(filepath=filepath)
    except Exception as e:
        print(f"[错误] 导入井室模型失败 {filepath}: {e}")
        return

    imported_objects = bpy.context.selected_objects

    if not imported_objects:
        print(f"[警告] {filepath} 导入后没有发现物体")
        return

    # 创建父节点进行统一变换
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    container = bpy.context.active_object
    container.name = name_prefix + "_Root"

    # 将导入的物体设为 container 的子级并链接到集合 (保持修复后的链接逻辑)
    for obj in imported_objects:
        obj.parent = container

        for coll in list(obj.users_collection):
            if coll != main_collection:
                coll.objects.unlink(obj)

        if obj.name not in main_collection.objects:
            main_collection.objects.link(obj)

    # 将容器也链接到集合 (保持修复后的链接逻辑)
    try:
        default_collection = bpy.context.scene.collection
        if container.name in default_collection.objects:
            default_collection.objects.unlink(container)

        if container.name not in main_collection.objects:
            main_collection.objects.link(container)
    except Exception as e:
        print(f"[警告] 链接容器 {container.name} 到集合失败: {e}")

    # --- 应用变换 (关键修改点) ---
    # 1. 旋转: 移除 (0, math.radians(90), 0)
    #    只应用 Z 轴旋转 (rotation_z)
    container.rotation_euler = (0, 0, rotation_z)  # <--- 修正后的旋转

    # 2. 位移
    container.location = location

    bpy.ops.object.select_all(action='DESELECT')


# ==========================================
# 核心逻辑 (包含井室) (已实现管道平移)
# ==========================================
def run_pipeline(csv_path, output_path, manhole_path):
    print(f"--- 开始处理 ---")
    print(f"CSV文件: {csv_path}")
    print(f"输出路径: {output_path}")
    print(f"井室路径: {manhole_path}")

    if not os.path.exists(manhole_path):
        print(f"[致命错误] 井室模型文件不存在: {manhole_path}")
        return

    # *** 新的配置：全局原点偏移 ***
    # 假设井室连接圆孔在模型自身 X=+2.0m 处。
    # 要将该点设为 X=0，所有物体都需要整体向左平移 2.0m。
    GLOBAL_X_OFFSET = -2.0

    # 井室的半长，用于计算结束井室的位置。
    MANHOLE_HALF_LENGTH = 2.0

    print(f"[配置] 全局原点偏移 (X轴): {GLOBAL_X_OFFSET:.2f} 米")

    # 1. 预处理 CSV 数据 (保持不变)
    segments = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in reader.fieldnames]
            reader.fieldnames = headers

            for row in reader:
                try:
                    seg_id = row.get('管节序号')
                    length = float(row.get('管节长度', 0))
                    path = row.get('模型路径', '')

                    if path and os.path.exists(path):
                        segments.append({
                            'id': seg_id,
                            'length': length,
                            'path': path
                        })
                except ValueError:
                    continue
    except Exception as e:
        print(f"[错误] 读取 CSV 失败: {e}")
        return

    # 2. 清理 Blender 场景
    clean_scene()

    accumulated_length = 0.0

    # 创建一个总的集合来存放所有物体
    main_collection = bpy.data.collections.new("PipelineAndManholes")
    bpy.context.scene.collection.children.link(main_collection)

    # ==========================================
    # 3. 放置起始井室 (井室 A)
    # ==========================================
    # 井室原中心点在 X=0，平移后，新位置在 X = -2.0
    manhole_a_location = (GLOBAL_X_OFFSET, 0, 0)
    manhole_a_rotation_z = 0

    place_manhole(
        filepath=manhole_path,
        name_prefix="Manhole_Start",
        location=manhole_a_location,
        rotation_z=manhole_a_rotation_z,
        main_collection=main_collection
    )

    # ==========================================
    # 4. 循环导入并放置管道
    # ==========================================
    for seg in segments:
        model_path = seg['path']
        length = seg['length']
        seg_id = seg['id']

        # X位置计算： = 累计长度 + 管节中心 (没有额外的起始偏移，因为起始井室已经平移)
        # 管道的起点是 X=0，所以位置从 0 开始累计
        x_pos = accumulated_length + (length / 2.0)

        print(f"处理管节 {seg_id}: X位置={x_pos:.2f}, 长度={length:.2f}")

        # --- 导入 GLB ---
        try:
            bpy.ops.import_scene.gltf(filepath=model_path)
        except Exception as e:
            print(f"[错误] 导入失败 {model_path}: {e}")
            continue

        imported_objects = bpy.context.selected_objects

        if not imported_objects:
            print(f"[警告] {model_path} 导入后没有发现物体")
            continue

        # --- 创建父节点进行统一变换 ---
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        container = bpy.context.active_object
        container.name = f"Segment_{seg_id}_Root"

        # 关联父子级和集合 (保持修复后的链接逻辑)
        for obj in imported_objects:
            obj.parent = container

            for coll in list(obj.users_collection):
                if coll != main_collection:
                    coll.objects.unlink(obj)

            if obj.name not in main_collection.objects:
                main_collection.objects.link(obj)

        try:
            default_collection = bpy.context.scene.collection
            if container.name in default_collection.objects:
                default_collection.objects.unlink(container)

            if container.name not in main_collection.objects:
                main_collection.objects.link(container)
        except Exception as e:
            print(f"[警告] 链接容器 {container.name} 到集合失败: {e}")

        # --- 应用变换 ---
        container.rotation_euler = (0, math.radians(90), 0)
        container.location = (x_pos, 0, 0)

        accumulated_length += length
        bpy.ops.object.select_all(action='DESELECT')

    # ==========================================
    # 5. 放置结束井室 (井室 B)
    # ==========================================
    # 管道末端位于 accumulated_length 处。
    # 结束井室的中心应该位于：管道末端 + 井室半长。
    manhole_b_location = (accumulated_length + MANHOLE_HALF_LENGTH, 0, 0)

    # 结束井室的接口（X轴负方向）需要朝向管道
    manhole_b_rotation_z = math.radians(180)

    place_manhole(
        filepath=manhole_path,
        name_prefix="Manhole_End",
        location=manhole_b_location,
        rotation_z=manhole_b_rotation_z,
        main_collection=main_collection
    )

    print(f"--- 拼接完成，管道总长度: {accumulated_length:.2f} 米 ---")
    print(f"--- 最终模型总长度 (含井室): {manhole_b_location[0] + MANHOLE_HALF_LENGTH:.2f} 米 ---")

    # 6. 导出结果 (保持不变)
    print(f"正在导出至: {output_path}")
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=True,
            export_yup=True,
            export_apply=True
        )
        print("[成功] 导出完成")
    except Exception as e:
        print(f"[失败] 导出出错: {e}")

# ==========================================
# 参数解析与入口
# ==========================================
if __name__ == "__main__":
    try:
        argv = sys.argv
        if "--" in argv:
            args = argv[argv.index("--") + 1:]
        else:
            args = []

        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', required=True, help='CSV文件路径')
        parser.add_argument('--output', required=True, help='输出GLB路径')
        parser.add_argument('--manhole', required=True, help='井室GLB模型文件路径')

        args = parser.parse_args(args)

        run_pipeline(args.csv, args.output, args.manhole)

    except Exception as e:
        # 打印详细错误信息有助于调试
        import traceback

        traceback.print_exc()
        print(f"脚本运行错误: {e}")
        sys.exit(1)
