# src/pipeline_splice/modeling/matcher.py
import os
from pathlib import Path
import pandas as pd
import re
from functools import lru_cache


# ========== 中文 -> 英文 对照表 ==========
TYPE_MAP = {
    "破裂": "PL",
    "腐蚀": "FS",
    "错口": "CK",
    # 可继续补充
}


def normalize_diameter(d):
    """ 保证管径 '0.3m' 格式并去除末尾零 """
    if not d:
        return ""
    d = str(d).strip().lower()
    if d.endswith("m"):
        d = d[:-1]
    try:
        val = float(d)
        return f"{val:g}m"
    except ValueError:
        return d + "m" if d else ""


def find_matching_glb(diameter, defect_type, severity, model, all_glb):
    """
    正常匹配逻辑（除错口外）
    """
    diameter_str = diameter.lower()
    severity_str = (defect_type + severity).lower()

    pattern_str = rf"{diameter_str}.*{model.lower()}.*{severity_str}.*\.glb"
    pattern = re.compile(pattern_str, re.IGNORECASE)

    for f in all_glb:
        if pattern.search(f["name"]):
            return f["path"]
    return ""


FILENAME_INDEX_PATTERN = re.compile(
    r"^(?P<diameter>\d+(?:\.\d+)?m)(?P<model>[a-z]+).*?_(?P<code>(?:pl|fs|ck)\d)_",
    re.IGNORECASE,
)


@lru_cache(maxsize=8)
def load_glb_catalog(dataset_dir):
    all_glb = []
    direct_index = {}

    for root, _, files in os.walk(dataset_dir):
        for filename in files:
            if not filename.lower().endswith(".glb"):
                continue

            full_path = os.path.join(root, filename)
            entry = {
                "name": filename,
                "name_lc": filename.lower(),
                "path": full_path,
            }
            all_glb.append(entry)

            match = FILENAME_INDEX_PATTERN.search(filename)
            if not match:
                continue

            key = (
                match.group("diameter").lower(),
                match.group("model").lower(),
                match.group("code").lower(),
            )
            direct_index.setdefault(key, full_path)

    return all_glb, direct_index


# ========== 这是最终暴露给外部使用的函数 ==========
def process_csv(input_csv, dataset_dir, output_csv, default_model="QKG", skip_ck=True, one_per_segment=True):
    df = pd.read_csv(input_csv, dtype=str).fillna("")

    all_glb, direct_index = load_glb_catalog(os.path.abspath(dataset_dir))

    new_paths = []

    for idx, row in df.iterrows():
        raw_type = row.get("模型类型", "")
        raw_severity = row.get("严重等级", "")
        raw_diameter = row.get("管节外径", "")

        defect_code = TYPE_MAP.get(raw_type, raw_type)
        d_match = normalize_diameter(raw_diameter)

        if skip_ck and defect_code.upper() == "CK":
            new_paths.append("")
            continue

        index_key = (d_match.lower(), default_model.lower(), (defect_code + raw_severity).lower())
        matched = direct_index.get(index_key)
        if not matched:
            matched = find_matching_glb(
                d_match, defect_code, raw_severity, default_model, all_glb
            )
        new_paths.append(matched)

    df["模型路径"] = new_paths

    if one_per_segment and "管节序号" in df.columns:
        for seg_id, group in df.groupby("管节序号"):
            non_empty_idx = group[group["模型路径"] != ""].index.tolist()
            if len(non_empty_idx) > 1:
                for idx_to_clear in non_empty_idx[1:]:
                    df.at[idx_to_clear, "模型路径"] = ""
    elif one_per_segment:
        print("警告：没有管节序号字段")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("CSV 处理完成：", output_csv)
    return output_csv
