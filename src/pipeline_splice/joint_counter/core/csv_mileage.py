"""从 CSV 自动识别总里程列并读取管道总里程。"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# 优先检查的列：第 3 列、第 6 列（0-based 索引 2、5）
MILEAGE_COLUMN_PRIORITY = (2, 5)


@dataclass
class CsvMileageInfo:
  total_mileage_m: float
  mileage_column_index: int
  source_row: int = 0


def _parse_float(value: str) -> float:
  text = value.strip().replace(",", "")
  match = re.search(r"[-+]?\d*\.?\d+", text)
  if not match:
    raise ValueError(f"无法解析数值: {value!r}")
  return float(match.group())


def _read_csv_rows(csv_path: Path) -> list[list[str]]:
  rows: list[list[str]] = []
  for encoding in ("utf-8-sig", "utf-8", "gbk"):
    try:
      with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row and any(cell.strip() for cell in row)]
      break
    except UnicodeDecodeError:
      rows = []
  if not rows:
    raise ValueError(f"CSV 为空或无法读取: {csv_path}")
  return rows


def _try_read_column(rows: list[list[str]], col_idx: int) -> list[float] | None:
  values: list[float] = []
  for row in rows:
    if col_idx >= len(row):
      return None
    cell = row[col_idx].strip()
    if not cell:
      return None
    try:
      values.append(_parse_float(cell))
    except ValueError:
      return None
  return values if values else None


def _column_numeric_series(rows: list[list[str]], col_idx: int) -> list[float] | None:
  """读取整列数值；若首行非数值（表头）则自动跳过首行再试。"""
  for start in (0, 1):
    if start >= len(rows):
      continue
    values = _try_read_column(rows[start:], col_idx)
    if values is not None:
      return values
  return None


def _is_strict_increasing_mileage_column(values: list[float]) -> bool:
  """里程列：全程严格递增、非负，末值>0 且为最大。"""
  if len(values) < 2:
    return False
  if any(v < 0 for v in values):
    return False
  if values[-1] <= 0:
    return False
  return all(values[i] < values[i + 1] for i in range(len(values) - 1))


def find_mileage_column_index(rows: list[list[str]]) -> int:
  """
  识别总里程列：从头至尾严格递增且均为正数。
  优先第 3 列、第 6 列，否则扫描其余列。
  """
  if not rows:
    raise ValueError("CSV 无有效数据行")

  max_cols = max(len(row) for row in rows)
  others = [j for j in range(max_cols) if j not in MILEAGE_COLUMN_PRIORITY]
  search_order = list(MILEAGE_COLUMN_PRIORITY) + others

  for col_idx in search_order:
    if col_idx >= max_cols:
      continue
    series = _column_numeric_series(rows, col_idx)
    if series is not None and _is_strict_increasing_mileage_column(series):
      return col_idx

  raise ValueError(
      "未在 CSV 中找到符合规则的里程列（需全程严格递增且均为正数；"
      "已优先检查第 3 列与第 6 列）"
  )


def read_mileage_from_csv(csv_path: str | Path) -> CsvMileageInfo:
  path = Path(csv_path)
  if not path.exists():
    raise FileNotFoundError(f"CSV 不存在: {path}")

  rows = _read_csv_rows(path)
  col_idx = find_mileage_column_index(rows)
  series = _column_numeric_series(rows, col_idx)
  assert series is not None

  return CsvMileageInfo(
      total_mileage_m=series[-1],
      mileage_column_index=col_idx,
      source_row=len(rows),
  )


def find_best_matching_csv(video_path: str | Path) -> Path | None:
  """在同目录下查找与视频文件名最接近的 CSV。"""
  video_path = Path(video_path)
  directory = video_path.parent
  if not directory.is_dir():
    return None

  exact = video_path.with_suffix(".csv")
  if exact.is_file():
    return exact

  candidates = sorted(directory.glob("*.csv"))
  if not candidates:
    return None

  video_stem = video_path.stem.lower()
  best = max(
      candidates,
      key=lambda p: SequenceMatcher(None, video_stem, p.stem.lower()).ratio(),
  )
  score = SequenceMatcher(None, video_stem, best.stem.lower()).ratio()
  return best if score >= 0.25 else None
