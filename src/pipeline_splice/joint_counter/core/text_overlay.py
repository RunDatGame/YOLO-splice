"""OpenCV 帧上的中文文字绘制（cv2.putText 不支持 CJK）。"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font_candidates() -> list[Path]:
  roots: list[Path] = []
  if sys.platform == "win32":
    roots.append(Path(r"C:\Windows\Fonts"))
  elif sys.platform == "darwin":
    roots.extend(
        Path(p)
        for p in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        )
    )
  else:
    roots.extend(
        Path(p)
        for p in (
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        )
    )

  names = (
      "msyh.ttc",
      "msyhbd.ttc",
      "simhei.ttf",
      "simsun.ttc",
      "simkai.ttf",
      "NotoSansCJK-Regular.ttc",
      "NotoSansSC-Regular.otf",
  )
  found: list[Path] = []
  for root in roots:
    if root.is_file():
      found.append(root)
      continue
    if not root.is_dir():
      continue
    for name in names:
      p = root / name
      if p.exists():
        found.append(p)
  assets = Path(__file__).resolve().parent.parent / "assets" / "fonts"
  for name in names:
    p = assets / name
    if p.exists():
      found.append(p)
  return found


@lru_cache(maxsize=4)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
  for path in _font_candidates():
    try:
      return ImageFont.truetype(str(path), size=size)
    except OSError:
      continue
  return ImageFont.load_default()


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
  b, g, r = color
  return (r, g, b)


def put_text(
    image_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_size: int,
    color_bgr: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_color_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
  """在 BGR 图像上绘制 Unicode 文字，返回同一数组（就地修改）。"""
  if not text:
    return image_bgr

  x, y = org
  font = _load_font(font_size)
  pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
  draw = ImageDraw.Draw(pil)
  fill = _bgr_to_rgb(color_bgr)
  stroke_fill = _bgr_to_rgb(stroke_color_bgr) if stroke_color_bgr else None
  draw.text(
      (x, y),
      text,
      font=font,
      fill=fill,
      anchor="lb",
      stroke_width=stroke_width,
      stroke_fill=stroke_fill,
  )
  image_bgr[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
  return image_bgr


def text_size(text: str, font_size: int) -> tuple[int, int]:
  font = _load_font(font_size)
  if not text:
    return 0, 0
  bbox = font.getbbox(text)
  return bbox[2] - bbox[0], bbox[3] - bbox[1]
