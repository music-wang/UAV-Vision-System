#!/usr/bin/env python3
"""把赛题附图切成可用的视觉素材。

输入：赛题附件《动物体态》整版图（透明底手绘动物，5 行 × 3 列，共 15 只）
输出：
  1. assets/templates/<species>/<species>_<n>.png   —— 去底、带 alpha、长边 <= 900px
  2. assets/terrain_preview.jpg                      —— 地貌图缩略图（文档/自检用）

切分方式不按等分网格，而是按 alpha 投影找"行带 → 每行的列分组"，
这样动物再高、尾巴再翘也不会被网格线切断（等分网格会把象背、虎尾切掉）。

注意：赛区实际打印的场地 = 地貌图 + 四周 25cm 边距 + 50cm 灰色网格 + 两条 1.8cm 黑线 + 动物贴纸。
网格/黑线/贴纸在原图里都没有，训练集必须自己合成，否则模型只见过白底插图。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# 行优先的类别表，按原图人工核对（第 1~2 行是孔雀与狼混排，不是按行分块）
LAYOUT = [
    ["peacock", "peacock", "wolf"],
    ["peacock", "wolf", "wolf"],
    ["monkey", "monkey", "monkey"],
    ["elephant", "elephant", "elephant"],
    ["tiger", "tiger", "tiger"],
]

ALPHA_MIN = 8      # 判为前景的最小 alpha
MIN_BAND = 50      # 行带最小高度
MIN_GROUP = 50     # 列分组最小宽度
PAD = 6            # 裁剪外扩像素
MAX_SIDE = 900     # 输出长边上限
MARKER = (255, 0, 255)


def rows_and_groups(mask: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    """返回 [[(x0, y0, x1, y1) × 3] × 5]，即每只动物的精确包围盒。"""
    rows = mask.any(axis=1)
    bands: list[tuple[int, int]] = []
    y = 0
    while y < len(rows):
        if rows[y]:
            y0 = y
            while y < len(rows) and rows[y]:
                y += 1
            if y - y0 >= MIN_BAND:
                bands.append((y0, y - 1))
        else:
            y += 1

    out = []
    for y0, y1 in bands:
        cols = mask[y0 : y1 + 1].any(axis=0)
        groups: list[tuple[int, int]] = []
        x = 0
        while x < len(cols):
            if cols[x]:
                x0 = x
                while x < len(cols) and cols[x]:
                    x += 1
                if x - x0 >= MIN_GROUP:
                    groups.append((x0, x - 1))
            else:
                x += 1
        out.append([(g0, y0, g1, y1) for g0, g1 in groups])
    return out


def cut(rgb_cell: Image.Image) -> Image.Image:
    """从包围盒里抠出动物：白底连通域变透明，动物内部的白（眼圈/牙齿）保留。"""
    w, h = rgb_cell.size
    img = rgb_cell.copy()
    d = ImageDraw.Draw(img)
    for seed in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if img.getpixel(seed) != MARKER:
            ImageDraw.floodfill(img, seed, MARKER, thresh=12)

    arr = np.asarray(img)
    keep = (arr != np.array(MARKER, dtype=arr.dtype)).any(axis=2)
    # 白底与动物之间有 1px 抗锯齿过渡带，会在地图上合成时留一圈白边；腐蚀 1px 去掉
    keep = keep & np.roll(keep, 1, 0) & np.roll(keep, -1, 0) \
                & np.roll(keep, 1, 1) & np.roll(keep, -1, 1)
    ys, xs = np.nonzero(keep)
    if len(xs) == 0:
        raise RuntimeError("空单元格：抠图后无前景")
    x0, x1 = max(xs.min() - PAD, 0), min(xs.max() + PAD + 1, w)
    y0, y1 = max(ys.min() - PAD, 0), min(ys.max() + PAD + 1, h)

    out = Image.new("RGBA", (x1 - x0, y1 - y0), (255, 255, 255, 0))
    out.paste(rgb_cell.crop((x0, y0, x1, y1)), (0, 0))
    alpha = Image.fromarray((keep[y0:y1, x0:x1] * 255).astype(np.uint8), "L")
    out.putalpha(alpha)

    if max(out.size) > MAX_SIDE:
        s = MAX_SIDE / max(out.size)
        size = (max(round(out.width * s), 1), max(round(out.height * s), 1))
        out = out.resize(size, Image.LANCZOS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True, help="动物体态 整版图路径")
    ap.add_argument("--terrain", help="地貌图路径（可选，生成缩略图）")
    ap.add_argument("--out", required=True, help="输出 assets 目录")
    ap.add_argument("--terrain-width", type=int, default=1600)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "templates").mkdir(parents=True, exist_ok=True)

    sheet = Image.open(args.sheet).convert("RGBA")
    mask = np.asarray(sheet.getchannel("A")) > ALPHA_MIN
    boxes = rows_and_groups(mask)
    print(f"sheet {sheet.width}x{sheet.height} -> {len(boxes)} 行带")
    if len(boxes) != len(LAYOUT):
        raise SystemExit(f"行带数 {len(boxes)} != 期望 {len(LAYOUT)}，请检查 ALPHA_MIN/MIN_BAND")

    for r, (row_boxes, row_cls) in enumerate(zip(boxes, LAYOUT)):
        print(f"  row{r+1}: {len(row_boxes)} 只")
        if len(row_boxes) != len(row_cls):
            raise SystemExit(f"第 {r+1} 行检测到 {len(row_boxes)} 只，期望 {len(row_cls)}")
        for (x0, y0, x1, y1), species in zip(row_boxes, row_cls):
            cell = sheet.crop((x0, y0, x1 + 1, y1 + 1))
            # 先合到白底，统一用"白底连通域"抠图
            on_white = Image.new("RGB", cell.size, (255, 255, 255))
            on_white.paste(cell, (0, 0), cell)
            tpl = cut(on_white)

            d = out / "templates" / species
            d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("*.png"))) + 1
            path = d / f"{species}_{n}.png"
            tpl.save(path, optimize=True)
            print(f"    -> {path.relative_to(out)}  {tpl.width}x{tpl.height}  "
                  f"{path.stat().st_size / 1024:.0f} KB")

    if args.terrain:
        t = Image.open(args.terrain).convert("RGB")
        w = args.terrain_width
        h = round(t.height * w / t.width)
        t.resize((w, h), Image.LANCZOS).save(out / "terrain_preview.jpg", quality=88, optimize=True)
        print(f"terrain_preview.jpg {w}x{h} "
              f"{(out / 'terrain_preview.jpg').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
