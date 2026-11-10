#!/usr/bin/env python3
"""按赛题要求的**真实尺寸**生成动物靶标打印页，用于实拍验证。

赛题说明 1(4)：象长 30cm，象:虎:狼:猴:孔雀 = 2:1:1:1:1（其余约 15cm），
彩色印刷、沿轮廓剪下贴到地图上。验证用的靶标必须按这个尺寸打，不能随手缩放。

用法：
  python 01_vision/scripts/make_print_sheet.py --paper A2          # 一张纸放得下 5 只
  python 01_vision/scripts/make_print_sheet.py --paper A3          # 自动分页（象占一页）
  python 01_vision/scripts/make_print_sheet.py --paper A3 --per-species 3

输出（runs/print_sheet/）：
  targets_A3_300dpi_p1.png ...   按 100% 比例打印，不要选"适应页面"
  README.txt                     打印/裁剪/拍摄说明
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[2]
SPECIES = ["peacock", "wolf", "monkey", "elephant", "tiger"]
SIZE_CM = {"elephant": 30.0}          # 其余按 15cm
DEFAULT_CM = 15.0
PAPERS = {"A4": (21.0, 29.7), "A3": (42.0, 29.7), "A2": (59.4, 42.0)}


def cm2px(cm: float, dpi: int) -> int:
    return round(cm / 2.54 * dpi)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default="A2", choices=sorted(PAPERS))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--per-species", type=int, default=1)
    ap.add_argument("--template-dir", default=str(HERE / "01_vision/assets/templates"))
    ap.add_argument("--out", default=str(HERE / "runs/print_sheet"))
    ap.add_argument("--margin-cm", type=float, default=1.2)
    ap.add_argument("--gap-cm", type=float, default=0.8)
    args = ap.parse_args()

    pw, ph = PAPERS[args.paper]
    W, H = cm2px(pw, args.dpi), cm2px(ph, args.dpi)
    margin, gap = cm2px(args.margin_cm, args.dpi), cm2px(args.gap_cm, args.dpi)
    usable_w, usable_h = W - 2 * margin, H - 2 * margin - cm2px(1.4, args.dpi)  # 底部留标尺

    f_label = font(round(cm2px(0.45, args.dpi)))

    # 收集靶标：优先用"侧视"模板（宽 >= 高），因为赛题按"长度"定尺寸
    items = []
    for sp in SPECIES:
        files = sorted((Path(args.template_dir) / sp).glob("*.png"))
        if not files:
            raise SystemExit(f"缺少模板：{args.template_dir / sp}")
        side = [f for f in files if Image.open(f).width >= Image.open(f).height] or files
        for k in range(min(args.per_species, len(side))):
            items.append((sp, side[k]))
    items.sort(key=lambda it: -SIZE_CM.get(it[0], DEFAULT_CM))

    pages: list[Image.Image] = []
    page = None
    x = y = row_h = 0

    for sp, f in items:
        size_cm = SIZE_CM.get(sp, DEFAULT_CM)
        img = Image.open(f).convert("RGBA")
        target = cm2px(size_cm, args.dpi)
        s = target / max(img.size)
        img = img.resize((max(round(img.width * s), 1), max(round(img.height * s), 1)),
                         Image.LANCZOS)

        pad = max(round(cm2px(0.15, args.dpi)), 2)     # 剪下时留的白边
        if img.height + 2 * pad > usable_h:            # 竖版模板按高度缩到能放下
            k = (usable_h - 2 * pad) / img.height
            img = img.resize((max(round(img.width * k), 1), usable_h - 2 * pad), Image.LANCZOS)
        cell_w, cell_h = img.width + 2 * pad, img.height + 2 * pad

        if page is not None and x + cell_w > margin + usable_w:   # 换行
            x = margin
            y += row_h + gap
            row_h = 0
        if page is None or y + cell_h > margin + usable_h:        # 换页
            if page is not None:
                pages.append(page)
                print(f"  - 第 {len(pages)} 页已满，开新页")
            page = Image.new("RGB", (W, H), (255, 255, 255))
            x, y, row_h = margin, margin, 0

        d = ImageDraw.Draw(page)
        d.rectangle([x, y, x + cell_w, y + cell_h], fill=(255, 255, 255),
                    outline=(150, 150, 150), width=2)
        page.paste(img, (x + pad, y + pad), img)
        d.text((x + pad + 4, y + pad + 4), f"{sp} {size_cm:.0f}cm",
               fill=(140, 140, 140), font=f_label)

        x += cell_w + gap
        row_h = max(row_h, cell_h)
    if page is not None:
        pages.append(page)

    # 每页都画 10cm 标尺，打印被缩放时一眼能看出来
    rl = cm2px(10, args.dpi)
    for i, pg in enumerate(pages, 1):
        d = ImageDraw.Draw(pg)
        ry = H - margin - cm2px(0.7, args.dpi)
        d.rectangle([margin, ry, margin + rl, ry + cm2px(0.2, args.dpi)], fill=(0, 0, 0))
        for k in range(11):
            xx = margin + k * rl // 10
            d.line([(xx, ry - cm2px(0.18, args.dpi)), (xx, ry)], fill=(0, 0, 0), width=2)
        d.text((margin + rl + cm2px(0.4, args.dpi), ry - cm2px(0.3, args.dpi)),
               "10cm ruler - measure after printing", fill=(0, 0, 0), font=f_label)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, pg in enumerate(pages, 1):
        p = out / f"targets_{args.paper}_{args.dpi}dpi_p{i}.png"
        pg.save(p, dpi=(args.dpi, args.dpi))
        print(f"[输出] {p}  {W}x{H}px")

    (out / "README.txt").write_text(
        f"打印说明\n"
        f"1) 纸张 {args.paper}，每页 {len(items)} 个靶标分 {len(pages)} 页\n"
        f"2) 打印比例必须选【实际大小 / 100%】，不要勾\"适应页面\"\n"
        f"3) 打完用尺子量页面底部 10cm 标尺，必须是 10.0cm，否则重打\n"
        f"4) 沿灰框剪下，保留 1~2mm 白边（真实赛题也是纸贴纸）\n"
        f"5) 贴到地面/喷绘布上，相机在约 1.2m 高度拍几张（含俯视和斜视）\n"
        f"6) 跑：python 01_vision/scripts/verify_real.py --images <照片目录> --labels <标注.csv>\n"
        f"\n尺寸：象 30cm，其余 15cm（赛题 2:1:1:1:1）\n",
        encoding="utf-8")
    print(f"[说明] {out / 'README.txt'}")


if __name__ == "__main__":
    main()

