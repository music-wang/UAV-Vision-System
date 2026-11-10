#!/usr/bin/env python3
"""合成 2025H 野生动物巡查系统的训练集。

为什么必须合成：
  赛区打印的场地 = 地貌图 + 四周 25cm 边距 + 50cm 灰色网格 + 两条 1.8cm 黑线
                   + 3 格灰色禁飞布 + 动物纸质贴纸（沿轮廓剪下）
  而赛题附件里的「地貌图」只有地貌，网格/黑线/禁飞布/动物全是打印时才加的。
  所以拿原图或白底插图训练，模型学到的分布和实拍场地完全不是一回事。

本脚本做三件事：
  1. 渲染一张「虚拟打印场地」（5000×4000 px，1px = 1mm）：
     地貌图 + 网格 + 黑线 + 禁飞布 + 随机动物贴纸
  2. 按机载相机几何（120cm 高、FOV 66°）从场地上采样视角，
     叠加机身偏航旋转、高度抖动、光照不均、色偏、模糊、噪声
  3. 输出 YOLO 格式 images/labels + manifest.json（含每只动物的格号/类别/真实坐标）

上游依赖：01_vision/assets/templates（由 split_templates.py 生成）
          datasets/raw/地貌图.png（原始附件，不入库）

用法：
  python 01_vision/scripts/synth_dataset.py --n-train 800 --n-val 100
  python 01_vision/scripts/synth_dataset.py --n-train 20 --n-val 5 --debug-overlay
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# ---------------- 场地参数（单位 cm，1px = 1mm） ----------------
MAT_W_CM, MAT_H_CM = 500, 400      # 喷绘布整幅
MARGIN_CM = 25                     # 四周留白
GRID_CM = 50                       # 网格间距
NX, NY = 9, 7                      # A1..A9 × B1..B7
BLACK_LINE_CM = 1.8                # 标志线宽
NOFLY_CELLS = 3                    # 禁飞区 = 3 个连续格
NOFLY_RGB = (180, 180, 180)        # 赛题给定的灰色

SPECIES = ["peacock", "wolf", "monkey", "elephant", "tiger"]  # 与 classes.txt 同序
NOMINAL_LEN_CM = {"elephant": 30.0}  # 其余按 1/2 处理（赛题 2:1:1:1:1）
DEFAULT_LEN_CM = 15.0


def cell_name(i: int, j: int) -> str:
    return f"A{i + 1}B{j + 1}"


def cell_rect(i: int, j: int) -> tuple[float, float, float, float]:
    """方格在场地坐标系（cm）中的矩形。"""
    x0 = MARGIN_CM + i * GRID_CM
    y0 = MARGIN_CM + j * GRID_CM
    return x0, y0, x0 + GRID_CM, y0 + GRID_CM


# ---------------- 场地渲染 ----------------


def render_mat(terrain_path: Path, px_per_cm: float) -> Image.Image:
    w, h = round(MAT_W_CM * px_per_cm), round(MAT_H_CM * px_per_cm)
    terrain = Image.open(terrain_path).convert("RGB").resize((w, h), Image.LANCZOS)
    return terrain


def draw_grid(img: Image.Image, px_per_cm: float) -> None:
    """50cm 灰色虚线，线宽 0.05cm（1mm 精度下就是 1px，用半透明压细）。"""
    d = ImageDraw.Draw(img, "RGBA")
    step = GRID_CM * px_per_cm
    x0, y0 = MARGIN_CM * px_per_cm, MARGIN_CM * px_per_cm
    x1, y1 = x0 + NX * step, y0 + NY * step
    color = (128, 128, 128, 200)
    dash, gap = 20, 16

    for k in range(NX + 1):
        x = x0 + k * step
        y = y0
        while y < y1:
            d.line([(x, y), (x, min(y + dash, y1))], fill=color, width=1)
            y += dash + gap
    for k in range(NY + 1):
        y = y0 + k * step
        x = x0
        while x < x1:
            d.line([(x, y), (min(x + dash, x1), y)], fill=color, width=1)
            x += dash + gap


def draw_black_lines(img: Image.Image, px_per_cm: float, pos: str = "outside") -> None:
    """上方长边 + 右侧短边各一条 1.8cm 黑线（辅助定位）。

    pos='outside'：画在 25cm 留白里、紧贴巡查区外沿（默认，不占格）
    pos='inside' ：画在巡查区内沿（若现场是压在格线上，改这个）
    """
    d = ImageDraw.Draw(img)
    lw = BLACK_LINE_CM * px_per_cm
    x0, y0 = MARGIN_CM * px_per_cm, MARGIN_CM * px_per_cm
    x1, y1 = x0 + NX * GRID_CM * px_per_cm, y0 + NY * GRID_CM * px_per_cm
    if pos == "outside":
        d.rectangle([x0, y0 - lw, x1, y0], fill=(0, 0, 0))
        d.rectangle([x1, y0, x1 + lw, y1], fill=(0, 0, 0))
    else:
        d.rectangle([x0, y0, x1, y0 + lw], fill=(0, 0, 0))
        d.rectangle([x1 - lw, y0, x1, y1], fill=(0, 0, 0))


def draw_no_fly(img: Image.Image, px_per_cm: float, rng: random.Random) -> set[str]:
    """随机选 3 个连续格贴灰色禁飞布，返回被占用的格名集合。"""
    horizontal = rng.random() < 0.5
    if horizontal:
        j = rng.randrange(NY)
        i0 = rng.randrange(NX - NOFLY_CELLS + 1)
        cells = [(i0 + k, j) for k in range(NOFLY_CELLS)]
    else:
        i = rng.randrange(NX)
        j0 = rng.randrange(NY - NOFLY_CELLS + 1)
        cells = [(i, j0 + k) for k in range(NOFLY_CELLS)]

    d = ImageDraw.Draw(img)
    for i, j in cells:
        x0, y0, x1, y1 = cell_rect(i, j)
        d.rectangle(
            [x0 * px_per_cm, y0 * px_per_cm, x1 * px_per_cm, y1 * px_per_cm],
            fill=NOFLY_RGB,
        )
    return {cell_name(i, j) for i, j in cells}


def load_templates(root: Path) -> dict[str, list[Image.Image]]:
    out: dict[str, list[Image.Image]] = {}
    for sp in SPECIES:
        files = sorted((root / sp).glob("*.png"))
        if not files:
            raise SystemExit(f"缺少模板：{root / sp}/*.png（先跑 split_templates.py）")
        out[sp] = [Image.open(f).convert("RGBA") for f in files]
    return out


def style_augment(tpl: Image.Image, rng: random.Random) -> Image.Image:
    """模拟"赛区打印出来的彩色动物贴纸"被相机拍到的样子。

    赛题说明 1(4)：动物体态图取自赛题附图、**彩色印刷**、沿轮廓剪下贴在地图上。
    所以这里要还原的是"打印 + 纸张 + 剪裁"的退化，而不是剪影化：
      · 饱和度下降、打印偏色（偏黄）、γ 与对比度抖动
      · 纸上白边（沿轮廓剪下会留下 1~3mm 纸边）
      · 轻度失焦/网点模糊
    """
    rgb = tpl.convert("RGB")
    a = tpl.getchannel("A")
    arr = np.asarray(rgb).astype(np.float32)

    sat = rng.uniform(0.55, 1.05)                   # 打印比屏幕更"灰"
    gray = arr.mean(axis=2, keepdims=True)
    arr = gray + (arr - gray) * sat
    arr *= np.array([rng.uniform(0.97, 1.05), rng.uniform(0.96, 1.02),
                     rng.uniform(0.88, 1.00)], dtype=np.float32)   # 偏暖/偏黄
    gamma = rng.uniform(0.85, 1.20)
    arr = 255.0 * np.clip(arr / 255.0, 0, 1) ** gamma
    arr = np.clip(arr * rng.uniform(0.90, 1.05) + rng.uniform(-12, 12), 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    out = img.convert("RGBA")
    out.putalpha(a)

    if rng.random() < 0.6:                          # 沿轮廓剪下留下的纸边
        k = rng.choice([3, 5, 7, 9])
        grown = a.filter(ImageFilter.MaxFilter(k))
        paper = Image.new("RGBA", tpl.size, (250, 248, 240, 255))
        paper.putalpha(grown)
        paper.paste(out, (0, 0), out)
        out = paper
    return out

def place_animals(
    img: Image.Image,
    px_per_cm: float,
    templates: dict[str, list[Image.Image]],
    n: int,
    blocked: set[str],
    rng: random.Random,
    style_aug: bool = True,
) -> list[dict]:
    free = [(i, j) for i in range(NX) for j in range(NY) if cell_name(i, j) not in blocked]
    chosen = rng.sample(free, k=min(n, len(free)))
    placed = []

    for i, j in chosen:
        sp = rng.choice(SPECIES)
        tpl = rng.choice(templates[sp])
        len_cm = NOMINAL_LEN_CM.get(sp, DEFAULT_LEN_CM) * rng.uniform(0.85, 1.15)
        target = len_cm * px_per_cm
        scale = target / max(tpl.size)
        w, h = max(round(tpl.width * scale), 8), max(round(tpl.height * scale), 8)
        tpl = tpl.resize((w, h), Image.LANCZOS)
        if style_aug:
            tpl = style_augment(tpl, rng)

        angle = rng.uniform(0, 360)
        rot = tpl.rotate(angle, expand=True, resample=Image.BICUBIC)

        x0, y0, x1, y1 = cell_rect(i, j)
        cx = (x0 + x1) / 2 + rng.uniform(-12, 12)   # 贴纸在格内，允许偏一点
        cy = (y0 + y1) / 2 + rng.uniform(-12, 12)
        px, py = cx * px_per_cm, cy * px_per_cm

        left, top = round(px - rot.width / 2), round(py - rot.height / 2)

        # 纸质贴纸压在地图上会有淡淡的投影；没有阴影的"硬贴"是合成集最明显的破绽
        shadow_a = rot.getchannel("A").point(lambda v: int(v * 0.38))
        shadow_a = shadow_a.filter(ImageFilter.GaussianBlur(3))
        shadow = Image.new("RGBA", rot.size, (0, 0, 0, 0))
        shadow.putalpha(shadow_a)
        img.paste(shadow, (left + 5, top + 6), shadow)
        img.paste(rot, (left, top), rot)

        # 标签用「真实不透明像素」的包围盒，而不是模板旋转后的矩形，
        # 否则贴纸斜着放会把框撑大一两成，模型学出来的框会系统性偏大
        solid = rot.getchannel("A").point(lambda v: 255 if v > 10 else 0)
        bb = solid.getbbox() or (0, 0, rot.width, rot.height)
        box_px = [left + bb[0], top + bb[1], left + bb[2], top + bb[3]]

        placed.append(
            {
                "cell": cell_name(i, j),
                "species": sp,
                "center_cm": [round(cx, 2), round(cy, 2)],
                "length_cm": round(len_cm, 2),
                "angle_deg": round(angle, 1),
                # 贴到场地上的真实包围盒（场地 px 坐标，后续用于算标签）
                "bbox_px": box_px,
            }
        )
    return placed


# ---------------- 视角采样 ----------------


def rotate_about(pts: np.ndarray, theta_deg: float, center: tuple[float, float]) -> np.ndarray:
    """y 轴向下坐标系里的旋转，与 PIL.Image.rotate 的正角度（视觉逆时针）一致。"""
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    x = pts[:, 0] - center[0]
    y = pts[:, 1] - center[1]
    return np.stack([center[0] + x * c + y * s, center[1] - x * s + y * c], axis=1)


def photometric(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """室内光照不均 + 白平衡漂移 + 顶灯渐晕 + 噪声。"""
    h, w = arr.shape[:2]
    x = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
    y = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]

    # 顶灯：一个随机的线性/对角亮度梯度
    gx, gy = rng.uniform(-0.14, 0.14), rng.uniform(-0.14, 0.14)
    arr = arr * (1.0 + gx * x + gy * y)[..., None]

    # 渐晕
    vig = rng.uniform(0.0, 0.18)
    if vig > 0:
        r2 = (x**2 + y**2) / 0.5
        arr = arr * (1.0 - vig * r2)[..., None]

    # 每个通道的增益漂移（拍哑光喷绘布的偏色）
    gain = np.array([rng.uniform(0.97, 1.03), rng.uniform(0.98, 1.02), rng.uniform(0.95, 1.05)],
                    dtype=np.float32)
    arr = arr * gain[None, None, :]
    return arr


def motion_blur(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """3 抽样点平移叠加，近似飞行时的运动模糊。"""
    if rng.random() > 0.5:
        return arr
    n = rng.choice([2, 3, 4])
    dx, dy = rng.uniform(-1, 1) * n, rng.uniform(-1, 1) * n
    acc = arr.copy()
    for k in range(1, n):
        acc += np.roll(np.roll(arr, round(dy * k / n), axis=0), round(dx * k / n), axis=1)
    return acc / n


def make_sample(
    mat_padded: Image.Image,
    pad: int,
    animals: list[dict],
    px_per_cm: float,
    out_w: int,
    out_h: int,
    alt_cm: float,
    fov_deg: float,
    anchor: float,
    rng: random.Random,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
    ground_w_cm = 2 * alt_cm * math.tan(math.radians(fov_deg / 2))
    crop_w = ground_w_cm * px_per_cm
    crop_h = crop_w * out_h / out_w
    s = out_w / crop_w                      # 场地 px -> 输出 px

    if animals and rng.random() < anchor:
        a = rng.choice(animals)
        cx, cy = a["center_cm"][0] * px_per_cm, a["center_cm"][1] * px_per_cm
        cx += rng.uniform(-0.3, 0.3) * crop_w
        cy += rng.uniform(-0.3, 0.3) * crop_h
    else:
        # 无人机只在巡查区格子上方飞，不会飞到 25cm 留白上，采样中心按格心范围取
        cx = rng.uniform(MARGIN_CM + GRID_CM / 2,
                         MAT_W_CM - MARGIN_CM - GRID_CM / 2) * px_per_cm
        cy = rng.uniform(MARGIN_CM + GRID_CM / 2,
                         MAT_H_CM - MARGIN_CM - GRID_CM / 2) * px_per_cm

    # 旋转必须在一个「对角线长度的正方形窗口」里做，否则四角会出现黑三角
    d = math.ceil(math.hypot(crop_w, crop_h)) + 2
    left, top = cx - d / 2, cy - d / 2
    crop = mat_padded.crop(
        (round(left) + pad, round(top) + pad, round(left) + pad + d, round(top) + pad + d)
    )

    theta = rng.uniform(-180, 180)          # 机身偏航
    rot = crop.rotate(theta, resample=Image.BICUBIC, expand=False)

    # 从旋转后的正方形里裁出真实的相机视场
    ox, oy = (d - crop_w) / 2, (d - crop_h) / 2
    rot = rot.crop((round(ox), round(oy), round(ox + crop_w), round(oy + crop_h)))
    if rot.size != (out_w, out_h):
        rot = rot.resize((out_w, out_h), Image.LANCZOS)

    # 标签：场地坐标 -> 相对窗口左上 -> 绕窗口中心旋转 -> 中心裁剪平移 -> 缩放 -> AABB
    labels = []
    for a in animals:
        bx = np.array(
            [[a["bbox_px"][0], a["bbox_px"][1]], [a["bbox_px"][2], a["bbox_px"][1]],
             [a["bbox_px"][2], a["bbox_px"][3]], [a["bbox_px"][0], a["bbox_px"][3]]],
            dtype=np.float64,
        )
        pts = bx - np.array([left, top])
        pts = rotate_about(pts, theta, (d / 2, d / 2))
        pts = (pts - np.array([ox, oy])) * s
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, out_w), min(y1, out_h)
        if x1c - x0c < 6 or y1c - y0c < 6:
            continue
        inter = (x1c - x0c) * (y1c - y0c)
        full = max((x1 - x0) * (y1 - y0), 1e-6)
        if inter / full < 0.35:             # 大部分在视野外就不算目标
            continue
        labels.append(
            (
                SPECIES.index(a["species"]),
                (x0c + x1c) / 2 / out_w,
                (y0c + y1c) / 2 / out_h,
                (x1c - x0c) / out_w,
                (y1c - y0c) / out_h,
            )
        )

    # 光度：亮度/对比度/饱和度扰动 + 顶灯梯度 + 渐晕 + 轻微运动模糊 + 噪声
    img = rot
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.12))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.90, 1.10))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.88, 1.12))

    arr = np.asarray(img).astype(np.float32)
    arr = photometric(arr, rng)
    arr = motion_blur(arr, rng)
    if rng.random() < 0.25:
        arr = np.asarray(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                         .filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7))))
    arr = arr + np.random.normal(0, rng.uniform(1.0, 3.5), arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return out, labels


# ---------------- 主流程 ----------------


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default=str(here / "datasets/raw/地貌图.png"))
    ap.add_argument("--templates", default=str(here / "01_vision/assets/templates"))
    ap.add_argument("--out", default=str(here / "datasets/yolo"))
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--size", default="1280x960", help="输出分辨率 WxH")
    ap.add_argument("--px-per-cm", type=float, default=10.0,
                    help="场地渲染精度：10 = 1px/mm（5000x4000），5 = 更快但更糊")
    ap.add_argument("--fov", type=float, default=66.0, help="相机水平视场角")
    ap.add_argument("--alt", type=float, default=120.0, help="巡航高度 cm（赛题 120±10）")
    ap.add_argument("--animals", type=int, default=8, help="场地上贴几只动物")
    ap.add_argument("--anchor", type=float, default=0.5, help="按目标居中采样的比例")
    ap.add_argument("--blackline-pos", choices=["outside", "inside"], default="outside")
    ap.add_argument("--no-style-aug", action="store_true",
                    help="关掉打印风格增强（默认开启：降饱和/偏色/纸边/失焦）")
    ap.add_argument("--preview", default="", help="整幅场地预览图输出路径")
    ap.add_argument("--debug-overlay", action="store_true", help="把标签画回图片，用于自查")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    out_w, out_h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)

    print(f"[1/3] 渲染虚拟场地 {MAT_W_CM}x{MAT_H_CM}cm @ {args.px_per_cm}px/cm ...")
    mat = render_mat(Path(args.terrain), args.px_per_cm)
    draw_grid(mat, args.px_per_cm)
    blocked = draw_no_fly(mat, args.px_per_cm, rng)
    draw_black_lines(mat, args.px_per_cm, args.blackline_pos)
    templates = load_templates(Path(args.templates))
    animals = place_animals(mat, args.px_per_cm, templates, args.animals, blocked, rng,
                            style_aug=not args.no_style_aug)
    print(f"      禁飞区 {sorted(blocked)}，动物 {len(animals)} 只："
          f"{ {a['species'] for a in animals} }")

    if args.preview:
        p = Path(args.preview)
        p.parent.mkdir(parents=True, exist_ok=True)
        mat.resize((1600, round(1600 * MAT_H_CM / MAT_W_CM)), Image.LANCZOS).save(
            p, quality=88, optimize=True)
        print(f"      预览 -> {p}")

    # 地图外：铺一层"体育馆地面"。真实场地里镜头越界看到的就是馆内地面，
    # 比把地图边缘拉伸过去更像实拍（而且打印布的边界本来就有一条硬边）。
    pad = 1000
    cw, ch = mat.width + 2 * pad, mat.height + 2 * pad
    luma = np.asarray(mat.crop((0, 0, 800, 640)).filter(ImageFilter.GaussianBlur(3))
                      .convert("L")).astype(np.float32)
    luma = 148 + (luma - luma.mean()) * 0.35          # 浅灰地面 + 地表纹理残留
    tile = Image.fromarray(np.clip(luma, 0, 255).astype(np.uint8)).convert("RGB")
    mat_padded = tile.resize((cw, ch), Image.BICUBIC)
    mat_padded.paste(mat, (pad, pad))

    print(f"[2/3] 采样视角：FOV {args.fov}° @ {args.alt}cm -> {out_w}x{out_h}")
    for split, n in (("train", args.n_train), ("val", args.n_val)):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for k in range(n):
            alt = args.alt + rng.uniform(-10, 10)          # 120±10cm
            img, labels = make_sample(mat_padded, pad, animals, args.px_per_cm,
                                      out_w, out_h, alt, args.fov, args.anchor, rng)
            stem = f"{split}_{k:05d}"
            if args.debug_overlay:
                dbg = img.convert("RGB").copy()
                d = ImageDraw.Draw(dbg)
                for cls, cx, cy, bw, bh in labels:
                    d.rectangle(
                        [(cx - bw / 2) * out_w, (cy - bh / 2) * out_h,
                         (cx + bw / 2) * out_w, (cy + bh / 2) * out_h],
                        outline=(255, 0, 0), width=2)
                    d.text(((cx - bw / 2) * out_w + 3, (cy - bh / 2) * out_h + 2),
                           SPECIES[cls], fill=(255, 255, 0))
                dbg.save(img_dir / f"{stem}.jpg", quality=92)
            else:
                img.convert("RGB").save(img_dir / f"{stem}.jpg", quality=92, optimize=True)
            with open(lbl_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
                for cls, cx, cy, bw, bh in labels:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            kept += 1
        print(f"      {split}: {kept} 张")

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "mat_px": [round(MAT_W_CM * args.px_per_cm), round(MAT_H_CM * args.px_per_cm)],
        "px_per_cm": args.px_per_cm,
        "camera": {"fov_deg": args.fov, "alt_cm": args.alt, "size": [out_w, out_h]},
        "no_fly_cells": sorted(blocked),
        "animals": animals,
        "species": SPECIES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"[3/3] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()

