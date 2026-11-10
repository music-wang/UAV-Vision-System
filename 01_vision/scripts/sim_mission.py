#!/usr/bin/env python3
"""离线"任务仿真"：按航线扫一遍场地，逐帧存图 + 位姿，用来验证
检测 → 归格 → 去重计数 的整条链路（不需要真飞）。

产出（默认 datasets/mission/）：
  images/*.jpg      每帧相机视角
  frames.json       [{file, x_cm, y_cm, yaw_deg, alt_cm}]  —— 相当于 SLAM 输出
  gt.json           {cell: {species: count}}               —— 场地真值
  mat_preview.jpg   整幅场地（含禁飞区与动物）
  scene.json        每只动物的格号/坐标/角度

用法：
  python 01_vision/scripts/sim_mission.py --cells 60 --frames-per-cell 2
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synth_dataset as S  # noqa: E402


def route_waypoints(blocked: set[str], yaw_mode: str, rng: random.Random):
    """牛耕式航线：逐行往返，途经每个可用格的格心；遇禁飞区整格跳过。"""
    pts = []
    for j in range(S.NY):
        order = range(S.NX) if j % 2 == 0 else reversed(range(S.NX))
        for i in order:
            if S.cell_name(i, j) in blocked:
                continue
            x0, y0, x1, y1 = S.cell_rect(i, j)
            yaw = 0.0
            if yaw_mode == "heading":
                yaw = 0.0 if j % 2 == 0 else 180.0
            pts.append(((x0 + x1) / 2, (y0 + y1) / 2, yaw))
    return pts


def render_view(mat_padded: Image.Image, pad: int, pose, out_w: int, out_h: int,
                fov_deg: float, alt_cm: float, ppc: float) -> Image.Image:
    """按位姿从场地截一帧，几何与 cell_mapper.px_to_map 严格互逆。"""
    gw = 2 * alt_cm * math.tan(math.radians(fov_deg / 2)) * ppc
    gh = gw * out_h / out_w
    d = math.ceil(math.hypot(gw, gh)) + 2
    left, top = pose[0] * ppc - d / 2, pose[1] * ppc - d / 2
    crop = mat_padded.crop((round(left) + pad, round(top) + pad,
                            round(left) + pad + d, round(top) + pad + d))
    crop = crop.rotate(pose[2], resample=Image.BICUBIC, expand=False)
    ox, oy = (d - gw) / 2, (d - gh) / 2
    crop = crop.crop((round(ox), round(oy), round(ox + gw), round(oy + gh)))
    return crop.resize((out_w, out_h), Image.LANCZOS)


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default=str(here / "datasets/raw/地貌图.png"))
    ap.add_argument("--templates", default=str(here / "01_vision/assets/templates"))
    ap.add_argument("--out", default=str(here / "datasets/mission"))
    ap.add_argument("--frames-per-cell", type=int, default=2)
    ap.add_argument("--size", default="1280x960")
    ap.add_argument("--fov", type=float, default=66.0)
    ap.add_argument("--alt", type=float, default=120.0)
    ap.add_argument("--px-per-cm", type=float, default=10.0)
    ap.add_argument("--animals", type=int, default=8)
    ap.add_argument("--yaw", choices=["heading", "zero"], default="heading")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_w, out_h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    print("[1/3] 布置场地 ...")
    mat = S.render_mat(Path(args.terrain), args.px_per_cm)
    S.draw_grid(mat, args.px_per_cm)
    blocked = S.draw_no_fly(mat, args.px_per_cm, rng)
    S.draw_black_lines(mat, args.px_per_cm, "outside")
    templates = S.load_templates(Path(args.templates))
    animals = S.place_animals(mat, args.px_per_cm, templates, args.animals, blocked, rng)

    gt: dict[str, dict[str, int]] = {}
    for a in animals:
        gt.setdefault(a["cell"], {}).setdefault(a["species"], 0)
        gt[a["cell"]][a["species"]] += 1

    mat.resize((1600, round(1600 * S.MAT_H_CM / S.MAT_W_CM)), Image.LANCZOS).save(
        out / "mat_preview.jpg", quality=88, optimize=True)

    pad = 1000
    cw, ch = mat.width + 2 * pad, mat.height + 2 * pad
    import numpy as np
    from PIL import ImageEnhance, ImageFilter
    luma = np.asarray(mat.crop((0, 0, 800, 640)).filter(ImageFilter.GaussianBlur(3))
                      .convert("L")).astype(np.float32)
    floor = Image.fromarray(np.clip(148 + (luma - luma.mean()) * 0.35, 0, 255)
                            .astype(np.uint8)).convert("RGB").resize((cw, ch), Image.BICUBIC)
    floor.paste(mat, (pad, pad))
    mat_padded = floor

    wps = route_waypoints(blocked, args.yaw, rng)
    print(f"[2/3] 航线 {len(wps)} 个航点 × {args.frames_per_cell} 帧 ...")
    frames = []
    k = 0
    for (x, y, yaw) in wps:
        for _ in range(args.frames_per_cell):
            alt = args.alt + rng.uniform(-8, 8)
            px = x + rng.uniform(-4, 4)
            py = y + rng.uniform(-4, 4)
            th = yaw + rng.uniform(-6, 6)
            img = render_view(mat_padded, pad, (px, py, th), out_w, out_h,
                              args.fov, alt, args.px_per_cm)
            name = f"f{k:04d}.jpg"
            img.save(out / "images" / name, quality=90, optimize=True)
            frames.append({"file": name, "x_cm": round(px, 2), "y_cm": round(py, 2),
                           "yaw_deg": round(th, 2), "alt_cm": round(alt, 2)})
            k += 1

    (out / "frames.json").write_text(json.dumps(frames, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    (out / "gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "scene.json").write_text(
        json.dumps({"no_fly": sorted(blocked), "animals": animals}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[3/3] {k} 帧 -> {out}")
    print("真值：", json.dumps(gt, ensure_ascii=False))


if __name__ == "__main__":
    main()
