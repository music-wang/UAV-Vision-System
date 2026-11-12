#!/usr/bin/env python3
"""对比"测试画面"和"训练集"的成像统计，用来判断这次拍的到底偏亮还是偏暗。

为什么需要它：模型对**亮度/对比度/边缘锐度**的分布很敏感。实拍训练集是在某个光照下拍的，
板端相机曝光一变，输入分布就偏移；偏移大到一定程度就开始漏检/误检。
这个脚本把这种偏移量化出来，而不是靠眼看。

用法：
  # 1) 先看训练集的基线（跑一次就记住这几个数）
  python 01_vision/scripts/domain_stats.py --ref D:\\UAV_Dataset\\images\\train --n 120

  # 2) 再拿板子抓的图对比（多张更有意义）
  python 01_vision/scripts/domain_stats.py --ref D:\\UAV_Dataset\\images\\train --test D:\\frames

输出：每张图的 平均亮度 / 对比度(std) / 5% 与 95% 分位 / 边缘密度，
     以及 test 相对 ref 的偏离（>15% 标 [!]）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def stats(path: Path) -> dict:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((320, 240)), dtype=np.float32)
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "edge": float((gx + gy) / 2),
    }


def collect(root: Path, n: int) -> list[dict]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)[:n]
    return [stats(p) for p in files]


def summarize(name: str, rows: list[dict]) -> dict:
    if not rows:
        return {}
    agg = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    print(f"\n[{name}] {len(rows)} 张（中位数）")
    print(f"   平均亮度 {agg['mean']:6.1f}   对比度 {agg['std']:6.1f}   "
          f"P5/P95 {agg['p5']:5.1f}/{agg['p95']:5.1f}   边缘密度 {agg['edge']:5.2f}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="训练集目录（参考分布）")
    ap.add_argument("--test", default="", help="要对比的画面（目录）")
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()

    ref = summarize("训练集", collect(Path(args.ref), args.n))
    if not args.test:
        print("\n（只给了 --ref；加 --test 目录即可得到偏离百分比）")
        return

    tst = summarize("测试画面", collect(Path(args.test), max(args.n, 50)))
    if not ref or not tst:
        raise SystemExit("有一边没读到图片")

    print("\n相对训练集的偏离（超过 15% 标 [!]）")
    print("   指标        训练集    测试      偏离")
    for k, label in (("mean", "平均亮度"), ("std", "对比度"), ("p95", "高光P95"), ("edge", "边缘密度")):
        base = ref[k] or 1e-6
        d = (tst[k] - base) / base * 100
        flag = "  [!]" if abs(d) > 15 else ""
        print(f"   {label:<9}{base:8.1f}{tst[k]:9.1f}{d:+8.1f}%{flag}")

    if abs((tst["mean"] - ref["mean"]) / (ref["mean"] or 1)) > 0.15:
        print("\n建议：调相机曝光/增益，让平均亮度接近训练集；再看对比度和边缘密度是否同时回正。")


if __name__ == "__main__":
    main()
