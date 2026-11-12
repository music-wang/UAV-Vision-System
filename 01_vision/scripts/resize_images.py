#!/usr/bin/env python3
"""批量缩小图片长边（训练用），标签不用改 —— YOLO 标签是归一化的，等比缩放不影响。

注意两点：
  * **不应用 EXIF 旋转**。打标时软件读的是原始像素方向，这里如果按 EXIF 转正，
    图和框就对不上了。所以只做等比缩放，像素方向保持不变。
  * 默认**原地替换**（先写临时文件再改名，失败不会留下半个文件）。想保留原图就用 --out。

用法：
  python 01_vision/scripts/resize_images.py --src D:\\UAV_Dataset\\images --dry-run
  python 01_vision/scripts/resize_images.py --src D:\\UAV_Dataset\\images --long-side 1280
  python 01_vision/scripts/resize_images.py --src D:\\photos --out D:\\photos_1280 --long-side 1280
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="图片所在目录（会递归）")
    ap.add_argument("--out", default="", help="输出目录；不填 = 原地替换")
    ap.add_argument("--long-side", type=int, default=1280)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        raise SystemExit(f"目录不存在：{src}")
    out_root = Path(args.out) if args.out else None

    files = [p for p in src.rglob("*") if p.suffix in IMG_EXT]
    if not files:
        raise SystemExit(f"{src} 下没有图片")

    before = sum(p.stat().st_size for p in files)
    print(f"[扫描] {len(files)} 张图，{before / 2**20:.0f} MB，长边目标 {args.long_side}")

    done = skipped = failed = 0
    after = 0
    t0 = time.perf_counter()
    for p in files:
        try:
            with Image.open(p) as im:
                w, h = im.size
                if max(w, h) <= args.long_side:
                    skipped += 1
                    after += p.stat().st_size
                    continue
                s = args.long_side / max(w, h)
                size = (max(round(w * s), 1), max(round(h * s), 1))
                if args.dry_run:
                    print(f"   {p.name}: {w}x{h} -> {size[0]}x{size[1]}")
                    done += 1
                    after += int(p.stat().st_size * (size[0] * size[1]) / (w * h))
                    continue
                im2 = im.convert("RGB").resize(size, Image.LANCZOS)
                target = (out_root / p.relative_to(src)) if out_root else p
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(target.suffix + ".tmp")
                # 临时文件后缀是 .tmp，PIL 认不出来，必须显式给 format
                fmt = "PNG" if target.suffix.lower() == ".png" else "JPEG"
                im2.save(tmp, format=fmt, quality=args.quality, optimize=True)
                os.replace(tmp, target)          # 原子替换，避免写坏原文件
                done += 1
                after += target.stat().st_size
        except Exception as e:
            failed += 1
            print(f"   ! {p.name}: {e}")

    dt = time.perf_counter() - t0
    print(f"\n[结果] 缩放 {done} 张，跳过 {skipped} 张（已经够小），失败 {failed} 张，用时 {dt:.0f}s")
    if not args.dry_run:
        print(f"[体积] {before / 2**20:.0f} MB -> {after / 2**20:.0f} MB（省 {100 * (1 - after / max(before, 1)):.0f}%）")
        print("[提醒] 标签是归一化的，不需要改；跑一遍 check_dataset.py 确认配对没坏")
    else:
        print(f"[dry-run] 预计 {before / 2**20:.0f} MB -> {after / 2**20:.0f} MB，未写任何文件")


if __name__ == "__main__":
    main()
