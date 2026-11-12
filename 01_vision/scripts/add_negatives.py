#!/usr/bin/env python3
"""把"纯背景（无目标）"照片当负样本并进数据集。

负样本在 YOLO 里就是**同名但内容为空的 .txt**。打标软件对没画框的图通常不导出 txt，
所以这里负责补出来。

流程：
  1. 扫描 --src 下所有图片，找出没有同名 .txt 的（= 漏导出的负样本）
  2. 在源目录补一个 0 字节同名 .txt（可用 --no-create-txt 关掉）
  3. 按 --val-ratio 比例，把图 + 空 txt 复制进 --dst 的 images/labels 的 train/val
  4. 和已有文件重名的跳过，绝不覆盖已标注数据

用法：
  python 01_vision/scripts/add_negatives.py --src C:\\Users\\andy\\Desktop\\labels --dry-run
  python 01_vision/scripts/add_negatives.py --src C:\\Users\\andy\\Desktop\\labels
  python 01_vision/scripts/add_negatives.py --src D:\\bg1 D:\\bg2 --dst D:\\UAV_Dataset
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def images_in(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="含背景图的目录（可多个）")
    ap.add_argument("--dst", default=r"D:\UAV_Dataset", help="目标数据集根目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-empty", action="store_true",
                    help="已经有空 txt 的图也算负样本（默认只找缺失 txt 的）")
    ap.add_argument("--no-create-txt", action="store_true",
                    help="不在源目录补空 txt，只在目标目录写")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dst = Path(args.dst)
    if not (dst / "images" / "train").is_dir():
        raise SystemExit(f"目标数据集不像已初始化：{dst}（先跑 prepare_real.py）")

    existing = {p.stem for split in ("train", "val")
                for p in (dst / "images" / split).glob("*") if p.suffix.lower() in IMG_EXT}

    negs: list[Path] = []
    for s in args.src:
        root = Path(s)
        if not root.is_dir():
            raise SystemExit(f"目录不存在：{root}")
        imgs = images_in(root)
        unpaired = []
        for p in imgs:
            lab = p.with_suffix(".txt")
            if not lab.is_file():
                unpaired.append(p)
            elif args.include_empty and lab.read_text(encoding="utf-8", errors="replace").strip() == "":
                unpaired.append(p)
        print(f"[扫描] {root}: {len(imgs)} 张图 -> 负样本候选 {len(unpaired)} 张")
        negs.extend(unpaired)

    # 去掉与已有数据重名的（避免覆盖真实标注）
    dup = [p for p in negs if p.stem in existing]
    negs = [p for p in negs if p.stem not in existing]
    if dup:
        print(f"[跳过] 与目标数据集重名 {len(dup)} 张：{[p.name for p in dup][:5]}")

    if not negs:
        print("\n没有可加入的负样本。确认背景图的目录是否正确？")
        return

    rng = random.Random(args.seed)
    order = list(range(len(negs)))
    rng.shuffle(order)
    n_val = max(1, round(len(negs) * args.val_ratio)) if len(negs) > 1 else 0
    val_idx = set(order[:n_val])

    print(f"\n[计划] 共 {len(negs)} 张负样本 -> train {len(negs) - n_val} / val {n_val}")
    if args.dry_run:
        for i, p in enumerate(negs[:10]):
            print(f"   {'val ' if i in val_idx else 'train'}  {p.name}")
        if len(negs) > 10:
            print(f"   ... 其余 {len(negs) - 10} 张")
        print("[dry-run] 未写任何文件")
        return

    added = {"train": 0, "val": 0}
    for i, p in enumerate(negs):
        split = "val" if i in val_idx else "train"
        if not args.no_create_txt:
            lab = p.with_suffix(".txt")
            if not lab.exists():
                lab.write_text("", encoding="utf-8")      # 0 字节负样本
        shutil.copy2(p, dst / "images" / split / p.name)
        (dst / "labels" / split / (p.stem + ".txt")).write_text("", encoding="utf-8")
        added[split] += 1

    print(f"[写入] train +{added['train']}  val +{added['val']}")

    for split in ("train", "val"):
        imgs = [x for x in (dst / "images" / split).glob("*") if x.suffix.lower() in IMG_EXT]
        empt = sum(1 for q in (dst / "labels" / split).glob("*.txt")
                   if not q.read_text(encoding="utf-8", errors="replace").strip())
        print(f"   {split}: 共 {len(imgs)} 张，其中负样本 {empt} 张（{empt / max(len(imgs), 1):.1%}）")
    print("\n下一步：python 01_vision/scripts/check_dataset.py")


if __name__ == "__main__":
    main()
