#!/usr/bin/env python3
"""把「拍好的照片 + 标注」整理成 Ultralytics 能直接训练的数据集。

支持两种输入：
  A. 图片和同名 .txt 放一起：  src/  ├─ IMG_0001.jpg
                                    └─ IMG_0001.txt
  B. 图片和标签分目录：        src/images/*.jpg + src/labels/*.txt
                              也兼容 labelImg 的 src/*.jpg + src/*.txt

用法：
  # 1) 先看一眼数据对不对（不复制，只统计）
  python 01_vision/scripts/prepare_real.py --src D:\\photos --dry-run

  # 2) 正式整理成 datasets/real（8:2 划分）
  python 01_vision/scripts/prepare_real.py --src D:\\photos --val-ratio 0.2

  # 3) 校验：任何一个类别 0 实例都会报错退出
  python 01_vision/scripts/check_dataset.py --data 01_vision/configs/data_real.yaml

设计要点：
  * **没标框的图当负样本**（空 txt）——这是抑制误检最有效的手段，别删；
  * 类别缺失会大声报警（我们踩过：一个类 0 实例，模型把它判成别的类）；
  * 只复制，不动原图，可反复跑（--clean 才清空目标目录）。
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(src: Path) -> list[tuple[Path, Path | None]]:
    """返回 [(图片, 标签或 None)]。"""
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    pairs = []
    for img in imgs:
        for cand in (img.with_suffix(".txt"),
                     src / "labels" / (img.stem + ".txt"),
                     img.parent / "labels" / (img.stem + ".txt")):
            if cand.is_file():
                pairs.append((img, cand))
                break
        else:
            pairs.append((img, None))
    return pairs


def count_labels(files: list[Path]) -> tuple[Counter, int, int]:
    """统计类别实例数、空标签数、每图目标数。"""
    cnt: Counter = Counter()
    empty = 0
    per_img = []
    for f in files:
        lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty += 1
        per_img.append(len(lines))
        for ln in lines:
            try:
                cnt[int(ln.split()[0])] += 1
            except Exception:
                pass
    return cnt, empty, (sum(per_img) / len(per_img) if per_img else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="照片（和标签）所在目录")
    ap.add_argument("--out", default="datasets/real", help="输出数据集根目录")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只统计，不复制")
    ap.add_argument("--clean", action="store_true", help="先清空输出目录")
    ap.add_argument("--unlabeled-as-negative", action="store_true", default=True,
                    help="没有 txt 的图当作负样本（空标签），默认开")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"目录不存在：{src}")

    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"{src} 里没有图片（支持 {sorted(IMG_EXT)}）")
    no_label = [p for p, l in pairs if l is None]
    print(f"[扫描] 图片 {len(pairs)} 张，其中有标签 {len(pairs) - len(no_label)}，"
          f"无标签 {len(no_label)}（按负样本处理）")

    labeled = [l for _, l in pairs if l is not None]
    if labeled:
        cnt, empty, avg = count_labels(labeled)
        print(f"[标签] 实例 {sum(cnt.values())}，空标签 {empty}，每图平均 {avg:.2f} 个")
        for i, name in enumerate(CLASSES):
            flag = "  <- [!] 没有样本！" if cnt[i] == 0 else ""
            print(f"   {name:<9}{cnt[i]:>6}{flag}")

    rng = random.Random(args.seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_val = max(1, round(len(pairs) * args.val_ratio)) if len(pairs) > 1 else 0
    val_idx = set(order[:n_val])

    if args.dry_run:
        print(f"[dry-run] 将划分 train {len(pairs) - n_val} / val {n_val}，未写文件")
        return

    if args.clean and out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stat = {"train": {"images": 0, "empty": 0}, "val": {"images": 0, "empty": 0}}
    for i, (img, lab) in enumerate(pairs):
        split = "val" if i in val_idx else "train"
        shutil.copy2(img, out / "images" / split / img.name)
        dst = out / "labels" / split / (img.stem + ".txt")
        if lab is not None:
            shutil.copy2(lab, dst)
            if not dst.read_text(encoding="utf-8").strip():
                stat[split]["empty"] += 1
        else:
            dst.write_text("", encoding="utf-8")     # 无标注 = 负样本
            stat[split]["empty"] += 1
        stat[split]["images"] += 1

    print(f"\n[输出] {out.resolve()}")
    for split in ("train", "val"):
        print(f"   {split}: {stat[split]['images']} 张（其中负样本 {stat[split]['empty']}）"
              f" -> images/{split}, labels/{split}")

    # 汇总每类实例数，写进 summary.json，训练前一看就知道有没有漏类
    summary = {"source": str(src.resolve()), "classes": CLASSES, "splits": {}}
    for split in ("train", "val"):
        labs = sorted((out / "labels" / split).glob("*.txt"))
        cnt, empty, avg = count_labels(labs)
        summary["splits"][split] = {"images": len(labs), "empty": empty,
                                    "per_class": {CLASSES[i]: cnt[i] for i in range(len(CLASSES))},
                                    "instances": sum(cnt.values())}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    missing = [c for c in CLASSES if summary["splits"]["train"]["per_class"][c] == 0]
    if missing:
        print(f"\n[!] 训练集里这些类别 0 个实例：{missing}")
        print("  模型会把这些类判成别的类。先补拍/补标，再训练。")
    print(f"\n下一步：python 01_vision/scripts/check_dataset.py --data 01_vision/configs/data_real.yaml")


if __name__ == "__main__":
    main()
