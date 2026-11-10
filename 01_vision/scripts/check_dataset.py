#!/usr/bin/env python3
"""训练前体检：任何一类 0 实例、图片/标签对不上、框越界，都会直接报错退出。

我们踩过的坑：800 张合成图里 wolf 一个实例都没有，模型只能把狼判成孔雀，
而 mAP 依然显示 0.99（因为验证集里也没有狼）。这个脚本就是为了拦住这种问题。

用法：
  python 01_vision/scripts/check_dataset.py                       # 默认查真实数据集
  python 01_vision/scripts/check_dataset.py --data 01_vision/configs/data_synth.yaml
  python 01_vision/scripts/check_dataset.py --allow-missing wolf   # 确实没有某类时才放行

退出码：0 通过；1 有阻塞问题。
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve(data_yaml: Path) -> dict:
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(str(cfg.get("path", ".")))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    cfg["_root"] = root
    return cfg


def scan_split(root: Path, split_rel: str, names: list[str]) -> dict:
    img_dir = root / split_rel
    lbl_dir = root / split_rel.replace("images", "labels", 1)   # images/train -> labels/train
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT) if img_dir.is_dir() else []
    cnt: Counter = Counter()
    empty = missing = bad_box = 0
    orphan = 0
    img_stems = {p.stem for p in imgs}

    for p in imgs:
        lab = lbl_dir / (p.stem + ".txt")
        if not lab.is_file():
            missing += 1
            continue
        lines = [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            empty += 1
            continue
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                bad_box += 1
                continue
            try:
                c = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                bad_box += 1
                continue
            if not all(0.0 <= v <= 1.0 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                bad_box += 1
        cnt.update(int(ln.split()[0]) for ln in lines if ln.split()[0].isdigit())

    if lbl_dir.is_dir():
        orphan = len([p for p in lbl_dir.glob("*.txt") if p.stem not in img_stems])
    return {"images": len(imgs), "empty": empty, "missing_label": missing,
            "orphan_label": orphan, "bad_box": bad_box, "per_class": cnt,
            "dir": str(img_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "01_vision/configs/data_real.yaml"))
    ap.add_argument("--allow-missing", nargs="*", default=[],
                    help="允许缺失的类别名（默认任何一类缺失都算失败）")
    ap.add_argument("--max-empty-ratio", type=float, default=0.5,
                    help="负样本占比上限，过高说明大部分图没标到目标")
    args = ap.parse_args()

    cfg = resolve(Path(args.data))
    names = list(cfg.get("names", []))
    root = cfg["_root"]
    print(f"[数据] {args.data}\n       根目录 {root}\n       类别 {names}\n")

    problems: list[str] = []
    if not root.is_dir():
        print(f"[X] 数据集根目录不存在：{root}")
        raise SystemExit(1)

    total = {n: 0 for n in names}
    for split_key in ("train", "val"):
        rel = cfg.get(split_key)
        if not rel:
            continue
        s = scan_split(root, str(rel), names)
        if s["images"] == 0:
            problems.append(f"{split_key}: 0 张图（{s['dir']}）")
            print(f"[X] {split_key}: 0 张图 -> {s['dir']}")
            continue
        print(f"[{split_key}] {s['images']} 张图  负样本 {s['empty']} ({s['empty']/s['images']:.1%})"
              f"  缺标签 {s['missing_label']}  多余标签 {s['orphan_label']}  坏框 {s['bad_box']}")
        for i, n in enumerate(names):
            c = s["per_class"][i]
            if split_key == "train":
                total[n] += c
            flag = "  [!] 0" if c == 0 else ""
            print(f"      {n:<9}{c:>6}{flag}")
        if s["missing_label"]:
            problems.append(f"{split_key}: {s['missing_label']} 张图没有对应 txt")
        if s["bad_box"]:
            problems.append(f"{split_key}: {s['bad_box']} 行标签格式/坐标非法")
        if s["empty"] / s["images"] > args.max_empty_ratio:
            problems.append(f"{split_key}: 负样本占 {s['empty']/s['images']:.0%}，"
                            f"高于 {args.max_empty_ratio:.0%}")

    print("\n[训练集合计]")
    for n in names:
        print(f"   {n:<9}{total[n]:>6}")
    missing = [n for n in names if total[n] == 0 and n not in args.allow_missing]
    if missing:
        problems.append(f"训练集缺少类别：{missing}")

    if problems:
        print("\n[X] 体检不通过：")
        for p in problems:
            print(f"   - {p}")
        print("\n修完再训练。确实没有某类时用 --allow-missing <类名> 显式放行。")
        raise SystemExit(1)
    print("\n[OK] 体检通过，可以训练。")


if __name__ == "__main__":
    main()
