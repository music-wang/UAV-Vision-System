#!/usr/bin/env python3
"""清洗 YOLO 标注：删掉非法类别 id、越界/退化框、格式错误的行；空 txt（负样本）保留。

典型用途：标注软件里多出来的"幽灵类"（比如误加的 r，对应 class_id 5），
导出的 txt 里可能残留 id=5 的框；或者有框超出了 [0,1]。

用法：
  python 01_vision/scripts/clean_labels.py --dir C:\\labels            # 只体检，不改文件
  python 01_vision/scripts/clean_labels.py --dir C:\\labels --apply    # 真的改（自动备份）
  python 01_vision/scripts/clean_labels.py --dir C:\\labels --write-classes --apply

规则（--nc 默认 5）：
  * class_id 不在 [0, nc) 的行 -> 删除
  * 任意坐标超出 [0,1]、或 w/h <= 0 的行 -> 删除
  * 字段数不等于 5 的行 -> 删除
  * 删空了的 txt 保留（它变成负样本，这是有用的）
"""
from __future__ import annotations

import argparse
import shutil
import time
from collections import Counter
from pathlib import Path

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]


def check_line(line: str, nc: int) -> tuple[str | None, str]:
    """返回 (保留的行或 None, 问题类型)。"""
    parts = line.split()
    if len(parts) != 5:
        return None, "字段数不是 5"
    try:
        cid = int(parts[0])
        vals = [float(x) for x in parts[1:]]
    except ValueError:
        return None, "不是数字"
    if not (0 <= cid < nc):
        return None, f"class_id={cid} 越界"
    if any(v < 0.0 or v > 1.0 for v in vals):
        return None, "坐标越界"
    if vals[2] <= 0 or vals[3] <= 0:
        return None, "宽或高 <= 0"
    return line.rstrip(), ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="标注 txt 所在目录（会递归）")
    ap.add_argument("--nc", type=int, default=len(CLASSES), help="类别数，默认 5")
    ap.add_argument("--apply", action="store_true", help="真的写回文件（默认只报告）")
    ap.add_argument("--backup", default="", help="备份目录，默认 <dir>/_backup_<时间>")
    ap.add_argument("--write-classes", action="store_true",
                    help="同时写一份 classes.txt 到该目录（标注软件会优先读同目录的）")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        raise SystemExit(f"目录不存在：{root}")
    txts = sorted(root.rglob("*.txt"))
    # 别把 classes.txt 当标注文件
    txts = [p for p in txts if p.name.lower() not in ("classes.txt", "predefined_classes.txt")]
    if not txts:
        raise SystemExit(f"{root} 下没有 .txt 标注文件")

    bad: list[tuple[Path, int, str, str]] = []
    changed_files: list[tuple[Path, list[str]]] = []
    empty = 0
    ids_before: Counter = Counter()
    total_lines = 0

    for p in txts:
        raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
        keep: list[str] = []
        dirty = False
        has_content = False
        for i, line in enumerate(raw, 1):
            if not line.strip():
                continue
            has_content = True
            total_lines += 1
            good, why = check_line(line, args.nc)
            if good is None:
                dirty = True
                bad.append((p, i, why, line.strip()))
            else:
                keep.append(good)
                ids_before[int(good.split()[0])] += 1
        if not has_content:
            empty += 1
        if dirty:
            changed_files.append((p, keep))

    print(f"[扫描] {len(txts)} 个 txt，{total_lines} 个框，空 txt（负样本）{empty} 个")
    print(f"[问题] 需要删除的框 {len(bad)} 个，涉及 {len(changed_files)} 个文件")

    if bad:
        by_reason = Counter(b[2] for b in bad)
        for reason, n in by_reason.most_common():
            print(f"   - {reason}: {n} 个")
        for p, i, why, line in bad[:10]:
            print(f"     {p.name}:{i}  [{why}]  {line}")

    print("\n[清洗后类别分布]")
    for i, name in enumerate(CLASSES[: args.nc]):
        n = ids_before.get(i, 0)
        flag = "   <- 0 个实例！" if n == 0 else ""
        print(f"   {i} {name:<9}{n:>6}{flag}")

    if not args.apply:
        print("\n[dry-run] 未修改任何文件。加 --apply 才会写回（会自动备份）。")
        if args.write_classes:
            print("[dry-run] 也不会写 classes.txt")
        return

    if bad:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bdir = Path(args.backup) if args.backup else root / f"_backup_{stamp}"
        bdir.mkdir(parents=True, exist_ok=True)
        for p, _ in changed_files:
            dst = bdir / p.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        print(f"\n[备份] 改动前的原文件 -> {bdir}")

        for p, keep in changed_files:
            # 末尾保留一个换行，符合 Ultralytics 习惯
            p.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        print(f"[写入] 已重写 {len(changed_files)} 个 txt")

    if args.write_classes:
        cpath = root / "classes.txt"
        cpath.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
        print(f"[classes] 已写入 {cpath}（{len(CLASSES)} 类）")

    print("\n完成。建议接着跑：python 01_vision/scripts/check_dataset.py")


if __name__ == "__main__":
    main()
