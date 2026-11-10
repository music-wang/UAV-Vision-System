#!/usr/bin/env python3
"""实拍验证：拿真实照片 + 你手标的每图目标数，对比模型检出，给出误差报告。

为什么不做"归格"而是先比对"每张图各类目标数"：
  实拍照片通常没有位姿（手机/相机随手拍），没法把像素投回场地坐标。
  先把"检测准不准"这件事独立验证清楚，归格等真的飞起来有 SLAM 位姿了再测
  （那时用 eval_mission.py 那套）。

用法：

  # 1) 先给图片目录生成一张空的标注表（Excel/WPS 都能填）
  python 01_vision/scripts/verify_real.py --images D:\\photos --init-labels

  # 2) 填好 labels.csv（每张图里各类别的个数，不填=0），然后跑对比
  python 01_vision/scripts/verify_real.py --images D:\\photos --labels D:\\photos\\labels.csv

  # 3) 如果你有位姿和整场真值，可以再叠加格级验证
  python 01_vision/scripts/verify_real.py --images D:\\photos --labels labels.csv `
      --pose pose.csv --gt-cells gt_cells.csv --fov 66

输出（runs/verify_real/）：
  annotated/*.jpg   画了框的图
  report.csv        逐图逐类的 手工值 / 预测值 / 差值
  summary.json      各类总误差、完全一致的图占比、漏检/多检图数（含格级指标）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def read_table(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_int(v) -> int:
    try:
        return int(float(str(v).strip() or 0))
    except Exception:
        return 0


def init_labels(images: list[Path], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", *CLASSES])
        for p in images:
            w.writerow([p.name, 0, 0, 0, 0, 0])
    print(f"[模板] {out}  共 {len(images)} 行，用 Excel 填每张图里各类个数（没有就留 0）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="实拍图片目录")
    ap.add_argument("--labels", default="", help="手工标注 csv：image + 各类列")
    ap.add_argument("--init-labels", action="store_true", help="生成空标注表后退出")
    ap.add_argument("--weights", default=str(HERE / "runs/2025h_synth/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(HERE / "runs/verify_real"))
    # 可选：位姿 + 整场真值，走格级验证
    ap.add_argument("--pose", default="", help="pose.csv: image,x_cm,y_cm,yaw_deg,alt_cm")
    ap.add_argument("--gt-cells", default="", help="gt_cells.csv: cell,各类列")
    ap.add_argument("--fov", type=float, default=66.0)
    args = ap.parse_args()

    images = list_images(Path(args.images))
    if not images:
        raise SystemExit(f"目录里没有图片：{args.images}")
    print(f"[输入] {len(images)} 张图")

    if args.init_labels:
        init_labels(images, Path(args.labels or (Path(args.images) / "labels.csv")))
        return
    if not args.labels:
        raise SystemExit("缺少 --labels（先用 --init-labels 生成模板）")

    manual: dict[str, dict[str, int]] = {}
    for row in read_table(Path(args.labels)):
        key = Path(str(row.get("image", "")).strip()).stem
        if not key:
            continue
        manual[key] = {c: to_int(row.get(c, 0)) for c in CLASSES}

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    (out_dir / "annotated").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    pred_cells: dict[str, dict[str, int]] = {}
    mapper = None
    poses: dict[str, tuple[float, float, float, float]] = {}
    if args.pose and args.gt_cells:
        from cell_mapper import CellMapper

        mapper = CellMapper(fov_deg=args.fov, min_conf=args.conf, confirm_hits=1)
        for row in read_table(Path(args.pose)):
            k = Path(str(row.get("image", "")).strip()).stem
            poses[k] = (float(row["x_cm"]), float(row["y_cm"]),
                        float(row["yaw_deg"]), float(row.get("alt_cm", 120)))

    per_class_err = Counter()
    exact_images = 0
    miss_images = 0
    over_images = 0

    for p in images:
        stem = p.stem
        r = model.predict(str(p), imgsz=args.imgsz, conf=args.conf, verbose=False,
                          save=True, project=str(out_dir), name="annotated",
                          exist_ok=True, max_det=50, line_width=2)[0]
        pred = Counter()
        if r.boxes is not None:
            for c in r.boxes.cls.tolist():
                pred[model.names[int(c)]] += 1
        got = {c: pred.get(c, 0) for c in CLASSES}
        want = manual.get(stem, {c: 0 for c in CLASSES})

        if got == want:
            exact_images += 1
        if any(got[c] > want[c] for c in CLASSES):
            over_images += 1
        if any(got[c] < want[c] for c in CLASSES):
            miss_images += 1
        for c in CLASSES:
            per_class_err[c] += abs(got[c] - want[c])
            rows.append({"image": p.name, "class": c, "manual": want[c],
                         "pred": got[c], "diff": got[c] - want[c]})

        if mapper is not None and stem in poses:
            dets = [(int(b.cls[0]), *((b.xyxy[0][:2] + b.xyxy[0][2:]) / 2).tolist(),
                     float(b.conf[0])) for b in (r.boxes if r.boxes is not None else [])]
            x, y, yaw, alt = poses[stem]
            mapper.update(dets, (x, y, yaw), alt_cm=alt)

    with open(out_dir / "report.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "class", "manual", "pred", "diff"])
        w.writeheader()
        w.writerows(rows)

    n = len(images)
    summary = {
        "images": n,
        "weights": str(args.weights),
        "conf": args.conf,
        "per_class_abs_error": dict(per_class_err),
        "per_class_mae_per_image": {c: round(per_class_err[c] / n, 3) for c in CLASSES},
        "total_manual": sum(c["manual"] for c in rows),
        "total_pred": sum(c["pred"] for c in rows),
        "exact_match_images": exact_images,
        "exact_match_rate": round(exact_images / n, 4),
        "images_with_false_positive": over_images,
        "images_with_miss": miss_images,
    }

    if mapper is not None:
        pred_cells = mapper.report()
        gt = {}
        for row in read_table(Path(args.gt_cells)):
            cell = str(row.get("cell", "")).strip()
            if cell:
                gt[cell] = {c: to_int(row.get(c, 0)) for c in CLASSES}
        tp = fp = fn = 0
        for cell in set(pred_cells) | set(gt):
            for c in CLASSES:
                p = pred_cells.get(cell, {}).get(c, 0)
                g = gt.get(cell, {}).get(c, 0)
                tp += min(p, g)
                fp += max(p - g, 0)
                fn += max(g - p, 0)
        summary["cell_level"] = {"tp": tp, "fp": fp, "fn": fn,
                                 "accuracy": round(tp / max(tp + fp + fn, 1), 4)}
        summary["predicted_cells"] = pred_cells
        summary["gt_cells"] = gt

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    print(f"\n[逐类] 图数 {n}   完全一致 {exact_images} 张（{summary['exact_match_rate']:.1%}）")
    print(f"{'类别':<10}{'手工总数':>8}{'预测总数':>8}{'总绝对误差':>10}{'每图MAE':>9}")
    for c in CLASSES:
        print(f"{c:<10}{sum(r['manual'] for r in rows if r['class'] == c):>8}"
              f"{sum(r['pred'] for r in rows if r['class'] == c):>8}"
              f"{per_class_err[c]:>10}{summary['per_class_mae_per_image'][c]:>9}")
    print(f"\n漏检图 {miss_images} 张，多检图 {over_images} 张")
    if "cell_level" in summary:
        cl = summary["cell_level"]
        print(f"格级：TP={cl['tp']} FP={cl['fp']} FN={cl['fn']}  正确率 {cl['accuracy']:.3f}")
    print(f"\n[输出] {out_dir}（annotated/ , report.csv, summary.json）")


if __name__ == "__main__":
    main()
