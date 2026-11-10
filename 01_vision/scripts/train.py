#!/usr/bin/env python3
"""训练 2025H 动物检测器（Ultralytics YOLO）。

用法：
  python 01_vision/scripts/train.py --epochs 60 --batch 16
  python 01_vision/scripts/train.py --model yolo11s.pt --epochs 100 --imgsz 960

产物（runs/<name>/）：
  weights/best.pt  最佳权重
  weights/last.pt  最后一轮
  summary.json     mAP / 每类 AP / 速度
  *_demo.jpg       在验证集上的预测可视化（--demo 张数）

数据集来自 scripts/synth_dataset.py 合成的 datasets/yolo/，配置见 configs/data.yaml。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def serial_scan_patch() -> None:
    """把 ultralytics 建标签缓存用的线程池换成串行。

    受限环境（如 DSH 沙箱）里 multiprocessing 的 Pipe/SimpleQueue 会被拒绝，
    报 WinError 5；900 张图串行扫描只要几秒，代价可以接受。
    """
    class _SerialPool:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def imap(self, func, iterable):
            for x in iterable:
                yield func(x)

    import ultralytics.data.dataset as ud
    import ultralytics.utils as uu

    ud.ThreadPool = _SerialPool
    if hasattr(uu, "ThreadPool"):
        uu.ThreadPool = _SerialPool


def resolve_data(data_yaml: Path, here: Path) -> Path:
    """把 data.yaml 里相对本文件所在目录的 path 解析成绝对路径。

    Ultralytics 会把相对 path 当成相对它的 datasets_dir（机器相关），所以这里先
    按"相对 yaml 文件"解析，再写一份 runs/data_resolved.yaml 给训练器用。
    仓库里保留相对路径，换机器不用改。
    """
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    p = Path(str(cfg.get("path", ".")))
    if not p.is_absolute():
        p = (data_yaml.parent / p).resolve()
    cfg["path"] = str(p)
    out = here / "runs" / "data_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"[data] {data_yaml.name} -> {p}")
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here / "01_vision/configs/data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="预训练权重或 yaml")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0' 用 GPU，'cpu' 用 CPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="2025h_synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--demo", type=int, default=12, help="验证集上出几张预测图，0 = 不出")
    ap.add_argument("--cache", action="store_true", help="把图片缓存进内存（数据小时更快，800 张 1280x960 约 3GB）")
    args = ap.parse_args()

    serial_scan_patch()
    from ultralytics import YOLO

    data_yaml = resolve_data(Path(args.data), here)
    run_dir = here / "runs" / args.name
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(here / "runs"),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        patience=args.patience,
        cos_lr=True,
        cache=args.cache,
        plots=True,
        verbose=True,
    )

    best = run_dir / "weights" / "best.pt"
    print(f"\n[训练完成] best.pt -> {best}")

    m = model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, device=args.device,
                  verbose=False)
    box = m.box
    names = getattr(m, "names", {}) or {}
    per_class = {}
    _maps = getattr(box, "maps", None)
    for i, ap_i in enumerate([] if _maps is None else _maps):
        per_class[names.get(i, str(i))] = round(float(ap_i), 4)

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_mAP50-95": per_class,
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in (m.speed or {}).items()},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.demo:
        demo_dir = here / "runs" / f"{args.name}_demo"
        model.predict(
            source=str(Path(args.data).resolve().parent.parent / "datasets/yolo/images/val"),
            imgsz=args.imgsz,
            conf=0.25,
            save=True,
            project=str(here / "runs"),
            name=f"{args.name}_demo",
            exist_ok=True,
            max_det=50,
            line_width=2,
        )
        print(f"[可视化] {demo_dir}")


if __name__ == "__main__":
    main()



