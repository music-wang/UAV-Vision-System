#!/usr/bin/env python3
"""单独出预测图：训练完可以重复跑，不用重新训练。

用法：
  python 01_vision/scripts/predict_demo.py                        # 验证集全部
  python 01_vision/scripts/predict_demo.py --limit 12            # 只出 12 张
  python 01_vision/scripts/predict_demo.py --weights runs/2025h_synth/weights/best.pt --conf 0.3
  python 01_vision/scripts/predict_demo.py --source datasets/mission   # 也可指定任意图片目录

结果保存在 runs/<name>_demo/，并打印各类别检出数量。
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from train import resolve_data


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(here / "runs/2025h_synth/weights/best.pt"))
    ap.add_argument("--data", default=str(here / "01_vision/configs/data_real.yaml"),
                    help="用它的 path/val 作为图片目录")
    ap.add_argument("--source", default="", help="直接指定图片目录，优先于 --data")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="最多出几张，0 = 全部")
    ap.add_argument("--name", default="2025h_synth_demo")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"权重不存在：{w}")

    if args.source:
        src = Path(args.source)
    else:
        import yaml

        cfg = yaml.safe_load(resolve_data(Path(args.data), here).read_text(encoding="utf-8"))
        src = Path(str(cfg["path"])) / str(cfg.get("val", "images/val"))
    if not src.is_dir():
        raise SystemExit(f"图片目录不存在：{src}")

    from ultralytics import YOLO

    model = YOLO(str(w))
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if args.limit:
        files = files[: args.limit]
    print(f"[输入] {src}  共 {len(files)} 张")

    counter: Counter[str] = Counter()
    for f in files:
        r = model.predict(str(f), imgsz=args.imgsz, conf=args.conf, verbose=False,
                          save=True, project=str(here / "runs"), name=args.name,
                          exist_ok=True, max_det=50, line_width=2)[0]
        for c in r.boxes.cls.tolist() if r.boxes is not None else []:
            counter[model.names[int(c)]] += 1

    print(f"[输出] {here / 'runs' / args.name}")
    if counter:
        print("[检出] " + "  ".join(f"{k}×{v}" for k, v in sorted(counter.items())))
    else:
        print("[检出] 无目标（检查 --conf 或权重）")


if __name__ == "__main__":
    main()
