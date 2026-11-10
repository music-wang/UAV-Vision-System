#!/usr/bin/env python3
"""跑通"检测 → 归格 → 去重计数"，并与场地真值对比。

输入：sim_mission.py 产出的 mission 目录（images/ + frames.json + gt.json）
输出：控制台对比表 + report.json

用法：
  python 01_vision/scripts/eval_mission.py --weights runs/2025h_synth/weights/best.pt
  python 01_vision/scripts/eval_mission.py --weights runs/2025h_synth/weights/best.pt --conf 0.3

真值来自场地布置（每格贴了什么），检测来自模型，位姿来自 frames.json
（真实系统里这一项换成 SLAM 的 /uav/pose，其余代码不用改）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_mapper import CellMapper, detections_from_yolo  # noqa: E402


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(here / "runs/2025h_synth/weights/best.pt"))
    ap.add_argument("--mission", default=str(here / "datasets/mission"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--fov", type=float, default=66.0)
    ap.add_argument("--match-cm", type=float, default=25.0)
    ap.add_argument("--confirm-hits", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 帧，0 = 全部")
    args = ap.parse_args()

    mission = Path(args.mission)
    frames = json.loads((mission / "frames.json").read_text(encoding="utf-8"))
    gt = json.loads((mission / "gt.json").read_text(encoding="utf-8"))
    scene = json.loads((mission / "scene.json").read_text(encoding="utf-8"))
    if args.limit:
        frames = frames[: args.limit]

    from PIL import Image
    from ultralytics import YOLO

    with Image.open(mission / "images" / frames[0]["file"]) as im:
        size = im.size

    model = YOLO(args.weights)
    mapper = CellMapper(fov_deg=args.fov, size=size, match_cm=args.match_cm,
                        confirm_hits=args.confirm_hits, min_conf=args.conf)

    t0 = time.perf_counter()
    total_det = 0
    for fr in frames:
        r = model.predict(str(mission / "images" / fr["file"]), imgsz=args.imgsz,
                          conf=args.conf, verbose=False)[0]
        dets = detections_from_yolo(r, conf=args.conf)
        total_det += len(dets)
        mapper.update(dets, (fr["x_cm"], fr["y_cm"], fr["yaw_deg"]), alt_cm=fr["alt_cm"])
    dt = time.perf_counter() - t0

    pred = mapper.report()
    report = {
        "frames": len(frames),
        "detections": total_det,
        "seconds": round(dt, 2),
        "ms_per_frame": round(dt / max(len(frames), 1) * 1000, 1),
        "predicted": pred,
        "ground_truth": gt,
        "no_fly": scene.get("no_fly", []),
    }

    # 逐格逐类对比
    keys = sorted({(c, s) for c in (set(pred) | set(gt))
                   for s in (set(pred.get(c, {})) | set(gt.get(c, {})))})
    tp = fp = fn = 0
    rows = []
    for c, s in keys:
        p = pred.get(c, {}).get(s, 0)
        g = gt.get(c, {}).get(s, 0)
        tp += min(p, g)
        fp += max(p - g, 0)
        fn += max(g - p, 0)
        if p != g:
            rows.append((c, s, g, p))

    report["tp"], report["fp"], report["fn"] = tp, fp, fn
    report["cell_accuracy"] = round(tp / max(tp + fp + fn, 1), 4)
    (mission / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print(f"帧数 {len(frames)}  检测框 {total_det}  {report['ms_per_frame']} ms/帧")
    print(f"格级正确率 {report['cell_accuracy']:.3f}   TP={tp} FP={fp} FN={fn}")
    print("\n真值：", json.dumps(gt, ensure_ascii=False))
    print("预测：", json.dumps(pred, ensure_ascii=False))
    if rows:
        print("\n不一致的格：")
        for c, s, g, p in rows:
            print(f"  {c:6s} {s:9s} 真值={g} 预测={p}")
    else:
        print("\n所有格与真值一致 ✅")


if __name__ == "__main__":
    main()
