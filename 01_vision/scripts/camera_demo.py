#!/usr/bin/env python3
"""用电脑摄像头实时看检测效果（调试用，比赛链路不走这个）。

用法：
  python 01_vision/scripts/camera_demo.py                      # 默认摄像头 0 + best.pt
  python 01_vision/scripts/camera_demo.py --cam 1 --conf 0.4
  python 01_vision/scripts/camera_demo.py --model runs/2025h_synth/weights/best.onnx

按键：q 退出，s 存一张当前画面到 runs/camera_shots/
窗口里显示每帧耗时和 FPS，方便初步判断端侧算力够不够。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "runs/2025h_synth/weights/best.pt"))
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit(f"打不开摄像头 {args.cam}")

    shots = HERE / "runs" / "camera_shots"
    print("运行中：q 退出，s 存图")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读不到画面，退出")
                break
            t0 = time.perf_counter()
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            ms = (time.perf_counter() - t0) * 1000

            n = 0 if r.boxes is None else len(r.boxes)
            cv2.putText(r.plot(), f"{ms:.0f} ms  {1000 / max(ms, 1):.1f} FPS  objs={n}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("UAV-Vision-System", r.plot())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                shots.mkdir(parents=True, exist_ok=True)
                p = shots / f"shot_{int(time.time())}.jpg"
                cv2.imwrite(str(p), r.plot())
                print(f"已保存 {p}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
