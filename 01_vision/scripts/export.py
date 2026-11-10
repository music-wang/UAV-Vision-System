#!/usr/bin/env python3
"""把训练好的权重导出成部署格式。

用法：
  python 01_vision/scripts/export.py                                  # ONNX
  python 01_vision/scripts/export.py --format engine --half            # TensorRT FP16（本机 GPU）
  python 01_vision/scripts/export.py --format onnx --imgsz 960 --dynamic

说明：
  - ONNX 是通用中转格式，RKNN / TensorRT / OpenVINO 一般都要先过 ONNX；
  - engine（TensorRT）与机器/驱动绑定，换机器要重导；
  - int8 需要校准数据（ultralytics 会用数据集里的图片，务必在真实分布上调）；
  - rknn 需要额外安装 rknn-toolkit2，且只能在 x86 上做转换。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(here / "runs/2025h_synth/weights/best.pt"))
    ap.add_argument("--format", default="onnx",
                    choices=["onnx", "engine", "torchscript", "openvino", "ncnn", "rknn"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--simplify", action="store_true", default=True)
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"权重不存在：{w}（先跑 train.py）")

    from ultralytics import YOLO

    model = YOLO(str(w))
    out = model.export(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )
    print(f"[导出完成] {args.format} -> {out}")


if __name__ == "__main__":
    main()
