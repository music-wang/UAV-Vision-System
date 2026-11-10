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
    ap.add_argument("--opset", type=int, default=19,
                    help="ONNX opset；转 RKNN 时若报不支持，降到 12 或 13")
    ap.add_argument("--batch", type=int, default=1, help="静态 batch；RKNN 只支持 1")
    ap.add_argument("--half", action="store_true", help="FP16（RKNN 路径不要开）")
    ap.add_argument("--int8", action="store_true", help="需要 --data 校准集")
    ap.add_argument("--dynamic", action="store_true", help="动态 shape（RKNN 不支持）")
    ap.add_argument("--simplify", action="store_true", default=True)
    ap.add_argument("--data", default="", help="int8 校准用 data.yaml，默认自动解析 configs/data_real.yaml")
    args = ap.parse_args()

    w = Path(args.weights)
    if not w.is_file():
        raise SystemExit(f"权重不存在：{w}（先跑 train.py）")

    kwargs = dict(
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        batch=args.batch,
    )
    if args.int8:
        from train import resolve_data

        data_yaml = Path(args.data) if args.data else here / "01_vision/configs/data_real.yaml"
        kwargs["data"] = str(resolve_data(data_yaml, here))

    from ultralytics import YOLO

    model = YOLO(str(w))
    out = model.export(**kwargs)
    print(f"[导出完成] {args.format} -> {out}")
    if args.format == "onnx":
        print("提示：转 RKNN 需要 静态 shape + batch=1；int8 量化要带 --data 校准集。")


if __name__ == "__main__":
    main()
