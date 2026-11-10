#!/usr/bin/env python3
"""在 Ubuntu x86_64 上把 ultralytics 导出的 best.onnx 转成 RK3588 可用的 best.rknn。

必须在 x86_64 Linux + rknn-toolkit2 环境里跑（Windows / ARM 都不行）。

用法：
  python convert_rknn.py --onnx best.onnx --out best.rknn --calib-dir ~/calib
  python convert_rknn.py --onnx best.onnx --out best_fp.rknn --no-quant     # 先验 FP16
  python convert_rknn.py --onnx best.onnx --out best.rknn --calib-dir ~/calib --analysis

要点：
  * 输入按 RGB uint8 喂（ultralytics 训练用 RGB），所以 mean=0 / std=255；
  * 量化校准集用**真实场地照片**最好，没有就用合成集 val（100~300 张足够）；
  * 先 --no-quant 跑通、确认精度，再开 int8；int8 掉点明显时把 --analysis 的敏感层
    加进 hybrid 量化（见 README）。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def build_calib(calib_dir: Path, out_txt: Path, limit: int, seed: int = 0) -> int:
    files = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"校准目录里没有图片：{calib_dir}")
    random.Random(seed).shuffle(files)
    files = files[:limit]
    out_txt.write_text("\n".join(str(p.resolve()) for p in files) + "\n", encoding="utf-8")
    return len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True, help="输出的 .rknn 路径")
    ap.add_argument("--calib-dir", default="", help="量化校准图片目录")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--no-quant", action="store_true", help="不做 int8 量化（FP16）")
    ap.add_argument("--platform", default="rk3588")
    ap.add_argument("--analysis", action="store_true", help="导出逐层量化误差分析")
    ap.add_argument("--analysis-image", default="", help="分析用的单张图，默认校准集第一张")
    args = ap.parse_args()

    from rknn.api import RKNN

    onnx = Path(args.onnx).resolve()
    if not onnx.is_file():
        raise SystemExit(f"onnx 不存在：{onnx}")
    out = Path(args.out).resolve()

    calib_txt = None
    if not args.no_quant:
        if not args.calib_dir:
            raise SystemExit("int8 量化需要 --calib-dir（或先用 --no-quant）")
        calib_txt = Path("calib.txt").resolve()
        n = build_calib(Path(args.calib_dir), calib_txt, args.calib_n)
        print(f"[校准] {n} 张 -> {calib_txt}")
        if n < 50:
            print("  ⚠ 校准图少于 50 张，量化误差可能偏大")

    rknn = RKNN(verbose=True)
    print("[config] mean=[0,0,0] std=[255,255,255] RGB")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        quant_img_RGB2BGR=False,      # 训练是 RGB，这里必须保持 RGB
        optimization_level=3,
    )

    print(f"[load] {onnx}")
    if rknn.load_onnx(model=str(onnx)) != 0:
        raise SystemExit("load_onnx 失败")

    print("[build] ...")
    if rknn.build(do_quantization=not args.no_quant,
                  dataset=str(calib_txt) if calib_txt else None,
                  rknn_batch_size=1) != 0:
        raise SystemExit("build 失败（多半是量化数据集或算子不支持，试 --no-quant）")

    if args.analysis:
        img = args.analysis_image or str(
            next(p for p in Path(args.calib_dir).rglob("*") if p.suffix.lower() in IMG_EXT))
        print(f"[analysis] 对照图 {img}")
        rknn.accuracy_analysis(inputs=[img], output_dir="./snapshot")

    print(f"[export] {out}")
    if rknn.export_rknn(str(out)) != 0:
        raise SystemExit("export_rknn 失败")
    rknn.release()

    size = out.stat().st_size / 2**20
    print(f"\n完成：{out}  {size:.1f} MiB")
    print("板端自检：python rknn_inference.py --model %s --source test.jpg" % out.name)


if __name__ == "__main__":
    main()
