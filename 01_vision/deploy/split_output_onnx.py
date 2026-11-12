#!/usr/bin/env python3
"""把 YOLO ONNX 的单个输出 [1, 4+nc, 8400] 拆成两个输出：boxes 和 scores。

为什么要拆：
  RKNN 对输出张量做 **per-tensor** 量化。原图最后是
      Concat(box[0~640 px], score[0~1])  -> output0
  框的量程把整张张量的 scale 撑到约 640/127≈5，于是分数 0.46 除以 5 再取整就是 0
  —— 这就是 int8 下"类别分数全 0、框还在"的机制。
  拆开后 boxes 用自己的 scale（精度 ±2.5px@640），scores 用自己的 scale（±0.004），
  互不干扰，**不需要 hybrid 量化，也不需要改模型结构**。

用法：
  python split_output_onnx.py --onnx best.onnx
  python split_output_onnx.py --onnx best.onnx --out best_2out.onnx --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


def find_output_concat(model: onnx.ModelProto):
    """找到产出唯一输出的那个节点（通常是 Concat，输入 = 框张量 + 分数张量）。"""
    if len(model.graph.output) != 1:
        raise SystemExit(f"期望 1 个输出，实际 {len(model.graph.output)} 个，"
                         f"这个模型可能已经是拆分版")
    out_name = model.graph.output[0].name
    for node in model.graph.node:
        if out_name in node.output:
            return node, out_name
    raise SystemExit(f"找不到产出 {out_name} 的节点")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true", help="用 onnxruntime 对比拆分前后输出")
    ap.add_argument("--verify-image", default="", help="验证用的图片，默认用随机张量")
    args = ap.parse_args()

    src = Path(args.onnx)
    if not src.is_file():
        raise SystemExit(f"文件不存在：{src}")
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_2out.onnx")

    model = onnx.load(str(src))
    node, out_name = find_output_concat(model)

    if node.op_type != "Concat" or len(node.input) != 2:
        raise SystemExit(f"输出节点是 {node.op_type} / {len(node.input)} 输入，"
                         f"这个脚本只处理 Concat(框, 分数) 的情况")

    # 形状推断，拿到两个中间张量的确切形状（RKNN 需要静态 shape）
    inferred = onnx.shape_inference.infer_shapes(model)
    vinfo = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}

    box_t, score_t = node.input[0], node.input[1]
    missing = [t for t in (box_t, score_t) if t not in vinfo]
    if missing:
        raise SystemExit(f"形状推断拿不到这些张量：{missing}")

    def vi(name: str, new_name: str):
        v = vinfo[name]
        dims = [d.dim_value if d.HasField("dim_value") else d.dim_param
                for d in v.type.tensor_type.shape.dim]
        return helper.make_tensor_value_info(new_name, onnx.TensorProto.FLOAT, dims), dims

    box_vi, box_dims = vi(box_t, "boxes")
    score_vi, score_dims = vi(score_t, "scores")
    print(f"[图] 输出节点 {node.name}")
    print(f"     boxes  <- {box_t}  {box_dims}")
    print(f"     scores <- {score_t}  {score_dims}")

    # 改图：删掉合并节点，用 Identity 把两个中间张量变成带正常名字的输出
    # （直接改 output 名字会触发 checker 报错：output 必须是某个节点的输出）
    model.graph.node.remove(node)
    model.graph.node.extend([
        helper.make_node("Identity", [box_t], ["boxes"], name="rename_boxes"),
        helper.make_node("Identity", [score_t], ["scores"], name="rename_scores"),
    ])
    del model.graph.output[:]
    model.graph.output.extend([box_vi, score_vi])

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    size = dst.stat().st_size / 2 ** 20
    print(f"[写出] {dst}  {size:.2f} MB  输出: boxes{box_dims} + scores{score_dims}")

    if not args.verify:
        return

    # ---------- 验证：拆分前后解码结果必须一致 ----------
    import onnxruntime as ort
    from PIL import Image

    sess_old = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    sess_new = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])

    if args.verify_image:
        img = Image.open(args.verify_image).convert("RGB").resize((640, 640))
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    else:
        rng = np.random.default_rng(0)
        x = rng.random((1, 3, 640, 640), dtype=np.float32)

    old = np.squeeze(sess_old.run(None, {"images": x})[0])
    b, s = [np.squeeze(a) for a in sess_new.run(None, {"images": x})]
    new = np.concatenate([b, s], axis=0)

    diff = float(np.abs(old - new).max())
    print(f"[验证] 拼接后与原输出最大绝对误差 = {diff:.3e}")
    if diff > 1e-5:
        print("[X] 不一致，不要用这个文件", file=sys.stderr)
        raise SystemExit(2)
    print("[OK] 拆分前后数值完全一致；boxes/scores 各自独立量化，int8 不会再吃掉分数")


if __name__ == "__main__":
    main()
