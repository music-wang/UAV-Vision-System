#!/usr/bin/env python3
"""鲁班猫 4（RK3588）上的 YOLO11 推理参考实现 —— rknnlite + 手写解码/NMS。

模型输出的实际情况（已用你导出的 best.onnx 实测）：
    output0: [1, 9, 8400]  float32
    第 0~3 通道 = xywh，**已经是 640 输入下的像素坐标**（不是归一化、不是格子单位）
    第 4~8 通道 = 5 类的分数，**已经过 sigmoid**（0~1）
也就是说 ultralytics 导出时已经把 DFL 和 sigmoid 烘进图里了，板端**不需要再解 DFL**，
只差：转置 → 阈值筛选 → xywh→xyxy → NMS。

用法（板端）：
  python rknn_inference.py --model best.rknn --source test.jpg
  python rknn_inference.py --model best.rknn --source ~/photos --out ~/result
  python rknn_inference.py --model best.rknn --source test.jpg --benchmark 100

依赖：rknnlite（板子出厂镜像一般自带）、opencv-python、numpy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
INPUT_SIZE = 640
PAD_COLOR = (114, 114, 114)


def letterbox(img: np.ndarray, size: int = INPUT_SIZE):
    """等比缩放 + 居中填充，返回 (图, 缩放比, (dw, dh))。与 ultralytics 一致。"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = img
    return canvas, r, (dw, dh)


def xywh2xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """单类 NMS（板端目标少，够用；要更快可换 cv2.dnn.NMSBoxes）。"""
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode(out: np.ndarray, conf_thr: float, iou_thr: float,
           ratio: float, pad: tuple[int, int], orig_hw: tuple[int, int]):
    """[1, 4+nc, N] -> [(x1,y1,x2,y2,score,cls)]，坐标已还原到原图。"""
    # rknn.inference / onnxruntime 都可能返回 [array]，统一剥成二维
    pred = out[0] if isinstance(out, (list, tuple)) else out
    pred = np.squeeze(np.asarray(pred))
    if pred.ndim != 2:
        raise ValueError(f"期望二维输出，实际 {pred.shape}")
    if pred.shape[0] < pred.shape[1]:          # (4+nc, N) -> (N, 4+nc)
        pred = pred.T
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]
    cls_id = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(cls_id)), cls_id]

    mask = scores >= conf_thr
    if not mask.any():
        return []
    boxes = xywh2xyxy(boxes_xywh[mask])
    scores = scores[mask]
    cls_id = cls_id[mask]

    # 去掉 letterbox 的缩放和填充，回到原图坐标
    dw, dh = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    h, w = orig_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)

    dets = []
    for c in np.unique(cls_id):
        idx = np.where(cls_id == c)[0]
        for k in nms(boxes[idx], scores[idx], iou_thr):
            b = boxes[idx][k]
            dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                         float(scores[idx][k]), int(c)))
    dets.sort(key=lambda d: -d[4])
    return dets


def draw(img: np.ndarray, dets, names) -> np.ndarray:
    for x1, y1, x2, y2, s, c in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0, 200, 0), 2)
        label = f"{names[c] if c < len(names) else c} {s:.2f}"
        cv2.putText(img, label, (p1[0], max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
    return img


def run_npu(rknn, inp: np.ndarray):
    """显式声明 NHWC 输入。

    模型元数据里布局标注有两处（内部 NHWC、原始 attrs NCHW）且不一致，
    显式声明可消除歧义；老版本 rknnlite 不接受该参数时自动回退。
    """
    try:
        return rknn.inference(inputs=[inp], data_format="nhwc")
    except TypeError:
        return rknn.inference(inputs=[inp])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="best.rknn")
    ap.add_argument("--source", required=True, help="图片 / 目录 / 视频")
    ap.add_argument("--out", default="", help="结果输出目录，默认不保存")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--core", default="012", help="NPU 核心：0/1/2/012")
    ap.add_argument("--benchmark", type=int, default=0, help="重复推理 N 次测速")
    ap.add_argument("--names", default=",".join(CLASSES))
    ap.add_argument("--json", default="", help="把检出结果写成 json，便于和 PC 端逐框对比")
    args = ap.parse_args()

    names = [s.strip() for s in args.names.split(",")]

    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        raise SystemExit("load_rknn 失败")
    masks = {"0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1,
             "2": RKNNLite.NPU_CORE_2, "012": RKNNLite.NPU_CORE_0_1_2}
    if rknn.init_runtime(core_mask=masks.get(args.core, RKNNLite.NPU_CORE_0_1_2)) != 0:
        raise SystemExit("init_runtime 失败")
    print(f"[ok] 模型已加载，NPU 核心 {args.core}")

    src = Path(args.source)
    files = ([src] if src.is_file() and src.suffix.lower() in IMG_EXT
             else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
             if src.is_dir() else [src])

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_ms = 0.0
    results: list[dict] = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        # cv2 读进来是 BGR，训练用的是 RGB —— 不转通道盒子还能对上，但类别会全错
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lb, ratio, pad = letterbox(rgb)
        inp = lb[None, ...]                       # NHWC uint8，模型内部做 /255

        t0 = time.perf_counter()
        out = run_npu(rknn, inp)
        dt = (time.perf_counter() - t0) * 1000

        n = args.benchmark if args.benchmark else 1
        if n > 1:
            ts = []
            for _ in range(n):
                t1 = time.perf_counter()
                run_npu(rknn, inp)
                ts.append((time.perf_counter() - t1) * 1000)
            print(f"[bench] {np.mean(ts):.1f} ms/帧  ({1000 / np.mean(ts):.1f} FPS)   "
                  f"min {np.min(ts):.1f}  max {np.max(ts):.1f}")
            break

        dets = decode(out, args.conf, args.iou, ratio, pad, (h, w))
        total_ms += dt
        print(f"[{i + 1}/{len(files)}] {f.name}  推理 {dt:.1f} ms  检出 {len(dets)}")
        for x1, y1, x2, y2, s, c in dets:
            print(f"    {names[c]:<9} {s:.2f}  ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
        if out_dir:
            cv2.imwrite(str(out_dir / f"{Path(f).stem}_pred.jpg"), draw(img.copy(), dets, names))
        if args.json:
            results.append({"image": f.name,
                            "detections": [{"cls": names[c] if c < len(names) else str(c),
                                            "score": round(s, 4),
                                            "xyxy": [round(x1, 1), round(y1, 1),
                                                     round(x2, 1), round(y2, 1)]}
                                           for x1, y1, x2, y2, s, c in dets]})

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"[json] {args.json}")
    if args.benchmark == 0 and len(files) > 1:
        print(f"[合计] {len(files)} 帧，平均 {total_ms / max(len(files), 1):.1f} ms/帧")
    rknn.release()


if __name__ == "__main__":
    main()



