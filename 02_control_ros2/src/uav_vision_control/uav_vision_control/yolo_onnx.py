#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO11 ONNX 推理与解码（与 01_vision/deploy/rknn_inference.py 保持同一套逻辑）。

同时支持两种导出：
  A) 单输出 [1, 4+nc, 8400]                       （ultralytics 原始导出）
  B) 双输出 boxes[1,4,8400] + scores[1,5,8400]    （split_output_onnx.py 拆分版）
"""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]
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
    """单类 NMS。"""
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


def decode(out, conf_thr: float, iou_thr: float,
           ratio: float, pad: tuple[int, int], orig_hw: tuple[int, int]):
    """模型输出 -> [(x1, y1, x2, y2, score, cls_id)]，坐标已还原到原图。"""
    if isinstance(out, (list, tuple)) and len(out) == 2:
        b = np.squeeze(np.asarray(out[0])).astype(np.float32)
        sc = np.squeeze(np.asarray(out[1])).astype(np.float32)
        if b.ndim != 2 or sc.ndim != 2:
            raise ValueError(f"期望二维输出，实际 {b.shape} / {sc.shape}")
        if b.shape[0] == 4:
            b = b.T
        if sc.shape[0] < sc.shape[1]:
            sc = sc.T
        pred = np.concatenate([b, sc], axis=1)
    else:
        pred = out[0] if isinstance(out, (list, tuple)) else out
        pred = np.squeeze(np.asarray(pred)).astype(np.float32)
        if pred.ndim != 2:
            raise ValueError(f"期望二维输出，实际 {pred.shape}")
        if pred.shape[0] < pred.shape[1]:
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


class YoloOnnx:
    """onnxruntime 封装：输入 BGR 帧，输出模型原始输出 + letterbox 参数。"""

    def __init__(self, model_path: str, num_threads: int = 4):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def infer(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lb, ratio, pad = letterbox(rgb)
        blob = lb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        outs = self.session.run(None, {self.input_name: blob})
        return outs, ratio, pad
