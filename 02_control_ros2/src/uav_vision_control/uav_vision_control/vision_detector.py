#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉检测节点：USB 摄像头 -> YOLO11(ONNX) -> /vision/detection。

运行（PC 调试，onnxruntime）：
  ros2 run uav_vision_control vision_detector --ros-args \
      -p model_path:=/home/andy/uav/best_2out.onnx -p video_source:=0

运行（鲁班猫 4，NPU）：
  ros2 run uav_vision_control vision_detector --ros-args \
      -p backend:=rknn -p model_path:=/home/cat/uav/best_int8.rknn -p video_source:=0

参数：
  backend        str   onnx | rknn，默认 onnx（板上用 rknn 走 NPU）
  model_path     str   必需，.onnx 或 .rknn 路径（单输出/双输出都支持）
  rknn_core      str   rknn 后端的 NPU 核心：0/1/2/012，默认 012（三核）
  classes_path   str   可选，每行一个类别名；默认内置 5 类
  video_source   str   摄像头序号 "0" / 视频文件 / 图片路径（图片用于离线调试）
  conf_threshold float 置信度阈值，默认 0.25
  iou_threshold  float NMS 阈值，默认 0.45
  publish_rate   float 定时器频率 Hz，默认 10.0
  frame_id       str   Header 里的坐标系，默认 "camera"
  num_threads    int   onnxruntime 线程数，默认 4
"""
from __future__ import annotations

import os

import cv2
import rclpy
from rclpy.node import Node
from uav_vision_interfaces.msg import Detection, DetectionArray

from .yolo_onnx import CLASSES, create_backend, decode

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class VisionDetector(Node):
    def __init__(self):
        super().__init__("vision_detector")

        self.declare_parameter("backend", "onnx")
        self.declare_parameter("rknn_core", "012")
        self.declare_parameter("model_path", "")
        self.declare_parameter("classes_path", "")
        self.declare_parameter("video_source", "0")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("frame_id", "camera")
        self.declare_parameter("num_threads", 4)

        model_path = self.get_parameter("model_path").value
        if not model_path or not os.path.isfile(model_path):
            raise SystemExit(
                f"模型不存在：{model_path!r}\n"
                "onnx 后端传 .onnx（用 -p backend:=onnx）；"
                "板上用 -p backend:=rknn 并传 .rknn")

        self.conf = float(self.get_parameter("conf_threshold").value)
        self.iou = float(self.get_parameter("iou_threshold").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.classes = self._load_classes(self.get_parameter("classes_path").value)

        backend = str(self.get_parameter("backend").value).lower()
        model_path = str(model_path)
        if backend == "rknn" and not model_path.lower().endswith(".rknn"):
            self.get_logger().warning(f"backend=rknn 但模型不是 .rknn：{model_path}")
        self.get_logger().info(f"加载模型（backend={backend}）：{model_path}")
        self.backend = backend
        self.model = create_backend(
            backend, model_path,
            num_threads=int(self.get_parameter("num_threads").value),
            rknn_core=str(self.get_parameter("rknn_core").value))

        self._open_source(str(self.get_parameter("video_source").value))

        self.pub = self.create_publisher(DetectionArray, "/vision/detection", 10)
        rate = max(float(self.get_parameter("publish_rate").value), 0.1)
        self.timer = self.create_timer(1.0 / rate, self.timer_cb)

        self._frames = 0
        self._last_log = self.get_clock().now()
        self.get_logger().info("节点就绪，发布 /vision/detection")

    def _load_classes(self, path: str):
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                names = [ln.strip() for ln in f if ln.strip()]
            if names:
                self.get_logger().info(f"类别来自 {path}（{len(names)} 类）")
                return names
        return list(CLASSES)

    def _open_source(self, src: str):
        self.capture = None
        self.static_image = None

        if os.path.isfile(src) and src.lower().endswith(IMAGE_EXTS):
            img = cv2.imread(src)
            if img is None:
                raise SystemExit(f"图片读不出来：{src}")
            self.static_image = img
            self.get_logger().info(f"静态图片模式：{src} {img.shape[1]}x{img.shape[0]}")
            return

        # 数字 = 摄像头序号；其它 = 视频文件
        try:
            cam_id = int(src)
        except ValueError:
            cam_id = None

        if cam_id is not None:
            self.capture = cv2.VideoCapture(cam_id)
            if not self.capture.isOpened():
                raise SystemExit(f"打不开摄像头 {cam_id}，检查 USB 连接和 /dev/video* 权限")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            # 只保留最新一帧：否则 10Hz 轮询 30fps 相机会一直读到排队里的旧帧，
            # 无人机移动时检测坐标会明显滞后
            try:
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:       # noqa: BLE001 - 某些后端不支持该属性
                pass
            self.get_logger().info(f"摄像头 {cam_id} 已打开")
        else:
            self.capture = cv2.VideoCapture(src)
            if not self.capture.isOpened():
                raise SystemExit(f"打不开视频源：{src}")
            self.get_logger().info(f"视频文件模式：{src}")

    def timer_cb(self):
        if self.static_image is not None:
            frame = self.static_image.copy()
        else:
            ok, frame = self.capture.read()
            if not ok:
                self.get_logger().warning("读帧失败", throttle_duration_sec=2.0)
                return

        try:
            outs, ratio, pad = self.model.infer(frame)
            h, w = frame.shape[:2]
            dets = decode(outs, self.conf, self.iou, ratio, pad, (h, w))
        except Exception as exc:  # noqa: BLE001 - 节点不应因单帧异常退出
            self.get_logger().error(f"推理失败：{exc}", throttle_duration_sec=2.0)
            return

        msg = DetectionArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.image_width = int(w)
        msg.image_height = int(h)
        for x1, y1, x2, y2, score, cls_id in dets:
            det = Detection()
            det.class_id = cls_id
            det.label = self.classes[cls_id] if cls_id < len(self.classes) else str(cls_id)
            det.confidence = score
            det.x1, det.y1, det.x2, det.y2 = x1, y1, x2, y2
            det.cx = (x1 + x2) / 2.0
            det.cy = (y1 + y2) / 2.0
            msg.detections.append(det)
        self.pub.publish(msg)

        self._frames += 1
        now = self.get_clock().now()
        if (now - self._last_log).nanoseconds > 2_000_000_000:
            self.get_logger().info(
                f"{self._frames} 帧 | 本帧 {len(dets)} 个目标"
                + (f" | 首个：{msg.detections[0].label} {msg.detections[0].confidence:.2f}"
                   if dets else ""))
            self._frames = 0
            self._last_log = now

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
        if hasattr(self.model, "release"):
            self.model.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VisionDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
