#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把检测结果归到 9×7 方格并去重计数，发布 /vision/cells。

订阅：
  /vision/detection   uav_vision_interfaces/DetectionArray   （vision_detector 发）
  /uav/pose           uav_vision_interfaces/UavPose          （SLAM / 光流 / 标签定位发）

发布：
  /vision/cells       uav_vision_interfaces/CellReport       （逐格 + 合计）

服务：
  ~/reset             std_srvs/Trigger                        （每次飞行前清空统计）

用法：
  ros2 run uav_vision_control cell_mapper_node --ros-args \
      -p fov_deg:=66.0 -p alt_default_cm:=120.0

  # 或者连 detector 一起起
  ros2 launch uav_vision_control mission.launch.py \
      backend:=rknn model_path:=$HOME/uav/best_int8.rknn

原理：像素 -> （相机 FOV + 高度 + 机身位姿）-> 场地 cm -> 格号 -> 按地面坐标跟踪去重。
没有位姿（或位姿过期）时**不丢帧、也不瞎猜**：这一帧直接跳过，避免把目标归到错误的格。
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from uav_vision_interfaces.msg import CellCount, CellReport, DetectionArray, UavPose

from .cell_mapper import SPECIES, CellMapper


class CellMapperNode(Node):
    def __init__(self):
        super().__init__("cell_mapper_node")

        self.declare_parameter("fov_deg", 66.0)
        self.declare_parameter("alt_default_cm", 120.0)
        self.declare_parameter("match_cm", 25.0)
        self.declare_parameter("confirm_hits", 3)
        self.declare_parameter("max_miss", 3)
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("pose_timeout", 0.5)
        self.declare_parameter("publish_rate", 2.0)

        self.fov = float(self.get_parameter("fov_deg").value)
        self.alt_default = float(self.get_parameter("alt_default_cm").value)
        self.match_cm = float(self.get_parameter("match_cm").value)
        self.confirm_hits = max(1, int(self.get_parameter("confirm_hits").value))
        self.max_miss = max(0, int(self.get_parameter("max_miss").value))
        self.min_conf = float(self.get_parameter("min_conf").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout").value)

        self._mapper: CellMapper | None = None
        self._size: tuple[int, int] | None = None
        self._pose = None            # (x, y, yaw, alt, 到达时刻纳秒)
        self._frames = 0
        self._skipped_no_pose = 0

        self.create_subscription(DetectionArray, "/vision/detection", self.cb_det, 10)
        self.create_subscription(UavPose, "/uav/pose", self.cb_pose, 10)
        self.pub = self.create_publisher(CellReport, "/vision/cells", 10)
        self.create_service(Trigger, "~/reset", self.srv_reset)

        rate = max(float(self.get_parameter("publish_rate").value), 0.2)
        self.create_timer(1.0 / rate, self.timer_cb)

        self.get_logger().info(
            f"归格节点就绪：fov={self.fov}° confirm_hits={self.confirm_hits} "
            f"match={self.match_cm}cm pose_timeout={self.pose_timeout}s")

    # ---------------- 回调 ----------------

    def cb_pose(self, msg: UavPose):
        if not msg.valid:
            self._pose = None
            return
        alt = float(msg.alt_cm) if msg.alt_cm > 1.0 else self.alt_default
        self._pose = (float(msg.x_cm), float(msg.y_cm), float(msg.yaw_deg), alt,
                      self.get_clock().now().nanoseconds)

    def _pose_is_fresh(self) -> bool:
        if self._pose is None:
            return False
        age = (self.get_clock().now().nanoseconds - self._pose[4]) / 1e9
        return age <= self.pose_timeout

    def _ensure_mapper(self, size: tuple[int, int]) -> CellMapper:
        # 分辨率变了就地重建（不同相机切换时会遇到，重建会清空已有统计）
        if self._mapper is None or self._size != size:
            if self._mapper is not None:
                self.get_logger().warning(f"图像尺寸变化 {self._size} -> {size}，重建统计")
            self._mapper = CellMapper(fov_deg=self.fov, alt_cm=self.alt_default, size=size,
                                      match_cm=self.match_cm, confirm_hits=self.confirm_hits,
                                      max_miss=self.max_miss, min_conf=self.min_conf)
            self._size = size
        return self._mapper

    def cb_det(self, msg: DetectionArray):
        if not msg.detections and msg.image_width == 0:
            return
        if not self._pose_is_fresh():
            self._skipped_no_pose += 1
            self.get_logger().warning(
                "没有新鲜位姿，跳过该帧（检查 /uav/pose 是否在发、pose_timeout 是否太短）",
                throttle_duration_sec=2.0)
            return

        size = (int(msg.image_width), int(msg.image_height))
        if size[0] <= 0 or size[1] <= 0:
            self.get_logger().warning("DetectionArray 里没有图像尺寸，无法归格",
                                      throttle_duration_sec=5.0)
            return

        mapper = self._ensure_mapper(size)
        x, y, yaw, alt, _ = self._pose
        # CellMapper 需要 (cls, u, v, conf)；u/v 用框心
        dets = [(int(d.class_id), float(d.cx), float(d.cy), float(d.confidence))
                for d in msg.detections if float(d.confidence) >= self.min_conf]
        mapper.update(dets, (x, y, yaw), alt_cm=alt)
        self._frames += 1

    # ---------------- 输出 ----------------

    def _label_of(self, sp: str) -> tuple[int, str]:
        if sp in SPECIES:
            return SPECIES.index(sp), sp
        return 255, sp

    def build_report(self) -> CellReport:
        msg = CellReport()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.served_frames = min(self._frames, 65535)

        if self._mapper is None:
            msg.confirmed_cells = 0
            return msg

        # (格号, 类别) -> [只数, 置信度和]
        agg: dict[tuple[str, str], list[float]] = {}
        for cell, sp, cnt, conf in self._mapper.report_detail():
            key = (cell, sp)
            slot = agg.setdefault(key, [0.0, 0.0])
            slot[0] += cnt
            slot[1] += conf

        totals: dict[str, float] = {}
        for (cell, sp), (cnt, conf_sum) in sorted(agg.items()):
            class_id, label = self._label_of(sp)
            item = CellCount()
            item.cell = cell
            item.class_id = class_id
            item.label = label
            item.count = int(cnt)
            item.mean_confidence = float(conf_sum / max(cnt, 1))
            msg.cells.append(item)
            totals[sp] = totals.get(sp, 0.0) + cnt

        msg.confirmed_cells = len({cell for cell, _ in agg})
        for sp, cnt in sorted(totals.items()):
            class_id, label = self._label_of(sp)
            item = CellCount()
            item.cell = ""
            item.class_id = class_id
            item.label = label
            item.count = int(cnt)
            item.mean_confidence = 0.0
            msg.totals.append(item)
        return msg

    def timer_cb(self):
        msg = self.build_report()
        self.pub.publish(msg)
        if msg.totals:
            summary = " ".join(f"{t.label}×{t.count}" for t in msg.totals)
            self.get_logger().info(
                f"{msg.served_frames} 帧 | {msg.confirmed_cells} 格 | {summary}",
                throttle_duration_sec=5.0)

    def srv_reset(self, request, response):
        self._mapper = None
        self._size = None
        self._frames = 0
        self._skipped_no_pose = 0
        response.success = True
        response.message = "归格统计已清空"
        self.get_logger().info("收到 reset，统计已清空")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CellMapperNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
