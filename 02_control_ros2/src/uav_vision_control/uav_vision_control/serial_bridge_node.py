#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""串口桥：机载 ROS 2 ←→ 地面站单片机。

上行（机载 → 地面站）：订阅 /vision/cells，把新增的「格号 + 类别 + 数量」打包成
    $EV,<seq>,<cell>,<cls>,<cnt>,<conf>*<CRC8>\\n
    只有某格的计数**变大**时才发，避免每帧刷重复数据。

下行（地面站 → 机载）：读串口解析
    $CMD,<NAME>,<ARG>*<CRC8>\\n      NAME ∈ TAKEOFF/ESTOP/LAND/RESET/PING
    校验通过后发布 /gcs/command，并回一帧
    $ACK,<NAME>*<CRC8>\\n

协议细节（含 CRC-8 算法与测试向量）见 05_ground_station/README.md。

运行：
  ros2 run uav_vision_control serial_bridge --ros-args \
      -p port:=/dev/ttyUSB0 -p baud:=115200

  # 没有硬件时先看它要发什么
  ros2 run uav_vision_control serial_bridge --ros-args -p simulate:=true
"""
from __future__ import annotations

import queue
import threading
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from uav_vision_interfaces.msg import CellReport, GcsCommand

from .serial_proto import CMD_NAMES, encode_ack, encode_ev, parse_frame

CMD_ENUM = {
    "TAKEOFF": GcsCommand.CMD_TAKEOFF,
    "ESTOP": GcsCommand.CMD_ESTOP,
    "LAND": GcsCommand.CMD_LAND,
    "RESET": GcsCommand.CMD_RESET,
    "PING": GcsCommand.CMD_PING,
}


class SerialBridge(Node):
    def __init__(self):
        super().__init__("serial_bridge")

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("simulate", False)
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("send_ack", True)
        self.declare_parameter("mapper_reset_service", "/cell_mapper_node/reset")

        self.port = str(self.get_parameter("port").value)
        self.baud = int(self.get_parameter("baud").value)
        self.simulate = bool(self.get_parameter("simulate").value)
        self.min_conf = float(self.get_parameter("min_conf").value)
        self.send_ack = bool(self.get_parameter("send_ack").value)

        self._tx_lock = threading.Lock()
        self._rx_q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        self._serial = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

        self._seq = 0
        self._sent: dict[tuple[str, int], int] = {}     # (cell, cls) -> 已发数量
        self._bad_frames = 0
        self._tx_frames = 0

        if not self.simulate:
            self._open_serial()

        self.pub_cmd = self.create_publisher(GcsCommand, "/gcs/command", 10)
        self.create_subscription(CellReport, "/vision/cells", self.cb_cells, 10)
        self.create_timer(0.05, self._drain_rx)          # 20 Hz

        self._reset_cli = None
        svc = str(self.get_parameter("mapper_reset_service").value)
        if svc:
            self._reset_cli = self.create_client(Trigger, svc)

        mode = "simulate（只打印，不占串口）" if self.simulate else f"{self.port} @ {self.baud}"
        self.get_logger().info(f"串口桥就绪：{mode}")

    # ---------------- 串口 ----------------

    def _open_serial(self):
        try:
            import serial
        except ImportError as exc:
            raise SystemExit("缺少 pyserial：pip install pyserial") from exc
        try:
            self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        except Exception as exc:                            # noqa: BLE001
            raise SystemExit(
                f"打不开串口 {self.port}：{exc}\n"
                "排查：ls /dev/ttyUSB*（或 /dev/ttyACM*）；"
                "权限：sudo usermod -aG dialout $USER 后重新登录；"
                "没有硬件时加 -p simulate:=true") from exc

        self._reader = threading.Thread(target=self._read_loop, name="serial_rx", daemon=True)
        self._reader.start()

    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self._serial.read(256)
            except Exception as exc:                        # noqa: BLE001
                self.get_logger().error(f"串口读异常：{exc}", throttle_duration_sec=2.0)
                time.sleep(0.1)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    try:
                        self._rx_q.put_nowait(line)
                    except queue.Full:
                        self._bad_frames += 1           # 处理不过来就丢弃最新帧

    def _write(self, frame: bytes):
        if self.simulate:
            self.get_logger().info(f"[SIM] TX {frame!r}")
        else:
            with self._tx_lock:
                try:
                    self._serial.write(frame)
                    self._serial.flush()
                except Exception as exc:                    # noqa: BLE001
                    self.get_logger().error(f"串口写失败：{exc}", throttle_duration_sec=2.0)
                    return
        self._tx_frames += 1

    # ---------------- 上行：视觉 -> 地面站 ----------------

    def cb_cells(self, msg: CellReport):
        for item in msg.cells:
            if float(item.mean_confidence) and float(item.mean_confidence) < self.min_conf:
                continue
            cnt = int(item.count)
            key = (item.cell, int(item.class_id))
            if cnt <= self._sent.get(key, -1):
                continue                                    # 只在计数变大时发
            self._sent[key] = cnt
            self._seq = (self._seq + 1) & 0xFFFF or 1
            frame = encode_ev(self._seq, item.cell, int(item.class_id), cnt,
                              float(item.mean_confidence))
            self._write(frame)
            self.get_logger().info(
                f"TX seq={self._seq} {item.cell} {item.label} ×{cnt} "
                f"(conf {float(item.mean_confidence):.2f})")

    # ---------------- 下行：地面站 -> 机载 ----------------

    def _drain_rx(self):
        for _ in range(16):
            try:
                line = self._rx_q.get_nowait()
            except queue.Empty:
                return
            parsed = parse_frame(line)
            if parsed is None:
                self._bad_frames += 1
                self.get_logger().warning(f"丢弃非法帧 {line!r}（CRC 错/格式错）",
                                          throttle_duration_sec=2.0)
                continue
            kind, fields = parsed
            if kind != "CMD" or not fields:
                continue                                    # 其它类型暂不处理
            name = fields[0].strip().upper()
            arg = 0
            if len(fields) > 1:
                try:
                    arg = int(fields[1])
                except ValueError:
                    arg = 0
            self._handle_cmd(name, arg)

    def _handle_cmd(self, name: str, arg: int):
        if self.send_ack:
            self._write(encode_ack(name if name in CMD_NAMES else "UNKNOWN"))

        if name not in CMD_ENUM:
            self.get_logger().warning(f"未知命令 {name!r}")
            return

        msg = GcsCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.command = CMD_ENUM[name]
        msg.name = name
        msg.arg = int(arg)
        self.pub_cmd.publish(msg)
        self.get_logger().info(f"RX CMD {name} arg={arg} -> /gcs/command")

        if name == "RESET":
            self._sent.clear()
            self._seq = 0
            if self._reset_cli is not None:
                if self._reset_cli.service_is_ready():
                    self._reset_cli.call_async(Trigger.Request())
                    self.get_logger().info("已请求归格节点清空统计")
                else:
                    self.get_logger().warning("归格节点的 reset 服务不可用（未启动？）")

    # ---------------- 退出 ----------------

    def destroy_node(self):
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:                               # noqa: BLE001
                pass
        self.get_logger().info(f"统计：TX {self._tx_frames} 帧，丢弃 {self._bad_frames} 帧")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SerialBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
