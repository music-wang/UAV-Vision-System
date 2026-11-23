#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PX4 飞控控制节点（uXRCE-DDS，不用 MAVROS）。

做什么：
  订阅 /vision/detection（目标框心像素），把"目标偏离画面中心多少像素"
  换算成"机身该往哪个方向挪多少米"，再用 P 控制器变成速度指令，
  发给 PX4 的 /fmu/in/trajectory_setpoint（Offboard 速度模式）。

  视觉超时（默认 1s）或一帧里没有目标 -> 输出 0 速度（悬停），不清零高度。

坐标与符号约定（最容易搞错的地方，务必核对你的相机安装方向）：
  · 相机**垂直向下**安装，画面"上"对应机身**前方**（默认；装反了就把
    camera_top_is_forward 设成 false）
  · 像素偏差 -> 相机系偏移（米）:
        dx_cam = (u - cx) / fx * alt      # 画面右为正
        dy_cam = (v - cy) / fy * alt      # 画面下为正
  · 相机系 -> 机体系（x 前 / y 右 / z 下）:
        机体前 = -dy_cam（画面往上 = 往前）
        机体右 = +dx_cam
  · 机体系 -> NED（TrajectorySetpoint 的 velocity 是 NED，不是机体系！）:
        vN =  cos(yaw)*前 - sin(yaw)*右
        vE =  sin(yaw)*前 + cos(yaw)*右
  · 控制方向：目标在右侧 -> 机体往右移动才能把它拉回画面中心（负反馈）

安全起飞序列（顺序不能乱，PX4 有硬性要求）：
  1) 等 /fmu/out/vehicle_status 与 /fmu/out/vehicle_local_position 有数据
  2) 以 setpoint_rate 持续发 OffboardControlMode(velocity=True) 当**心跳**
  3) 心跳连续发满 prestream_sec（默认 1s）后，发 DO_SET_MODE 切 Offboard
  4) 发 COMPONENT_ARM_DISARM 解锁
  5) 轮询 vehicle_status：arming_state==ARMED 且 nav_state==OFFBOARD 才算就绪
  6) 进入控制循环；任何时刻视觉超时 -> 零速度

运行：
  ros2 run uav_flight_control flight_control --ros-args \
      -p fx:=985.6 -p fy:=985.6 -p auto_arm:=true

  # 只发零速度、不解锁（地面联调 / 看话题）
  ros2 run uav_flight_control flight_control --ros-args -p dry_run:=true
"""
from __future__ import annotations

import math
import time

import rclpy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                          VehicleLocalPosition, VehicleStatus)
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from uav_vision_interfaces.msg import DetectionArray

# PX4 常量（写死数字，避免不同版本 px4_msgs 里常量名差异导致编译不过）
# 来源：PX4-Autopilot release/1.17
#   src/modules/commander/px4_custom_mode.h   enum PX4_CUSTOM_MAIN_MODE
#   msg/VehicleStatus.msg                     ARMING_STATE_ARMED / NAVIGATION_STATE_OFFBOARD
#   msg/VehicleCommand.msg                    VEHICLE_CMD_*
ARMING_STATE_ARMED = 2
NAV_STATE_OFFBOARD = 14
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
# 注意：这个值不在 px4_msgs 里（是固件内部枚举），只能读 PX4 源码核对。
#   MANUAL=1 ALTCTL=2 POSCTL=3 AUTO=4 ACRO=5 OFFBOARD=6 STABILIZED=7 ...
# 写错会导致 DO_SET_MODE 被判为非法模式、nav_state 永远不进 14。
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6

# 发出 DO_SET_MODE 后等待 nav_state 变成 OFFBOARD 的最长时间（秒）。
# 超过仍不是 14 -> 记录错误并停止后续解锁流程（不会带着"以为在 offboard"的风险起飞）。
OFFBOARD_CONFIRM_TIMEOUT_S = 2.0


class FlightControl(Node):
    def __init__(self):
        super().__init__("flight_control")

        # ---- 相机 / 几何 ----
        self.declare_parameter("fx", 985.6)      # 1280 宽、水平 FOV 66° 时约 985.6
        self.declare_parameter("fy", 985.6)
        self.declare_parameter("alt_default", 1.20)      # 拿不到高度时用（米）
        self.declare_parameter("camera_top_is_forward", True)

        # ---- 控制 ----
        self.declare_parameter("kp", 0.8)               # 速度 = kp * 偏差(米)
        self.declare_parameter("max_vel", 0.5)          # 限幅 m/s（赛题安全）
        self.declare_parameter("vision_timeout", 1.0)   # 视觉超时 -> 悬停
        self.declare_parameter("target_select", "center")  # center|conf|largest
        self.declare_parameter("min_conf", 0.25)
        self.declare_parameter("setpoint_rate", 30.0)
        self.declare_parameter("desired_alt", -1.0)     # >0 时做高度保持，否则 vz=0

        # ---- 安全序列 ----
        self.declare_parameter("auto_arm", False)
        self.declare_parameter("prestream_sec", 1.0)
        self.declare_parameter("dry_run", False)

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.alt_default = float(self.get_parameter("alt_default").value)
        self.top_is_forward = bool(self.get_parameter("camera_top_is_forward").value)
        self.kp = float(self.get_parameter("kp").value)
        self.max_vel = abs(float(self.get_parameter("max_vel").value))
        self.vision_timeout = float(self.get_parameter("vision_timeout").value)
        self.target_select = str(self.get_parameter("target_select").value)
        self.min_conf = float(self.get_parameter("min_conf").value)
        rate = max(float(self.get_parameter("setpoint_rate").value), 2.0)
        self.desired_alt = float(self.get_parameter("desired_alt").value)
        self.prestream = float(self.get_parameter("prestream_sec").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)

        # PX4 的 /fmu/in 与 /fmu/out 都要求 BEST_EFFORT + VOLATILE
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_offboard = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", qos)
        self.pub_setpoint = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos)
        self.pub_cmd = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status",
                                 self.cb_status, qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position",
                                 self.cb_local_pos, qos)
        self.create_subscription(DetectionArray, "/vision/detection",
                                 self.cb_detection, 10)

        self.status: VehicleStatus | None = None
        self.local_pos: VehicleLocalPosition | None = None
        self.det: DetectionArray | None = None
        self.det_stamp = 0.0

        self.armed = False
        self.offboard = False
        self.stream_start = 0.0
        self.mode_sent = False
        self.mode_sent_at = 0.0
        self.arm_sent = False
        self.sequence_started = False
        self.mode_failed = False
        self.hb_count = 0

        self.create_service(Trigger, "~/start", self.srv_start)
        self.create_timer(1.0 / rate, self.loop)

        self.get_logger().info(
            f"flight_control 就绪：fx={self.fx:.1f} fy={self.fy:.1f} kp={self.kp} "
            f"max_vel={self.max_vel}m/s timeout={self.vision_timeout}s dry_run={self.dry_run}")

        if bool(self.get_parameter("auto_arm").value):
            self.sequence_started = True
            self.get_logger().info("auto_arm=true，开始安全起飞序列")

    # ---------------- 回调 ----------------

    def cb_status(self, msg: VehicleStatus):
        self.status = msg
        self.armed = (msg.arming_state == ARMING_STATE_ARMED)
        self.offboard = (msg.nav_state == NAV_STATE_OFFBOARD)

    def cb_local_pos(self, msg: VehicleLocalPosition):
        self.local_pos = msg

    def cb_detection(self, msg: DetectionArray):
        self.det = msg
        self.det_stamp = time.monotonic()

    # ---------------- 工具 ----------------

    def _ts_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _altitude(self) -> float:
        """离地高度（米）。PX4 的 local position z 是 NED（向下为正），所以取 -z。"""
        if self.local_pos is not None and math.isfinite(self.local_pos.z):
            alt = -float(self.local_pos.z)
            if 0.05 < alt < 30.0:
                return alt
        return self.alt_default

    def _yaw(self) -> float:
        if self.local_pos is not None and hasattr(self.local_pos, "heading"):
            h = float(self.local_pos.heading)
            if math.isfinite(h):
                return h
        return 0.0

    def _pick_target(self, msg: DetectionArray):
        """挑一个目标：默认取离画面中心最近的（不会在多个目标间来回跳）。"""
        if not msg.detections:
            return None
        cx0, cy0 = msg.image_width / 2.0, msg.image_height / 2.0
        cands = [d for d in msg.detections if float(d.confidence) >= self.min_conf]
        if not cands:
            return None
        if self.target_select == "conf":
            return max(cands, key=lambda d: float(d.confidence))
        if self.target_select == "largest":
            return max(cands, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
        return min(cands, key=lambda d: (d.cx - cx0) ** 2 + (d.cy - cy0) ** 2)

    # ---------------- 控制律 ----------------

    def _velocity_command(self) -> tuple[float, float, float]:
        """返回 NED 速度 (vN, vE, vD)。视觉失效时全 0 = 悬停。"""
        if self.det is None:
            return 0.0, 0.0, 0.0
        if (time.monotonic() - self.det_stamp) > self.vision_timeout:
            return 0.0, 0.0, 0.0

        tgt = self._pick_target(self.det)
        if tgt is None:                                 # 目标丢失
            return 0.0, 0.0, 0.0

        alt = self._altitude()
        dx_cam = (float(tgt.cx) - self.det.image_width / 2.0) / self.fx * alt
        dy_cam = (float(tgt.cy) - self.det.image_height / 2.0) / self.fy * alt

        # 画面往上 = 往前（默认）；相机装反了就把 camera_top_is_forward 设 false
        fwd = -dy_cam if self.top_is_forward else dy_cam
        right = dx_cam

        v_fwd = self._clamp(self.kp * fwd)
        v_right = self._clamp(self.kp * right)

        yaw = self._yaw()
        vn = math.cos(yaw) * v_fwd - math.sin(yaw) * v_right
        ve = math.sin(yaw) * v_fwd + math.cos(yaw) * v_right

        vz = 0.0
        if self.desired_alt > 0:                       # 需要定高时再加一点 P（默认不用）
            vz = self._clamp(0.5 * (self.desired_alt - alt))
        return vn, ve, vz

    def _clamp(self, v: float) -> float:
        return max(-self.max_vel, min(self.max_vel, v))

    # ---------------- 发布 ----------------

    def _publish_offboard_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self._ts_us()
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.hb_count += 1
        self.pub_offboard.publish(msg)

    def _publish_setpoint(self, vn: float, ve: float, vz: float):
        sp = TrajectorySetpoint()
        sp.timestamp = self._ts_us()
        sp.position = [math.nan, math.nan, math.nan]     # 不控位置
        sp.velocity = [float(vn), float(ve), float(vz)]  # NED，m/s
        sp.acceleration = [math.nan, math.nan, math.nan]
        sp.jerk = [math.nan, math.nan, math.nan]
        sp.yaw = math.nan                                # 不控朝向
        sp.yawspeed = math.nan
        self.pub_setpoint.publish(sp)

    def _send_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0):
        cmd = VehicleCommand()
        cmd.timestamp = self._ts_us()
        cmd.command = command
        cmd.param1 = float(param1)
        cmd.param2 = float(param2)
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.source_system = 1
        cmd.source_component = 1
        cmd.from_external = True
        self.pub_cmd.publish(cmd)

    # ---------------- 主循环 ----------------

    def loop(self):
        # ① 心跳必须每个周期都发，PX4 靠它判断 offboard 是否还活着
        self._publish_offboard_mode()

        vn, ve, vz = self._velocity_command()
        if self.dry_run:
            vn = ve = vz = 0.0
        self._publish_setpoint(vn, ve, vz)          # ② 每周期都发 setpoint

        if self.dry_run:
            return

        # ③ 安全起飞序列（非阻塞状态机）
        if self.sequence_started:
            self._safety_sequence()

    def _safety_sequence(self):
        now = time.monotonic()
        if self.stream_start == 0.0:
            self.stream_start = now
            self.get_logger().info(
                f"开始发 offboard 心跳，需连续 {self.prestream:.1f}s 后才允许切模式…")
            return

        # ④ PX4 硬性要求：先持续收到合规的 OffboardControlMode 心跳 ≥1s，才能请求 OFFBOARD。
        #    这里除了时间，还要求确实发出过心跳帧（防止时间到了但一帧都没发）。
        if not self.mode_sent and not self.mode_failed:
            elapsed = now - self.stream_start
            if elapsed < self.prestream or self.hb_count == 0:
                return
            self.get_logger().info(
                f"心跳已连续 {elapsed:.1f}s / {self.hb_count} 帧 -> 发 DO_SET_MODE"
                f"(main_mode={PX4_CUSTOM_MAIN_MODE_OFFBOARD} -> OFFBOARD)")
            self._send_vehicle_command(
                VEHICLE_CMD_DO_SET_MODE, 1.0, float(PX4_CUSTOM_MAIN_MODE_OFFBOARD))
            self.mode_sent = True
            self.mode_sent_at = now
            return

        # ⑤ 等 nav_state 真正变成 OFFBOARD，最多 OFFBOARD_CONFIRM_TIMEOUT_S 秒。
        #    等不到就报错并停在这里：绝不带着"以为在 offboard"的状态去解锁。
        if self.mode_sent and not self.offboard and not self.mode_failed:
            waited = now - self.mode_sent_at
            if waited >= OFFBOARD_CONFIRM_TIMEOUT_S:
                nav = self.status.nav_state if self.status is not None else -1
                self.get_logger().error(
                    f"切换 OFFBOARD 失败：{waited:.1f}s 内 nav_state 仍为 {nav}"
                    f"（期望 {NAV_STATE_OFFBOARD}），已停止解锁流程")
                self.mode_failed = True
                self.sequence_started = False
            return

        # ⑥ 确认处于 OFFBOARD 之后再解锁（顺序反过来 PX4 会拒绝或跳模式）
        if self.offboard and not self.arm_sent and not self.mode_failed:
            self.get_logger().info("已进入 OFFBOARD -> 发 COMPONENT_ARM_DISARM 解锁")
            self._send_vehicle_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
            self.arm_sent = True
            return

        if self.armed and self.offboard and not getattr(self, "_ready_logged", False):
            self._ready_logged = True
            self.get_logger().info("已解锁且处于 OFFBOARD，进入视觉引导速度控制")

    def srv_start(self, request, response):
        self.sequence_started = True
        self.stream_start = 0.0
        self.mode_sent = False
        self.mode_sent_at = 0.0
        self.arm_sent = False
        self.mode_failed = False
        self.hb_count = 0
        self._ready_logged = False
        response.success = True
        response.message = "起飞序列已启动"
        self.get_logger().info("收到 start，起飞序列启动")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = FlightControl()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node._publish_setpoint(0.0, 0.0, 0.0)   # 退出前给一帧零速度
            except Exception:                           # noqa: BLE001
                pass
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
