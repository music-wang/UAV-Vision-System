# uav_flight_control — PX4 飞控接口（uXRCE-DDS）

视觉引导的速度控制 + 安全起飞序列。**不用 MAVROS**，直接走 PX4 官方的 uXRCE-DDS 桥。

```
/vision/detection ──► flight_control ──► /fmu/in/trajectory_setpoint (TrajectorySetpoint)
                                     └─► /fmu/in/offboard_control_mode (心跳)
                                     └─► /fmu/in/vehicle_command      (切模式/解锁)
/fmu/out/vehicle_status ─┐
/fmu/out/vehicle_local_position ─┘──► 高度、yaw、是否已解锁
```

## 1. 依赖与编译

```bash
cd ~/UAV-Vision-System/02_control_ros2
source /opt/ros/humble/setup.bash

# px4_msgs 的分支必须和你的 PX4 固件版本一致！
#   PX4 v1.15 -> release/1.15      PX4 v1.14 -> release/1.14
git clone https://github.com/PX4/px4_msgs.git -b release/1.15 src/px4_msgs

colcon build --symlink-install --packages-up-to uav_flight_control
source install/setup.bash
```

**为什么版本要对齐**：`px4_msgs` 是"PX4 消息定义的镜像"，固件升级时消息字段会变
（例如 `TrajectorySetpoint` 新增/移除字段）。分支错位会导致字段缺失或 `KeyError`，
甚至能编译但运行时报错。

> 视觉那几个包（`uav_vision_control` / `uav_vision_interfaces`）**不依赖 px4_msgs**，
> 所以即使没装 PX4 那套，`colcon build --packages-up-to uav_vision_control` 照样能过。

## 2. 跑通 uXRCE-DDS

### 机载电脑侧：装 micro-ROS agent

```bash
# 源码编译（推荐，版本可控）
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
sudo ldconfig /usr/local/lib/
```

```bash
# 串口连飞控（TELEM2）
MicroXRCEAgent serial --dev /dev/ttyUSB0 -b 921600

# 或走网口 / 仿真
MicroXRCEAgent udp4 -p 8888
```

### 飞控侧：两个参数

| 参数 | 值（串口 TELEM2） | 说明 |
|---|---|---|
| `UXRCE_DDS_CFG` | `TELEM2` | DDS 走哪个口（网口用 `Ethernet`） |
| `SER_TEL2_BAUD` | `921600` | 必须和 agent 的 `-b` 一致 |

设完重启飞控，然后在机载电脑上验证：

```bash
ros2 topic list | grep fmu
# /fmu/in/offboard_control_mode  /fmu/in/trajectory_setpoint  /fmu/in/vehicle_command
# /fmu/out/vehicle_status        /fmu/out/vehicle_local_position ...
ros2 topic echo /fmu/out/vehicle_status --once
```

看到 `arming_state` 就说明桥通了。**看不到就先别碰飞行**，先解决 DDS。

## 3. 安全起飞序列（代码里的顺序，不能改）

PX4 对 Offboard 有硬性要求：**必须先有持续的 setpoint 流，才允许切入 Offboard**。
所以顺序是"先心跳 → 再切模式 → 再解锁"。

| 步 | 动作 | 代码位置 | 判据 |
|---|---|---|---|
| 0 | 等 `vehicle_status` 与 `vehicle_local_position` 有数据 | 订阅回调 | 两个话题都有消息 |
| 1 | 每周期发 `OffboardControlMode(velocity=True)` 当心跳 | `_publish_offboard_mode()` | 持续 ≥ `prestream_sec`（1s） |
| 2 | 发 `DO_SET_MODE(1, 22)` 切 Offboard | `_send_vehicle_command(176, 1, 22)` | `nav_state == 14` |
| 3 | 发 `COMPONENT_ARM_DISARM(1)` 解锁 | `_send_vehicle_command(400, 1, 0)` | `arming_state == 2` |
| 4 | 进入视觉控制 | `_velocity_command()` | 已解锁 且 offboard |
| 5 | 视觉超时 / 无目标 | 返回 `(0,0,0)` | 零速度悬停 |

**为什么心跳和 setpoint 要每个周期都发**：PX4 检测到 `/fmu/in/trajectory_setpoint`
断流超过约 0.5s 就会触发 Offboard 失效保护并自动切模式（可能直接掉落）。
本节点在 `loop()` 里两件事都做，30 Hz。

**为什么先切模式再解锁**：在 PX4 里，未解锁时切 Offboard 是允许的（会保持待命），
但**先解锁再切模式**可能被拒，或者在切换瞬间触发模式跳变。按上表顺序最稳。

**手动触发序列**（不想自动起飞时）：

```bash
ros2 service call /flight_control/start std_srvs/srv/Trigger
```

## 4. 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `fx` / `fy` | `985.6` | 焦距（像素）。见下方算法 |
| `alt_default` | `1.20` | 拿不到飞控高度时用的高度（米） |
| `camera_top_is_forward` | `true` | 画面"上"是否对应机身前。装反了设 `false` |
| `kp` | `0.8` | 速度 = kp × 偏差(米)，单位 1/s |
| `max_vel` | `0.5` | 速度限幅 m/s（赛题安全值，别超） |
| `vision_timeout` | `1.0` | 超过这么久没新检测 → 悬停 |
| `target_select` | `center` | `center` 取离画面中心最近 / `conf` 取最自信 / `largest` 取最大 |
| `min_conf` | `0.25` | 低于此置信度的框不参与控制 |
| `setpoint_rate` | `30.0` | 心跳 + setpoint 频率 Hz |
| `desired_alt` | `-1.0` | ≤0 时不做定高（vz=0）；>0 时加一个缓慢的高度 P |
| `auto_arm` | `false` | 启动即执行安全起飞序列 |
| `prestream_sec` | `1.0` | 心跳持续多久后才切模式 |
| `dry_run` | `false` | 只发零速度，不发切模式/解锁（地面联调用） |

### fx / fy 怎么算

```
fx = (图像宽度 / 2) / tan(水平视场角 / 2)
fy = (图像高度 / 2) / tan(垂直视场角 / 2)
```

例：1280×960，水平 FOV 66°：
`fx = 640 / tan(33°) = 640 / 0.6494 = 985.6`；方形像素时 `fy ≈ fx`。

> 不确定就用标定板实测：`fx = 像素距离 / 实际距离 × 物距`。

## 5. 控制律与坐标约定

```
像素偏差 → 物理偏差（米）:
  dx_cam = (u - cx) / fx * alt        画面右为正
  dy_cam = (v - cy) / fy * alt        画面下为正

相机系 → 机体系（x 前 / y 右）:
  前 = -dy_cam （画面往上 = 往前）
  右 = +dx_cam

机体系 → NED（TrajectorySetpoint 要的是 NED，不是机体系！）:
  vN =  cos(yaw)*前 - sin(yaw)*右
  vE =  sin(yaw)*前 + cos(yaw)*右

控制: v = clamp(kp * 偏差, -max_vel, +max_vel)
```

**符号自检**：目标出现在画面右侧 → 机体往右移动 → 目标回到中心。如果实测是
"越飞越偏"，先查 `camera_top_is_forward`，再查 `yaw` 的符号（PX4 的 `heading` 是
NED 下的偏航角，逆时针为正在 ENU 里对应正，NED 里是顺时针为正——用四元数更保险）。

## 6. 调试顺序（强烈建议照这个来）

```bash
# ① 地面：只发零速度，确认话题、频率、消息类型都对
ros2 run uav_flight_control flight_control --ros-args -p dry_run:=true
ros2 topic hz /fmu/in/trajectory_setpoint        # 应约 30 Hz
ros2 topic echo /fmu/in/trajectory_setpoint --once

# ② 地面：验证视觉→速度的换算（不出桨），把机子举起来对着靶标看日志
#    在 loop 里加一行 debug 日志，或直接看下面的公式手算

# ③ 系留试飞：auto_arm:=false，手动用遥控切 Offboard + 解锁，只让视觉给速度

# ④ 自由飞行：auto_arm:=true
```

## 7. 安全注意

- `max_vel` 默认 0.5 m/s，比赛场地 4.5×3.5 m，**别调大**。
- 视觉丢目标 → 零速度悬停，但**不会自动降落**；降落要由任务状态机发
  `LAND`（或遥控切模式）。这是有意的：自动降落更危险。
- `desired_alt` 默认关掉。高度环由 PX4 自己在 Offboard 速度模式里维持
  （vz=0 表示"别动高度"），比我们自己加高度环更稳。
- 退出节点前会补发一帧零速度，但仍建议**先切回定高模式再关节点**。
