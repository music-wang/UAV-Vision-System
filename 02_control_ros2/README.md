# 02_control_ros2 — 飞行控制（ROS 2）

## 职责

1. **飞控桥接**：与 PX4 / ArduPilot 通信（MAVROS 或 uXRCE-DDS），收发姿态、位置、状态。
2. **定高巡航**：保持 120±10cm，按航线执行航点，速度 ≥0.25m/s 以满足 300s。
3. **航线执行**：接收地面站下发的航线（按 50cm 格心/格边展开），支持禁飞区绕行。
4. **45°±5° 降落（发挥）**：位置与高度 1:1 联动的斜降轨迹，落点中心落在红色起飞区内。
5. **LED 指示（发挥）**：降落开始时闪烁。
6. **状态回报**：当前格、进度、模式、异常，供地面站显示。

## 输入 / 输出

| 方向 | 内容 | 来源/去向 |
|---|---|---|
| 输入 | 航线（格序列 / 航点序列）、禁飞区 | 地面站 |
| 输入 | 位姿（`map`→`base_link`） | `03_slam` |
| 输入 | 动物事件、光斑偏差 | `01_vision` |
| 输出 | 速度/位置 setpoint | 飞控 |
| 输出 | 巡查进度、当前格、告警 | 地面站 |

## 建议话题（草案，随实现调整）

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/vision/detection` | `uav_vision_interfaces/DetectionArray` | 入 | 检测结果（含 `class_id`、`image_width/height`） |
| `/vision/cells` | `uav_vision_interfaces/CellReport` | 内部 | 逐格计数，`cell_mapper_node` 输出 |
| `/uav/pose` | `uav_vision_interfaces/UavPose` | 入 | 场地坐标位姿，`cell_mapper_node` 用；**当前无节点发布**（03_slam 待做） |
| `/gcs/command` | `uav_vision_interfaces/GcsCommand` | 入 | 地面站命令，`serial_bridge` 解析串口后发布 |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | 出 | Offboard 速度指令（NED） |
| `/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | 出 | 心跳，断流会触发 PX4 失效保护 |
| `/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | 出 | 切 Offboard / 解锁 |
| `/fmu/out/vehicle_status`、`/fmu/out/vehicle_local_position` | `px4_msgs` | 入 | 飞控状态与位置 |

服务：`/flight_control/start`（`std_srvs/Trigger`，启动安全起飞序列）、
`/cell_mapper_node/reset`（`std_srvs/Trigger`，清空逐格统计）。

## 关键点

- **光斑轨迹 = 航线**：赛题用光斑判偏航，所以位置环要跟住，转弯处别抄近路切角（切角会被判"轨迹与航线不一致"）。
- 转弯策略：`stop-and-turn`（每格停一下）稳但慢，`fly-through` 快要靠角点前馈；按 300s 预算取舍。
- 45° 斜降：`vz : vxy = 1 : 1`，并且要在红区上方就对准，不能到低空再修水平位。
- 全程无人工干预，一键起飞后要能自主完成收尾与降落。

## 现状

三个包，PC 上可编译、可运行（**尚未上机联飞**）：

| 包 | 内容 |
|---|---|
| `uav_vision_interfaces` | 自定义消息：`Detection` / `DetectionArray` / `CellCount` / `CellReport` / `UavPose` / `GcsCommand` |
| `uav_vision_control` | `vision_detector`（onnx / rknn 双后端）、`cell_mapper_node`（逐格计数）、`serial_bridge`（地面站串口） |
| `uav_flight_control` | PX4 uXRCE-DDS：30Hz 心跳 + 速度 setpoint + 安全起飞序列（依赖 `px4_msgs`，需按固件版本 clone） |

## 待办

- [x] 选定 PX4 + uXRCE-DDS（不用 MAVROS），飞控节点已写
- [ ] 实机跑通 uXRCE-DDS（pixhawk + MicroXRCEAgent），确认 `/fmu/*` 话题
- [ ] 定高环调参（120cm，气垫效应与地面效应）
- [ ] 航线执行器 + 禁飞区绕行
- [ ] 45° 斜降轨迹与落点精度验证
- [ ] LED 灯效
- [ ] `/uav/pose` 真实来源（`03_slam`）接入，打通"检测 → 归格"链路
