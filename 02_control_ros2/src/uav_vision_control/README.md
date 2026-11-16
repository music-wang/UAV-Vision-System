# uav_vision_control

USB 摄像头 → YOLO11(ONNX) → `/vision/detection`。

## 依赖

```bash
# ROS 侧（Ubuntu 22.04 + Humble）
sudo apt install -y python3-opencv python3-numpy

# onnxruntime 不在 rosdep 里，单独装（不要带依赖，避免 pip 升级 numpy 破坏 ROS 环境）
pip3 install --user --no-deps onnxruntime==1.16.3
```

已用 `numpy 1.21.5 + onnxruntime 1.16.3` 验证可跑 `best.onnx`（opset 19）。

## 编译与运行

```bash
cd ~/UAV-Vision-System/02_control_ros2
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# 摄像头（序号 0）
ros2 run uav_vision_control vision_detector --ros-args \
    -p model_path:=$HOME/uav/best.onnx -p video_source:=0

# 或图片/视频，便于无摄像头时调试
ros2 run uav_vision_control vision_detector --ros-args \
    -p model_path:=$HOME/uav/best_2out.onnx \
    -p video_source:=$HOME/UAV-Vision-System/01_vision/assets/predict_demo.jpg

# 或 launch
ros2 launch uav_vision_control vision_detector.launch.py \
    model_path:=$HOME/uav/best.onnx video_source:=0
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `backend` | `onnx` | `onnx`（PC 调试）或 `rknn`（鲁班猫 4 的 NPU） |
| `model_path` | 必填 | onnx 后端传 `.onnx`，rknn 后端传 `.rknn`；单输出 `[1,4+nc,8400]` 或双输出 `boxes/scores` 均可 |
| `rknn_core` | `012` | rknn 后端的 NPU 核心：`0`/`1`/`2`/`012` |
| `classes_path` | 空 | 每行一个类别名；空则用内置 5 类 |
| `video_source` | `0` | 摄像头序号 / 视频文件 / 图片路径 |
| `conf_threshold` | `0.25` | 置信度阈值 |
| `iou_threshold` | `0.45` | NMS 阈值 |
| `publish_rate` | `10.0` | 定时器频率 Hz |
| `frame_id` | `camera` | Header 坐标系 |
| `num_threads` | `4` | onnxruntime 线程数（rknn 后端忽略） |

## 两种后端：PC 调试用 onnx，板上用 rknn

同一个节点、同一套话题，只换 `backend` 和模型文件：

```bash
# PC / Ubuntu：onnxruntime CPU
ros2 run uav_vision_control vision_detector --ros-args \
    -p backend:=onnx -p model_path:=$HOME/uav/best_2out.onnx -p video_source:=0

# 鲁班猫 4：RKNN NPU（yolo11n@640 int8 约 12~20 ms/帧）
ros2 run uav_vision_control vision_detector --ros-args \
    -p backend:=rknn -p model_path:=$HOME/uav/best_int8.rknn \
    -p rknn_core:=012 -p video_source:=0
```

为什么必须分两种：RK3588 的 **CPU 跑 yolo11n@640 大约 100~200 ms/帧（5~10 fps）**，
达不到比赛要求；NPU 才是量产路径。PC 上用 onnx 只是为了让调试不依赖板子。

板上还需要：

```bash
python3 -c "from rknnlite.api import RKNNLite; print('rknnlite ok')"   # 出厂镜像一般自带
```

## 话题

`/vision/detection`（`uav_vision_interfaces/msg/DetectionArray`）

```yaml
header:
  frame_id: camera
image_width: 1280      # 本帧尺寸，下游把像素投影到地图/归一化时要用
image_height: 960
detections:
- label: peacock
  class_id: 0          # 类别序号，下游统计/归格用这个，别比较字符串
  confidence: 0.43
  x1: 805.5   # 原图像素坐标
  y1: 640.3
  x2: 986.3
  y2: 807.4
  cx: 895.9
  cy: 723.8
```

## 注意

- 解码逻辑与 `01_vision/deploy/rknn_inference.py` 一致：letterbox、`/255`、RGB、NMS、坐标还原。
- INT8 模型建议用 `best_2out.onnx` 转换（框和分数分开量化，避免分数被吃掉）。
- 相机打开后设了 `CAP_PROP_BUFFERSIZE=1`，只保留最新帧——否则 10 Hz 轮询 30 fps 相机会一直
  读到排队旧帧，无人机移动时坐标会明显滞后。
- 摄像头打不开时检查 `ls -l /dev/video*` 和当前用户是否在 `video` 组。

## 归格计数节点：`cell_mapper_node`

把 `/vision/detection` 和 `/uav/pose` 合起来，输出赛题基本要求 3 要的「格号 + 类别 + 数量」。

```
/vision/detection ─┐
                   ├─► cell_mapper_node ─► /vision/cells (CellReport)
/uav/pose ─────────┘
```

```bash
# 和检测节点一起起
ros2 launch uav_vision_control mission.launch.py \
    backend:=rknn model_path:=$HOME/uav/best_int8.rknn video_source:=0

# 或者单独起（检测节点已在跑）
ros2 run uav_vision_control cell_mapper_node --ros-args -p fov_deg:=66.0

# 每次起飞前清空统计
ros2 service call /cell_mapper_node/reset std_srvs/srv/Trigger
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `fov_deg` | `66.0` | 相机水平视场角，决定像素→地面的换算比例 |
| `alt_default_cm` | `120.0` | 位姿里没给高度时用这个（赛题 120±10） |
| `match_cm` | `25.0` | 同一目标的地面距离阈值，用于跨帧去重 |
| `confirm_hits` | `3` | 至少被看到几帧才算确认（防误检） |
| `max_miss` | `3` | 连续丢几帧后允许新建轨迹 |
| `min_conf` | `0.25` | 低于此置信度的框不参与归格 |
| `pose_timeout` | `0.5` | 位姿超过这么久没更新就跳过该帧 |
| `publish_rate` | `2.0` | `/vision/cells` 发布频率 Hz |

### 位姿约定（`UavPose`）

| 字段 | 含义 |
|---|---|
| `x_cm` / `y_cm` | 场地坐标，原点 = 喷绘布左上角，y 向下为正 |
| `yaw_deg` | 机身偏航角，与「图像逆时针为正」一致 |
| `alt_cm` | 离地高度，≤1 时回落到 `alt_default_cm` |
| `valid` | false 时该帧不参与归格 |

### 两个边界处理

1. **没有新鲜位姿就不归格**：宁可少统计，也不把目标算到错误的格子里（日志会节流提示）。
2. **图像尺寸变化会重建统计**：换相机/分辨率时 `image_width/height` 变了，节点会清空并从零累计，
   同时打一条 warning——比赛现场不要中途换分辨率。

### 输出示例

```
$ ros2 topic echo /vision/cells --once
header: {frame_id: camera}
served_frames: 62
confirmed_cells: 3
cells:
- {cell: A3B5, class_id: 3, label: elephant, count: 1, mean_confidence: 0.88}
- {cell: A4B4, class_id: 1, label: wolf,     count: 1, mean_confidence: 0.91}
totals:
- {cell: '', class_id: 3, label: elephant, count: 1, mean_confidence: 0.0}
- {cell: '', class_id: 1, label: wolf,     count: 1, mean_confidence: 0.0}
```

> `cell_mapper.py` 是 `01_vision/scripts/cell_mapper.py` 的副本（板端只部署 ROS 工作空间，
> 无法跨目录 import）。改动请先改 `01_vision/scripts` 下那份，再复制过来。
