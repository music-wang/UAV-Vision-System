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
