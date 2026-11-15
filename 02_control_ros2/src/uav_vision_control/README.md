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
| `model_path` | 必填 | 单输出 `[1,4+nc,8400]` 或双输出 `boxes/scores` 均可 |
| `classes_path` | 空 | 每行一个类别名；空则用内置 5 类 |
| `video_source` | `0` | 摄像头序号 / 视频文件 / 图片路径 |
| `conf_threshold` | `0.25` | 置信度阈值 |
| `iou_threshold` | `0.45` | NMS 阈值 |
| `publish_rate` | `10.0` | 定时器频率 Hz |
| `frame_id` | `camera` | Header 坐标系 |
| `num_threads` | `4` | onnxruntime 线程数 |

## 话题

`/vision/detection`（`uav_vision_interfaces/msg/DetectionArray`）

```yaml
header:
  frame_id: camera
detections:
- label: peacock
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
- 摄像头打不开时检查 `ls -l /dev/video*` 和当前用户是否在 `video` 组。
