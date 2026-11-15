"""启动视觉检测节点。

用法：
  ros2 launch uav_vision_control vision_detector.launch.py \
      model_path:=/home/andy/uav/best_2out.onnx video_source:=0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument(
            "model_path", default_value="/home/andy/uav/best_2out.onnx",
            description="YOLO ONNX 模型路径"),
        DeclareLaunchArgument(
            "video_source", default_value="0",
            description="摄像头序号 / 视频文件 / 图片路径"),
        DeclareLaunchArgument("conf_threshold", default_value="0.25"),
        DeclareLaunchArgument("iou_threshold", default_value="0.45"),
        DeclareLaunchArgument("publish_rate", default_value="10.0"),
    ]

    return LaunchDescription(args + [
        Node(
            package="uav_vision_control",
            executable="vision_detector",
            name="vision_detector",
            output="screen",
            parameters=[{
                "model_path": LaunchConfiguration("model_path"),
                "video_source": LaunchConfiguration("video_source"),
                "conf_threshold": LaunchConfiguration("conf_threshold"),
                "iou_threshold": LaunchConfiguration("iou_threshold"),
                "publish_rate": LaunchConfiguration("publish_rate"),
            }],
        ),
    ])
