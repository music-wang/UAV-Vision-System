#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务级 launch：视觉检测 + 归格计数一起起。

用法：
  ros2 launch uav_vision_control mission.launch.py \
      backend:=rknn model_path:=$HOME/uav/best_int8.rknn video_source:=0

  # PC 调试
  ros2 launch uav_vision_control mission.launch.py \
      backend:=onnx model_path:=$HOME/uav/best_2out.onnx video_source:=0
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("backend", default_value="onnx",
                              description="onnx（PC 调试）或 rknn（鲁班猫 NPU）"),
        DeclareLaunchArgument("model_path", default_value="/home/andy/uav/best_2out.onnx",
                              description="onnx 传 .onnx，rknn 传 .rknn"),
        DeclareLaunchArgument("rknn_core", default_value="012",
                              description="rknn 后端的 NPU 核心 0/1/2/012"),
        DeclareLaunchArgument("video_source", default_value="0",
                              description="摄像头序号 / 视频文件 / 图片路径"),
        DeclareLaunchArgument("conf_threshold", default_value="0.25"),
        DeclareLaunchArgument("fov_deg", default_value="66.0",
                              description="相机水平视场角，影响像素->地面换算"),
    ]

    detector = Node(
        package="uav_vision_control",
        executable="vision_detector",
        name="vision_detector",
        output="screen",
        parameters=[{
            "backend": LaunchConfiguration("backend"),
            "model_path": LaunchConfiguration("model_path"),
            "rknn_core": LaunchConfiguration("rknn_core"),
            "video_source": LaunchConfiguration("video_source"),
            "conf_threshold": LaunchConfiguration("conf_threshold"),
        }],
    )

    mapper = Node(
        package="uav_vision_control",
        executable="cell_mapper_node",
        name="cell_mapper_node",
        output="screen",
        parameters=[{
            "fov_deg": LaunchConfiguration("fov_deg"),
        }],
    )

    return LaunchDescription(args + [detector, mapper])
