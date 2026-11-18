#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""飞控节点 launch。

  ros2 launch uav_flight_control flight_control.launch.py fx:=985.6 fy:=985.6 auto_arm:=true
  ros2 launch uav_flight_control flight_control.launch.py dry_run:=true   # 只发零速度
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("fx", default_value="985.6", description="焦距（像素）"),
        DeclareLaunchArgument("fy", default_value="985.6"),
        DeclareLaunchArgument("kp", default_value="0.8"),
        DeclareLaunchArgument("max_vel", default_value="0.5", description="限幅 m/s"),
        DeclareLaunchArgument("auto_arm", default_value="false", description="自动起飞序列"),
        DeclareLaunchArgument("dry_run", default_value="false", description="只发零速度，不解锁"),
        DeclareLaunchArgument("alt_default", default_value="1.20"),
    ]

    node = Node(
        package="uav_flight_control",
        executable="flight_control",
        name="flight_control",
        output="screen",
        parameters=[{
            "fx": LaunchConfiguration("fx"),
            "fy": LaunchConfiguration("fy"),
            "kp": LaunchConfiguration("kp"),
            "max_vel": LaunchConfiguration("max_vel"),
            "auto_arm": LaunchConfiguration("auto_arm"),
            "dry_run": LaunchConfiguration("dry_run"),
            "alt_default": LaunchConfiguration("alt_default"),
        }],
    )

    return LaunchDescription(args + [node])
