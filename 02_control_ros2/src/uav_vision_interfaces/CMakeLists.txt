cmake_minimum_required(VERSION 3.8)
project(uav_vision_interfaces)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/CellCount.msg"
  "msg/CellReport.msg"
  "msg/UavPose.msg"
  "msg/GcsCommand.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
