# 一帧图像里的全部检测结果
std_msgs/Header header
Detection[] detections

# 本帧图像尺寸：下游把像素坐标投影到地图/归一化时要用
uint32 image_width
uint32 image_height
