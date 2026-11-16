# 无人机位姿（场地坐标系）
# 由 SLAM / 光流 / 标签定位节点发布；cell_mapper_node 用它把像素坐标投到地图上
std_msgs/Header header

float32 x_cm      # 场地坐标 x（cm），原点 = 喷绘布左上角
float32 y_cm      # 场地坐标 y（cm），向下为正
float32 yaw_deg   # 机身偏航角（deg），与图像逆时针为正的约定一致
float32 alt_cm    # 离地高度（cm），赛题要求 120±10

bool valid        # 该帧位姿是否可用；false 时 cell_mapper 会跳过这一帧
