# 逐格计数报告（赛题基本要求 3 要发给地面站的内容）
std_msgs/Header header

uint16 served_frames     # 已参与归格的检测帧数
uint16 confirmed_cells   # 有目标且已确认的格数

CellCount[] cells        # 逐格逐类，例如 A3B5 / elephant / 1
CellCount[] totals       # 全场合计（cell 字段留空）
