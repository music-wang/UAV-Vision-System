# 地面站下行命令（serial_bridge_node 从串口解析后发布）
uint8 CMD_UNKNOWN=0
uint8 CMD_TAKEOFF=1
uint8 CMD_ESTOP=2
uint8 CMD_LAND=3
uint8 CMD_RESET=4
uint8 CMD_PING=5

std_msgs/Header header

uint8 command       # 见上面的常量
string name         # 原始命令字，如 "TAKEOFF"（排查用）
int32 arg           # 附带参数；无参数为 0
