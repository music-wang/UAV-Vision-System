# 单个检测框（原图像素坐标系，左上角为原点）
string label          # 类别名，如 "tiger"
float32 confidence    # 置信度 0.0 ~ 1.0

float32 x1            # 左上角 x
float32 y1            # 左上角 y
float32 x2            # 右下角 x
float32 y2            # 右下角 y

float32 cx            # 框中心 x
float32 cy            # 框中心 y
