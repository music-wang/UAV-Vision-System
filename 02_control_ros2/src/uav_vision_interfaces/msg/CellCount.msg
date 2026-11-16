# 某个格子里某一类动物的统计结果
string cell            # 格号，如 "A3B5"；totals 里该项为空
uint8 class_id         # 类别序号 0~4（peacock/wolf/monkey/elephant/tiger）
string label           # 类别名，方便人看
uint8 count            # 该格该类只数
float32 mean_confidence  # 该类目标的平均置信度
