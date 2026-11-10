# 01_vision/configs

## classes.txt —— 类序冻结

```
0 peacock   孔雀
1 wolf      狼
2 monkey    猴
3 elephant  象
4 tiger     虎
```

**类序是接口**：训练标签、导出模型、端侧推理、地面站显示类名，四处必须完全一致。
改类序等于换模型，必须同步重新训练/导出，否则会出现"象被报成虎"这种静默错误。
新增类别只允许**追加**到末尾。

## data.yaml —— YOLO 数据集配置（已加入）

```yaml
path: D:/UAV_Dataset
train: images/train
val: images/val
nc: 5
names: ['peacock', 'wolf', 'monkey', 'elephant', 'tiger']
```

- `names` 与 `classes.txt` **顺序一致**（改一个必须改另一个，否则标签错位）。
- ⚠️ `path` 是**本机绝对路径**，换机器/换目录直接失效。建议改成相对仓库的路径，
  例如 `path: ../datasets/yolo`，数据放 `datasets/yolo/`（已 gitignore），
  这样队友拉下来也能直接跑训练。
- 目录按 Ultralytics 约定：`images/{train,val}` + `labels/{train,val}`，一一对应同名 `.txt`。

## 计划加入的配置

| 文件 | 用途 |
|---|---|
| `model.yaml` | 模型结构/版本、输入尺寸、类别数（必须 = 5） |
| `infer.yaml` | 置信度阈值、NMS IoU、最小框尺寸（滤掉噪点） |
| `calib.yaml` | 相机内参 `fx, fy, cx, cy` + 畸变系数 + 安装倾角，供像素→地图投影 |
| `station.yaml` | 地面站侧映射：类别 id → 中文名（与 classes.txt 对齐） |

> 阈值类参数不要写死在代码里，统一放这里，方便现场调。
