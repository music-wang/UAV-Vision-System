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

## 数据配置：data_real.yaml（主）/ data_synth.yaml（辅）

```yaml
# data_real.yaml —— 实拍数据集，正式训练用这个
path: ../../datasets/real
train: images/train
val: images/val
nc: 5
names: ['peacock', 'wolf', 'monkey', 'elephant', 'tiger']
```

- **两份配置各管一条路线**：`data_real.yaml` -> `datasets/real`（实拍，主训练用）；
  `data_synth.yaml` -> `datasets/yolo_synth`（合成，辅助/兜底）。流程见 `docs/真实数据流程.md`。
- `path` 写**相对本文件所在目录**的路径（如 `../../datasets/real`）。注意 Ultralytics
  会把相对 path 当成相对它的 `datasets_dir`，所以 `scripts/train.py` 会先按"相对本文件"
  解析成绝对路径再交给训练器；直接用 `yolo train data=...` 时请写绝对路径。
- `names` 必须与 `classes.txt` 同序（改一个必须改另一个）。
- 目录按 Ultralytics 约定：`images/{train,val}` + `labels/{train,val}`，同名一一对应。
- 训练前先跑 `scripts/check_dataset.py`：任何一类 0 实例都会直接失败（我们踩过这个坑）。

## 计划加入的配置

| 文件 | 用途 |
|---|---|
| `model.yaml` | 模型结构/版本、输入尺寸、类别数（必须 = 5） |
| `infer.yaml` | 置信度阈值、NMS IoU、最小框尺寸（滤掉噪点） |
| `calib.yaml` | 相机内参 `fx, fy, cx, cy` + 畸变系数 + 安装倾角，供像素→地图投影 |
| `station.yaml` | 地面站侧映射：类别 id → 中文名（与 classes.txt 对齐） |

> 阈值类参数不要写死在代码里，统一放这里，方便现场调。
