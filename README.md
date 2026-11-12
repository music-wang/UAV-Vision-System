# UAV-Vision-System

[![最新版本](https://img.shields.io/github/v/release/music-wang/UAV-Vision-System?label=%E6%9C%80%E6%96%B0%E6%A8%A1%E5%9E%8B&color=2ea44f)](https://github.com/music-wang/UAV-Vision-System/releases/latest)
[![mAP50-95](https://img.shields.io/badge/mAP50--95-0.730-blue)](https://github.com/music-wang/UAV-Vision-System/releases/tag/v3.0)

2025 年全国大学生电子设计竞赛（TI 杯）**H 题：野生动物巡查系统**（本科组）的工程仓库。

> 无人机在 450cm×350cm 的 9×7 网格地图上自主巡查，绕开 3 格禁飞区，机载视觉识别 5 类动物
> （象 / 虎 / 狼 / 猴 / 孔雀），统计"哪一格、什么动物、几只"并实时回传地面站；
> 巡查中激光笔垂直向下打点（轨迹即航线），结束后以 45°±5° 俯角降回红色起飞区。

## 当前状态：v3.0（真实数据版）

| 项目 | 值 |
|---|---|
| 模型 | yolo11n，5 类，输入 640 |
| 数据 | **实拍 249 张 + 7 张纯背景负样本**（199 train / 50 val） |
| 指标 | mAP50 **0.995** / mAP50-95 **0.730** / P 0.993 / R 0.991 |
| 下载 | [v3.0 Release](https://github.com/music-wang/UAV-Vision-System/releases/tag/v3.0)（`best.onnx` + `best.pt` + 指标图） |

**版本演进**（v1.0 跑通 → v2.0 指标虚高 → v3.0 数据可信）：见 [CHANGELOG.md](CHANGELOG.md)。

## 工作流（先看这条）

```
拍真实照片 → prepare_real.py 划分/统计 → check_dataset.py 体检 → train.py 训练
           → predict_demo.py 看效果 → export.py 导出 ONNX → deploy/ 转 RKNN → 板端推理
```

| 步骤 | 命令 | 说明 |
|---|---|---|
| ① 整理实拍 | `python 01_vision/scripts/prepare_real.py --src D:\photos` | 8:2 划分，无标注图自动转负样本 |
| ② 训练前体检 | `python 01_vision/scripts/check_dataset.py` | **某类 0 实例直接失败**（踩过这个坑） |
| ③ 训练 | `python 01_vision/scripts/train.py --device 0 --epochs 150` | 300 张约 150 轮 |
| ④ 看效果 | `python 01_vision/scripts/predict_demo.py --limit 12` | 出预测图 + 各类检出数 |
| ⑤ 导出 | `python 01_vision/scripts/export.py --format onnx --imgsz 640` | RKNN 用这份 |
| ⑥ 部署 | `01_vision/deploy/` | ONNX→RKNN、板端 rknnlite 推理 |

合成数据是**辅助/兜底**，不是主路线：`scripts/synth_dataset.py`（必须给 `--scenes`）。

## 文档

| 文件 | 内容 |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | v1.0 → v2.0 → v3.0 版本演进与指标对比 |
| [docs/真实数据流程.md](docs/真实数据流程.md) | **主路线**：拍什么、怎么打标、300 张怎么训、怎么验 |
| [docs/2025H-赛题原文.md](docs/2025H-赛题原文.md) | 赛题全文转录（任务 / 要求 / 说明 / 评分） |
| [docs/2025H-约束与验收清单.md](docs/2025H-约束与验收清单.md) | 硬指标表、推论指标、扣分点、现场流程、需求→模块映射 |

## 目录结构

| 目录 | 职责 | 状态 |
|---|---|---|
| `01_vision/` | 机载视觉：5 类动物检测、逐格归并计数、光斑落点估计 | 训练/导出/计数已跑通 |
| `01_vision/deploy/` | RK3588 部署：ONNX→RKNN 转换、板端 rknnlite 推理 + 解码/NMS | 板端待实测 |
| `02_control_ros2/` | ROS 2 飞行控制：定高 120cm、航点/航线执行、45° 降落、LED | 骨架 |
| `03_slam/` | 室内定位与建图：黑线视觉 / 雷达 / 标签等定位方案 | 骨架 |
| `04_hardware/` | 机体、桨叶防护、激光笔与相机安装、供电与载荷 | 骨架 |
| `05_ground_station/` | 地面站：MCU + ≥6 吋屏 + 按键，9×7 航线显示、禁飞区设置、收显/存档/回看 | 骨架（占位，可按需合并） |
| `docs/` | 赛题、真实数据流程、方案、测试记录 | 已有 3 篇 |

## 数据集约定

```
datasets/                    # 已被 .gitignore 忽略，不进仓库
├── raw/                     # 赛题附件原图（动物体态 / 地貌图）
├── real/                    # ★ 实拍 + 手标（主训练集）
│   ├── images/{train,val}   labels/{train,val}
│   └── summary.json         # 每类实例数
└── yolo_synth/              # 合成集（辅助/兜底）
```

配置：`01_vision/configs/data_real.yaml`（主）、`data_synth.yaml`（辅）。

## 硬约束速查（动手前先看）

- 地面站只能用 **MCU + ≥6 吋屏 + 按键**，禁止 PC 和通用键盘。
- **除无人机外不得有其它摄像头**；辅助定位装置须在地图区域及其上方空间之外。
- 飞行高度 **120±10cm**，全程 **≤300s**，单次测试不许换电池。
- 激光光斑 **≤1cm**、必须垂直向下；光斑轨迹不一致即判偏离航线，漏格扣分。
- 轴距 ≤45cm，桨叶必须全防护。
- 识别结果要带 **格名（AnBm）+ 类别 + 数量**，能实时显示、能存档回看、能汇总。

## 提交约定

- 分支：`main` 保持可复现；改动大时开 `feat/xxx` 分支。
- 提交信息：`feat|fix|docs|chore|test: 简短说明`（中文可）。
- **大文件不进库**：数据集、权重、录包已由 `.gitignore` 排除，模型权重发 Release 附件。
- 报告/文档用图请用 **PNG**（`.gitignore` 不再全局忽略 jpg，但 `datasets/`、`runs/` 整目录忽略）。
- 每次改动同步更新对应子目录的 `README.md`（现状 / 待办），避免目录树和实际脱节。
