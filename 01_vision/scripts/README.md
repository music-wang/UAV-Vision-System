# 01_vision/scripts

计划脚本清单（尚未实现）：

| 脚本 | 用途 |
|---|---|
| `split_templates.py` | **已实现**：把赛题《动物体态》整版图切成 15 张去底模板 + 地貌图缩略图 |
| `train.py` | 训练入口（YOLO 系列），产出权重与指标曲线 |
| `export.py` | 导出端侧格式（ONNX / RKNN / TensorRT engine） |
| `infer_video.py` | 离线跑视频，输出逐帧检测与可视化，用于回归对比 |
| `eval_map.py` | 在固定验证集上算 mAP / 召回，防止改模型后指标悄悄退化 |
| `calibrate.py` | 相机内参标定，产出 `configs/calib.yaml` |

约定：

- 所有脚本从 `configs/` 读参数，不硬编码路径。
- 输出统一写到 `../runs/`（已被 `.gitignore` 忽略），不要提交训练产物。
