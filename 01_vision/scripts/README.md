# 01_vision/scripts

| 脚本 | 状态 | 用途 |
|---|---|---|
| `split_templates.py` | ✅ | 把赛题《动物体态》整版图切成 15 张去底模板 + 地貌图缩略图 |
| `synth_dataset.py` | ✅ | 渲染虚拟场地并采样相机视角，合成 YOLO 训练集 |
| `train.py` | ✅ | 训练 + 验证 + 预测可视化，输出 `runs/<name>/summary.json` |
| `export.py` | ✅ | 导出 ONNX / TensorRT / OpenVINO 等部署格式 |
| `cell_mapper.py` | ✅ | **归格计数**：像素→场地坐标→AnBm 格，按地图坐标跟踪去重（`--selftest` 可自检） |
| `sim_mission.py` | ✅ | 按航线扫场，生成带逐帧位姿的任务序列（可离线验证整条链路） |
| `predict_demo.py` | ✅ | 单独出预测图并统计检出（训练完可反复跑，不用重训） |
| `eval_mission.py` | ✅ | 跑检测 + 归格计数，与场地真值对比，输出 `report.json` |
| `infer_video.py` | ⬜ | 离线跑视频，输出逐帧检测与可视化，用于回归对比 |
| `calibrate.py` | ⬜ | 相机内参标定，产出 `configs/calib.yaml` |

## 本机 GPU 正式训练（推荐）

前提：装了带 CUDA 的 PyTorch（`python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`），
数据在 `datasets/yolo/`（`synth_dataset.py` 生成，已被 gitignore）。

```powershell
conda activate yolo                      # 你的 ultralytics/torch 装哪个环境就用哪个
python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"

python 01_vision/scripts/train.py --device 0 --epochs 60 --batch 16 --imgsz 640 --workers 8
```

- `--batch 32` 显存够（≥12GB）再上；8GB 笔记本卡用 16，OOM 就降到 8。
- 内存 ≥32GB 可加 `--cache`，每个 epoch 少读盘。
- 只在受限环境（沙箱禁多进程）才需要 `--serial-scan` 和 `--workers 0`，正常本机别加。
- 产物：`runs/2025h_synth/weights/best.pt`、`summary.json`、`runs/2025h_synth_demo/*.jpg`。
- 参考耗时：RTX 5070 Laptop + yolo11n + 800 张，约 10~15s/epoch，60 轮约 10~15 分钟。

## 导出 ONNX（给 RKNN 用）

```powershell
python 01_vision/scripts/export.py --format onnx --imgsz 640
# RKNN 报 opset 不支持时：
python 01_vision/scripts/export.py --format onnx --imgsz 640 --opset 12
```

默认就是 RKNN 需要的形状：**batch=1、静态 shape、simplify=True**。不要加 `--half` 或 `--dynamic`。
需要 INT8 量化校准集时加 `--int8`（会自动把 `configs/data.yaml` 解析成绝对路径传给 ultralytics）。

## split_templates.py

```powershell
python 01_vision/scripts/split_templates.py `
  --sheet datasets/raw/动物体态.png `
  --terrain datasets/raw/地貌图.png `
  --out 01_vision/assets
```

## synth_dataset.py

```powershell
# 正式生成（实测约 2.4s/张，800+100 张约 35 分钟）
python 01_vision/scripts/synth_dataset.py --n-train 800 --n-val 100

# 小批量自查：把标签画回图片，并输出整幅场地预览
python 01_vision/scripts/synth_dataset.py --n-train 20 --n-val 5 --debug-overlay `
  --preview datasets/yolo/mat_preview.jpg

# 相机/场地参数不是默认值时手动覆盖
python 01_vision/scripts/synth_dataset.py --fov 82 --alt 120 --size 1920x1080 --animals 10
```

输出（默认 `datasets/yolo/`，已被 `.gitignore` 忽略）：

```
datasets/yolo/
├── images/train/*.jpg      labels/train/*.txt
├── images/val/*.jpg        labels/val/*.txt
├── manifest.json           # 禁飞区格号、每只动物的格号/类别/真实坐标(cm)/长度/角度
└── mat_preview.jpg         # 传了 --preview 时才有：整幅场地渲染
```

`manifest.json` 是后面做**打点精度评测**和**地面站联调**的真值来源，别删。

参数说明（都影响"像不像实拍"）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--px-per-cm` | 10 | 场地渲染精度，10 = 1px/mm。降到 5 更快但动物更糊 |
| `--fov` | 66 | 相机水平视场角，决定单帧看到多大范围（按你的镜头改） |
| `--alt` | 120 | 巡航高度 cm，采样时在 ±10cm 内抖动 |
| `--size` | 1280x960 | 输出分辨率，与训练/部署分辨率一致 |
| `--animals` | 8 | 场地上一共贴几只动物（60 格里 5~10 只比较接近实测） |
| `--anchor` | 0.5 | 按目标居中采样的比例，太低会导致负样本压倒性多 |
| `--blackline-pos` | outside | 1.8cm 黑线画在 25cm 留白里（外侧）还是压在格线内侧 |

## 约定

- 所有脚本从命令行 / `configs/` 取参数，不硬编码本机路径。
- 训练产物写 `runs/`、合成数据写 `datasets/`，两者都被 `.gitignore` 忽略，不要提交。


