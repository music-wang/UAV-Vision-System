# v2.0 — 真实数据版（249 张实拍）

> 这一天开始，训练集从"程序合成的虚拟场地"换成**真实拍摄 + 手工标注**。
> 旧版合成集有两个致命问题（整个数据集只有一张场地 → 狼类 0 实例、孔雀占 45% 成默认类），
> 这一版全部修掉。

## 模型指标（val 51 张 / 214 框）

| 指标 | 值 |
|---|---|
| mAP50 | **0.995** |
| mAP50-95 | **0.730** |
| Precision | 0.993 |
| Recall | 0.991 |
| 每类 mAP50-95 | 猴 0.772 / 虎 0.767 / 孔雀 0.711 / 狼 0.709 / 象 0.688 |
| GPU 推理（5070 Laptop, 640） | 4.8 ms/帧 |

五类均衡（最大/最小差 0.08），**没有塌缩类**；val 中的纯背景负样本零误检。

## 数据集

| 项目 | 值 |
|---|---|
| 实拍照片 | 249 张（199 train / 50 val，8:2） |
| 标注框 | 1089 个，平均 4.37 个/图 |
| 类别分布 | 孔雀 249 / 狼 251 / 猴 196 / 象 205 / 虎 188 |
| 负样本 | 7 张纯背景（绿垫子、蓝篷布、桌面、书本、地面纹理） |
| 图片尺寸 | 长边 1280（由 16MP 原图缩至，加速 3 倍、省 87% 空间） |

## 新增工具链

| 脚本 | 用途 |
|---|---|
| `scripts/prepare_real.py` | 拍完的照片整理成数据集（8:2 划分、无标注图转负样本） |
| `scripts/check_dataset.py` | **训练前体检**：某类 0 实例 / 图标签不匹配 / 框越界 → 直接失败 |
| `scripts/clean_labels.py` | 清洗标注：幽灵类 id、越界框、坏行（默认 dry-run） |
| `scripts/resize_images.py` | 批量缩图（长边 1280，原地原子替换） |
| `scripts/add_negatives.py` | 把纯背景照片补成空 txt 负样本并入数据集 |
| `scripts/make_print_sheet.py` | 按赛题真实尺寸打印靶标（象 30cm、其余 15cm，带 10cm 标尺） |
| `scripts/verify_real.py` | 实拍验证：真实照片 vs 手标真值，出误差表 |
| `deploy/convert_rknn.py` | ONNX → RKNN（FP16 / int8 + 校准集） |
| `deploy/rknn_inference.py` | 鲁班猫 4 板端推理（rknnlite + 解码 + NMS） |

## 附件说明

| 文件 | 说明 |
|---|---|
| `best.pt` | PyTorch 权重（5.25 MB），继续训练或导出用 |
| `best.onnx` | ONNX 10.2 MB，输入 `[1,3,640,640]`、输出 `[1,9,8400]`（已含 DFL + sigmoid，板端无需再解 DFL） |
| `results.png` | 训练曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `BoxPR_curve.png` | PR 曲线 |
| `summary.json` | 指标汇总 |
| `results.csv` | 逐轮指标 |

## 快速开始

```powershell
# 训练（数据放在 D:\UAV_Dataset）
python 01_vision/scripts/check_dataset.py
python 01_vision/scripts/train.py --data D:/UAV_Dataset/data.yaml --device 0 --epochs 200 --cache

# 导出
python 01_vision/scripts/export.py --weights runs/2025h_real_v2/weights/best.pt --format onnx --imgsz 640
```

```bash
# 转 RKNN（Ubuntu x86 + rknn-toolkit2 2.3.x）
python convert_rknn.py --onnx best.onnx --out best_fp.rknn --no-quant
```

## 已知限制

- **val 与 train 同一次拍摄**（同手机、同光照、同场地），0.73 是同域指标；
  跨拍摄的泛化能力**尚未验证**，需用 `verify_real.py` 在另一次拍摄上测。
- RKNN **尚未在鲁班猫 4 上实测**（FP16 模型待板上 benchmark）。
- 负样本仅 7 张（占 2.9%），对误检的抑制有限，建议补到 40~60 张（15~20%）。
- 兼容性：Ultralytics 8.4.x、torch 2.11+cu128、rknn-toolkit2 2.3.x（板端 runtime 需匹配）。
