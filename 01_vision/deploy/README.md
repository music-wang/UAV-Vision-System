# 01_vision/deploy — RK3588（鲁班猫 4）部署

把 `runs/2025h_synth/weights/best.onnx` 转成 `best.rknn`，并在板子上跑推理。

## 0. 先看清这个模型的实际输出

用你导出的 `best.onnx`（10.2 MB）实测：

```
INPUT   images  [1, 3, 640, 640]   float32, RGB, 0~1
OUTPUT  output0 [1, 9, 8400]       float32
        通道 0~3 = xywh，单位是 640 输入下的**像素**（不是归一化、不是格子）
        通道 4~8 = 5 类分数，**已经过 sigmoid**（0~1）
```

也就是说 `simplify=True` 导出时 **DFL 和 sigmoid 已经烘进图里**了。板端**不需要再解 DFL**，
只差四步：转置 → 阈值筛选 → xywh→xyxy → NMS。`rknn_inference.py` 就是按这个写的，
并且已经和 ultralytics 的结果逐框对齐验证过（同一张图：elephant 0.95 / peacock 0.95 /
tiger 0.94 / peacock 0.92，坐标误差 ≤2px）。

> 如果你的 ONNX 输出是 `[1, 84, 8400]` 之外的其他形状（比如多输出、或带 NMS 的 3 输出），
> 用 `netron` 打开确认；本脚本只按上面这一种写。

## 1. 转换环境（Ubuntu x86_64）

`rknn-toolkit2` 只有 **x86_64 Linux** 的轮子，Windows / ARM 上都装不了。

```bash
# 建议独立环境，python 3.10（2.3.x 支持 3.8~3.12）
conda create -n rknn python=3.10 -y
conda activate rknn

# 方式一：从 airockchip 仓库拿对应版本的 whl（推荐，版本要对齐板端 runtime）
git clone https://github.com/airockchip/rknn-toolkit2
pip install rknn-toolkit2/packages/rknn_toolkit2-2.3.*-cp310-cp310-linux_x86_64.whl

# 方式二：pip 直装
pip install rknn-toolkit2==2.3.2

# 转换过程要读图片做校准，装个 opencv
pip install opencv-python-headless numpy
```

## 2. 准备量化校准集

**优先用真实场地照片**（打印好的喷绘布 + 真实光照），没有就先拿合成集：

```bash
mkdir -p ~/calib && cp <你的实拍照片>/*.jpg ~/calib/     # 100~300 张
# 或者：
cp -r datasets/yolo/images/val ~/calib
```

校准集要覆盖：不同格子的地貌（森林/河流/干裂地）、有无动物、不同亮度。

## 3. 转换

```bash
# 先不量化，确认能转、精度不掉 —— 这一步排除"算子不支持"的问题
python convert_rknn.py --onnx best.onnx --out best_fp.rknn --no-quant

# 再开 int8 量化
python convert_rknn.py --onnx best.onnx --out best.rknn --calib-dir ~/calib --calib-n 200
```

关键参数（脚本里已固定，别改错）：

| 项 | 值 | 原因 |
|---|---|---|
| `mean_values` | `[[0,0,0]]` | 训练时只做了 /255，没有减均值 |
| `std_values` | `[[255,255,255]]` | 等价于输入 /255 |
| `quant_img_RGB2BGR` | `False` | 训练用 RGB，不能转 BGR |
| `target_platform` | `rk3588` | 鲁班猫 4 |
| batch | 1 | RKNN 动态 batch 不支持 |

## 4. 量化掉点怎么办

```bash
python convert_rknn.py --onnx best.onnx --out best.rknn --calib-dir ~/calib --analysis
# 会在 ./snapshot 生成逐层误差报告，看哪几层误差大
```

- 误差集中在检测头（最后几层 Conv）→ 用 hybrid 量化，把敏感层留在 FP16；
- 换更多真实照片做校准（合成图只有 15 种动物 + 1 张地貌，分布偏窄）；
- 实在不行就 `--no-quant` 出 FP16 模型：速度慢一些，但精度稳。

## 5. 板端运行（鲁班猫 4 / RK3588）

```bash
# rknn_toolkit_lite2 一般出厂镜像自带；没有就装对应 aarch64 whl
python3 -c "from rknnlite.api import RKNNLite; print('ok')"

python3 rknn_inference.py --model best.rknn --source test.jpg
python3 rknn_inference.py --model best.rknn --source ~/photos --out ~/result
python3 rknn_inference.py --model best.rknn --source test.jpg --benchmark 100
```

`--core 0/1/2/012` 选 NPU 核心，默认三核。yolo11n 640 int8 在 RK3588 上大致是
**每帧十几到几十毫秒**（跟核心数、散热、内存带宽有关，以 `--benchmark` 实测为准）。

## 6. 两个必踩的坑

1. **BGR / RGB**：`cv2.imread` 读的是 BGR，模型要 RGB。不转换的话框还能对上、类别会全错
   —— 我在验证时就踩了：同一批框，`peacock 0.76` vs 正确的 `elephant 0.95`。
   `rknn_inference.py` 里已经 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`。
2. **letterbox 要还原**：模型在 640×640 上推理，框是 letterbox 后的坐标。
   脚本里按 `ratio` 和 `(dw, dh)` 反算回原图，漏了这步框会整体偏移+缩放。

## 7. 接到比赛链路

`rknn_inference.py` 的输出是 `(x1, y1, x2, y2, score, cls)` 原图像素坐标，
取框心 `((x1+x2)/2, (y1+y2)/2)` 加上 SLAM 位姿，直接喂 `01_vision/scripts/cell_mapper.py`：

```python
from cell_mapper import CellMapper
mapper = CellMapper(fov_deg=你的视场角, alt_cm=120, size=(1280, 960))
mapper.update([(cls, cx, cy, score)], (x_cm, y_cm, yaw_deg), alt_cm=alt)
...
mapper.report()   # -> {'A3B5': {'elephant': 1}, ...}
```

这就是赛题基本要求 3 要发给地面站的"格号 + 类别 + 数量"。
