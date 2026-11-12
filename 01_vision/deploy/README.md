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

## 8. 板上验证清单

目标：确认「这个 .rknn 在鲁班猫 4 上能加载、推理结果与 PC 一致、速度够 30fps」。

### 8.0 一次配好免密（只需做一次）

板子挂同一网段时先确认能通：

```powershell
ping 192.168.137.2
ssh cat@192.168.137.2 "echo ok"     # 要输一次密码
```

把本仓库准备的部署公钥装进板子（之后所有 scp/ssh 都不再要密码）：

```powershell
type "D:\deepseek harness work space\deploy_key.pub" | ssh cat@192.168.137.2 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

# 验证免密
ssh -i "D:\deepseek harness work space\_deploy_key" -o BatchMode=yes cat@192.168.137.2 "echo key-ok"
```

### 8.1 在 Ubuntu x86 上产出 .rknn

```bash
python convert_rknn.py --onnx best.onnx --out best_fp.rknn --no-quant      # 先 FP16
python convert_rknn.py --onnx best.onnx --out best.rknn --calib-dir ~/calib --calib-n 200
```

### 8.2 一键推送 + 验证（PC 上跑）

```powershell
cd "D:\deepseek harness work space\UAV-Vision-System"
powershell -ExecutionPolicy Bypass -File 01_vision\deploy\board_test.ps1 `
  -Rknn "E:\best_fp16_rk3588.rknn" `
  -Image "01_vision\assets\predict_demo.jpg"
```

脚本会自动做完：连通性 → 传文件 → 版本与 md5 校验 → 单图推理（出 json）→ 三核 benchmark，
最后打印一张 `PASS / FAIL` 汇总表。

### 8.3 手工执行（脚本挂了时的兜底）

```bash
# ① 环境：三样版本必须能对上
cat /sys/kernel/debug/rknpu/version                     # NPU 驱动
strings /usr/lib/librknnrt.so | grep -E "^2\.[0-9]+\.[0-9]+"   # runtime
python3 -c "import rknnlite, sys; print(rknnlite.__file__)"    # lite

# ② 文件完整性
md5sum best_fp.rknn                                    # 与 PC 端对比

# ③ 单图推理（同时产出 json 供与 PC 逐框比对）
python3 rknn_inference.py --model best_fp.rknn --source test.jpg --json board.json

# ④ 三核速度（先 warmup 再计时，避免把首次加载算进去）
python3 rknn_inference.py --model best_fp.rknn --source test.jpg --core 012 --benchmark 200
for c in 0 1 2; do
  echo "--- core $c ---"
  python3 rknn_inference.py --model best_fp.rknn --source test.jpg --core $c --benchmark 200
done

# ⑤ 是否降频
cat /sys/class/thermal/thermal_zone*/temp
cat /sys/kernel/debug/rknpu/load
```

### 8.4 判定标准

| 检查项 | 通过标准 | 不通过怎么办 |
|---|---|---|
| 版本 | runtime / lite 均 ≥ 2.3.x，与转模型用的 toolkit 同版本 | 换 rknn-toolkit-lite2 版本，或回 x86 用同版本重转 |
| md5 | 与 PC 端一致 | 重新 scp（别用二进制模式被改写的 FTP） |
| 单图检出 | 类别/数量与 PC 上 `best.onnx` 一致（差 1 个框可接受） | 见 8.5 |
| 速度（FP16 三核） | ≤ 40 ms/帧（≥25 fps） | 走 int8，或降到 512 输入 |
| 速度（int8 三核） | ≤ 20 ms/帧（≥50 fps） | 检查是否只跑单核、是否降频 |
| 温度 | benchmark 前后温差 < 10°C，未触发降频 | 加散热/风扇，缩短连续推理 |

### 8.5 板端对不上 PC 时的排查顺序

1. **类别全错但框对** → 几乎一定是 **BGR/RGB**（§6 第 1 条）。
2. **框整体偏移/缩放** → letterbox 没还原，或 `data_format` 传错（`nhwc` vs `nchw`）。
3. **全 0 检出** → 多半是 **int8 量化把 sigmoid 输出压成 0**：换 FP16 验证；
   要 int8 就用 `export.py --format rknn` 重导（后处理进图，输出变成小张量）。
4. **检出数量少一半** → 置信度阈值不同（PC 上 0.25，板上默认也是 0.25，检查是否被 `--conf` 改过）。
5. **速度只有单核的 1/3** → `--core` 没生效，或模型不支持多核切分（小模型正常）。

## 9. 相机曝光与训练分布的匹配

模型对**亮度/对比度/边缘锐度**敏感，板端曝光一变，输入就偏离训练分布。实测基线
（D:\\UAV_Dataset\\images\\train，150 张中位数）：

| 指标 | 训练集基线 |
|---|---|
| 平均亮度 | **124** |
| 对比度（灰度 std） | **46** |
| 高光 P95 | **200** |
| 边缘密度 | **5.8** |

已知可用的一组板端参数（2026-11-13 实测，剪影/彩印卡都能出框）：

`
exposure = 2200     gain = 30/50      # 画面偏暗，但亮度落在训练分布内
`

比这更亮的设置会让白纸过曝、卡片糊成一块高亮，检测反而变差。

用脚本量化对比，而不是靠眼睛：

```powershell
# 训练集基线
python 01_vision/scripts/domain_stats.py --ref D:\UAV_Dataset\images\train --n 150

# 板端抓 20~30 张图放一个目录，再比
python 01_vision/scripts/domain_stats.py --ref D:\UAV_Dataset\images\train --test D:\frames
```

判定：**平均亮度、对比度、边缘密度三项都在 ±15% 以内**，说明相机设置进入了训练分布；
某一项偏得多就对应调曝光（亮度）/光照均匀性（对比度）/对焦（边缘密度）。

## 10. INT8 分数全 0 的根治：把输出拆成两个张量

### 病因

原模型最后一个节点是 `Concat(box[0~640px], score[0~1]) -> output0`。
RKNN 对输出做 **per-tensor** 量化，框的量程把整张张量的 scale 撑到约 640/127≈5，
于是分数 0.46 先除以 5 再取整就是 **0** —— 这就是 int8 下"框还在、类别分数全 0"的机制。
FP16 没有量化，所以一切正常。

### 解法（不改模型、不用 hybrid 量化）

拆成两个张量，各用各的刻度：`boxes[1,4,8400]` 精度约 ±2.5px，`scores[1,5,8400]` 约 ±0.004。

```powershell
# Windows 上（一次性）
python 01_vision/deploy/split_output_onnx.py --onnx best.onnx --verify
# -> best_2out.onnx，输出 boxes[1,4,8400] + scores[1,5,8400]
```

脚本会用 onnxruntime 对比拆分前后的数值，**必须看到 `[OK] ... 完全一致`** 再用。

```bash
# Ubuntu 上重新转 int8
python convert_rknn.py --onnx best_2out.onnx --out best_int8.rknn --calib-dir ~/uav/calib --calib-n 200

# 建议同时出一份 FP16 作为对照基线
python convert_rknn.py --onnx best_2out.onnx --out best_fp_2out.rknn --no-quant
```

### 板端不用改

`rknn_inference.py` 会看 `rknn.inference()` 返回几个数组：1 个按单输出解码，
2 个按 `boxes + scores` 解码。两种模型同一个命令。

```bash
python3 rknn_inference.py --model best_int8.rknn --source test.jpg --json int8.json
python3 rknn_inference.py --model best_fp_2out.rknn --source test.jpg --json fp.json
```

### 验收

| 检查 | 通过标准 |
|---|---|
| 分数是否恢复 | int8 输出里有 `peacock 0.8x` 这种分数，而不是全 0 |
| 与 FP16 一致 | 同一张图 `--json` 的类别/数量相同，分数差 < 0.05 |
| 速度 | 三核 int8 到 **12~20 ms/帧**（50~80 fps） |

> 若 int8 分数仍异常，再用 `--analysis` 看逐层误差，多半是校准集太少或与场景差异太大。
