# 05_ground_station — 地面站

> ⚠️ 本目录是新增的占位目录（原仓库只有 01~04）。地面站在基本要求里占两条（设置禁飞区 + 画 9×7 航线、
> 实时显示与存档回看），没有落点不合适；若你打算放到别的目录，直接删掉本目录即可。

## 硬约束

- 组成：**微控制器 + ≥6 吋显示屏 + 按键输入设备**。
- **不允许使用通用键盘和 PC 机**（赛题一）。
- 需要在屏上按 9×7 画出航线，且航线覆盖禁飞区以外的所有方格。

## 功能

| 编号 | 功能 | 对应要求 |
|---|---|---|
| G1 | 按键输入禁飞区格码（3 个连续格），屏上标灰 | 基本 1 |
| G2 | 生成并绘制 9×7 航线（覆盖除禁飞区外 60 格） | 基本 1 |
| G3 | 实时接收并显示"格名 + 动物名 + 数量" | 基本 3 |
| G4 | 事件存档，支持事后调出显示 | 基本 3 |
| G5 | 巡查结束显示各类动物总数 | 基本 4 |
| G6 | 下发航线、起飞/急停指令，显示飞行进度 | 基本 2 / 流程 |

## 建议硬件

| 部件 | 方案 |
|---|---|
| MCU | STM32F4/F1 或 ESP32（若走 WiFi 直连） |
| 屏 | ≥6 吋串口屏 / SPI TFT（6~9 吋在器材清单内） |
| 按键 | 矩阵键盘或 5 向键 + 确认/取消 |
| 存储 | Flash / SD 卡存事件表（掉电不丢） |
| 无线 | 与机载链路（WiFi / 数传 / 2.4G） |

## 通信协议（v1）

机载侧实现：`02_control_ros2/src/uav_vision_control/uav_vision_control/serial_bridge_node.py`，
协议纯逻辑在 `serial_proto.py`（可单测）。**本文档是给单片机同学的唯一依据。**

### 物理层

| 项 | 值 |
|---|---|
| 波特率 | 115200，8 数据位、无校验、1 停止位（8N1），无流控 |
| 编码 | 全 ASCII 可见字符 |
| 帧结束 | `\n`（0x0A）。MCU 侧按行缓冲，收到 `\n` 再解析 |
| 大小写 | 命令字/类型字**大写**，格号大写 |

### 帧格式

```
$<body>*<CRC8>\n
```

CRC 的计算范围是 **`$` 与 `*` 之间的 `body`**（不含 `$`、`*` 和换行），
按 body 的 ASCII 字节逐个计算。

### CRC-8（单片机照抄这段即可）

- 多项式 **0x07**，初值 **0x00**，输入/输出**不反转**，结果**不异或**（等价 CRC-8/ATM）
- **标准校验：`crc8("123456789") == 0xF4`** —— 先跑通这条再对接

```c
#include <stdint.h>
#include <stddef.h>

/* CRC-8: poly=0x07, init=0x00, 不反转, 无 xorout */
uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            if (crc & 0x80) crc = (uint8_t)((crc << 1) ^ 0x07);
            else            crc = (uint8_t)(crc << 1);
        }
    }
    return crc;
}
```

解析一行的建议流程：

```c
// line = 收到的一行（已去掉 '\n'），以 '$' 开头
// 1) 找到 '*'，它后面必须正好 2 位大写十六进制
// 2) body = line[1] .. '*' 之前（不含两端）
// 3) 若 crc8(body, len) != 收到的值 -> 整帧丢弃，不要回 ACK
// 4) 否则按 ',' 切分 body：第 0 段是类型（EV / ACK / CMD）
```

### 上行：机载 → 地面站

**检测事件**（只在某格计数**变大**时发送，不会重复刷同一格）：

```
$EV,<seq>,<cell>,<cls>,<cnt>,<conf>*<crc>\n
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `seq` | 1~65535 | 事件序号，循环递增。地面站可用于去重/发现丢帧 |
| `cell` | 字符串 | `AnBm`，如 `A3B5`；A1~A9 横向、B1~B7 纵向 |
| `cls` | 0~4 | 0 孔雀 / 1 狼 / 2 猴 / 3 象 / 4 虎（与 `01_vision/configs/classes.txt` 一致） |
| `cnt` | 整数 | 该格该类**当前累计**数量（不是增量） |
| `conf` | 0.00~1.00 | 该类目标的平均置信度，两位小数 |

**命令确认**（收到下行 CMD 后回一帧）：

```
$ACK,<CMD_NAME>*<crc>\n
```

### 下行：地面站 → 机载

```
$CMD,<NAME>,<ARG>*<crc>\n
```

| NAME | 含义 | ARG |
|---|---|---|
| `TAKEOFF` | 一键起飞（收到后开始自主巡查） | 0 |
| `ESTOP` | 急停（立即悬停/降落，最高优先级） | 0 |
| `LAND` | 返航降落（45° 俯角降回起飞区） | 0 |
| `RESET` | 清空统计与序号，准备下一次飞行 | 0 |
| `PING` | 心跳/链路检查，机载回 ACK 即可 | 0 |

约定：

- MCU 发出 CMD 后 **1 秒内**收不到 `ACK` 就重发，最多 3 次；3 次都失败按链路故障处理（屏上报警）。
- `ESTOP` 不做重发等待，连发 3 帧（间隔 20ms）更稳。
- 机载收到未知 NAME 时回 `$ACK,UNKNOWN*<crc>`，并在日志里告警。

### 测试向量（两边对拍用）

| body | CRC | 完整帧 |
|---|---|---|
| `123456789`（标准校验） | `0xF4` | —— |
| `EV,1,A3B5,3,1,0.88` | `0x18` | `$EV,1,A3B5,3,1,0.88*18` |
| `EV,2,A1B7,0,2,0.91` | `0xCE` | `$EV,2,A1B7,0,2,0.91*CE` |
| `EV,3,A9B3,4,1,0.74` | `0xC0` | `$EV,3,A9B3,4,1,0.74*C0` |
| `ACK,TAKEOFF` | `0xF1` | `$ACK,TAKEOFF*F1` |
| `CMD,TAKEOFF,0` | `0x66` | `$CMD,TAKEOFF,0*66` |
| `CMD,ESTOP,0` | `0xD1` | `$CMD,ESTOP,0*D1` |
| `CMD,LAND,0` | `0x96` | `$CMD,LAND,0*96` |
| `CMD,RESET,0` | `0x67` | `$CMD,RESET,0*67` |
| `CMD,PING,0` | `0x47` | `$CMD,PING,0*47` |

> 单片机同学只要保证：这几个 body 算出来的 CRC 与表中一致，就能和机载端互通。
> 两侧代码都有自测：机载侧 `python .../serial_proto.py` 会打印同样的向量。

### 速率与容量

- EV 帧平均 25 字节，115200 下约 4.6 ms/帧；一分钟最多约 200 条事件足够用。
- 地面站按 `(cell, cls)` 存表，`cnt` 用**覆盖**而不是累加（机载发的是累计值）。

## 联调（没有单片机时）

`scripts/gcs_sim.py` 就是"假单片机"：它收 EV/ACK、维护事件表、发 CMD 并按约定等 ACK，
**协议用的是机载侧同一份 `serial_proto.py`**，所以不会出现两边 CRC 不一致。

### ① 零硬件自测（先做这个 30 秒）

```powershell
python 05_ground_station/scripts/gcs_sim.py --self-test
```

会打印本文上面那张测试向量表并逐条比对，全 `OK` 且最后出现 `[PASS]` 才继续。

### ② 本机虚拟串口（Linux / 板端，推荐，不用装 socat）

```bash
python3 05_ground_station/scripts/gcs_sim.py --loopback
# [OK] 本机虚拟串口：/dev/pts/3
#      让机载节点连这个路径：
#       ros2 run uav_vision_control serial_bridge --ros-args -p port:=/dev/pts/3
```

另开一个终端，把上面那行 `serial_bridge` 跑起来，两边就通了。敲 `t` 发 TAKEOFF，
机载侧会回 ACK，本脚本打印 `[OK] TAKEOFF 已确认`。

### ③ socat 手工建串口对（Linux）

```bash
sudo apt install socat
socat -d -d pty,raw,echo=0,link=/tmp/ttyGCS pty,raw,echo=0,link=/tmp/ttyUAV

# 终端 A（机载侧）
ros2 run uav_vision_control serial_bridge --ros-args -p port:=/tmp/ttyUAV
# 终端 B（地面站模拟）
python3 05_ground_station/scripts/gcs_sim.py --port /tmp/ttyGCS
```

### ④ Windows：com0com

装 [com0com](https://sourceforge.net/projects/com0com/)，建一对（如 `COM10 <-> COM11`），
`gcs_sim.py --port COM11`。另一端 `COM10` 给机载节点——但 ROS 2 一般跑在 Linux，
Windows 这边通常只用来做 `--self-test` 和看帧格式。

### 交互与自动化

| 按键 | 作用 |
|---|---|
| `t` / `e` / `l` / `r` / `p` | 发 TAKEOFF / ESTOP / LAND / RESET / PING |
| `s` | 打印事件表与汇总 |
| `q` | 退出（退出时自动打统计与汇总） |

```bash
# 自动跑一遍流程：2 秒后起飞，15 秒后降落
python3 gcs_sim.py --loopback --auto --auto-land 15
```

### 期望输出

```
[OK] 本机虚拟串口：/dev/pts/3
[TX] b'$CMD,TAKEOFF,0*66\n'
[RX] ACK TAKEOFF（无待确认命令）        ← 机载侧回了 ACK
[OK] TAKEOFF 已确认（第 1 次发送）
[RX] #1    A3B5 elephant ×1 (conf 0.88)
[RX] #2    A1B7 peacock ×2 (conf 0.91)
[RX] #4    A9B3 tiger ×5 (conf 0.70)
```

`s` 打印：

```
---- 事件表（格号 / 类别 / 数量 / 置信度）----
   A1B7  peacock   ×2   conf=0.91
   A3B5  elephant  ×1   conf=0.88
   A9B3  tiger     ×5   conf=0.70
   汇总: peacock×2  elephant×1  tiger×5
   序号 last=4 丢帧=0 重复=1
```

### 换真单片机时改什么

**什么都不用改**：把 `--port` 从虚拟串口换成真实串口（`/dev/ttyUSB0`）即可，
协议、重发、表格逻辑全部不变。

## 界面草图（待画）

```
┌──────────────────────────────┐
│ 禁飞区: A4B3 A4B4 A4B5  [设] │
│ ┌──────────────────────────┐ │
│ │ 9×7 网格 + 航线 + 进度   │ │
│ └──────────────────────────┘ │
│ 事件: A3B5 象 ×1             │
│ 汇总: 象4 虎2 狼2 猴1 孔雀1  │
└──────────────────────────────┘
```

## 现状

- 协议 **v1 已定稿**（见上：帧格式、CRC-8、测试向量、ACK/重发约定）。
- 机载侧桥接已实现：`02_control_ros2` 的 `serial_bridge` 节点
  （`/vision/cells` → 串口 EV 帧；串口 CMD → `/gcs/command`）。
- 地面站固件（9×7 屏上绘制、按键交互、事件存档回看）**尚未开始**。

## 待办

- [ ] 定屏与 MCU，搭最小系统点亮屏
- [ ] 9×7 网格绘制 + 禁飞区输入交互
- [ ] 航线生成算法（boustrophedon + 禁飞区绕行），与机载侧同一份实现/同一份数据
- [ ] 事件表存储与回看界面
- [x] 协议定稿（v1；CRC-8 标准校验 `"123456789" -> 0xF4`）
- [ ] 单片机侧按 v1 写解析/组帧，先用测试向量对拍
- [x] 地面站模拟器 scripts/gcs_sim.py（自测 / loopback / socat / com0com）
