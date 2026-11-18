#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地面站串口协议（纯逻辑，不依赖 ROS / pyserial，方便单测和对拍）。

帧格式：  $<body>*<CRC8 两位大写十六进制>\\n

CRC：    CRC-8，多项式 0x07，初值 0x00，不反转，无异或输出
         （等价于 CRC-8/ATM，很多单片机例程里就叫 "crc8"）
         计算范围 = **$ 与 * 之间的 body 的 ASCII 字节**（不含 '$' 和 '*'）

上行（机载 → 地面站）
    EV,<seq>,<cell>,<cls>,<cnt>,<conf>     检测事件，seq 从 1 递增
    ACK,<CMD>                              对下行命令的确认

下行（地面站 → 机载）
    CMD,<NAME>,<ARG>                       NAME: TAKEOFF/ESTOP/LAND/RESET/PING

示例：
    $EV,1,A3B5,3,1,0.88*4F\\n
    $CMD,TAKEOFF,0*9A\\n
"""
from __future__ import annotations

FRAME_START = "$"
FRAME_SEP = "*"
FRAME_END = b"\n"

# 下行命令字
CMD_NAMES = ("TAKEOFF", "ESTOP", "LAND", "RESET", "PING")


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """CRC-8/ATM：poly=0x07, init=0x00, 无反转, 无 xorout。"""
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def build_frame(body: str) -> bytes:
    """body -> '$body*HH\\n'（CRC 只算 body）。"""
    payload = body.encode("ascii")
    return f"{FRAME_START}{body}{FRAME_SEP}{crc8(payload):02X}".encode("ascii") + FRAME_END


def encode_ev(seq: int, cell: str, cls: int, cnt: int, conf: float) -> bytes:
    """检测事件帧。conf 保留两位小数（0.00~1.00）。"""
    body = f"EV,{int(seq) & 0xFFFF},{cell},{int(cls)},{int(cnt)},{float(conf):.2f}"
    return build_frame(body)


def encode_ack(cmd_name: str) -> bytes:
    return build_frame(f"ACK,{cmd_name}")


def parse_frame(line: bytes) -> tuple[str, list[str]] | None:
    """解析一行（可带或不带结尾换行）。

    返回 (kind, fields)，kind 是 body 的第一段（EV/ACK/CMD）；
    CRC 不对、格式不对、非 ASCII 都返回 None（调用方丢弃该帧）。
    """
    s = line.strip(b"\r\n\x00 ")
    if not s.startswith(b"$"):
        return None
    try:
        text = s.decode("ascii")
    except UnicodeDecodeError:
        return None
    if FRAME_SEP not in text:
        return None
    body, _, crc_text = text[1:].rpartition(FRAME_SEP)
    if len(crc_text) != 2:
        return None
    try:
        want = int(crc_text, 16)
    except ValueError:
        return None
    if crc8(body.encode("ascii")) != want:
        return None
    fields = body.split(",")
    return fields[0], fields[1:]


# ---------------- 自测 ----------------

# 与 README 里给单片机队友的测试向量保持一致，改协议时必须同步更新
TEST_VECTORS = [
    ("EV,1,A3B5,3,1,0.88", 0x00),   # 占位，下面 __main__ 会打印真实值
    ("CMD,TAKEOFF,0", 0x00),
    ("CMD,ESTOP,0", 0x00),
]


def _selftest() -> None:
    print("CRC-8 (poly=0x07, init=0x00) 测试向量：")
    for body, _ in TEST_VECTORS:
        print(f"    {body!r:<28} -> 0x{crc8(body.encode()):02X}")

    # 往返：组帧 -> 解析 -> 字段一致
    frame = encode_ev(7, "A3B5", 3, 1, 0.88)
    print("\n组帧示例：", frame)
    kind, fields = parse_frame(frame)
    assert (kind, fields) == ("EV", ["7", "A3B5", "3", "1", "0.88"]), (kind, fields)

    # 坏 CRC 必须被拒绝
    bad = bytearray(frame)
    bad[-3] = ord("0") if bad[-3] != ord("0") else ord("1")
    assert parse_frame(bytes(bad)) is None, "坏 CRC 没被拒绝"

    # 下行命令
    cmd = build_frame("CMD,TAKEOFF,0")
    print("下行示例：", cmd, "->", parse_frame(cmd))
    assert parse_frame(cmd) == ("CMD", ["TAKEOFF", "0"])

    print("\n[OK] 协议自测通过")


if __name__ == "__main__":
    _selftest()
