#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""地面站模拟器：在 PC 上扮演单片机，把「机载 → 地面站」这条链路先跑通。

它自己就是协议的另一端：收 EV/ACK 帧、维护事件表、发 CMD 帧并按约定等 ACK（1s 超时重发 3 次）。
协议实现直接复用机载侧同一份 `serial_proto.py`（只读导入，不装 ROS 也能跑），
所以**两边不会各写一套 CRC**。

三种用法：

  1) 零硬件自测（先验证算法与帧格式）
     python 05_ground_station/scripts/gcs_sim.py --self-test

  2) 接虚拟串口联调（Linux）
     socat -d -d pty,raw,echo=0,link=/tmp/ttyGCS pty,raw,echo=0,link=/tmp/ttyUAV
     # 机载节点连 /tmp/ttyUAV，本脚本连 /tmp/ttyGCS
     python 05_ground_station/scripts/gcs_sim.py --port /tmp/ttyGCS

  3) 等真实单片机（Windows 用 com0com 建 COM10<->COM11 对）
     python 05_ground_station/scripts/gcs_sim.py --port COM11

交互：t 起飞 / e 急停 / l 降落 / r 复位 / p ping / s 打印汇总 / q 退出
自动化：--auto 会在 2s 后发 TAKEOFF，--auto-land 15 表示 15s 后发 LAND。
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

# ---- 复用机载侧的协议实现（不依赖 ROS）----
REPO = Path(__file__).resolve().parents[2]
PROTO_DIR = REPO / "02_control_ros2" / "src" / "uav_vision_control" / "uav_vision_control"
if not (PROTO_DIR / "serial_proto.py").is_file():
    raise SystemExit(f"找不到 serial_proto.py：{PROTO_DIR}")
sys.path.insert(0, str(PROTO_DIR))
import serial_proto as proto  # noqa: E402

CLASSES = ["peacock", "wolf", "monkey", "elephant", "tiger"]

KEY_CMD = {"t": "TAKEOFF", "e": "ESTOP", "l": "LAND", "r": "RESET", "p": "PING"}


class PtyPort:
    """本机虚拟串口（Linux 才有）：直接开一对 pty，省掉 socat。

    用法：gcs_sim.py --loopback
      -> 打印形如 /dev/pts/3 的路径，让 serial_bridge 连这个路径即可。
    """

    def __init__(self) -> None:
        import os
        import pty
        import tty

        self.master, self.slave = pty.openpty()
        tty.setraw(self.master)
        tty.setraw(self.slave)
        self.path = os.ttyname(self.slave)

    def read(self, n: int = 256) -> bytes:
        import os
        import select

        ready, _, _ = select.select([self.master], [], [], 0.1)
        return os.read(self.master, n) if ready else b""

    def write(self, data: bytes) -> int:
        import os

        return os.write(self.master, data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        import os

        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass


class EventTable:
    """地面站侧的事件表：cnt 用覆盖（机载发的是累计值）。"""

    def __init__(self) -> None:
        self.data: dict[tuple[str, int], int] = {}
        self.conf: dict[tuple[str, int], float] = {}
        self.last_seq = 0
        self.gaps = 0
        self.dups = 0

    def update(self, seq: int, cell: str, cls: int, cnt: int, conf: float) -> str:
        if seq <= self.last_seq:
            # 序号不增长：重复帧或乱序。仍按"覆盖"更新，但要报出来
            self.dups += 1
            flag = " (重复/乱序)"
        else:
            if self.last_seq and seq > self.last_seq + 1:
                self.gaps += seq - self.last_seq - 1
            self.last_seq = seq
            flag = ""
        self.data[(cell, cls)] = cnt
        self.conf[(cell, cls)] = conf
        return flag

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {c: 0 for c in CLASSES}
        for (cell, cls), cnt in self.data.items():
            name = CLASSES[cls] if 0 <= cls < len(CLASSES) else str(cls)
            out[name] = out.get(name, 0) + cnt
        return out

    def dump(self) -> None:
        print("\n---- 事件表（格号 / 类别 / 数量 / 置信度）----")
        if not self.data:
            print("   （空）")
        for (cell, cls), cnt in sorted(self.data.items()):
            name = CLASSES[cls] if 0 <= cls < len(CLASSES) else str(cls)
            print(f"   {cell:<5} {name:<9} ×{cnt}   conf={self.conf.get((cell, cls), 0):.2f}")
        tot = {k: v for k, v in self.totals().items() if v}
        print("   汇总:", "  ".join(f"{k}×{v}" for k, v in tot.items()) or "无")
        print(f"   序号 last={self.last_seq} 丢帧={self.gaps} 重复={self.dups}\n")


class GcsSim:
    def __init__(self, args) -> None:
        self.args = args
        self.table = EventTable()
        self.tx_frames = 0
        self.rx_frames = 0
        self.crc_err = 0
        self.pending: dict[str, dict] = {}      # name -> {'deadline': ts, 'tries': n}
        self.cmd_q: queue.Queue[str] = queue.Queue()
        self.ser = None
        self.stop = threading.Event()
        self.lock = threading.Lock()

    # ---------- 串口 ----------

    def open(self) -> None:
        if self.args.loopback:
            try:
                self.ser = PtyPort()
            except ImportError:
                raise SystemExit("--loopback 只在 Linux/板端可用；Windows 请用 com0com 建一对虚拟串口")
            print(f"[OK] 本机虚拟串口：{self.ser.path}")
            print(f"     让机载节点连这个路径：")
            print(f"       ros2 run uav_vision_control serial_bridge --ros-args -p port:={self.ser.path}")
            return

        import serial

        try:
            self.ser = serial.Serial(self.args.port, self.args.baud, timeout=0.1)
        except Exception as exc:                    # noqa: BLE001
            raise SystemExit(
                f"打不开串口 {self.args.port}：{exc}\n"
                "Linux 先建虚拟串口对：\n"
                "  socat -d -d pty,raw,echo=0,link=/tmp/ttyGCS pty,raw,echo=0,link=/tmp/ttyUAV\n"
                "Windows 用 com0com 建一对（如 COM10<->COM11），本脚本连其中一个") from exc
        print(f"[OK] 串口已打开 {self.args.port} @ {self.args.baud}")

    def send(self, name: str, arg: int = 0) -> None:
        frame = proto.build_frame(f"CMD,{name},{arg}")
        with self.lock:
            try:
                self.ser.write(frame)
                self.ser.flush()
                self.tx_frames += 1
            except Exception as exc:                # noqa: BLE001
                print(f"[X] 写串口失败：{exc}")
                return
        print(f"[TX] {frame!r}")

    # ---------- 收发 ----------

    def reader(self) -> None:
        buf = b""
        while not self.stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as exc:                # noqa: BLE001
                print(f"[X] 读串口异常：{exc}")
                time.sleep(0.1)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                self.handle_line(line)

    def handle_line(self, line: bytes) -> None:
        parsed = proto.parse_frame(line)
        if parsed is None:
            self.crc_err += 1
            print(f"[!] 非法帧（CRC/格式）：{line!r}")
            return
        kind, fields = parsed
        self.rx_frames += 1

        if kind == "EV" and len(fields) == 5:
            try:
                seq, cell, cls, cnt, conf = (int(fields[0]), fields[1], int(fields[2]),
                                             int(fields[3]), float(fields[4]))
            except ValueError:
                print(f"[!] EV 字段解析失败：{fields}")
                return
            flag = self.table.update(seq, cell, cls, cnt, conf)
            name = CLASSES[cls] if 0 <= cls < len(CLASSES) else str(cls)
            print(f"[RX] #{seq:<4} {cell} {name} ×{cnt} (conf {conf:.2f}){flag}")
        elif kind == "ACK" and fields:
            name = fields[0].upper()
            with self.lock:
                p = self.pending.pop(name, None)
            if p:
                print(f"[OK] {name} 已确认（第 {p['tries']} 次发送）")
            else:
                print(f"[RX] ACK {name}（无待确认命令）")
        else:
            print(f"[RX] {kind} {fields}")

    def stdin_loop(self) -> None:
        while not self.stop.is_set():
            try:
                line = input()
            except EOFError:
                return
            key = line.strip().lower()[:1]
            if key == "q":
                self.stop.set()
                return
            if key == "s":
                self.table.dump()
                continue
            if key in KEY_CMD:
                self.cmd_q.put(KEY_CMD[key])
            elif key:
                print("   t 起飞 / e 急停 / l 降落 / r 复位 / p ping / s 汇总 / q 退出")

    def issue(self, name: str) -> None:
        """发命令并登记等待 ACK；ESTOP 特殊：连发 3 帧不等确认。"""
        if name == "ESTOP":
            for _ in range(3):
                self.send("ESTOP")
                time.sleep(0.02)
            return
        self.send(name)
        with self.lock:
            self.pending[name] = {"deadline": time.time() + 1.0, "tries": 1}

    # ---------- 主循环 ----------

    def run(self) -> None:
        self.open()
        threading.Thread(target=self.reader, daemon=True).start()
        threading.Thread(target=self.stdin_loop, daemon=True).start()

        if self.args.auto:
            time.sleep(2.0)
            self.issue("TAKEOFF")
        if self.args.auto_land:
            threading.Timer(self.args.auto_land, lambda: self.issue("LAND")).start()

        print("就绪。直接敲 t/e/l/r/p 发命令，s 看汇总，q 退出。")
        try:
            while not self.stop.is_set():
                try:
                    name = self.cmd_q.get(timeout=0.05)
                    self.issue(name)
                except queue.Empty:
                    pass
                # ACK 超时重发
                now = time.time()
                retry = []
                with self.lock:
                    for name, p in list(self.pending.items()):
                        if now >= p["deadline"]:
                            retry.append((name, p))
                for name, p in retry:
                    if p["tries"] >= 3:
                        print(f"[X] {name} 连续 3 次无 ACK，按链路故障处理")
                        with self.lock:
                            self.pending.pop(name, None)
                        continue
                    print(f"[!] {name} 超时，重发（第 {p['tries'] + 1} 次）")
                    self.send(name)
                    with self.lock:
                        p["tries"] += 1
                        p["deadline"] = time.time() + 1.0
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            time.sleep(0.05)
            print(f"\n[统计] RX {self.rx_frames} 帧，TX {self.tx_frames} 帧，非法帧 {self.crc_err}")
            self.table.dump()
            try:
                self.ser.close()
            except Exception:                       # noqa: BLE001
                pass


def self_test() -> None:
    """零硬件：验证 CRC、帧格式、解析、以及测试向量与 README 一致。"""
    print("== 协议自测（与 05_ground_station/README 的测试向量一致）==")
    checks = [
        ("123456789", 0xF4),
        ("EV,1,A3B5,3,1,0.88", 0x18),
        ("EV,2,A1B7,0,2,0.91", 0xCE),
        ("EV,3,A9B3,4,1,0.74", 0xC0),
        ("ACK,TAKEOFF", 0xF1),
        ("CMD,TAKEOFF,0", 0x66),
        ("CMD,ESTOP,0", 0xD1),
        ("CMD,LAND,0", 0x96),
        ("CMD,RESET,0", 0x67),
        ("CMD,PING,0", 0x47),
    ]
    ok = True
    for body, want in checks:
        got = proto.crc8(body.encode())
        flag = "OK " if got == want else "FAIL"
        ok &= got == want
        print(f"   [{flag}] {body!r:<24} crc=0x{got:02X}  期望 0x{want:02X}")

    ev = proto.encode_ev(1, "A3B5", 3, 1, 0.88)
    print(f"\n   组帧: {ev!r}")
    ok &= ev == b"$EV,1,A3B5,3,1,0.88*18\n"
    ok &= proto.parse_frame(ev) == ("EV", ["1", "A3B5", "3", "1", "0.88"])

    bad = b"$EV,1,A3B5,3,1,0.88*00\n"
    ok &= proto.parse_frame(bad) is None
    print("   坏 CRC 被拒绝:", proto.parse_frame(bad) is None)

    print("\n[PASS] 协议自测通过，可以接串口了" if ok else "\n[FAIL] 有向量不一致，别接硬件")
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="零硬件自测 CRC 与帧格式")
    ap.add_argument("--port", default="", help="串口：/tmp/ttyGCS 或 COM11")
    ap.add_argument("--loopback", action="store_true",
                    help="本机开一对 PTY 当虚拟串口（Linux/板端，无需 socat）")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--auto", action="store_true", help="2 秒后自动发 TAKEOFF")
    ap.add_argument("--auto-land", type=float, default=0.0,
                    help="N 秒后自动发 LAND（例如 15）")
    args = ap.parse_args()

    if args.self_test:
        self_test()
    if not args.port and not args.loopback:
        ap.error("需要 --port（或用 --loopback / --self-test）")
    GcsSim(args).run()


if __name__ == "__main__":
    main()
