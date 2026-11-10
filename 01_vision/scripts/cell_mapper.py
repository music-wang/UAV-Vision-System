#!/usr/bin/env python3
"""把逐帧检测结果归到 9×7 方格并去重计数（赛题基本要求 3 的核心）。

链路：
  像素坐标 --(相机几何 + 机身位姿)--> 场地坐标(cm) --> 方格号 AnBm --> 目标跟踪去重 --> 每格计数

为什么在"场地坐标"里跟踪，而不是在图像里做 IoU 跟踪：
  无人机在移动，同一只动物在相邻帧里的像素位置会大幅跳动；但它在地面上的位置不动。
  所以在地图坐标系里按距离匹配（同一目标 20~25cm 内视为同一只）比图像 IoU 稳得多。

对外接口：
  mapper = CellMapper(fov_deg=66, alt_cm=120, size=(1280, 960))
  for each frame: mapper.update(dets, pose=(x_cm, y_cm, yaw_deg))
  report = mapper.report()      # {cell: {species: count}}

依赖：只用标准库，方便在机载端直接跑。
自测：python 01_vision/scripts/cell_mapper.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field

# 场地常量（cm），与 synth_dataset.py / 赛题说明保持一致
MAT_W_CM, MAT_H_CM = 500.0, 400.0
MARGIN_CM = 25.0
GRID_CM = 50.0
NX, NY = 9, 7
SPECIES = ["peacock", "wolf", "monkey", "elephant", "tiger"]


def cell_of(x_cm: float, y_cm: float) -> str | None:
    """场地坐标 -> 格号；落在 25cm 留白里返回 None。"""
    i = math.floor((x_cm - MARGIN_CM) / GRID_CM)
    j = math.floor((y_cm - MARGIN_CM) / GRID_CM)
    if 0 <= i < NX and 0 <= j < NY:
        return f"A{i + 1}B{j + 1}"
    return None


def cell_center(cell: str) -> tuple[float, float]:
    i = int(cell[1]) - 1
    j = int(cell[3]) - 1
    return (MARGIN_CM + (i + 0.5) * GRID_CM, MARGIN_CM + (j + 0.5) * GRID_CM)


@dataclass
class Track:
    cls: int
    x: float
    y: float
    hits: int = 1
    misses: int = 0
    conf_sum: float = 0.0
    cells: dict[str, int] = field(default_factory=dict)

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(self.hits, 1)

    def best_cell(self) -> str | None:
        if not self.cells:
            return None
        return max(self.cells.items(), key=lambda kv: kv[1])[0]


class CellMapper:
    """把像素检测投到地面并累积成"每格每类几只"。"""

    def __init__(
        self,
        fov_deg: float = 66.0,
        alt_cm: float = 120.0,
        size: tuple[int, int] = (1280, 960),
        match_cm: float = 25.0,
        confirm_hits: int = 3,
        max_miss: int = 3,
        min_conf: float = 0.25,
    ) -> None:
        self.fov = fov_deg
        self.alt = alt_cm
        self.w, self.h = size
        self.match_cm = match_cm
        self.confirm_hits = confirm_hits
        self.max_miss = max_miss
        self.min_conf = min_conf

        self.ground_w = 2 * alt_cm * math.tan(math.radians(fov_deg / 2))
        self.ground_h = self.ground_w * self.h / self.w
        self.tracks: list[Track] = []
        self.frames = 0

    # ---------- 几何 ----------

    def px_to_map(self, u: float, v: float, pose: tuple[float, float, float],
                  alt_cm: float | None = None) -> tuple[float, float]:
        """像素 -> 场地坐标。pose=(x_cm, y_cm, yaw_deg)，yaw 为机身偏航（图像逆时针为正）。

        alt_cm 传入则按该帧高度重算地面覆盖（定高有 ±10cm 波动时用）。
        """
        gw, gh = self.ground_w, self.ground_h
        if alt_cm is not None:
            gw = 2 * alt_cm * math.tan(math.radians(self.fov / 2))
            gh = gw * self.h / self.w
        dx = (u - self.w / 2) / self.w * gw                  # 相机系，右为正
        dy = (v - self.h / 2) / self.h * gh                  # 相机系，下为正
        t = math.radians(-pose[2])                          # 反变换：图像旋转 theta <=> 地图旋转 -theta
        c, s = math.cos(t), math.sin(t)
        # y 轴向下坐标系下的旋转矩阵
        rx = dx * c + dy * s
        ry = -dx * s + dy * c
        return pose[0] + rx, pose[1] + ry

    # ---------- 跟踪 ----------

    def update(self, dets, pose: tuple[float, float, float],
               alt_cm: float | None = None) -> None:
        """dets: [(cls:int, u, v, conf)] 或 [(cls, u, v, w, h, conf)]。"""
        self.frames += 1
        mapped = []
        for d in dets:
            cls, u, v, conf = d[0], d[1], d[2], d[-1]
            if conf < self.min_conf:
                continue
            x, y = self.px_to_map(u, v, pose, alt_cm)
            cell = cell_of(x, y)
            if cell is None:          # 落在留白/场地外，不计
                continue
            mapped.append((cls, x, y, conf, cell))

        used = [False] * len(self.tracks)
        for cls, x, y, conf, cell in mapped:
            best, best_d = None, self.match_cm
            for k, t in enumerate(self.tracks):
                if used[k] or t.cls != cls:
                    continue
                d = math.hypot(t.x - x, t.y - y)
                if d < best_d:
                    best, best_d = k, d
            if best is None:
                t = Track(cls=cls, x=x, y=y, conf_sum=conf)
                self.tracks.append(t)
                used.append(True)
            else:
                t = self.tracks[best]
                used[best] = True
                a = 0.7                                        # 位置平滑，抗抖动
                t.x = a * t.x + (1 - a) * x
                t.y = a * t.y + (1 - a) * y
                t.hits += 1
                t.misses = 0
                t.conf_sum += conf
            t.cells[cell] = t.cells.get(cell, 0) + 1

        for k, t in enumerate(self.tracks):
            if not used[k]:
                t.misses += 1
        # 注意：这里**不删**已确认的目标——动物一旦被看到，离开视野后仍要计入结果。
        # 只把长期失联的丢掉，防止误检把列表撑爆。
        if len(self.tracks) > 500:
            self.tracks = [t for t in self.tracks if t.misses <= self.max_miss]

    # ---------- 结果 ----------

    def report(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        # 同一只动物离开视野再回来后可能生成新轨迹；按"同类 + 地面距离 < 20cm"合并，
        # 否则一只动物会被数成两三只。
        kept: list[Track] = []
        for t in sorted(self.tracks, key=lambda x: -x.hits):
            if t.hits < self.confirm_hits:
                continue
            if any(t.cls == o.cls and math.hypot(t.x - o.x, t.y - o.y) < 20.0 for o in kept):
                continue
            kept.append(t)
        for t in kept:
            cell = t.best_cell()
            if cell is None:
                continue
            sp = SPECIES[t.cls] if 0 <= t.cls < len(SPECIES) else str(t.cls)
            out.setdefault(cell, {}).setdefault(sp, 0)
            out[cell][sp] += 1
        return out

    def totals(self) -> dict[str, int]:
        tot: dict[str, int] = {}
        for cell in self.report().values():
            for sp, n in cell.items():
                tot[sp] = tot.get(sp, 0) + n
        return tot


def detections_from_yolo(result, conf: float = 0.25):
    """把 ultralytics 的 Results 转成本模块需要的 (cls, u, v, conf)。"""
    dets = []
    if result.boxes is None:
        return dets
    for b in result.boxes:
        c = float(b.conf[0])
        if c < conf:
            continue
        cls = int(b.cls[0])
        x0, y0, x1, y1 = (float(v) for v in b.xyxy[0])
        dets.append((cls, (x0 + x1) / 2, (y0 + y1) / 2, c))
    return dets


# ---------------- 自测 ----------------

def _selftest() -> None:
    """构造一个已知答案的场景，验证几何与去重是否自洽。"""
    mapper = CellMapper(fov_deg=66, alt_cm=120, size=(1280, 960))
    # 一只"象"在 A3B5 格心，一只"狼"在 A4B4 格心
    targets = [(3, cell_center("A3B5")), (1, cell_center("A4B4"))]
    # 让机身分别掠过两个目标（每个目标 4 帧，保证 hits >= confirm_hits）
    poses = []
    for tx, ty in (t for _, t in targets):
        for k in range(4):
            poses.append((tx + (k - 1.5) * 12, ty + (k - 1.5) * 9, 20 * k - 30))
    for pose in poses:
        dets = []
        for cls, (tx, ty) in targets:
            # 正演：地图 -> 像素，必须与 px_to_map 互逆（M(+yaw) 对 M(-yaw)）
            t = math.radians(pose[2])
            c, s = math.cos(t), math.sin(t)
            dx, dy = tx - pose[0], ty - pose[1]
            rx = dx * c + dy * s
            ry = -dx * s + dy * c
            u = rx / mapper.ground_w * mapper.w + mapper.w / 2
            v = ry / mapper.ground_h * mapper.h + mapper.h / 2
            if 0 <= u < mapper.w and 0 <= v < mapper.h:
                dets.append((cls, u, v, 0.9))
        mapper.update(dets, pose)

    rep = mapper.report()
    print("自测结果：", json.dumps(rep, ensure_ascii=False), " 合计:", mapper.totals())
    ok = rep.get("A3B5", {}).get("elephant") == 1 and rep.get("A4B4", {}).get("wolf") == 1
    print("PASS" if ok else "FAIL：投影或去重有偏差")
    raise SystemExit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fov", type=float, default=66.0)
    ap.add_argument("--alt", type=float, default=120.0)
    ap.add_argument("--match-cm", type=float, default=25.0)
    ap.add_argument("--confirm-hits", type=int, default=3)
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    print("作为库使用：from cell_mapper import CellMapper")


if __name__ == "__main__":
    main()


