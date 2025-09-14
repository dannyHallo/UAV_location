"""
Straight-line trajectory generator (returns small consecutive segments as (line_a, line_b))
- Inside the quadrilateral (scaled inner area)
- Free direction straight motion, constant speed
- Each produced trajectory is sampled at fixed time interval and exported as
  line_a = P[0:-1], line_b = P[1:]
- Speed fixed at 30 m/s by requirement; min segment length >= 50 m
"""

import random
from typing import List, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon as SPolygon, LineString, Point
from matplotlib import pyplot as plt

import src.config as config


class StraightLineTrajectoryGenerator:
    """Generate straight-line trajectories as consecutive (line_a, line_b) small segments."""

    def __init__(self, time_interval: float = None):
        # 固定速度 30 m/s（按需求）
        self.velocity: float = 30.0
        # 采样时间间隔，默认使用项目配置
        self.time_interval: float = (
            config.time_interval if time_interval is None else float(time_interval)
        )

    # ===================== Public API =====================
    def generate_line_pairs(
        self,
        quad_vertices: List[np.ndarray],
        num_lines: int = 10,
        seed: int = 42,
        inner_scale: float = 0.8,
        min_length: float = 50.0,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate num_lines straight trajectories inside an inner-scaled quadrilateral,
        and return lists of (line_a, line_b) where:
          - line_a: sampled_points[:-1]  (shape [N-1, 2])
          - line_b: sampled_points[1:]   (shape [N-1, 2])
        """
        np.random.seed(seed)
        random.seed(seed)

        inner_vertices = self._get_inner_quad_vertices(quad_vertices, inner_scale)
        poly = SPolygon(inner_vertices)

        list_a: List[np.ndarray] = []
        list_b: List[np.ndarray] = []

        tries_per_line = 200
        for _ in range(num_lines):
            ok = False
            for _ in range(tries_per_line):
                start = self._random_point_in_polygon(poly)
                theta = np.random.uniform(0.0, np.pi)
                direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                seg = self._clip_infinite_line_with_polygon(poly, start, direction)
                if seg is None:
                    continue
                pA, pB = seg
                full_len = float(np.linalg.norm(pB - pA))
                if full_len < min_length:
                    continue

                # Choose a subsegment length L in [min_length, full_len]
                L = float(np.random.uniform(min_length, full_len))
                margin = full_len - L
                offset = 0.0 if margin <= 0 else np.random.uniform(0.0, margin)
                unit_dir = (pB - pA) / full_len
                seg_start = pA + unit_dir * offset
                seg_end = seg_start + unit_dir * L

                sampled = self._sample_straight_segment(seg_start, seg_end)
                if len(sampled) >= 2:
                    list_a.append(sampled[:-1])
                    list_b.append(sampled[1:])
                    ok = True
                    break
            if not ok:
                # skip if failed to find a valid segment after tries
                continue

        return list_a, list_b

    # ===================== Helpers =====================
    def _get_inner_quad_vertices(
        self, quad_vertices: List[np.ndarray], scale: float
    ) -> List[np.ndarray]:
        centroid = np.mean(quad_vertices, axis=0)
        return [centroid + (np.array(v) - centroid) * scale for v in quad_vertices]

    def _random_point_in_polygon(self, poly: SPolygon) -> np.ndarray:
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(100):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            p = Point(x, y)
            if poly.contains(p):
                return np.array([x, y], dtype=float)
        c = poly.representative_point()
        return np.array([c.x, c.y], dtype=float)

    def _clip_infinite_line_with_polygon(
        self, poly: SPolygon, start: np.ndarray, direction: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Intersect a very long line with polygon, return segment endpoints inside polygon."""
        big = 1e6
        p1 = start - direction * big
        p2 = start + direction * big
        inf_line = LineString([tuple(p1), tuple(p2)])
        inter = poly.intersection(inf_line)
        if inter.is_empty:
            return None
        if isinstance(inter, LineString):
            coords = np.array(inter.coords, dtype=float)
            return coords[0], coords[-1]
        try:
            # Multi geometries: pick the longest LineString
            max_seg = max(
                (g for g in inter.geoms if isinstance(g, LineString)),
                key=lambda g: g.length,
                default=None,
            )
        except Exception:
            max_seg = None
        if max_seg is None:
            return None
        coords = np.array(max_seg.coords, dtype=float)
        return coords[0], coords[-1]

    def _sample_straight_segment(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """
        等间隔采样（包含首尾），采样点数按 floor(duration/dt)+1 计算：
          - 若 duration = 0，返回 [p0]
          - 否则产生恰好 N = floor(duration/dt)+1 个采样点
          - 这样当计算得到 10 个采样点时，line_a = P[0:9]，line_b = P[1:10]
        """
        length = float(np.linalg.norm(p1 - p0))
        if length <= 0:
            return np.array([p0], dtype=float)
        duration = length / self.velocity
        dt = float(self.time_interval)
        # 至少两个点（首尾）。常规为 floor(T/dt)+1 个点，包含终点
        num_points = int(np.floor(duration / dt)) + 1
        if num_points < 2:
            num_points = 2
        # 使用 linspace 保证包含首尾
        t = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
        pts = (1 - t) * p0.reshape(1, 2) + t * p1.reshape(1, 2)
        return pts.astype(float)


# =============== Simple test (optional) ===============
if __name__ == "__main__":
    # quick sanity run: random quad via region generator
    from src.detecting_region_info_generator import generate_detecting_region_infos

    infos = generate_detecting_region_infos(
        num_configurations=1, seed=config.region_seed
    )
    info = infos[0]
    quad = [
        info.transmittor_position,
        info.receiver_position_1,
        info.receiver_position_2,
        info.receiver_position_3,
    ]

    gen = StraightLineTrajectoryGenerator(time_interval=config.time_interval)
    la, lb = gen.generate_line_pairs(
        quad, num_lines=8, seed=42, inner_scale=0.7, min_length=50.0
    )
    print(f"generated pairs: {len(la)}\n")
    for i, (a, b) in enumerate(zip(la, lb), start=1):
        print(f"trajectory #{i}")
        print(f"line_a shape={a.shape}\n{a}")
        print(f"line_b shape={b.shape}\n{b}\n")
