from typing import List
import numpy as np
from scipy.spatial import ConvexHull
import src.detecting_region_info as detecting_region_info
import numpy as np


def generate_detecting_region_infos(
    num_configurations=500, seed=42
) -> List[detecting_region_info.DetectingRegionInfo]:
    """
    生成指定数量的四点配置，满足以下条件：
    - 第一个点是原点(0,0)
    - 四个点之间的距离在100到200之间
    - 四个点形成凸四边形
    - 四边形不是矩形（没有直角）

    参数:
        num_configurations: 要生成的配置数量, 默认为500
        seed: 随机数种子，确保可重现性

    返回:
        嵌套列表，形如 [[[0, 0], [x1, y1], [x2, y2], [x3, y3]], ...]
    """

    # 设置随机种子
    np.random.seed(seed)

    def calculate_distance(p1, p2):
        """计算两点之间的欧氏距离"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def is_convex(points):
        """检查四个点是否构成凸四边形"""
        try:
            hull = ConvexHull(points)
            return len(hull.vertices) == 4
        except:
            return False

    def is_rectangular(points, tolerance=0.1):
        """检查四边形是否近似于矩形（通过检查角度）"""
        # 计算四边形的所有内角
        angles = []
        n = len(points)
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]

            # 计算两个向量
            v1 = [p1[0] - p2[0], p1[1] - p2[1]]
            v2 = [p3[0] - p2[0], p3[1] - p2[1]]

            # 计算角度（弧度）
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            norm_v1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
            norm_v2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
            cos_angle = dot_product / (norm_v1 * norm_v2)

            # 处理数值精度问题
            cos_angle = min(1.0, max(-1.0, cos_angle))
            angle = np.arccos(cos_angle)
            angles.append(angle)

        # 检查是否有角度接近90度（π/2弧度）
        right_angle = np.pi / 2
        for angle in angles:
            if abs(angle - right_angle) < tolerance:
                return True

        return False

    def check_constraints(points):
        """检查四个点是否满足所有限制条件"""
        # 检查点对之间的距离
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = calculate_distance(points[i], points[j])
                if dist < 100 or dist > 200:
                    return False

        # 检查是否为凸四边形
        if not is_convex(points):
            return False

        # 检查是否为矩形
        if is_rectangular(points):
            return False

        return True

    def generate_valid_quadrilateral():
        """生成一个满足所有条件的四边形"""
        # 固定原点在(0,0)
        origin = [0, 0]

        max_attempts = 1000

        for attempt in range(max_attempts):
            # 使用极坐标生成三个额外的点
            for sub_attempt in range(100):  # 尝试不同的极坐标组合
                rho_values = np.random.uniform(100, 150, 3)  # ρ取100-150之间的值
                beta_values = np.random.uniform(0, 2 * np.pi, 3)  # β在0到2π之间

                # 排序β值以确保点是按顺序排列的（有助于形成凸四边形）
                beta_values.sort()

                # 生成三个点
                extra_points = []
                for rho, beta in zip(rho_values, beta_values):
                    x = rho * np.cos(beta)
                    y = rho * np.sin(beta)
                    extra_points.append([float(x), float(y)])

                # 合并所有点
                all_points = [origin] + extra_points

                # 检查约束
                if check_constraints(all_points):
                    return all_points

        return None

    # 生成指定数量的符合条件的四边形
    all_quadrilaterals = []

    successful_generations = 0
    total_attempts = 0
    max_total_attempts = 100000  # 防止无限循环

    while (
        successful_generations < num_configurations
        and total_attempts < max_total_attempts
    ):
        total_attempts += 1

        # 生成有效的四边形
        quad_points = generate_valid_quadrilateral()

        if quad_points is None:
            continue

        all_quadrilaterals.append(quad_points)
        successful_generations += 1

    detecting_region_infos = []
    for quad_points in all_quadrilaterals:
        # Convert to numpy array for calculations
        points_arr = np.array(quad_points, dtype=float)

        # The transmitter is fixed at (0,0)
        T = np.array([0.0, 0.0], dtype=float)

        # Choose R1 as the point with the largest positive projection on +X (tie-break by max norm)
        # This enforces that the local X axis will align with T->R1.
        candidates = [p for p in points_arr if not np.allclose(p, T)]
        if len(candidates) < 3:
            continue
        # pick the one with maximal x (and then by norm) as R1
        candidates_sorted = sorted(
            candidates, key=lambda p: (p[0], np.hypot(p[0], p[1])), reverse=True
        )
        R1 = np.array(candidates_sorted[0], dtype=float)

        # The other two receivers
        others = [p for p in candidates if not np.allclose(p, R1)]
        if len(others) != 2:
            continue
        P2 = np.array(others[0], dtype=float)
        P3 = np.array(others[1], dtype=float)

        # Build local orthonormal basis from T->R1
        vec_TR1 = R1 - T
        norm_TR1 = np.hypot(vec_TR1[0], vec_TR1[1])
        if norm_TR1 < 1e-8:
            # degenerate, skip
            continue
        u = vec_TR1 / norm_TR1  # +X (unit)
        v = np.array([-u[1], u[0]], float)  # +Y (CCW 90°)
        R = np.stack([u, v], axis=1)  # columns are basis vectors

        # Transform points to local frame: p_local = R^T (p - T)
        def to_local(p):
            return (p - T) @ R

        T_local = to_local(T)
        R1_local = to_local(R1)
        P2_local = to_local(P2)
        P3_local = to_local(P3)

        # Ensure counter-clockwise order starting from T, then R1.
        pts_local = [T_local, R1_local, P2_local, P3_local]
        centroid = np.mean(np.stack(pts_local, axis=0), axis=0)
        # sort P2/P3 CCW around centroid, keeping T at index 0 and R1 at index 1
        tail = sorted(
            [P2_local, P3_local],
            key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]),
        )
        v1, v2, v3, v4 = T_local, R1_local, tail[0], tail[1]

        detecting_region_infos.append(
            detecting_region_info.DetectingRegionInfo(
                transmittor_position=v1,
                receiver_position_1=v2,
                receiver_position_2=v3,
                receiver_position_3=v4,
            )
        )

    return detecting_region_infos
