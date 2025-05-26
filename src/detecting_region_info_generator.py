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

    # First 5 elements of all_quadrilaterals:
    # [[0, 0], [107.01878572092521, 53.00947095264386], [17.42663234205817, 121.18813320162731], [-78.23541667557139, 79.97330130502992]]
    # [[0, 0], [83.30014032487031, 80.10271420059038], [-18.490630333630683, 129.20024280395432], [-116.26018079288995, 70.1944198260255]]
    # [[0, 0], [106.58356928846854, 74.3748627437956], [11.196573160113298, 135.9726058558487], [-90.64257842755903, 72.01524932319562]]
    # [[0, 0], [-95.56013906609807, 52.52602721906696], [-123.33439448104394, -66.15769423122934], [-13.99418136314565, -115.8676052882015]]
    # [[0, 0], [-112.80956816400752, -0.041294315802823826], [-50.47409785419665, -98.89688198979262], [67.6838375888345, -82.7517223516432]]

    detecting_region_infos = []
    for config in all_quadrilaterals:
        # config[0] is transmitter
        transmitter = np.array(config[0])

        # take the other three points
        others = list(config[1:])  # make a mutable copy
        # random.shuffle(others)        # randomly permute them

        # first of the shuffled others is the transmitter
        receiver1 = np.array(others[0])
        # the remaining two are receivers 2 & 3
        receiver2 = np.array(others[1])
        receiver3 = np.array(others[2])

        detecting_region_infos.append(
            detecting_region_info.DetectingRegionInfo(
                transmittor_position=transmitter,
                receiver_position_1=receiver1,
                receiver_position_2=receiver2,
                receiver_position_3=receiver3,
            )
        )

    return detecting_region_infos
