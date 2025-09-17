import numpy as np
from scipy.optimize import least_squares
from src.detecting_region_info import DetectingRegionInfo


def get_features(
    phis1234,
    w,
    doppler,
):
    """
    Returns:
      features: np.ndarray, shape (N, 6 + w_dim + d_dim)
      labels:   np.ndarray, shape (N, 4) == phis1234
    """
    # ———— 1) phis1234 -> ndarray 并检查
    phis1234 = np.asarray(phis1234, dtype=float)
    if phis1234.ndim != 2 or phis1234.shape[1] != 4:
        raise ValueError(f"phis1234 应该是 (N,4)，但得到 {phis1234.shape}")
    N = phis1234.shape[0]

    # ———— 2) 将 w, doppler 规范成 (N, *)
    def prep_matrix(x, name):
        arr = np.asarray(x, dtype=float)
        # 一维： (N,) -> (N,1)
        if arr.ndim == 1:
            if arr.shape[0] != N:
                raise ValueError(f"{name}.shape={arr.shape}，第一维应为 {N}")
            return arr.reshape(N, 1)
        # 二维：若 (N, k) 保持；若 (k, N) 转置
        if arr.ndim == 2:
            if arr.shape[0] == N:
                return arr
            if arr.shape[1] == N:
                return arr.T
            raise ValueError(f"{name}.shape={arr.shape}，既不是 (N,*) 也不是 (*,N)")
        raise ValueError(f"{name} 不支持 ndim={arr.ndim}")

    w = prep_matrix(w, "w")
    doppler = prep_matrix(doppler, "doppler")

    # ———— 5) 横向拼接
    features = np.concatenate(
        [
            w,  # (N, w_dim)
            doppler,  # (N, d_dim)
        ],
        axis=1,
    )
    return features


def get_labels(detecting_region_info: DetectingRegionInfo, coords_b):
    labels = []
    for c in coords_b:
        labels.append(_get_d_sin_cos_phi(detecting_region_info, c))
    return np.array(labels)


# 定义计算距离的函数
def calculate_distances(train_label, detecting_region_info):
    # 提取接收器位置
    receiver_1 = detecting_region_info.receiver_position_1
    receiver_2 = detecting_region_info.receiver_position_2
    receiver_3 = detecting_region_info.receiver_position_3

    # 计算 train_label 到 receiver_1, receiver_2, receiver_3 的欧几里得距离
    distances_to_r1 = np.sqrt(
        (train_label[:, 0] - receiver_1[0]) ** 2
        + (train_label[:, 1] - receiver_1[1]) ** 2
    )
    distances_to_r2 = np.sqrt(
        (train_label[:, 0] - receiver_2[0]) ** 2
        + (train_label[:, 1] - receiver_2[1]) ** 2
    )
    distances_to_r3 = np.sqrt(
        (train_label[:, 0] - receiver_3[0]) ** 2
        + (train_label[:, 1] - receiver_3[1]) ** 2
    )

    # 将结果组合成 (n, 3) 形状的数组
    distances = np.stack((distances_to_r1, distances_to_r2, distances_to_r3), axis=1)

    return distances


def _get_d_sin_cos_phi(detecting_region_info: DetectingRegionInfo, coord_a):
    # 使用以 T 为原点、T→RX1 为 X 轴的局部坐标系
    p_local = detecting_region_info.transform_point_to_tx_rx1(coord_a)
    phi = np.arctan2(p_local[1], p_local[0])
    if phi < 0:
        phi += 2 * np.pi
    d = np.hypot(p_local[0], p_local[1])
    return [d, np.sin(phi), np.cos(phi)]


def _extract_coords_from_lines(lines):
    """
    Extract all coordinates from m different lines. Each line l1, l2, ..., consists of n1, n2, n3 different coordinates.
    Each coordinate is a 2D point.
    Args:
        lines: A list of lines, where each line is a list of coordinates.
    Returns:
        A numpy array containing all coordinates from the lines.
    """
    coords = []
    for line in lines:
        coords.extend(line)
    return np.array(coords)


# estimate_positions_optimized 函数：
# 输入：距离表和检测区域信息。
# 方法：
# 对于每一组距离，定义一个残差函数，表示估计位置与三个接收器的距离之差。
# 使用 scipy.optimize.least_squares 进行非线性最小二乘优化，寻找使残差最小的 (x, y)。
# 初始猜测采用接收器1的位置，可以根据需要调整。
# 如果优化成功，则使用优化结果作为估计位置；否则，使用接收器1的位置作为默认值。
# 输出：预测的位置数组，形状为 (n, 2)。
def _estimate_positions_optimized(distance_table, detecting_region_info):
    """
    反向计算函数，使用非线性最小二乘法根据到三个接收器的距离估计 (x, y) 位置。

    参数：
        distance_table (numpy.ndarray): 距离表，形状为 (n, 3)，每行对应到三个接收器的距离。
        detecting_region_info (DetectingRegionInfo): 检测区域信息，包含三个接收器的位置。

    返回：
        numpy.ndarray: 预测的位置，形状为 (n, 2)。
    """
    # 获取接收器的位置
    r1 = np.array(detecting_region_info.receiver_position_1)
    r2 = np.array(detecting_region_info.receiver_position_2)
    r3 = np.array(detecting_region_info.receiver_position_3)

    receivers = np.array([r1, r2, r3])

    predicted_positions = []

    for distances in distance_table:
        # 定义残差函数
        def residuals(vars):
            x, y = vars
            return (
                np.sqrt((x - receivers[:, 0]) ** 2 + (y - receivers[:, 1]) ** 2)
                - distances
            )

        # 初始猜测：使用接收器1的位置
        initial_guess = r1

        # 使用最小二乘法进行优化
        result = least_squares(residuals, initial_guess)

        if result.success:
            estimated_position = result.x
        else:
            # 如果优化失败，使用接收器1的位置作为默认值
            estimated_position = r1

        predicted_positions.append(estimated_position)

    return np.array(predicted_positions)
