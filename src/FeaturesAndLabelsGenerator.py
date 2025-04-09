import numpy as np
from scipy.optimize import least_squares


def get_d_phi(detecting_region_info, coord_a):
    ref = detecting_region_info.transmittor_position
    phi = np.arctan2(coord_a[1] - ref[1], coord_a[0] - ref[0])
    # change range from -pi to pi to 0 to 2pi
    if phi < 0:
        phi += 2 * np.pi
    d = np.sqrt((coord_a[0] - ref[0]) ** 2 + (coord_a[1] - ref[1]) ** 2)
    return [d, phi]


def get_labels(detecting_region_info, lines_a):
    labels = []
    for line in lines_a:
        for coord_a in line:
            labels.append(get_d_phi(detecting_region_info, coord_a))
    return np.array(labels)


def extract_coords_from_lines(lines):
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


def generateFeaturesAndLabels(
    detecting_region_info,
    lines_a,
    w,
    doppler,
    num_of_lines_to_generate,
    step_count_per_line,
):
    features = np.concatenate((w, doppler), axis=1)
    labels = extract_coords_from_lines(lines_a)
    reshaped_features = features.reshape(
        num_of_lines_to_generate, step_count_per_line, 6
    )
    reshaped_labels = labels.reshape(
        num_of_lines_to_generate, step_count_per_line, 2)

    # drop the sequence of the label
    reshaped_labels = reshaped_labels[:, -1, :]

    return [reshaped_features, reshaped_labels]


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
    distances = np.stack(
        (distances_to_r1, distances_to_r2, distances_to_r3), axis=1)

    return distances


# estimate_positions_optimized 函数：
# 输入：距离表和检测区域信息。
# 方法：
# 对于每一组距离，定义一个残差函数，表示估计位置与三个接收器的距离之差。
# 使用 scipy.optimize.least_squares 进行非线性最小二乘优化，寻找使残差最小的 (x, y)。
# 初始猜测采用接收器1的位置，可以根据需要调整。
# 如果优化成功，则使用优化结果作为估计位置；否则，使用接收器1的位置作为默认值。
# 输出：预测的位置数组，形状为 (n, 2)。

def estimate_positions_optimized(distance_table, detecting_region_info):
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
            return np.sqrt((x - receivers[:, 0])**2 + (y - receivers[:, 1])**2) - distances

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
