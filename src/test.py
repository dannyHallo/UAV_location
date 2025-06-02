import numpy as np
from scipy.optimize import least_squares

# 定义检测区域信息的类


class DetectingRegionInfo:
    def __init__(self, receiver_position_1, receiver_position_2, receiver_position_3):
        self.receiver_position_1 = receiver_position_1  # 接收器1的位置 (x1, y1)
        self.receiver_position_2 = receiver_position_2  # 接收器2的位置 (x2, y2)
        self.receiver_position_3 = receiver_position_3  # 接收器3的位置 (x3, y3)


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

# 测试函数


def test_estimate_positions_optimized():
    # 定义三个接收器的位置
    receiver_pos1 = (200, 0)
    receiver_pos2 = (100, 100)
    receiver_pos3 = (300, 150)

    detecting_region = DetectingRegionInfo(
        receiver_pos1, receiver_pos2, receiver_pos3)

    # 定义一些训练标签的位置
    true_positions = np.array([
        [120, 80],
        [50, 50],
        [200, 200],
        [300, 100],
        [250, 150]
    ])

    # 使用 calculate_distances 函数计算到三个接收器的距离
    distances = calculate_distances(true_positions, detecting_region)

    # 添加一些噪声模拟实际情况中的不准确性
    np.random.seed(42)  # 固定随机种子
    noise = np.random.normal(0, 0.1, distances.shape)
    noisy_distances = distances + noise

    # 使用 estimate_positions_optimized 函数估计位置
    estimated_positions = estimate_positions_optimized(
        noisy_distances, detecting_region)

    # 打印结果
    for i, (true, est) in enumerate(zip(true_positions, estimated_positions)):
        print(f"样本 {i+1}:")
        print(f"  真实位置: (x={true[0]:.3f}, y={true[1]:.3f})")
        print(f"  估计位置: (x={est[0]:.3f}, y={est[1]:.3f})\n")

# 定义 calculate_distances 函数（用户提供的函数）


def calculate_distances(train_label, detecting_region_info):
    """
    计算训练标签到三个接收器的欧几里得距离。

    参数：
        train_label (numpy.ndarray): 训练标签的位置，形状为 (n, 2)。
        detecting_region_info (DetectingRegionInfo): 检测区域信息，包含三个接收器的位置。

    返回：
        numpy.ndarray: 距离表，形状为 (n, 3)。
    """
    receiver_1 = detecting_region_info.receiver_position_1
    receiver_2 = detecting_region_info.receiver_position_2
    receiver_3 = detecting_region_info.receiver_position_3

    # 计算到接收器1的距离
    distances_to_r1 = np.sqrt(
        (train_label[:, 0] - receiver_1[0]) ** 2
        + (train_label[:, 1] - receiver_1[1]) ** 2
    )
    # 计算到接收器2的距离
    distances_to_r2 = np.sqrt(
        (train_label[:, 0] - receiver_2[0]) ** 2
        + (train_label[:, 1] - receiver_2[1]) ** 2
    )
    # 计算到接收器3的距离
    distances_to_r3 = np.sqrt(
        (train_label[:, 0] - receiver_3[0]) ** 2
        + (train_label[:, 1] - receiver_3[1]) ** 2
    )

    # 将结果组合成 (n, 3) 形状的数组
    distances = np.stack(
        (distances_to_r1, distances_to_r2, distances_to_r3), axis=1)

    return distances


if __name__ == "__main__":
    test_estimate_positions_optimized()
