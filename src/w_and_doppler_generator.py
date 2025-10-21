import numpy as np
import math
import src.detecting_region_info as detecting_region_info
import src.doppler_info as doppler_info


def _calculate_instantaneous_speeds(coord_a, coord_b, time_interval):
    """
    计算瞬时速度大小 v = |coord_b - coord_a| / Δt

    参数：
      coord_a        单个点 (d,) 或 多个点 (n,d)
      coord_b        与 coord_a 同形状，表示 Δt 后的位置
      time_interval  时间间隔 Δt（标量）

    返回：
      如果输入是一维 (d,), 返回标量速度；
      如果输入是二维 (n,d), 返回形状 (n,) 的速度数组。
    """
    # 转成 numpy 数组
    coord_a = np.asarray(coord_a, dtype=float)
    coord_b = np.asarray(coord_b, dtype=float)

    # 形状检查
    if coord_a.shape != coord_b.shape:
        raise ValueError(
            f"coord_a{coord_a.shape} 与 coord_b{coord_b.shape} 必须形状一致"
        )

    # 计算位移
    disp = coord_b - coord_a

    # 单点情况
    if coord_a.ndim == 1:
        # 欧氏距离 / Δt
        return np.linalg.norm(disp) / time_interval

    # 多点情况
    if coord_a.ndim == 2:
        # 每行一个点，axis=1 计算每行范数
        mags = np.linalg.norm(disp, axis=1)
        return mags / time_interval

    # 其余不支持
    raise ValueError(f"不支持 ndim={coord_a.ndim} 的输入")


def calculate_distance(x, y, a, b):
    """Calculate the Euclidean distance between two points (x, y) and (a, b)."""
    return math.sqrt((a - x) ** 2 + (b - y) ** 2)


# 定义计算夹角的函数，直接返回cosθ值，避免度数转换的数值误差
def _calculate_cos_angle_with_x_axis(x, y, a, b):
    """
    计算向量与x轴夹角的余弦值，直接返回cosθ避免度数转换误差

    参数：
        x, y: 起点坐标
        a, b: 终点坐标

    返回：
        cosθ值，范围[-1, 1]
    """
    # 向量p1p2
    vector = np.array([a - x, b - y])
    # x轴正方向的向量
    x_axis = np.array([1, 0])

    # 向量的点乘
    dot_product = np.dot(vector, x_axis)
    # 向量的模长
    norm_vector = np.linalg.norm(vector)

    # 防止除以零
    if norm_vector == 0:
        return 1.0  # 零向量与任何向量夹角为0，cos(0)=1

    # 计算夹角的余弦值
    cos_angle = dot_product / norm_vector
    # 余弦值的范围是[-1, 1]，可能由于浮点数误差超出这个范围
    # 这里将其限制在[-1, 1]内
    cos_angle = np.clip(cos_angle, -1, 1)

    return cos_angle


# 计算路径差和相位差的新方法
def _calculate_path_difference_and_phase_shift(
    detecting_region_info, coord_a, coord_b, wavelength
):
    """
    计算路径差和相位差

    参数：
        detecting_region_info: 检测区域信息
        coord_a: 上一时刻A的位置
        coord_b: 下一时刻B的位置
        wavelength: 波长 λ

    返回：
        四个接收器的相位差 [Δφ1, Δφ2, Δφ3, Δφ4]
    """

    # 计算各点到发射器和接收器的距离
    def calc_distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # 发射器位置
    T = detecting_region_info.transmittor_position
    # 3个接收器位置
    R1 = detecting_region_info.receiver_position_1
    R2 = detecting_region_info.receiver_position_2
    R3 = detecting_region_info.receiver_position_3

    # 计算路径差 Δd_ab = |TA|+|AR1| − (|TB|+|BR1|)
    def calc_path_diff(T, R, A, B):
        TA = calc_distance(T, A)
        AR = calc_distance(A, R)
        TB = calc_distance(T, B)
        BR = calc_distance(B, R)
        return (TA + AR) - (TB + BR)

    # 计算四个接收器的路径差
    delta_d1 = calc_path_diff(T, R1, coord_a, coord_b)
    delta_d2 = calc_path_diff(T, R2, coord_a, coord_b)
    delta_d3 = calc_path_diff(T, R3, coord_a, coord_b)

    # 计算相位差 Δφ_k(t) = 2π/λ · Δd_ab
    delta_phi1 = 2 * np.pi * delta_d1 / wavelength
    delta_phi2 = 2 * np.pi * delta_d2 / wavelength
    delta_phi3 = 2 * np.pi * delta_d3 / wavelength

    return [delta_phi1, delta_phi2, delta_phi3]


# 改进的角度计算方法，使用向量点积直接计算cosθ
def _get_cos_angles(detecting_region_info, coord_a, coord_b):
    """
    计算角度变化，返回cos值而不是度数，避免数值误差

    参数：
        detecting_region_info: 检测区域信息
        coord_a: 上一时刻位置
        coord_b: 下一时刻位置

    返回：
        四个接收器对应的cos角度变化值 [cos_θ1, cos_θ2, cos_θ3, cos_θ4]
    """
    # 在以 T 为原点、T→RX1 为 X 轴的局部坐标系下计算
    a_local = detecting_region_info.transform_point_to_tx_rx1(coord_a)
    b_local = detecting_region_info.transform_point_to_tx_rx1(coord_b)
    v1_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v1)
    v2_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v2)
    v3_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v3)
    v4_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v4)

    # 计算各接收器到A点和B点的cos角度
    cos_theta_1a = _calculate_cos_angle_with_x_axis(
        v1_local[0], v1_local[1], a_local[0], a_local[1]
    )
    cos_theta_1b = _calculate_cos_angle_with_x_axis(
        v1_local[0], v1_local[1], b_local[0], b_local[1]
    )
    cos_theta_2a = _calculate_cos_angle_with_x_axis(
        v2_local[0], v2_local[1], a_local[0], a_local[1]
    )
    cos_theta_2b = _calculate_cos_angle_with_x_axis(
        v2_local[0], v2_local[1], b_local[0], b_local[1]
    )
    cos_theta_3a = _calculate_cos_angle_with_x_axis(
        v3_local[0], v3_local[1], a_local[0], a_local[1]
    )
    cos_theta_3b = _calculate_cos_angle_with_x_axis(
        v3_local[0], v3_local[1], b_local[0], b_local[1]
    )
    cos_theta_4a = _calculate_cos_angle_with_x_axis(
        v4_local[0], v4_local[1], a_local[0], a_local[1]
    )
    cos_theta_4b = _calculate_cos_angle_with_x_axis(
        v4_local[0], v4_local[1], b_local[0], b_local[1]
    )

    # 计算角度变化的cos值（使用余弦差公式）
    # cos(θ_b - θ_a) = cos(θ_a)cos(θ_b) + sin(θ_a)sin(θ_b)
    # 但我们需要的是角度差的绝对值，所以使用 |cos(θ_b - θ_a)|
    cos_delta_theta1 = abs(
        cos_theta_1a * cos_theta_1b
        + np.sqrt(1 - cos_theta_1a**2) * np.sqrt(1 - cos_theta_1b**2)
    )
    cos_delta_theta2 = abs(
        cos_theta_2a * cos_theta_2b
        + np.sqrt(1 - cos_theta_2a**2) * np.sqrt(1 - cos_theta_2b**2)
    )
    cos_delta_theta3 = abs(
        cos_theta_3a * cos_theta_3b
        + np.sqrt(1 - cos_theta_3a**2) * np.sqrt(1 - cos_theta_3b**2)
    )
    cos_delta_theta4 = abs(
        cos_theta_4a * cos_theta_4b
        + np.sqrt(1 - cos_theta_4a**2) * np.sqrt(1 - cos_theta_4b**2)
    )

    return [cos_delta_theta1, cos_delta_theta2, cos_delta_theta3, cos_delta_theta4]


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


def generate_w_and_doppler(
    detecting_region_info: detecting_region_info,
    doppler_info: doppler_info,
    coords_a,
    coords_b,
):
    """
    使用基于路径差的相位定义：
      - w[k] = Δφ_k = 2π/λ · Δd_ab (弧度)
      - doppler[k] = f_D = 1/(2π) · Δφ_k / Δt (Hz)

    参数：
        detecting_region_info: 检测区域信息
        doppler_info: Doppler信息（包含c, fc, time_interval）
        coords_a: 上一时刻位置数组
        coords_b: 下一时刻位置数组

    返回：
        [w, doppler]: w单位为弧度，doppler单位为Hz，形状均为 (N,3)
    """

    data_length = np.shape(coords_a)[0]
    w = np.zeros([data_length, 3])
    doppler = np.zeros([data_length, 3])

    # 计算波长 λ = c / fc
    wavelength = doppler_info.c / doppler_info.fc

    # 遍历每一个coords_a
    for i in range(data_length):
        coord_a = coords_a[i]
        coord_b = coords_b[i]

        # 使用新的基于路径差的相位差计算方法
        [delta_phi1, delta_phi2, delta_phi3] = (
            _calculate_path_difference_and_phase_shift(
                detecting_region_info, coord_a, coord_b, wavelength
            )
        )

        # w = Δφ，各通道分别对应 R1、R2、R3
        w[i][0] = delta_phi1
        w[i][1] = delta_phi2
        w[i][2] = delta_phi3

        # doppler = f_D = 1/(2π) · Δφ/Δt (Hz)
        doppler[i][0] = delta_phi1 / (2 * np.pi * doppler_info.time_interval)
        doppler[i][1] = delta_phi2 / (2 * np.pi * doppler_info.time_interval)
        doppler[i][2] = delta_phi3 / (2 * np.pi * doppler_info.time_interval)

    return [w, doppler]


# 保留原有的实现用于比较测试
def generate_w_and_doppler_original(
    detecting_region_info: detecting_region_info,
    doppler_info: doppler_info,
    coords_a,
    coords_b,
    phis_1234,
):
    """
    原有的w和doppler计算方法（保留用于比较）
    """
    data_length = np.shape(coords_a)[0]
    w = np.zeros([data_length, 3])
    doppler = np.zeros([data_length, 3])

    # 遍历每一个coords_a
    for i in range(data_length):
        coord_a = coords_a[i]
        coord_b = coords_b[i]

        [phi1, phi2, phi3, phi4] = phis_1234[i]
        [theta1, theta2, theta3, theta4] = _get_angle_theta_original(
            detecting_region_info, coord_a, coord_b
        )

        # 预先计算速度大小
        speeds = _calculate_instantaneous_speeds(
            coord_a, coord_b, doppler_info.time_interval
        )

        x = coord_a[0]
        y = coord_a[1]
        x1 = coord_b[0]
        y1 = coord_b[1]

        d11 = calculate_distance(
            x, y, detecting_region_info.v1[0], detecting_region_info.v1[1]
        )
        d12 = calculate_distance(
            x1, y1, detecting_region_info.v1[0], detecting_region_info.v1[1]
        )
        d21 = calculate_distance(
            x, y, detecting_region_info.v2[0], detecting_region_info.v2[1]
        )
        d22 = calculate_distance(
            x1, y1, detecting_region_info.v2[0], detecting_region_info.v2[1]
        )
        d31 = calculate_distance(
            x, y, detecting_region_info.v3[0], detecting_region_info.v3[1]
        )
        d32 = calculate_distance(
            x1, y1, detecting_region_info.v3[0], detecting_region_info.v3[1]
        )
        d41 = calculate_distance(
            x, y, detecting_region_info.v3[0], detecting_region_info.v3[1]
        )
        d42 = calculate_distance(
            x1, y1, detecting_region_info.v3[0], detecting_region_info.v3[1]
        )

        w12 = (
            d11
            * theta1
            * (np.cos(phi1) + np.cos(phi2))
            * doppler_info.fc
            / doppler_info.c
            * (theta1 * np.cos(phi1) + np.sin(phi1))
        )
        w13 = (
            d31
            * theta3
            * (np.cos(phi1) + np.cos(phi3))
            * doppler_info.fc
            / doppler_info.c
            * (theta3 * np.cos(phi3) + np.sin(phi3))
        )
        w14 = (
            d41
            * theta4
            * (np.cos(phi1) + np.cos(phi3))
            * doppler_info.fc
            / doppler_info.c
            * (theta4 * np.cos(phi4) + np.sin(phi4))
        )

        # 使用第i个速度值
        v12 = speeds * doppler_info.fc * (np.cos(phi1) + np.cos(phi2)) / doppler_info.c
        v13 = speeds * doppler_info.fc * (np.cos(phi1) + np.cos(phi3)) / doppler_info.c
        v14 = speeds * doppler_info.fc * (np.cos(phi1) + np.cos(phi4)) / doppler_info.c

        w[i][0] = w12
        w[i][1] = w13
        w[i][2] = w14

        doppler[i][0] = v12
        doppler[i][1] = v13
        doppler[i][2] = v14

    return [w, doppler]


# 保留原有的角度计算函数用于比较
def _get_angle_theta_original(detecting_region_info, coord_a, coord_b):
    """原有的角度计算方法"""
    # 在以 T 为原点、T→RX1 为 X 轴的局部坐标系下计算
    a_local = detecting_region_info.transform_point_to_tx_rx1(coord_a)
    b_local = detecting_region_info.transform_point_to_tx_rx1(coord_b)
    v1_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v1)
    v2_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v2)
    v3_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v3)
    v4_local = detecting_region_info.transform_point_to_tx_rx1(detecting_region_info.v4)

    n_theta_1a = _calculate_angle_with_x_axis_original(
        v1_local[0], v1_local[1], a_local[0], a_local[1]
    )
    n_theta_1b = _calculate_angle_with_x_axis_original(
        v1_local[0], v1_local[1], b_local[0], b_local[1]
    )
    n_theta_2a = _calculate_angle_with_x_axis_original(
        v2_local[0], v2_local[1], a_local[0], a_local[1]
    )
    n_theta_2b = _calculate_angle_with_x_axis_original(
        v2_local[0], v2_local[1], b_local[0], b_local[1]
    )
    n_theta_3a = _calculate_angle_with_x_axis_original(
        v3_local[0], v3_local[1], a_local[0], a_local[1]
    )
    n_theta_3b = _calculate_angle_with_x_axis_original(
        v3_local[0], v3_local[1], b_local[0], b_local[1]
    )
    n_theta_4a = _calculate_angle_with_x_axis_original(
        v4_local[0], v4_local[1], a_local[0], a_local[1]
    )
    n_theta_4b = _calculate_angle_with_x_axis_original(
        v4_local[0], v4_local[1], b_local[0], b_local[1]
    )

    theta1 = abs(n_theta_1a - n_theta_1b)
    theta2 = abs(n_theta_2a - n_theta_2b)
    theta3 = abs(n_theta_3a - n_theta_3b)
    theta4 = abs(n_theta_4a - n_theta_4b)

    return [
        np.radians(theta1),
        np.radians(theta2),
        np.radians(theta3),
        np.radians(theta4),
    ]


def _calculate_angle_with_x_axis_original(x, y, a, b):
    """原有的角度计算函数"""
    # 向量p1p2
    vector = np.array([a - x, b - y])
    # x轴正方向的向量
    x_axis = np.array([1, 0])

    # 向量的点乘
    dot_product = np.dot(vector, x_axis)
    # 向量的模长
    norm_vector = np.linalg.norm(vector)

    # 防止除以零
    if norm_vector == 0:
        return 0

    # 计算夹角的余弦值
    cos_angle = dot_product / norm_vector
    # 余弦值的范围是[-1, 1]，可能由于浮点数误差超出这个范围
    # 这里将其限制在[-1, 1]内
    cos_angle = np.clip(cos_angle, -1, 1)

    # 计算夹角（弧度转换为度）
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)

    # 考虑向量在y轴上的方向，如果向量第二个分量是负的，则它在x轴的下方，夹角应该是360°-计算出的角度
    if vector[1] < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees


def compare_implementations(
    detecting_region_info, doppler_info, coords_a, coords_b, phis_1234
):
    """
    比较新旧实现的数值差异

    参数：
        detecting_region_info: 检测区域信息
        doppler_info: Doppler信息
        coords_a: 上一时刻位置数组
        coords_b: 下一时刻位置数组
        phis_1234: 相位信息

    返回：
        比较结果字典
    """
    # 使用新实现
    w_new, doppler_new = generate_w_and_doppler(
        detecting_region_info, doppler_info, coords_a, coords_b, phis_1234
    )

    # 使用原实现
    w_old, doppler_old = generate_w_and_doppler_original(
        detecting_region_info, doppler_info, coords_a, coords_b, phis_1234
    )

    # 计算差异
    w_diff = np.abs(w_new - w_old)
    doppler_diff = np.abs(doppler_new - doppler_old)

    # 计算相对误差
    w_rel_error = np.divide(w_diff, np.abs(w_old) + 1e-10)
    doppler_rel_error = np.divide(doppler_diff, np.abs(doppler_old) + 1e-10)

    return {
        "w_new": w_new,
        "w_old": w_old,
        "w_absolute_diff": w_diff,
        "w_relative_error": w_rel_error,
        "doppler_new": doppler_new,
        "doppler_old": doppler_old,
        "doppler_absolute_diff": doppler_diff,
        "doppler_relative_error": doppler_rel_error,
        "max_w_abs_diff": np.max(w_diff),
        "max_w_rel_error": np.max(w_rel_error),
        "max_doppler_abs_diff": np.max(doppler_diff),
        "max_doppler_rel_error": np.max(doppler_rel_error),
    }
