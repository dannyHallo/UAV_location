import numpy as np
import math
import src.detecting_region_info as detecting_region_info
import src.doppler_info as doppler_info
import src.get_phi_info as get_phi_info

# def calculate_angles_phi(array_A, array_B, origin):
#     # Function to calculate the angle between vectors OA and AB for each pair of points
#     angles = []
#     for A, B in zip(array_A, array_B):
#         vector_OA = A - origin
#         vector_AB = B - A
#         dot_product = np.dot(vector_OA, vector_AB)
#         norm_OA = np.linalg.norm(vector_OA)
#         norm_AB = np.linalg.norm(vector_AB)
#         # Avoid division by zero in case one of the vectors is zero
#         if norm_OA == 0 or norm_AB == 0:
#             angle = 0
#         else:
#             # Clip the cosine value to avoid numerical errors beyond the range [-1, 1]
#             cos_angle = dot_product / (norm_OA * norm_AB)
#             cos_angle_clipped = np.clip(cos_angle, -1.0, 1.0)
#             angle = np.arccos(cos_angle_clipped)
#         angles.append(angle)
#     return angles


# def get_angle_phi(detecting_region_info, coords_A, coords_B):
#     # Call the function and pass the arrays
#     phi1 = calculate_angles_phi(coords_A, coords_B, detecting_region_info.v1)
#     phi2 = calculate_angles_phi(coords_A, coords_B, detecting_region_info.v2)
#     phi3 = calculate_angles_phi(coords_A, coords_B, detecting_region_info.v3)
#     phi4 = calculate_angles_phi(coords_A, coords_B, detecting_region_info.v4)

#     return [phi1, phi2, phi3, phi4]


import numpy as np


def calculate_instantaneous_speeds(coord_a, coord_b, time_interval):
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


# 定义计算夹角的函数，以计算与x轴的夹角
def calculate_angle_with_x_axis(x, y, a, b):
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


# 继续使用之前定义的点和距离计算方法
def get_angle_theta(detecting_region_info, coord_a, coord_b):

    # 计算夹角
    n_theta_1a = calculate_angle_with_x_axis(
        detecting_region_info.v1[0], detecting_region_info.v1[1], coord_a[0], coord_a[1]
    )
    n_theta_1b = calculate_angle_with_x_axis(
        detecting_region_info.v1[0], detecting_region_info.v1[1], coord_b[0], coord_b[1]
    )
    n_theta_2a = calculate_angle_with_x_axis(
        detecting_region_info.v2[0], detecting_region_info.v2[1], coord_a[0], coord_a[1]
    )
    n_theta_2b = calculate_angle_with_x_axis(
        detecting_region_info.v2[0], detecting_region_info.v2[1], coord_b[0], coord_b[1]
    )
    n_theta_3a = calculate_angle_with_x_axis(
        detecting_region_info.v3[0], detecting_region_info.v3[1], coord_a[0], coord_a[1]
    )
    n_theta_3b = calculate_angle_with_x_axis(
        detecting_region_info.v3[0], detecting_region_info.v3[1], coord_b[0], coord_b[1]
    )
    n_theta_4a = calculate_angle_with_x_axis(
        detecting_region_info.v4[0], detecting_region_info.v4[1], coord_a[0], coord_a[1]
    )
    n_theta_4b = calculate_angle_with_x_axis(
        detecting_region_info.v4[0], detecting_region_info.v4[1], coord_b[0], coord_b[1]
    )

    theta1 = abs(n_theta_1a - n_theta_1b)
    theta2 = abs(n_theta_2a - n_theta_2b)
    theta3 = abs(n_theta_3a - n_theta_3b)
    theta4 = abs(n_theta_4a - n_theta_4b)

    return [
        np.radians(theta1),  # theta1
        np.radians(theta2),  # theta2
        np.radians(theta3),  # theta3
        np.radians(theta4),  # theta4
    ]


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


def generateWAndDoppler(
    detecting_region_info: detecting_region_info,
    doppler_info: doppler_info,
    coords_a,
    coords_b,
    phis_1234,
):
    """
    lines 输入格式: 见 NewLinesGenerator.py
    """

    data_length = np.shape(coords_a)[0]
    w = np.zeros([data_length, 3])
    doppler = np.zeros([data_length, 3])

    # 遍历每一个coords_a
    for i in range(data_length):
        coord_a = coords_a[i]
        coord_b = coords_b[i]

        [phi1, phi2, phi3, phi4] = phis_1234[i]
        [theta1, theta2, theta3, theta4] = get_angle_theta(
            detecting_region_info, coord_a, coord_b
        )

        # 预先计算速度大小
        speeds = calculate_instantaneous_speeds(
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
        # dd12 = np.abs(d11+d21-d12-d22)
        # dd13 = np.abs(d11+d31-d12-d32)

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
