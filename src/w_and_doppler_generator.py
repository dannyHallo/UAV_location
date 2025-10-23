import numpy as np
from scipy import signal
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
    # 转成 numpy 数组 - 使用双精度避免数值误差
    coord_a = np.asarray(coord_a, dtype=np.float64)
    coord_b = np.asarray(coord_b, dtype=np.float64)

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


# calculate_distance函数已删除，使用numpy.linalg.norm替代


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

    # 计算路径差 Δd_ab = (AR-BR) - (TB-TA) - 差分重排减少数值误差
    def calc_path_diff(T, R, A, B):
        # 使用双精度计算
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        R = np.asarray(R, dtype=np.float64)

        TA = calc_distance(T, A)
        AR = calc_distance(A, R)
        TB = calc_distance(T, B)
        BR = calc_distance(B, R)

        # 差分重排：先计算中等量级数的差，再相减
        # 原来：(TA + AR) - (TB + BR) = (TA - TB) + (AR - BR)
        # 改进：(AR - BR) - (TB - TA) = (AR - BR) + (TA - TB)
        delta_d = (AR - BR) - (TB - TA)
        return delta_d

    # 计算四个接收器的路径差
    delta_d1 = calc_path_diff(T, R1, coord_a, coord_b)
    delta_d2 = calc_path_diff(T, R2, coord_a, coord_b)
    delta_d3 = calc_path_diff(T, R3, coord_a, coord_b)

    # 计算相位差 Δφ_k(t) = 2π/λ · Δd_ab
    delta_phi1 = 2 * np.pi * delta_d1 / wavelength
    delta_phi2 = 2 * np.pi * delta_d2 / wavelength
    delta_phi3 = 2 * np.pi * delta_d3 / wavelength

    return [delta_phi1, delta_phi2, delta_phi3]


# 旧的角度计算函数已删除


def apply_signal_filtering(w_raw, filter_type="moving_average", window_size=5):
    """
    对原始相位差信号进行滤波处理，减少振荡

    参数：
        w_raw: 原始相位差数据，形状 (N, 3)
        filter_type: 滤波类型 ('moving_average', 'lowpass', 'none')
        window_size: 滑动平均窗口大小

    返回：
        滤波后的相位差数据
    """
    if filter_type == "none":
        return w_raw

    w_filtered = np.zeros_like(w_raw, dtype=np.float64)

    if filter_type == "moving_average":
        # 滑动平均滤波
        kernel = np.ones(window_size) / window_size
        for k in range(w_raw.shape[1]):  # 对每个接收器通道
            w_filtered[:, k] = np.convolve(w_raw[:, k], kernel, mode="same")

    elif filter_type == "lowpass":
        # 低通滤波 (Butterworth)
        nyquist = 0.5  # 假设采样频率为1Hz
        cutoff = 0.1  # 截止频率
        b, a = signal.butter(4, cutoff / nyquist, btype="low")

        for k in range(w_raw.shape[1]):
            w_filtered[:, k] = signal.filtfilt(b, a, w_raw[:, k])

    return w_filtered


# EKF功能已移除以减少计算量


def generate_w_and_doppler_central_diff(
    detecting_region_info: detecting_region_info,
    doppler_info: doppler_info,
    trajectory_coords,
    apply_filtering=True,
    filter_type="moving_average",
    filter_window_size=5,
):
    """
    使用中心差分计算w和doppler：
    A = pos(t-Δt/2), B = pos(t+Δt/2)

    参数：
        detecting_region_info: 检测区域信息
        doppler_info: Doppler信息（包含c, fc, time_interval）
        trajectory_coords: 完整轨迹坐标数组，形状 (N, 2)
        apply_filtering: 是否应用信号滤波
        filter_type: 滤波类型 ('moving_average', 'lowpass', 'none')
        filter_window_size: 滑动平均窗口大小

    返回：
        [w, doppler]: w单位为弧度，doppler单位为Hz，形状均为 (N-1,3)
    """

    data_length = len(trajectory_coords)
    if data_length < 2:
        raise ValueError("轨迹数据至少需要2个点才能计算中心差分")

    # 中心差分：需要N-1个输出点
    output_length = data_length - 1
    w = np.zeros([output_length, 3], dtype=np.float64)
    doppler = np.zeros([output_length, 3], dtype=np.float64)

    # 计算波长 λ = c / fc
    wavelength = doppler_info.c / doppler_info.fc

    # 遍历轨迹点，计算中心差分
    for i in range(output_length):
        # 中心差分：A = pos(t-Δt/2), B = pos(t+Δt/2)
        # 对于轨迹点i，A是前一个点，B是后一个点
        coord_a = trajectory_coords[i]  # pos(t-Δt/2)
        coord_b = trajectory_coords[i + 1]  # pos(t+Δt/2)

        # 使用基于路径差的相位差计算方法
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
        # 注意：中心差分的时间间隔是Δt，不是Δt/2
        doppler[i][0] = delta_phi1 / (2 * np.pi * doppler_info.time_interval)
        doppler[i][1] = delta_phi2 / (2 * np.pi * doppler_info.time_interval)
        doppler[i][2] = delta_phi3 / (2 * np.pi * doppler_info.time_interval)

    # 应用信号滤波减少振荡
    if apply_filtering:
        w = apply_signal_filtering(w, filter_type, filter_window_size)
        doppler = apply_signal_filtering(doppler, filter_type, filter_window_size)

    return [w, doppler]


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
    apply_filtering=True,
    filter_type="moving_average",
    filter_window_size=5,
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
        apply_filtering: 是否应用信号滤波
        filter_type: 滤波类型 ('moving_average', 'lowpass', 'none')
        filter_window_size: 滑动平均窗口大小

    返回：
        [w, doppler]: w单位为弧度，doppler单位为Hz，形状均为 (N,3)
    """

    data_length = np.shape(coords_a)[0]
    w = np.zeros([data_length, 3], dtype=np.float64)
    doppler = np.zeros([data_length, 3], dtype=np.float64)

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

    # 应用信号滤波减少振荡
    if apply_filtering:
        w = apply_signal_filtering(w, filter_type, filter_window_size)
        doppler = apply_signal_filtering(doppler, filter_type, filter_window_size)

    return [w, doppler]


# 旧的实现已删除，只保留新的改进版本
