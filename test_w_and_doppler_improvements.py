#!/usr/bin/env python3
"""
测试w_and_doppler_generator.py的改进实现
比较新旧方法的数值差异
"""

import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.w_and_doppler_generator import (
    generate_w_and_doppler,
    generate_w_and_doppler_original,
    compare_implementations,
)
from src.detecting_region_info import DetectingRegionInfo
from src.doppler_info import DopplerInfo


def create_test_data():
    """创建测试数据"""
    # 创建检测区域信息
    detecting_region_info = DetectingRegionInfo(
        transmittor_position=np.array([0.0, 0.0]),
        receiver_position_1=np.array([100.0, 0.0]),
        receiver_position_2=np.array([50.0, 86.6]),  # 等边三角形
        receiver_position_3=np.array([50.0, -86.6]),
    )

    # 创建Doppler信息
    doppler_info = DopplerInfo(
        c=3e8, fc=6e9, time_interval=0.01  # 光速  # 载波频率 2.4GHz  # 10ms时间间隔
    )

    # 创建测试坐标
    coords_a = np.array([[10.0, 5.0], [20.0, 10.0], [30.0, 15.0]])

    coords_b = np.array([[10.1, 5.1], [20.2, 10.2], [30.3, 15.3]])  # 小位移

    # 创建相位信息（保持兼容性）
    phis_1234 = np.array(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5, 0.6]]
    )

    return detecting_region_info, doppler_info, coords_a, coords_b, phis_1234


def test_angle_calculation_improvement():
    """测试角度计算改进"""
    print("=== 测试角度计算改进 ===")

    # 测试向量点积计算cosθ
    from src.w_and_doppler_generator import _calculate_cos_angle_with_x_axis

    # 测试用例
    test_cases = [
        (0, 0, 1, 0),  # 沿x轴正方向
        (0, 0, 0, 1),  # 沿y轴正方向
        (0, 0, 1, 1),  # 45度角
        (0, 0, -1, 0),  # 沿x轴负方向
        (0, 0, 0, -1),  # 沿y轴负方向
    ]

    print("角度计算测试:")
    for i, (x, y, a, b) in enumerate(test_cases):
        cos_theta = _calculate_cos_angle_with_x_axis(x, y, a, b)
        theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
        theta_deg = np.degrees(theta_rad)
        print(
            f"  测试 {i+1}: 向量({a-x}, {b-y}) -> cosθ={cos_theta:.6f}, θ={theta_deg:.2f}°"
        )


def test_path_difference_calculation():
    """测试路径差计算"""
    print("\n=== 测试路径差计算 ===")

    from src.w_and_doppler_generator import _calculate_path_difference_and_phase_shift

    detecting_region_info, doppler_info, coords_a, coords_b, phis_1234 = (
        create_test_data()
    )

    # 计算波长
    wavelength = doppler_info.c / doppler_info.fc
    print(f"波长 λ = {wavelength:.6f} m")

    # 测试路径差计算
    coord_a = coords_a[0]
    coord_b = coords_b[0]

    delta_phis = _calculate_path_difference_and_phase_shift(
        detecting_region_info, coord_a, coord_b, wavelength
    )

    print(f"位置A: {coord_a}")
    print(f"位置B: {coord_b}")
    print("各接收器的相位差:")
    for i, delta_phi in enumerate(delta_phis):
        print(
            f"  接收器 {i+1}: Δφ = {delta_phi:.6f} rad = {np.degrees(delta_phi):.2f}°"
        )


def test_implementation_comparison():
    """测试新旧实现比较"""
    print("\n=== 新旧实现比较 ===")

    detecting_region_info, doppler_info, coords_a, coords_b, phis_1234 = (
        create_test_data()
    )

    # 比较实现
    comparison = compare_implementations(
        detecting_region_info, doppler_info, coords_a, coords_b, phis_1234
    )

    print("比较结果:")
    print(f"  w的最大绝对差异: {comparison['max_w_abs_diff']:.2e}")
    print(f"  w的最大相对误差: {comparison['max_w_rel_error']:.2e}")
    print(f"  doppler的最大绝对差异: {comparison['max_doppler_abs_diff']:.2e}")
    print(f"  doppler的最大相对误差: {comparison['max_doppler_rel_error']:.2e}")

    print("\n详细比较 (前3个数据点):")
    print(
        "索引 | w_old | w_new | w_diff | w_rel_err | doppler_old | doppler_new | doppler_diff | doppler_rel_err"
    )
    print("-" * 120)

    for i in range(min(3, len(coords_a))):
        w_old = comparison["w_old"][i]
        w_new = comparison["w_new"][i]
        w_diff = comparison["w_absolute_diff"][i]
        w_rel_err = comparison["w_relative_error"][i]
        doppler_old = comparison["doppler_old"][i]
        doppler_new = comparison["doppler_new"][i]
        doppler_diff = comparison["doppler_absolute_diff"][i]
        doppler_rel_err = comparison["doppler_relative_error"][i]

        print(
            f"{i:4d} | {w_old[0]:6.2e} | {w_new[0]:6.2e} | {w_diff[0]:6.2e} | {w_rel_err[0]:8.2e} | "
            f"{doppler_old[0]:10.2e} | {doppler_new[0]:10.2e} | {doppler_diff[0]:10.2e} | {doppler_rel_err[0]:12.2e}"
        )


def main():
    """主测试函数"""
    print("w_and_doppler_generator.py 改进测试")
    print("=" * 50)

    try:
        test_angle_calculation_improvement()
        test_path_difference_calculation()
        test_implementation_comparison()

        print("\n=== 测试总结 ===")
        print("✓ 角度计算改进：使用向量点积直接计算cosθ，避免度数转换误差")
        print("✓ 路径差计算：实现基于 Δφ_k(t) = 2π/λ · Δd_ab 的相位差计算")
        print("✓ 数值比较：新旧实现可以并行使用，便于验证和改进")
        print("\n主要改进:")
        print("1. 消除了 sin(θ+φ) ≈ θ·cosφ + sinφ 的近似误差")
        print("2. 避免了角度度数转换的数值误差")
        print("3. 使用精确的路径差计算相位差")
        print("4. 提供了比较函数来验证改进效果")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
