#!/usr/bin/env python3
"""
测试加噪功能
验证噪声是否正确添加到w和doppler数据中
"""
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos
import src.doppler_info as doppler_info_module
from src.trajectory_generator import generate_lines
from src.get_phi_info import get_angle_phi
from src.w_and_doppler_generator import generate_w_and_doppler, add_gaussian_noise_to_measurements


def test_noise_addition():
    """测试噪声添加功能"""
    print("=== 测试加噪功能 ===\n")

    # 生成检测区域
    region_infos = generate_detecting_region_infos(
        num_configurations=1, seed=config.region_seed
    )
    region_info = region_infos[0]

    # 创建doppler_info
    doppler_info = doppler_info_module.DopplerInfo(
        config.c, config.fc, config.time_interval
    )

    # 生成轨迹
    lines_a, lines_b = generate_lines(
        region_info,
        num_lines=5,
        time_interval=config.time_interval,
        seed=42,
    )

    if not lines_a or len(lines_a) == 0:
        print("错误: 未能生成轨迹")
        return

    # 取第一条轨迹的前10个点
    coords_a = lines_a[0][:10]
    coords_b = lines_b[0][:10]

    # 计算phi
    phis1234 = np.array([
        get_angle_phi(region_info, ca, cb)
        for ca, cb in zip(coords_a, coords_b)
    ])

    print(f"测试数据点数: {len(coords_a)}")
    print(f"SNR = {config.SNR_dB} dB")
    print(f"T_CPI = {config.T_CPI} s\n")

    # 1. 不加噪声的情况
    print("1. 不加噪声:")
    w_clean, doppler_clean = generate_w_and_doppler(
        region_info, doppler_info, coords_a, coords_b, phis1234,
        add_noise=False
    )
    print(f"   w[0] = {w_clean[0]}")
    print(f"   doppler[0] = {doppler_clean[0]}\n")

    # 2. 加噪声的情况
    print("2. 加噪声:")
    w_noisy, doppler_noisy = generate_w_and_doppler(
        region_info, doppler_info, coords_a, coords_b, phis1234,
        add_noise=True,
        snr_db=config.SNR_dB,
        t_cpi=config.T_CPI
    )
    print(f"   w[0] = {w_noisy[0]}")
    print(f"   doppler[0] = {doppler_noisy[0]}\n")

    # 3. 验证噪声是否独立
    print("3. 验证噪声独立性 (生成两次加噪数据):")
    w_noisy2, doppler_noisy2 = generate_w_and_doppler(
        region_info, doppler_info, coords_a, coords_b, phis1234,
        add_noise=True,
        snr_db=config.SNR_dB,
        t_cpi=config.T_CPI
    )

    # 计算差异
    w_diff = w_noisy - w_noisy2
    doppler_diff = doppler_noisy - doppler_noisy2

    print(f"   两次生成的w差异[0] = {w_diff[0]}")
    print(f"   两次生成的doppler差异[0] = {doppler_diff[0]}")
    print(f"   (如果噪声是独立的，这些差异应该不为零)\n")

    # 4. 计算理论标准差和实际标准差
    print("4. 验证噪声标准差:")

    # 理论标准差
    snr_linear = 10 ** (config.SNR_dB / 10)
    sigma_phase_theory = np.sqrt(1 / snr_linear)
    sigma_doppler_theory = np.sqrt(1 / (snr_linear * config.T_CPI**2))

    print(f"   理论相位差标准差 σ_Δφ = {sigma_phase_theory:.6f}")
    print(f"   理论多普勒标准差 σ_f = {sigma_doppler_theory:.6f}\n")

    # 生成多次来估计实际标准差
    num_trials = 1000
    w_noises = []
    doppler_noises = []

    for _ in range(num_trials):
        w_temp, doppler_temp = generate_w_and_doppler(
            region_info, doppler_info, coords_a, coords_b, phis1234,
            add_noise=True,
            snr_db=config.SNR_dB,
            t_cpi=config.T_CPI
        )
        w_noises.append(w_temp[0])  # 只取第一个点
        doppler_noises.append(doppler_temp[0])

    w_noises = np.array(w_noises)
    doppler_noises = np.array(doppler_noises)

    # 计算实际标准差 (噪声相对于干净信号)
    w_std = np.std(w_noises - w_clean[0], axis=0)
    doppler_std = np.std(doppler_noises - doppler_clean[0], axis=0)

    print(f"   实际w标准差 (w12, w13, w14) = {w_std}")
    print(f"   实际doppler标准差 (v12, v13, v14) = {doppler_std}\n")

    print(f"   相位差标准差相对误差: {np.abs(w_std - sigma_phase_theory) / sigma_phase_theory * 100}%")
    print(f"   多普勒标准差相对误差: {np.abs(doppler_std - sigma_doppler_theory) / sigma_doppler_theory * 100}%\n")

    # 5. 检查独立性
    print("5. 检查w12, w13, w14之间的相关性:")
    w_corr = np.corrcoef(w_noises.T)
    print(f"   w相关系数矩阵:\n{w_corr}\n")
    print(f"   (对角线应为1，非对角线应接近0表示独立)\n")

    print("6. 检查v12, v13, v14之间的相关性:")
    doppler_corr = np.corrcoef(doppler_noises.T)
    print(f"   doppler相关系数矩阵:\n{doppler_corr}\n")
    print(f"   (对角线应为1，非对角线应接近0表示独立)\n")

    print("=== 测试完成 ===")
    print(f"\n配置信息:")
    print(f"  add_noise = {config.add_noise}")
    print(f"  SNR_dB = {config.SNR_dB}")
    print(f"  T_CPI = {config.T_CPI}")


if __name__ == "__main__":
    test_noise_addition()
