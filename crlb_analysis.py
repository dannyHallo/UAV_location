#!/usr/bin/env python3
"""
CRLB理论极限分析
计算不同SNR下的Cramer-Rao Lower Bound (CRLB)理论极限
基于公式：CRLB_p = (G^T G)^(-1) * σ_ρ^2
其中 σ_ρ,u,i = λ / (2π * sqrt(2 * SNR_u,i))
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos


def compute_geometry_matrix(tx, rx_positions, p):
    """
    计算几何矩阵G

    参数：
        tx: 发射机位置 (2,)
        rx_positions: 接收机位置列表 [(2,), (2,), ...]
        p: 目标位置 (2,)

    返回：
        G: 几何矩阵 (3, 2)
    """
    rows = []
    for rx in rx_positions:
        # 计算单位向量
        u_t = (tx - p) / np.linalg.norm(tx - p)
        u_r = (rx - p) / np.linalg.norm(rx - p)
        rows.append(u_t + u_r)

    return np.vstack(rows)


def compute_sigma_rho(snr_db, wavelength):
    """
    计算相位测量噪声标准差

    参数：
        snr_db: 信噪比 (dB)
        wavelength: 波长 (m)

    返回：
        σ_ρ: 相位测量噪声标准差 (m)
    """
    # 将dB转换为线性值
    snr_linear = 10 ** (snr_db / 10)

    # 计算σ_ρ = λ / (2π * sqrt(2 * SNR))
    sigma_rho = wavelength / (2 * np.pi * np.sqrt(2 * snr_linear))

    return sigma_rho


def compute_crlb(tx, rx_positions, p, snr_db, wavelength):
    """
    计算CRLB理论极限

    参数：
        tx: 发射机位置 (2,)
        rx_positions: 接收机位置列表
        p: 目标位置 (2,)
        snr_db: 信噪比 (dB)
        wavelength: 波长 (m)

    返回：
        crlb_trace: CRLB矩阵的迹 (位置误差的平方和，单位：m²)
        crlb_det: CRLB矩阵的行列式 (单位：m⁴)
    """
    # 计算几何矩阵G
    G = compute_geometry_matrix(tx, rx_positions, p)

    # 计算相位测量噪声标准差
    sigma_rho = compute_sigma_rho(snr_db, wavelength)

    # 计算G^T G
    GTG = G.T @ G

    # 检查矩阵是否可逆
    if np.linalg.det(GTG) < 1e-12:
        print(f"Warning: GTG is nearly singular at position {p}")
        return np.inf, np.inf

    # 计算CRLB = (G^T G)^(-1) * σ_ρ^2
    # 注意：这里σ_ρ对所有接收器都是相同的
    crlb = np.linalg.inv(GTG) * (sigma_rho**2)

    # 计算CRLB的迹（位置误差的平方和，单位：m²）
    crlb_trace = np.trace(crlb)

    # 计算CRLB的行列式（单位：m⁴）
    crlb_det = np.linalg.det(crlb)

    return crlb_trace, crlb_det


def analyze_crlb_vs_snr():
    """
    分析不同SNR下的CRLB理论极限
    """
    # 生成检测区域信息
    region_infos = generate_detecting_region_infos(
        num_configurations=config.num_detecting_regions, seed=config.region_seed
    )

    if not region_infos:
        print("Failed to generate detecting region information")
        return

    region_info = region_infos[0]

    # 获取发射机和接收机位置
    tx = np.array(region_info.transmittor_position)
    rx1 = np.array(region_info.receiver_position_1)
    rx2 = np.array(region_info.receiver_position_2)
    rx3 = np.array(region_info.receiver_position_3)
    rx_positions = [rx1, rx2, rx3]

    print(f"Transmitter position T: {tx}")
    print(f"Receiver positions: R1={rx1}, R2={rx2}, R3={rx3}")

    # 获取外边界顶点
    outer_vertices = [region_info.v1, region_info.v2, region_info.v3, region_info.v4]

    # 计算内边界（绿色虚线区域，scale_factor=0.7）
    inner_scale_factor = 0.7
    centroid = np.mean(outer_vertices, axis=0)
    inner_vertices = [
        centroid + (vertex - centroid) * inner_scale_factor for vertex in outer_vertices
    ]

    print(f"Inner boundary vertices (computation region): {inner_vertices}")

    # 计算内边界的边界框
    inner_vertices_array = np.array(inner_vertices)
    xmin, ymin = inner_vertices_array.min(axis=0)
    xmax, ymax = inner_vertices_array.max(axis=0)

    print(f"Computation region: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")

    # 设置计算参数
    wavelength = config.c / config.fc  # 波长
    print(f"Wavelength λ = {wavelength:.4f} m")

    # 定义SNR范围
    snr_values_db = [0, 5, 10, 15, 20]  # dB
    print(f"SNR values: {snr_values_db} dB")

    # 创建更密集的网格用于采样
    nx, ny = 101, 101  # 增加网格密度
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    print(f"Grid density: {nx}×{ny} = {nx*ny} points")

    # 存储结果
    peb_avg = []

    # 先计算平均G矩阵
    print("\nComputing average G matrix...")
    G_matrices = []

    for x in xs:
        for y in ys:
            p = np.array([x, y])
            try:
                G = compute_geometry_matrix(tx, rx_positions, p)
                G_matrices.append(G)
            except:
                continue

    if G_matrices:
        # 计算平均G矩阵
        G_avg = np.mean(G_matrices, axis=0)
        print(f"Average G matrix (3×2):")
        print(G_avg)
        print(f"Number of valid G matrices: {len(G_matrices)}")
    else:
        print("No valid G matrices computed")
        return

    print("\nComputing PEB for different SNR values...")

    for snr_db in snr_values_db:
        print(f"Processing SNR = {snr_db} dB...")

        peb_values = []

        # 遍历网格点
        for x in xs:
            for y in ys:
                p = np.array([x, y])

                try:
                    crlb_trace, _ = compute_crlb(
                        tx, rx_positions, p, snr_db, wavelength
                    )

                    if np.isfinite(crlb_trace):
                        # PEB = √(tr(Σₚ)) = √(CRLB_trace)
                        peb = np.sqrt(crlb_trace)
                        peb_values.append(peb)

                except Exception as e:
                    # 跳过奇异点
                    continue

        # 计算平均值
        if peb_values:
            avg_peb = np.mean(peb_values)
            peb_avg.append(avg_peb)

            print(f"  SNR {snr_db:2d} dB: Average PEB = {avg_peb:.6f} m")
        else:
            print(f"  SNR {snr_db:2d} dB: No valid PEB values computed")
            peb_avg.append(np.nan)

    # 绘制结果
    plot_peb_vs_snr(snr_values_db, peb_avg)

    return snr_values_db, peb_avg


def plot_peb_vs_snr(snr_values_db, peb_avg):
    """
    绘制PEB vs SNR的图表
    PEB = √(tr(Σₚ))，其中Σₚ是协方差矩阵，单位m²，所以PEB单位是m
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # PEB vs SNR
    ax.semilogy(
        snr_values_db,
        peb_avg,
        "bo-",
        linewidth=2,
        markersize=8,
        label="Average PEB",
    )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PEB (m)")
    ax.set_title("Position Error Bound vs SNR\nPEB = √(tr(Σₚ))")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 添加数值标注
    for i, (snr, peb) in enumerate(zip(snr_values_db, peb_avg)):
        if not np.isnan(peb):
            ax.annotate(
                f"{peb:.3f}",
                (snr, peb),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    plt.tight_layout()
    plt.show()

    # 打印数值结果
    print(f"\n=== PEB Analysis Results ===")
    print(f"{'SNR (dB)':<10} {'Average PEB (m)':<20}")
    print("-" * 40)
    for snr, peb in zip(snr_values_db, peb_avg):
        if not np.isnan(peb):
            print(f"{snr:<10} {peb:<20.6f}")
        else:
            print(f"{snr:<10} {'N/A':<20}")


def verify_calculation():
    """
    验证计算是否正确，使用用户提供的具体例子
    """
    print("=== 验证计算 ===")

    # 用户提供的例子
    G = np.array(
        [
            [0.26860505, -1.30455283],
            [-0.1357932, 0.02204915],
            [-1.28643854, -0.17097403],
        ]
    )

    sigma_rho = 0.02  # m
    wavelength = 0.05  # 5 cm

    print(f"G矩阵:")
    print(G)
    print(f"σ_ρ = {sigma_rho} m")
    print(f"λ = {wavelength} m")

    # 计算G^T G
    GTG = G.T @ G
    print(f"\nG^T G:")
    print(GTG)

    # 计算(G^T G)^(-1)
    GTG_inv = np.linalg.inv(GTG)
    print(f"\n(G^T G)^(-1):")
    print(GTG_inv)

    # 计算CRLB = (G^T G)^(-1) * σ_ρ^2
    crlb = GTG_inv * (sigma_rho**2)
    print(f"\nCRLB = (G^T G)^(-1) * σ_ρ^2:")
    print(crlb)

    # 计算PEB = √(tr(CRLB))
    peb = np.sqrt(np.trace(crlb))
    print(f"\nPEB = √(tr(CRLB)) = {peb:.6f} m")
    print(f"期望值: 0.0215 m")
    print(f"差异: {abs(peb - 0.0215):.6f} m")

    # 检查我的SNR计算
    print(f"\n=== 检查SNR计算 ===")
    snr_db = 10
    snr_linear = 10 ** (snr_db / 10)
    sigma_rho_calculated = wavelength / (2 * np.pi * np.sqrt(2 * snr_linear))
    print(f"SNR = {snr_db} dB")
    print(f"SNR_linear = {snr_linear}")
    print(f"计算的σ_ρ = {sigma_rho_calculated:.6f} m")
    print(f"期望σ_ρ = {sigma_rho} m")
    print(f"差异: {abs(sigma_rho_calculated - sigma_rho):.6f} m")


# 单个位置分析函数已删除，只保留平均PEB分析


if __name__ == "__main__":
    print("=== CRLB Theory Limit Analysis ===")
    print("Computing Cramer-Rao Lower Bound for different SNR values")
    print("=" * 60)

    try:
        # 先验证计算
        verify_calculation()

        print(f"\n{'='*60}")
        # 分析平均PEB vs SNR
        snr_values, peb_values = analyze_crlb_vs_snr()

    except Exception as e:
        print(f"Error occurred during computation: {e}")
        import traceback

        traceback.print_exc()
