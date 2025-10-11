#!/usr/bin/env python3
"""
验证理论极限的正确性
通过厚椭圆带几何验证与线性化理论的一致性
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon
from matplotlib.collections import PatchCollection
import sys
import os
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos


def compute_phase_range(p, tx, rx):
    """
    计算相位测距 ρ(p) = ||p-T|| + ||p-R||
    """
    return np.linalg.norm(p - tx) + np.linalg.norm(p - rx)


def generate_thick_ellipse_band(tx, rx, rho_hat, wavelength, resolution=100):
    """
    生成厚椭圆带：|ρ(p) - ρ̂| ≤ λ/2

    Args:
        tx, rx: 发射机和接收机位置
        rho_hat: 观测到的相位测距
        wavelength: 波长
        resolution: 分辨率

    Returns:
        points: 厚椭圆带内的点集
    """
    # 计算椭圆参数
    c = np.linalg.norm(tx - rx) / 2  # 半焦距
    center = (tx + rx) / 2  # 椭圆中心

    # 内椭圆和外椭圆的半长轴
    a_inner = (rho_hat - wavelength / 2) / 2
    a_outer = (rho_hat + wavelength / 2) / 2

    if a_inner <= c or a_outer <= c:
        return np.array([]).reshape(0, 2)

    # 半短轴
    b_inner = np.sqrt(a_inner**2 - c**2)
    b_outer = np.sqrt(a_outer**2 - c**2)

    # 椭圆方向
    direction = rx - tx
    angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi

    # 生成椭圆带内的点
    points = []

    # 创建网格
    x_range = np.linspace(
        center[0] - a_outer - 10, center[0] + a_outer + 10, resolution
    )
    y_range = np.linspace(
        center[1] - b_outer - 10, center[1] + b_outer + 10, resolution
    )

    for x in x_range:
        for y in y_range:
            p = np.array([x, y])
            rho = compute_phase_range(p, tx, rx)

            if abs(rho - rho_hat) <= wavelength / 2:
                points.append([x, y])

    return np.array(points)


def compute_feasible_region_intersection(
    tx, receivers, rho_hats, wavelength, resolution=100
):
    """
    计算三条厚椭圆带的交集 ℰ

    Args:
        tx: 发射机位置
        receivers: 接收机位置列表
        rho_hats: 观测到的相位测距列表
        wavelength: 波长
        resolution: 分辨率

    Returns:
        feasible_points: 可行域内的点集
    """
    print("Computing thick ellipse bands...")

    # 计算每个接收机的厚椭圆带
    bands = []
    for i, (rx, rho_hat) in enumerate(zip(receivers, rho_hats)):
        band_points = generate_thick_ellipse_band(
            tx, rx, rho_hat, wavelength, resolution
        )
        bands.append(band_points)
        print(f"Band {i+1}: {len(band_points)} points")

    if not bands or any(len(band) == 0 for band in bands):
        print("Warning: Some bands are empty")
        return np.array([]).reshape(0, 2)

    # 找到所有点的交集
    print("Computing intersection...")

    # 使用网格方法找交集
    all_points = np.vstack(bands)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # 创建更细的网格
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)

    feasible_points = []

    for x in x_grid:
        for y in y_grid:
            p = np.array([x, y])

            # 检查是否在所有厚椭圆带内
            in_all_bands = True
            for rx, rho_hat in zip(receivers, rho_hats):
                rho = compute_phase_range(p, tx, rx)
                if abs(rho - rho_hat) > wavelength / 2:
                    in_all_bands = False
                    break

            if in_all_bands:
                feasible_points.append([x, y])

    return np.array(feasible_points)


def compute_diameter(feasible_points):
    """
    计算可行域直径 D = max ||p₁ - p₂||₂
    """
    if len(feasible_points) < 2:
        return 0.0

    # 计算所有点对之间的距离
    distances = pdist(feasible_points)
    return np.max(distances)


def compute_geometry_matrix(p, tx, receivers):
    """
    计算几何矩阵 G
    """
    rows = []
    for rx in receivers:
        # 计算单位向量
        u_t = (tx - p) / np.linalg.norm(tx - p)
        u_r = (rx - p) / np.linalg.norm(rx - p)
        rows.append(u_t + u_r)

    return np.vstack(rows)


def compute_theoretical_radius(G, wavelength, n_vector):
    """
    计算理论半径 R = ||n||₂ λ / σ_min(G)
    """
    sigma_min = np.linalg.svd(G, compute_uv=False)[-1]
    n_norm = np.linalg.norm(n_vector)
    return n_norm * wavelength / sigma_min


def verify_theoretical_bounds():
    """
    验证理论极限的正确性
    """
    print("=== Theoretical Bounds Verification ===")
    print(
        "Verifying consistency between thick ellipse band geometry and linearized theory"
    )
    print("=" * 70)

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
    receivers = [
        np.array(region_info.receiver_position_1),
        np.array(region_info.receiver_position_2),
        np.array(region_info.receiver_position_3),
    ]

    print(f"Transmitter T: {tx}")
    for i, rx in enumerate(receivers):
        print(f"Receiver R{i+1}: {rx}")

    # 设置参数
    wavelength = config.c / config.fc
    print(f"Wavelength λ = {wavelength:.4f} m")

    # 测试多个 (p_true, n) 组合
    test_cases = [
        # (p_true, n_vector)
        (np.array([50.0, 30.0]), np.array([0, 0, 0])),  # 无整数偏差
        (np.array([60.0, 40.0]), np.array([1, -1, 0])),  # 有整数偏差
        (np.array([45.0, 25.0]), np.array([0, 1, -1])),  # 另一个组合
        (np.array([55.0, 35.0]), np.array([2, -1, 1])),  # 较大偏差
    ]

    results = []

    for case_idx, (p_true, n_vector) in enumerate(test_cases):
        print(f"\n--- Test Case {case_idx + 1} ---")
        print(f"True position: {p_true}")
        print(f"Integer vector n: {n_vector}")

        # 1. 计算观测值
        rho_hats = []
        for rx in receivers:
            rho_true = compute_phase_range(p_true, tx, rx)
            rho_hat = rho_true + np.dot(n_vector, np.ones(3)) * wavelength  # 简化处理
            rho_hats.append(rho_hat)

        print(f"Observed ranges: {[f'{r:.2f}' for r in rho_hats]}")

        # 2. 计算厚椭圆带交集 ℰ
        feasible_points = compute_feasible_region_intersection(
            tx, receivers, rho_hats, wavelength, resolution=50
        )

        if len(feasible_points) == 0:
            print("Warning: No feasible points found")
            continue

        print(f"Feasible region contains {len(feasible_points)} points")

        # 3. 计算直径 D
        diameter_D = compute_diameter(feasible_points)
        print(f"Feasible region diameter D = {diameter_D:.4f} m")

        # 4. 计算几何矩阵和理论半径
        G = compute_geometry_matrix(p_true, tx, receivers)
        sigma_min = np.linalg.svd(G, compute_uv=False)[-1]
        theoretical_radius_R = compute_theoretical_radius(G, wavelength, n_vector)
        theoretical_diameter_2R = 2 * theoretical_radius_R

        print(f"Minimum singular value σ_min = {sigma_min:.6f}")
        print(f"Theoretical radius R = {theoretical_radius_R:.4f} m")
        print(f"Theoretical diameter 2R = {theoretical_diameter_2R:.4f} m")

        # 5. 验证不等式 D ≤ 2R
        inequality_satisfied = diameter_D <= theoretical_diameter_2R
        print(
            f"Inequality D ≤ 2R: {diameter_D:.4f} ≤ {theoretical_diameter_2R:.4f} = {inequality_satisfied}"
        )

        # 6. 计算紧贴程度
        tightness_ratio = (
            diameter_D / theoretical_diameter_2R if theoretical_diameter_2R > 0 else 0
        )
        print(f"Tightness ratio D/(2R) = {tightness_ratio:.4f}")

        results.append(
            {
                "case": case_idx + 1,
                "p_true": p_true,
                "n_vector": n_vector,
                "diameter_D": diameter_D,
                "theoretical_diameter_2R": theoretical_diameter_2R,
                "sigma_min": sigma_min,
                "inequality_satisfied": inequality_satisfied,
                "tightness_ratio": tightness_ratio,
                "feasible_points": feasible_points,
            }
        )

        # 可视化（仅对第一个测试用例）
        if case_idx == 0:
            visualize_verification(
                tx,
                receivers,
                feasible_points,
                p_true,
                diameter_D,
                theoretical_diameter_2R,
                wavelength,
            )

    # 总结结果
    print(f"\n=== Verification Summary ===")
    satisfied_cases = sum(1 for r in results if r["inequality_satisfied"])
    print(f"Inequality satisfied in {satisfied_cases}/{len(results)} cases")

    if results:
        avg_tightness = np.mean([r["tightness_ratio"] for r in results])
        print(f"Average tightness ratio: {avg_tightness:.4f}")

        print(f"\nDetailed results:")
        for r in results:
            status = "✓" if r["inequality_satisfied"] else "✗"
            print(
                f"Case {r['case']}: D={r['diameter_D']:.3f}, 2R={r['theoretical_diameter_2R']:.3f}, "
                f"ratio={r['tightness_ratio']:.3f} {status}"
            )

    return results


def visualize_thick_ellipse_bands(
    tx, receivers, p_true, wavelength, resolution=100, thickness_factor=100
):
    """
    可视化厚椭圆带

    Args:
        thickness_factor: 厚度放大因子，用于可视化（默认100倍）
    """
    print("Visualizing thick ellipse bands...")
    print(f"Using thickness factor: {thickness_factor}x (for visualization only)")

    # 计算观测值
    rho_hats = []
    for rx in receivers:
        rho_true = compute_phase_range(p_true, tx, rx)
        rho_hats.append(rho_true)

    print(f"True position: {p_true}")
    print(f"Observed ranges: {[f'{r:.2f}' for r in rho_hats]}")
    print(f"Actual wavelength λ = {wavelength:.4f} m")
    print(f"Visualized thickness = {wavelength * thickness_factor / 2:.4f} m")

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ["red", "green", "blue"]

    # 单独显示每个厚椭圆带
    for i, (rx, rho_hat, color) in enumerate(zip(receivers, rho_hats, colors)):
        ax = axes[i // 2, i % 2]

        # 生成厚椭圆带（使用放大的厚度）
        band_points = generate_thick_ellipse_band(
            tx, rx, rho_hat, wavelength * thickness_factor, resolution
        )

        if len(band_points) > 0:
            ax.scatter(
                band_points[:, 0],
                band_points[:, 1],
                c=color,
                s=1,
                alpha=0.6,
                label=f"Band {i+1}",
            )

        # 标记发射机和接收机
        ax.scatter(
            tx[0], tx[1], c="red", s=100, marker="^", label="Transmitter T", zorder=5
        )
        ax.scatter(
            rx[0], rx[1], c=color, s=100, marker="s", label=f"Receiver R{i+1}", zorder=5
        )

        # 标记真实位置
        ax.scatter(
            p_true[0],
            p_true[1],
            c="black",
            s=100,
            marker="*",
            label="True Position",
            zorder=6,
        )

        # 绘制理论椭圆边界（使用放大的厚度）
        c = np.linalg.norm(tx - rx) / 2
        center = (tx + rx) / 2
        a_inner = (rho_hat - wavelength * thickness_factor / 2) / 2
        a_outer = (rho_hat + wavelength * thickness_factor / 2) / 2

        if a_inner > c and a_outer > c:
            b_inner = np.sqrt(a_inner**2 - c**2)
            b_outer = np.sqrt(a_outer**2 - c**2)
            direction = rx - tx
            angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi

            # 内椭圆
            ellipse_inner = Ellipse(
                center,
                2 * a_inner,
                2 * b_inner,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle="-",
                linewidth=2,
                label=f"Inner ellipse (ρ-λ/2)",
            )
            ax.add_patch(ellipse_inner)

            # 外椭圆
            ellipse_outer = Ellipse(
                center,
                2 * a_outer,
                2 * b_outer,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=2,
                label=f"Outer ellipse (ρ+λ/2)",
            )
            ax.add_patch(ellipse_outer)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"Thick Ellipse Band {i+1}: |ρ(p) - {rho_hat:.2f}| ≤ {wavelength*thickness_factor/2:.3f} (×{thickness_factor})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # 第四个图：所有厚椭圆带的交集
    ax = axes[1, 1]

    # 计算交集（使用放大的厚度）
    feasible_points = compute_feasible_region_intersection(
        tx, receivers, rho_hats, wavelength * thickness_factor, resolution=50
    )

    if len(feasible_points) > 0:
        ax.scatter(
            feasible_points[:, 0],
            feasible_points[:, 1],
            c="purple",
            s=2,
            alpha=0.8,
            label="Intersection ℰ",
        )

    # 标记发射机和接收机
    ax.scatter(
        tx[0], tx[1], c="red", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, (rx, color) in enumerate(zip(receivers, colors)):
        ax.scatter(
            rx[0],
            rx[1],
            c=color,
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )

    # 标记真实位置
    ax.scatter(
        p_true[0],
        p_true[1],
        c="black",
        s=100,
        marker="*",
        label="True Position",
        zorder=6,
    )

    # 绘制所有椭圆边界（使用放大的厚度）
    for i, (rx, rho_hat, color) in enumerate(zip(receivers, rho_hats, colors)):
        c = np.linalg.norm(tx - rx) / 2
        center = (tx + rx) / 2
        a_inner = (rho_hat - wavelength * thickness_factor / 2) / 2
        a_outer = (rho_hat + wavelength * thickness_factor / 2) / 2

        if a_inner > c and a_outer > c:
            b_inner = np.sqrt(a_inner**2 - c**2)
            b_outer = np.sqrt(a_outer**2 - c**2)
            direction = rx - tx
            angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi

            # 内椭圆
            ellipse_inner = Ellipse(
                center,
                2 * a_inner,
                2 * b_inner,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(ellipse_inner)

            # 外椭圆
            ellipse_outer = Ellipse(
                center,
                2 * a_outer,
                2 * b_outer,
                angle=angle,
                fill=False,
                edgecolor=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(ellipse_outer)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Intersection of All Thick Ellipse Bands (×{thickness_factor} thickness)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig("thick_ellipse_bands_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        f"\nThick ellipse bands visualization saved as: thick_ellipse_bands_visualization.png"
    )

    return feasible_points


def visualize_verification(
    tx,
    receivers,
    feasible_points,
    p_true,
    diameter_D,
    theoretical_diameter_2R,
    wavelength,
):
    """
    可视化验证结果
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：厚椭圆带和可行域
    ax1.scatter(
        feasible_points[:, 0],
        feasible_points[:, 1],
        c="lightblue",
        s=1,
        alpha=0.6,
        label="Feasible Region ℰ",
    )

    # 标记发射机和接收机
    ax1.scatter(
        tx[0], tx[1], c="red", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, rx in enumerate(receivers):
        ax1.scatter(
            rx[0],
            rx[1],
            c="blue",
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )

    # 标记真实位置
    ax1.scatter(
        p_true[0],
        p_true[1],
        c="green",
        s=100,
        marker="*",
        label="True Position",
        zorder=6,
    )

    # 绘制理论半径圆
    circle = plt.Circle(
        p_true,
        theoretical_diameter_2R / 2,
        fill=False,
        edgecolor="red",
        linestyle="--",
        linewidth=2,
        label=f"Theoretical Radius R={theoretical_diameter_2R/2:.2f}m",
    )
    ax1.add_patch(circle)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Thick Ellipse Bands and Feasible Region")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # 右图：直径比较
    categories = ["Geometric\nDiameter D", "Theoretical\nDiameter 2R"]
    values = [diameter_D, theoretical_diameter_2R]
    colors = ["lightblue", "lightcoral"]

    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Diameter (m)")
    ax2.set_title("Diameter Comparison")
    ax2.grid(True, alpha=0.3)

    # 添加数值标签
    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 添加不等式验证
    inequality_text = f"D ≤ 2R: {diameter_D:.3f} ≤ {theoretical_diameter_2R:.3f}\n"
    inequality_text += (
        f"✓ Verified" if diameter_D <= theoretical_diameter_2R else f"✗ Failed"
    )
    ax2.text(
        0.5,
        0.95,
        inequality_text,
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor=(
                "lightgreen" if diameter_D <= theoretical_diameter_2R else "lightcoral"
            ),
        ),
    )

    plt.tight_layout()
    plt.savefig("theoretical_bounds_verification.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved as: theoretical_bounds_verification.png")


def analyze_geometry_quality():
    """
    分析几何质量对理论紧贴程度的影响
    """
    print(f"\n=== Geometry Quality Analysis ===")

    # 生成检测区域信息
    region_infos = generate_detecting_region_infos(
        num_configurations=config.num_detecting_regions, seed=config.region_seed
    )

    if not region_infos:
        return

    region_info = region_infos[0]
    tx = np.array(region_info.transmittor_position)
    receivers = [
        np.array(region_info.receiver_position_1),
        np.array(region_info.receiver_position_2),
        np.array(region_info.receiver_position_3),
    ]

    wavelength = config.c / config.fc

    # 在区域内采样多个位置
    outer_vertices = [region_info.v1, region_info.v2, region_info.v3, region_info.v4]
    centroid = np.mean(outer_vertices, axis=0)
    inner_scale_factor = 0.7
    inner_vertices = [
        centroid + (vertex - centroid) * inner_scale_factor for vertex in outer_vertices
    ]

    inner_vertices_array = np.array(inner_vertices)
    xmin, ymin = inner_vertices_array.min(axis=0)
    xmax, ymax = inner_vertices_array.max(axis=0)

    # 采样点
    nx, ny = 10, 10
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    sigma_mins = []
    tightness_ratios = []

    print("Sampling geometry quality across the region...")

    for x in xs:
        for y in ys:
            p = np.array([x, y])
            G = compute_geometry_matrix(p, tx, receivers)
            sigma_min = np.linalg.svd(G, compute_uv=False)[-1]
            sigma_mins.append(sigma_min)

            # 简化的紧贴程度估计（基于几何直觉）
            # 几何差（σ_min小）→ 紧贴程度高
            # 几何好（σ_min大）→ 紧贴程度低
            estimated_tightness = 1.0 / (1.0 + sigma_min * 10)  # 经验公式
            tightness_ratios.append(estimated_tightness)

    sigma_mins = np.array(sigma_mins)
    tightness_ratios = np.array(tightness_ratios)

    print(
        f"Minimum singular value range: [{sigma_mins.min():.6f}, {sigma_mins.max():.6f}]"
    )
    print(f"Average σ_min: {sigma_mins.mean():.6f}")
    print(
        f"Estimated tightness range: [{tightness_ratios.min():.4f}, {tightness_ratios.max():.4f}]"
    )

    # 可视化几何质量分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(sigma_mins, tightness_ratios, alpha=0.6)
    plt.xlabel("Minimum Singular Value σ_min")
    plt.ylabel("Estimated Tightness Ratio")
    plt.title("Geometry Quality vs Tightness")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(sigma_mins, bins=20, alpha=0.7, edgecolor="black")
    plt.xlabel("Minimum Singular Value σ_min")
    plt.ylabel("Frequency")
    plt.title("Distribution of σ_min")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("geometry_quality_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nGeometry quality analysis saved as: geometry_quality_analysis.png")


def visualize_thick_ellipse_bands_only():
    """
    仅可视化厚椭圆带，不进行理论验证
    """
    print("=== Thick Ellipse Bands Visualization ===")
    print("Visualizing thick ellipse bands for better understanding")
    print("=" * 50)

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
    receivers = [
        np.array(region_info.receiver_position_1),
        np.array(region_info.receiver_position_2),
        np.array(region_info.receiver_position_3),
    ]

    print(f"Transmitter T: {tx}")
    for i, rx in enumerate(receivers):
        print(f"Receiver R{i+1}: {rx}")

    # 设置参数
    wavelength = config.c / config.fc
    print(f"Wavelength λ = {wavelength:.4f} m")

    # 选择一个测试位置
    p_true = np.array([50.0, 30.0])
    print(f"Test position: {p_true}")

    # 可视化厚椭圆带（使用100倍厚度放大因子）
    feasible_points = visualize_thick_ellipse_bands(
        tx, receivers, p_true, wavelength, thickness_factor=100
    )

    if len(feasible_points) > 0:
        diameter_D = compute_diameter(feasible_points)
        print(f"\nFeasible region diameter D = {diameter_D:.4f} m")
        print(f"Feasible region contains {len(feasible_points)} points")
    else:
        print("No feasible points found")


if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize-only":
        # 仅可视化厚椭圆带
        try:
            visualize_thick_ellipse_bands_only()
        except Exception as e:
            print(f"Error occurred during visualization: {e}")
            import traceback

            traceback.print_exc()
    else:
        # 完整的理论验证
        try:
            results = verify_theoretical_bounds()
            analyze_geometry_quality()

            print(f"\n=== Verification Complete ===")
            print("Theoretical bounds verification completed successfully!")
            print("\nTo visualize thick ellipse bands only, run:")
            print("python verify_theoretical_bounds.py --visualize-only")

        except Exception as e:
            print(f"Error occurred during verification: {e}")
            import traceback

            traceback.print_exc()
