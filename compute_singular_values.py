#!/usr/bin/env python3
"""
计算最小奇异值分析
在绿色虚线区域内计算几何矩阵的最小奇异值，用于分析定位精度
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos


def compute_singular_values_in_region():
    """
    在绿色虚线区域内计算最小奇异值
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

    print(f"Transmitter position T: {tx}")
    print(f"Receiver position R1: {rx1}")
    print(f"Receiver position R2: {rx2}")
    print(f"Receiver position R3: {rx3}")

    # 获取外边界顶点
    outer_vertices = [region_info.v1, region_info.v2, region_info.v3, region_info.v4]

    # 计算内边界（绿色虚线区域，scale_factor=0.7）
    inner_scale_factor = 0.7
    centroid = np.mean(outer_vertices, axis=0)
    inner_vertices = [
        centroid + (vertex - centroid) * inner_scale_factor for vertex in outer_vertices
    ]

    print(f"Outer boundary vertices: {outer_vertices}")
    print(f"Inner boundary vertices (green dashed area): {inner_vertices}")

    # 计算内边界的边界框
    inner_vertices_array = np.array(inner_vertices)
    xmin, ymin = inner_vertices_array.min(axis=0)
    xmax, ymax = inner_vertices_array.max(axis=0)

    print(f"Computation region: x=[{xmin:.2f}, {xmax:.2f}], y=[{ymin:.2f}, {ymax:.2f}]")

    # 设置计算参数
    lam = config.c / config.fc  # 波长
    print(f"Wavelength λ = {lam:.4f} m")

    # 创建网格
    nx, ny = 201, 201
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    # 初始化最小奇异值映射
    sigma_min_map = np.empty((nx, ny))

    print("Starting singular value computation...")

    # 遍历网格点
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            p = np.array([x, y])

            # 构造几何矩阵 G
            rows = []
            for rx in [rx1, rx2, rx3]:
                # 计算单位向量
                u_t = (tx - p) / np.linalg.norm(tx - p)
                u_r = (rx - p) / np.linalg.norm(rx - p)
                rows.append(u_t + u_r)

            G = np.vstack(rows)

            # 计算最小奇异值
            sigma_min_map[ix, iy] = np.linalg.svd(G, compute_uv=False)[-1]

    # 找到最差位置
    sigma_min_worst = sigma_min_map.min()
    worst_idx = np.unravel_index(sigma_min_map.argmin(), sigma_min_map.shape)
    worst_x = xs[worst_idx[0]]
    worst_y = ys[worst_idx[1]]

    print(f"\nResults:")
    print(f"Worst minimum singular value σ_min = {sigma_min_worst:.6f}")
    print(f"Worst position: ({worst_x:.2f}, {worst_y:.2f})")
    print(f"Average minimum singular value: {sigma_min_map.mean():.6f}")
    print(f"Standard deviation: {sigma_min_map.std():.6f}")

    # 可视化结果
    plot_singular_values(
        xs,
        ys,
        sigma_min_map,
        outer_vertices,
        inner_vertices,
        tx,
        [rx1, rx2, rx3],
        worst_x,
        worst_y,
    )

    return sigma_min_map, xs, ys, worst_x, worst_y, sigma_min_worst


def plot_singular_values(
    xs,
    ys,
    sigma_min_map,
    outer_vertices,
    inner_vertices,
    tx,
    receivers,
    worst_x,
    worst_y,
):
    """
    绘制最小奇异值分布图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：最小奇异值分布
    im1 = ax1.contourf(xs, ys, sigma_min_map.T, levels=50, cmap="viridis")
    ax1.contour(
        xs, ys, sigma_min_map.T, levels=10, colors="white", alpha=0.5, linewidths=0.5
    )

    # 绘制边界
    outer_poly = Polygon(
        outer_vertices,
        fill=False,
        edgecolor="black",
        linewidth=2,
        label="Outer Boundary",
    )
    inner_poly = Polygon(
        inner_vertices,
        fill=False,
        edgecolor="green",
        linestyle="--",
        linewidth=2,
        label="Inner Boundary (Computation Region)",
    )
    ax1.add_patch(outer_poly)
    ax1.add_patch(inner_poly)

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

    # 标记最差位置
    ax1.scatter(
        worst_x,
        worst_y,
        c="red",
        s=150,
        marker="x",
        linewidth=3,
        label="Worst Position",
        zorder=6,
    )

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("Minimum Singular Value Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # 添加颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Minimum Singular Value σ_min")

    # 右图：3D 表面图
    X, Y = np.meshgrid(xs, ys)
    ax2 = fig.add_subplot(122, projection="3d")
    surf = ax2.plot_surface(X, Y, sigma_min_map.T, cmap="viridis", alpha=0.8)

    # 标记最差位置
    ax2.scatter(
        [worst_x],
        [worst_y],
        [sigma_min_map.min()],
        c="red",
        s=100,
        marker="x",
        linewidth=3,
    )

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Minimum Singular Value σ_min")
    ax2.set_title("Minimum Singular Value 3D Distribution")

    plt.tight_layout()
    plt.savefig("singular_values_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nImage saved as: singular_values_analysis.png")


def analyze_singular_value_statistics(sigma_min_map):
    """
    Analyze statistical properties of minimum singular values
    """
    print(f"\n=== Minimum Singular Value Statistical Analysis ===")
    print(f"Minimum: {sigma_min_map.min():.6f}")
    print(f"Maximum: {sigma_min_map.max():.6f}")
    print(f"Mean: {sigma_min_map.mean():.6f}")
    print(f"Median: {np.median(sigma_min_map):.6f}")
    print(f"Standard deviation: {sigma_min_map.std():.6f}")
    print(f"Coefficient of variation: {sigma_min_map.std() / sigma_min_map.mean():.6f}")

    # 分位数分析
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\nPercentile analysis:")
    for p in percentiles:
        value = np.percentile(sigma_min_map, p)
        print(f"  {p:2d}%: {value:.6f}")


if __name__ == "__main__":
    print("=== Minimum Singular Value Analysis ===")
    print("Computing minimum singular values of geometry matrix in green dashed region")
    print("=" * 70)

    try:
        sigma_min_map, xs, ys, worst_x, worst_y, sigma_min_worst = (
            compute_singular_values_in_region()
        )
        analyze_singular_value_statistics(sigma_min_map)

    except Exception as e:
        print(f"Error occurred during computation: {e}")
        import traceback

        traceback.print_exc()
