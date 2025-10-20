#!/usr/bin/env python3
"""
验证理论极限 (worst-case)
-----------------------------------------------------------
1) 允许的最大整周误差：每链路 |n_i| ≤ 6  ⇒  ‖n‖₂_max = 6√3
2) 飞行走廊 Ω：用 Tx+3×Rx 构成的四边形按 70% 等比收缩
3) 取 σ_min^worst = min_{p∈Ω} σ_min( G(p) )
4) 理论半径 R = λ · 6√3 / σ_min^worst
   用黄色虚线圆画出
-----------------------------------------------------------
依赖: numpy / scipy / matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.path import Path  # 新增
from scipy.optimize import root, minimize
from scipy.spatial.distance import pdist
import sys, os

# ---------- 若仍需工程内模块, 保持不变 ----------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos


# -------------------------------------------------
# 基本几何与工具函数
# -------------------------------------------------
def phase_range(p, tx, rx):
    return np.linalg.norm(p - tx) + np.linalg.norm(p - rx)


def ellipse_equation(p, tx, rx, rho):
    return phase_range(p, tx, rx) - rho


def geometry_matrix(p, tx, receivers):
    rows = []
    for rx in receivers:
        rows.append(
            (p - tx) / np.linalg.norm(p - tx) + (p - rx) / np.linalg.norm(p - rx)
        )
    return np.vstack(rows)


def find_ellipse_points(tx, rx, rho, num_points=361):
    """数值采样法画椭圆"""
    pts = []
    for k in range(num_points):
        theta = 2 * np.pi * k / num_points
        d = np.array([np.cos(theta), np.sin(theta)])
        lo, hi = 0.0, rho
        for _ in range(30):
            mid = (lo + hi) / 2
            if phase_range(tx + mid * d, tx, rx) > rho:
                hi = mid
            else:
                lo = mid
        pts.append(tx + mid * d)
    return np.array(pts)


def find_ellipse_intersection(tx, rx1, rx2, rho1, rho2, guess):
    def f(p):
        return (
            ellipse_equation(p, tx, rx1, rho1),
            ellipse_equation(p, tx, rx2, rho2),
        )

    sol = root(f, guess, method="hybr")
    return sol.x if sol.success else None


# ---------- 新增：整条飞行走廊内最坏 σ_min(G) ----------
def worst_sigma_in_area(tx, receivers, inner_vertices, grid_N=200):
    """
    在多边形 inner_vertices 围成的区域 Ω 中，网格采样求
    σ_min^worst = min_{p∈Ω} σ_min(G(p))
    """
    poly_path = Path(inner_vertices)
    xmin, ymin = inner_vertices.min(axis=0)
    xmax, ymax = inner_vertices.max(axis=0)

    xs = np.linspace(xmin, xmax, grid_N)
    ys = np.linspace(ymin, ymax, grid_N)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pts = pts[poly_path.contains_points(pts)]

    sigmas = []
    for p in pts:
        G = geometry_matrix(p, tx, receivers)
        sigmas.append(np.linalg.svd(G, compute_uv=False)[-1])
    return np.min(sigmas)


# -------------------------------------------------
# 主流程
# -------------------------------------------------
def verify_theoretical_bounds():
    print("\n=== Ellipse Intersection Analysis ===")
    print("分析椭圆组A/B/B'的交点情况和定位误差")

    # ---------- 获取场景 ----------
    region_infos = generate_detecting_region_infos(
        num_configurations=config.num_detecting_regions, seed=config.region_seed
    )
    region_info = region_infos[0]

    tx = np.array(region_info.transmittor_position)
    receivers = [
        np.array(region_info.receiver_position_1),
        np.array(region_info.receiver_position_2),
        np.array(region_info.receiver_position_3),
    ]

    wavelength = config.c / config.fc
    print(f"λ = {wavelength:.4f} m")

    # ---------- 生成飞行走廊 Ω ----------
    outer_vertices = [region_info.v1, region_info.v2, region_info.v3, region_info.v4]
    outer_vertices = np.array(outer_vertices)
    centroid = outer_vertices.mean(axis=0)
    inner_vertices = centroid + 0.7 * (outer_vertices - centroid)  # 70% 缩小

    # 位置A和B
    xmin, ymin = inner_vertices.min(axis=0)
    xmax, ymax = inner_vertices.max(axis=0)
    p_A = np.array([xmin + 0.3 * (xmax - xmin), ymin + 0.4 * (ymax - ymin)])
    direction = np.array([1.0, 0.5])
    direction /= np.linalg.norm(direction)
    p_B = p_A + 0.3 * direction

    # 整周误差向量
    n_vec = np.array([2, -3, 1])

    print(f"Position A (known): {p_A}")
    print(f"Position B (true): {p_B}")
    print(f"Integer cycle errors n: {n_vec}")

    # ---------- 量测 ρ ----------
    rho_A = [phase_range(p_A, tx, rx) for rx in receivers]  # 椭圆组A
    rho_B = [phase_range(p_B, tx, rx) for rx in receivers]  # 椭圆组B
    rho_Bp = [r + n * wavelength for r, n in zip(rho_B, n_vec)]  # 椭圆组B'

    print(f"Ellipse group A ranges: {[f'{r:.2f}' for r in rho_A]}")
    print(f"Ellipse group B ranges: {[f'{r:.2f}' for r in rho_B]}")
    print(f"Ellipse group B' ranges: {[f'{r:.2f}' for r in rho_Bp]}")

    # ---------- 计算理论半径 ----------
    sigma_worst = worst_sigma_in_area(tx, receivers, inner_vertices)
    N_WORST_NORM = 6 * np.sqrt(3)  # = 10.392  (最坏 ‖n‖₂)
    R_theory = wavelength * N_WORST_NORM / sigma_worst
    print(f"σ_min^worst over Ω = {sigma_worst:.6f}")
    print(f"R (worst case) = {R_theory:.3f} m")

    # ---------- 求椭圆组A的交点（应该交于位置A） ----------
    print("\n=== 椭圆组A的交点分析 ===")
    A_intersections = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        inter = find_ellipse_intersection(
            tx, receivers[i], receivers[j], rho_A[i], rho_A[j], p_A
        )
        if inter is not None:
            A_intersections.append(inter)
            print(f"R{i+1}-R{j+1} intersection: ({inter[0]:.3f}, {inter[1]:.3f})")

    # ---------- 求椭圆组B的交点（应该交于位置B） ----------
    print("\n=== 椭圆组B的交点分析 ===")
    B_intersections = []
    for i, j in pairs:
        inter = find_ellipse_intersection(
            tx, receivers[i], receivers[j], rho_B[i], rho_B[j], p_B
        )
        if inter is not None:
            B_intersections.append(inter)
            print(f"R{i+1}-R{j+1} intersection: ({inter[0]:.3f}, {inter[1]:.3f})")

    # ---------- 求椭圆组B'的交点（误差三角形） ----------
    print("\n=== 椭圆组B'的交点分析 ===")
    Bp_intersections = []
    for i, j in pairs:
        inter = find_ellipse_intersection(
            tx, receivers[i], receivers[j], rho_Bp[i], rho_Bp[j], p_B
        )
        if inter is not None:
            Bp_intersections.append(inter)
            print(f"R{i+1}-R{j+1} intersection: ({inter[0]:.3f}, {inter[1]:.3f})")

    # ---------- 误差分析 ----------
    if len(Bp_intersections) == 3:
        Bp_intersections = np.array(Bp_intersections)
        triangle_diameter = np.max(pdist(Bp_intersections))
        print(f"\nError triangle diameter: {triangle_diameter:.3f} m")

        # 计算BB'的极值
        bb_distances = [np.linalg.norm(vertex - p_B) for vertex in Bp_intersections]
        min_bb_distance = min(bb_distances)
        max_bb_distance = max(bb_distances)
        print(f"BB' distance range: [{min_bb_distance:.3f}, {max_bb_distance:.3f}] m")
        # print(
        #     f"Theoretical bound satisfied: {'✓' if max_bb_distance <= R_theory else '✗'}"
        # )

    # ---------- 四子图可视化 ----------
    visualize_four_panel_analysis(
        tx,
        receivers,
        p_A,
        p_B,
        rho_A,
        rho_B,
        rho_Bp,
        A_intersections,
        B_intersections,
        Bp_intersections,
        R_theory,
        wavelength,
        inner_vertices,
    )

    # ---------- 图5：以A点为原点的理论圆分析 ----------
    visualize_analysis_from_A(
        tx,
        receivers,
        p_A,
        p_B,
        rho_A,
        rho_B,
        rho_Bp,
        A_intersections,
        B_intersections,
        Bp_intersections,
        R_theory,
        wavelength,
        inner_vertices,
    )

    return {
        "A_intersections": A_intersections,
        "B_intersections": B_intersections,
        "Bp_intersections": Bp_intersections,
        "theoretical_radius": R_theory,
        "triangle_diameter": triangle_diameter if len(Bp_intersections) == 3 else None,
        "bb_distance_range": (
            [min_bb_distance, max_bb_distance] if len(Bp_intersections) == 3 else None
        ),
    }


def visualize_four_panel_analysis(
    tx,
    receivers,
    p_A,
    p_B,
    rho_A,
    rho_B,
    rho_Bp,
    A_intersections,
    B_intersections,
    Bp_intersections,
    R_theory,
    wavelength,
    inner_vertices,
):
    """
    四子图可视化：椭圆组A交点、椭圆组B交点、椭圆组B'交点、误差分析
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["red", "green", "blue"]

    # 子图1：椭圆组A的交点（应该交于位置A）
    ax = axes[0, 0]
    ax.set_title("Ellipse Group A Intersections (Known Position A)")

    # 绘制椭圆组A
    for i, (rx, rho, color) in enumerate(zip(receivers, rho_A, colors)):
        pts = find_ellipse_points(tx, rx, rho)
        ax.plot(pts[:, 0], pts[:, 1], c=color, alpha=0.7, linewidth=2)

    # 绘制交点
    if len(A_intersections) >= 3:
        for i, intersection in enumerate(A_intersections[:3]):
            ax.scatter(
                intersection[0],
                intersection[1],
                c="purple",
                s=100,
                marker="o",
                zorder=6,
                label="Intersections" if i == 0 else "",
            )

    # 关键点
    ax.scatter(
        tx[0], tx[1], c="black", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, rx in enumerate(receivers):
        ax.scatter(
            rx[0],
            rx[1],
            c=colors[i],
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )
    ax.scatter(
        p_A[0], p_A[1], c="black", s=150, marker="*", label="Position A", zorder=6
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 子图2：椭圆组B的交点（应该交于位置B）
    ax = axes[0, 1]
    ax.set_title("Ellipse Group B Intersections (True Position B)")

    # 绘制椭圆组B
    for i, (rx, rho, color) in enumerate(zip(receivers, rho_B, colors)):
        pts = find_ellipse_points(tx, rx, rho)
        ax.plot(pts[:, 0], pts[:, 1], c=color, alpha=0.7, linewidth=2, linestyle="--")

    # 绘制交点
    if len(B_intersections) >= 3:
        for i, intersection in enumerate(B_intersections[:3]):
            ax.scatter(
                intersection[0],
                intersection[1],
                c="purple",
                s=100,
                marker="o",
                zorder=6,
                label="Intersections" if i == 0 else "",
            )

    # 关键点
    ax.scatter(
        tx[0], tx[1], c="black", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, rx in enumerate(receivers):
        ax.scatter(
            rx[0],
            rx[1],
            c=colors[i],
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )
    ax.scatter(
        p_B[0], p_B[1], c="green", s=150, marker="*", label="Position B", zorder=6
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 子图3：椭圆组B'的交点（误差三角形）
    ax = axes[1, 0]
    ax.set_title("Ellipse Group B' Intersections (Integer Cycle Errors)")

    # 绘制椭圆组B'
    for i, (rx, rho, color) in enumerate(zip(receivers, rho_Bp, colors)):
        pts = find_ellipse_points(tx, rx, rho)
        ax.plot(pts[:, 0], pts[:, 1], c=color, alpha=0.7, linewidth=2)

    # 绘制误差三角形
    if len(Bp_intersections) >= 3:
        Bp_array = np.array(Bp_intersections[:3])
        triangle = Polygon(
            Bp_array,
            fill=False,
            edgecolor="purple",
            linewidth=2,
            linestyle=":",
            label="Error Triangle",
        )
        ax.add_patch(triangle)
        for i, intersection in enumerate(Bp_array):
            ax.scatter(
                intersection[0],
                intersection[1],
                c="purple",
                s=100,
                marker="o",
                zorder=6,
                label="Intersections" if i == 0 else "",
            )

    # 关键点
    ax.scatter(
        tx[0], tx[1], c="black", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, rx in enumerate(receivers):
        ax.scatter(
            rx[0],
            rx[1],
            c=colors[i],
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )
    ax.scatter(
        p_B[0], p_B[1], c="green", s=150, marker="*", label="Position B", zorder=6
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # 子图4：误差分析
    ax = axes[1, 1]
    ax.set_title("Positioning Error Analysis")

    # 绘制理论圆（橙色虚线）
    circle = Circle(
        p_B,
        R_theory,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
        label=f"Theoretical Bound R={R_theory:.3f}m",
    )
    ax.add_patch(circle)

    # 绘制误差三角形
    if len(Bp_intersections) >= 3:
        Bp_array = np.array(Bp_intersections[:3])
        triangle = Polygon(
            Bp_array,
            fill=False,
            edgecolor="purple",
            linewidth=2,
            linestyle=":",
            label="Error Triangle",
        )
        ax.add_patch(triangle)
        ax.scatter(
            Bp_array[:, 0],
            Bp_array[:, 1],
            c="purple",
            s=100,
            marker="o",
            label="B' Boundary Points",
            zorder=6,
        )

    # 关键点
    ax.scatter(
        p_B[0], p_B[1], c="green", s=150, marker="*", label="Position B", zorder=6
    )

    # 添加误差分析文本
    if len(Bp_intersections) >= 3:
        bb_distances = [np.linalg.norm(vertex - p_B) for vertex in Bp_intersections[:3]]
        min_bb_distance = min(bb_distances)
        max_bb_distance = max(bb_distances)
        triangle_diameter = np.max(pdist(Bp_array))

        result_text = (
            f"BB' Distance Range: [{min_bb_distance:.3f}, {max_bb_distance:.3f}] m\n"
        )
        result_text += f"Theoretical Bound: {R_theory:.3f} m\n"
        result_text += f"Triangle Diameter: {triangle_diameter:.3f} m\n"
        # result_text += f'Bound Satisfied: {"✓" if max_bb_distance <= R_theory else "✗"}'

        bound_satisfied = max_bb_distance <= R_theory
    else:
        result_text = "Insufficient intersections for analysis"
        bound_satisfied = False

    ax.text(
        0.05,
        0.95,
        result_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor="lightgreen" if bound_satisfied else "lightcoral",
        ),
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig("ellipse_analysis_4panels.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFour-panel analysis saved as: ellipse_analysis_4panels.png")


def visualize_analysis_from_A(
    tx,
    receivers,
    p_A,
    p_B,
    rho_A,
    rho_B,
    rho_Bp,
    A_intersections,
    B_intersections,
    Bp_intersections,
    R_theory,
    wavelength,
    inner_vertices,
):
    """
    图5：以A点为原点的理论圆分析
    分析误差三角形是否在以A为中心的理论圆内
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ["red", "green", "blue"]

    ax.set_title("Analysis from Position A (Theoretical Bound)")

    # 绘制椭圆组B'（在相交三角形处的边界）
    for i, (rx, rho, color) in enumerate(zip(receivers, rho_Bp, colors)):
        pts = find_ellipse_points(tx, rx, rho)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            c=color,
            alpha=0.6,
            linewidth=1.5,
            label=f"Ellipse B'{i+1}" if i == 0 else "",
        )

    # 绘制以A为中心的理论圆（橙色虚线）
    circle_A = Circle(
        p_A,
        R_theory,
        fill=False,
        edgecolor="orange",
        linewidth=2,
        linestyle="--",
        label=f"Theoretical Bound from A (R={R_theory:.3f}m)",
    )
    ax.add_patch(circle_A)

    # 绘制误差三角形
    if len(Bp_intersections) >= 3:
        Bp_array = np.array(Bp_intersections[:3])
        triangle = Polygon(
            Bp_array,
            fill=False,
            edgecolor="purple",
            linewidth=2,
            linestyle=":",
            label="Error Triangle",
        )
        ax.add_patch(triangle)
        ax.scatter(
            Bp_array[:, 0],
            Bp_array[:, 1],
            c="purple",
            s=100,
            marker="o",
            label="B' Boundary Points",
            zorder=6,
        )

    # 关键点
    ax.scatter(
        tx[0], tx[1], c="black", s=100, marker="^", label="Transmitter T", zorder=5
    )
    for i, rx in enumerate(receivers):
        ax.scatter(
            rx[0],
            rx[1],
            c=colors[i],
            s=80,
            marker="s",
            label=f"Receiver R{i+1}" if i == 0 else "",
            zorder=5,
        )
    ax.scatter(
        p_A[0], p_A[1], c="black", s=150, marker="*", label="Position A", zorder=6
    )
    ax.scatter(
        p_B[0], p_B[1], c="green", s=150, marker="*", label="Position B", zorder=6
    )

    # 添加分析文本
    if len(Bp_intersections) >= 3:
        Bp_array = np.array(Bp_intersections[:3])

        # 计算从A点到三角形顶点的距离
        aa_distances = [np.linalg.norm(vertex - p_A) for vertex in Bp_intersections[:3]]
        min_aa_distance = min(aa_distances)
        max_aa_distance = max(aa_distances)
        triangle_diameter = np.max(pdist(Bp_array))

        # 检查三角形是否在圆内
        triangle_in_circle = max_aa_distance <= R_theory

        result_text = (
            f"AA' Distance Range: [{min_aa_distance:.3f}, {max_aa_distance:.3f}] m\n"
        )
        result_text += f"Theoretical Bound from A: {R_theory:.3f} m\n"
        result_text += f"Triangle Diameter: {triangle_diameter:.3f} m\n"
        result_text += f'Triangle in Circle: {"✓" if triangle_in_circle else "✗"}'

        bound_satisfied = triangle_in_circle
    else:
        result_text = "Insufficient intersections for analysis"
        bound_satisfied = False

    ax.text(
        0.05,
        0.95,
        result_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round",
            facecolor="lightgreen" if bound_satisfied else "lightcoral",
        ),
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    plt.savefig("ellipse_analysis_from_A.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nAnalysis from A saved as: ellipse_analysis_from_A.png")


if __name__ == "__main__":
    verify_theoretical_bounds()
