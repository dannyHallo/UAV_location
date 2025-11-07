#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import LinAlgError
from scipy.spatial import ConvexHull
from matplotlib.path import Path

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# -----------------------------
# Geometry
# -----------------------------
T = np.array([0, 0])
R1 = np.array([200, 0])
R2 = np.array([100, 100])
R3 = np.array([300, 150])
receivers = [R1, R2, R3]


# -----------------------------
# Functions
# -----------------------------
def calculate_geometry_matrix(target, transmitter, receivers):
    """
    计算几何矩阵 G (即雅可比矩阵)
    G的第i行 = [∂ρᵢ/∂x, ∂ρᵢ/∂y]
    其中 ρᵢ = |T-p| + |Rᵢ-p| 是双基地距离
    """
    x, y = target
    x_t, y_t = transmitter

    G = []

    for receiver in receivers:
        x_r, y_r = receiver

        # 计算发射机到目标的距离
        R_T = np.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
        # 计算接收机到目标的距离
        R_R = np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2)

        # 避免除以零
        if R_T < 1e-12 or R_R < 1e-12:
            return None

        # 计算对x的偏导数
        dRho_dx = (x - x_t) / R_T + (x - x_r) / R_R
        # 计算对y的偏导数
        dRho_dy = (y - y_t) / R_T + (y - y_r) / R_R

        G.append([dRho_dx, dRho_dy])

    return np.array(G)


def calculate_GDOP(G):
    """
    计算几何精度衰减因子 GDOP
    GDOP = √(trace((G^T G)^(-1)))
    """
    if G is None:
        return np.inf, None, None

    GTG = G.T @ G
    try:
        GTG_inv = np.linalg.inv(GTG)
        GDOP = np.sqrt(np.trace(GTG_inv))
        return GDOP, GTG, GTG_inv
    except (LinAlgError, np.linalg.LinAlgError):
        # ill-conditioned -> return large number
        return np.inf, GTG, None


# -----------------------------
# Grid setup (cover the region enclosing T and receivers with margins)
# -----------------------------
all_pts = np.vstack([T, R1, R2, R3])
mins = all_pts.min(axis=0) - 20.0
maxs = all_pts.max(axis=0) + 20.0

# You can adjust resolution if needed
nx, ny = 401, 321  # ~ dense but still fast
xs = np.linspace(mins[0], maxs[0], nx)
ys = np.linspace(mins[1], maxs[1], ny)

GDOP_map = np.zeros((ny, nx), dtype=float)

# -----------------------------
# Compute GDOP map
# -----------------------------
for iy, y in enumerate(ys):
    for ix, x in enumerate(xs):
        p = np.array([x, y])
        # Skip exactly at sensor locations to avoid undefined gradients
        if (
            np.allclose(p, T)
            or np.allclose(p, R1)
            or np.allclose(p, R2)
            or np.allclose(p, R3)
        ):
            GDOP_map[iy, ix] = np.inf
            continue

        # Calculate geometry matrix
        G = calculate_geometry_matrix(p, T, receivers)

        # Calculate GDOP
        GDOP_val, _, _ = calculate_GDOP(G)
        GDOP_map[iy, ix] = GDOP_val

# Mask infinities or very large outliers
GDOP_map = np.where(np.isfinite(GDOP_map), GDOP_map, np.nan)

# -----------------------------
# Find min and max within convex hull region (optional mask)
# For simplicity, we report over whole plotted rectangle.
# -----------------------------
min_val = np.nanmin(GDOP_map)
max_val = np.nanmax(GDOP_map)

min_idx = np.nanargmin(GDOP_map)
max_idx = np.nanargmax(GDOP_map)
min_iy, min_ix = np.unravel_index(min_idx, GDOP_map.shape)
max_iy, max_ix = np.unravel_index(max_idx, GDOP_map.shape)
p_min = np.array([xs[min_ix], ys[min_iy]])
p_max = np.array([xs[max_ix], ys[max_iy]])

# -----------------------------
# Calculate average GDOP
# -----------------------------
# 1. 整个有效区域的平均GDOP（排除NaN值）
valid_gdop = GDOP_map[np.isfinite(GDOP_map)]
mean_gdop_all = np.mean(valid_gdop)
median_gdop_all = np.median(valid_gdop)
std_gdop_all = np.std(valid_gdop)

# 2. 凸包区域内的平均GDOP
# 构建传感器位置点（包括发射机和所有接收机）
sensor_points = np.vstack([T, R1, R2, R3])
# 计算凸包
hull = ConvexHull(sensor_points)
hull_points = sensor_points[hull.vertices]
# 创建凸包路径
hull_path = Path(hull_points)

# 创建网格点坐标
xx, yy = np.meshgrid(xs, ys)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# 检查哪些点在凸包内
inside_hull = hull_path.contains_points(grid_points)
inside_hull = inside_hull.reshape(GDOP_map.shape)

# 计算凸包内的GDOP统计
gdop_in_hull = GDOP_map[inside_hull & np.isfinite(GDOP_map)]
if len(gdop_in_hull) > 0:
    mean_gdop_hull = np.mean(gdop_in_hull)
    median_gdop_hull = np.median(gdop_in_hull)
    std_gdop_hull = np.std(gdop_in_hull)
else:
    mean_gdop_hull = np.nan
    median_gdop_hull = np.nan
    std_gdop_hull = np.nan

print("=" * 70)
print("几何配置")
print("=" * 70)
print(f"发射机位置 T: {T}")
print(f"接收机1位置 R1: {R1}")
print(f"接收机2位置 R2: {R2}")
print(f"接收机3位置 R3: {R3}")

print("\n" + "=" * 70)
print("GDOP 分布统计")
print("=" * 70)
print(f"最小 GDOP = {min_val:.6f} 在位置 {p_min}")
print(f"最大 GDOP = {max_val:.6f} 在位置 {p_max}")

print("\n--- 整个有效区域统计 ---")
print(f"平均 GDOP = {mean_gdop_all:.6f}")
print(f"中位数 GDOP = {median_gdop_all:.6f}")
print(f"标准差 = {std_gdop_all:.6f}")
print(f"有效点数 = {len(valid_gdop)}")

print("\n--- 凸包区域统计（传感器围成区域） ---")
if not np.isnan(mean_gdop_hull):
    print(f"平均 GDOP = {mean_gdop_hull:.6f}")
    print(f"中位数 GDOP = {median_gdop_hull:.6f}")
    print(f"标准差 = {std_gdop_hull:.6f}")
    print(f"有效点数 = {len(gdop_in_hull)}")
else:
    print("凸包区域内无有效数据点")

print("\n注: GDOP越小表示几何配置越好(测量噪声放大较小)。")

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Colormap with reasonable clipping
vmin = np.nanpercentile(GDOP_map, 5)
vmax = np.nanpercentile(GDOP_map, 95)
im = ax.imshow(
    GDOP_map,
    origin="lower",
    extent=[xs[0], xs[-1], ys[0], ys[-1]],
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
    aspect="equal",
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("GDOP (√(trace((G^T G)^(-1))))", fontsize=12)

# Mark sensors and optima
ax.scatter(*T, c="k", marker="*", s=120, label="发射机 T", zorder=5)
ax.scatter(*R1, c="C1", marker="^", s=80, label="接收机 R1", zorder=5)
ax.scatter(*R2, c="C2", marker="^", s=80, label="接收机 R2", zorder=5)
ax.scatter(*R3, c="C3", marker="^", s=80, label="接收机 R3", zorder=5)
ax.scatter(
    *p_min,
    c="w",
    edgecolors="k",
    marker="o",
    s=120,
    label=f"最小 GDOP ({min_val:.3f})",
    zorder=5,
    linewidths=2,
)
ax.scatter(
    *p_max,
    c="r",
    marker="x",
    s=100,
    label=f"最大 GDOP ({max_val:.3f})",
    zorder=5,
    linewidths=2,
)

# 绘制凸包边界
hull_points_closed = np.vstack([hull_points, hull_points[0]])  # 闭合路径
ax.plot(
    hull_points_closed[:, 0],
    hull_points_closed[:, 1],
    "r--",
    linewidth=2,
    alpha=0.7,
    label=f"凸包边界 (平均GDOP: {mean_gdop_hull:.3f})",
    zorder=4,
)

ax.set_title("双基地雷达定位系统 GDOP 分布图", fontsize=14, fontweight="bold")
ax.set_xlabel("x [m]", fontsize=12)
ax.set_ylabel("y [m]", fontsize=12)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("GDOP_map.png", dpi=150, bbox_inches="tight")
fig.show()
print("\n图表已保存为: GDOP_map.png")
