import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as EllipsePatch
from scipy.optimize import fsolve

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ======================== 系统参数 ========================
T = np.array([0, 0])
R1 = np.array([200, 0])
R2 = np.array([100, 100])
R3 = np.array([300, 150])
p_A = np.array([40, 30])  # 目标A位置
p_B_true = np.array([40.1, 30.1])  # 真实B点位置

# 系统参数
lambda_wave = 0.05  # 波长 (m)
T_CPI = 0.01  # 相干处理时间 (s)
v_max = 30  # 最大速度 (m/s)
d_max = v_max * T_CPI  # 最大位移 0.3m

# SNR参数 - 选择一个中等SNR来演示
SNR_dB = 5
SNR_linear = 10 ** (SNR_dB / 10)

print("=" * 80)
print("椭圆定位误差可视化 - 验证理论极限")
print("=" * 80)
print(f"目标A位置: {p_A}")
print(f"目标B真实位置: {p_B_true}")
print(f"A到B的真实位移: {p_B_true - p_A}")
print(f"SNR: {SNR_dB} dB")
print("=" * 80)

# ======================== 函数定义 ========================


def calculate_bistatic_range(point, transmitter, receiver):
    """计算双基地距离"""
    R_T = np.linalg.norm(point - transmitter)
    R_R = np.linalg.norm(point - receiver)
    return R_T + R_R


def calculate_doppler_shift(p1, p2, transmitter, receiver, T_CPI):
    """计算多普勒频移（Hz）
    由双基地距离变化与 (λ, T_CPI) 的关系得到，保持与生成器一致的物理关系。
    """
    R1_T = np.linalg.norm(p1 - transmitter)
    R1_R = np.linalg.norm(p1 - receiver)
    R2_T = np.linalg.norm(p2 - transmitter)
    R2_R = np.linalg.norm(p2 - receiver)

    delta_R = (R2_T + R2_R) - (R1_T + R1_R)
    f_d = delta_R / (lambda_wave * T_CPI)
    return f_d


def calculate_phase_difference(p1, p2, transmitter, receiver):
    """计算相位差（弧度）
    使用 Δφ = 2π·ΔR/λ，并将结果归一化到 [0, 2π)。
    """
    R1_T = np.linalg.norm(p1 - transmitter)
    R1_R = np.linalg.norm(p1 - receiver)
    R2_T = np.linalg.norm(p2 - transmitter)
    R2_R = np.linalg.norm(p2 - receiver)

    delta_R = (R2_T + R2_R) - (R1_T + R1_R)
    phase_diff_rad = (delta_R / lambda_wave) * (2 * np.pi)
    return phase_diff_rad % (2 * np.pi)


def measurement_error_doppler(SNR_linear, T_CPI, lambda_wave):
    """多普勒测量误差标准差（Hz）
    对齐生成器噪声模型：σ_f = sqrt(1 / (SNR * T_CPI^2))。
    """
    sigma_fd = np.sqrt(1 / (SNR_linear * (T_CPI**2)))
    return sigma_fd


def measurement_error_phase(SNR_linear):
    """相位差测量误差标准差（弧度）
    对齐生成器噪声模型：σ_Δφ = sqrt(1 / SNR)。
    返回弧度。
    """
    sigma_phi_rad = np.sqrt(1 / SNR_linear)
    return sigma_phi_rad


def find_ellipse_intersections(bistatic_range1, bistatic_range2, T, R1, R2):
    """找到两个双基地椭圆的交点"""

    def equations(p):
        x, y = p
        point = np.array([x, y])
        eq1 = calculate_bistatic_range(point, T, R1) - bistatic_range1
        eq2 = calculate_bistatic_range(point, T, R2) - bistatic_range2
        return [eq1, eq2]

    # 尝试多个初始点
    intersections = []
    for x0 in [0, 50, 100, 150]:
        for y0 in [0, 50, 100, 150]:
            try:
                sol = fsolve(equations, [x0, y0], full_output=True)
                if sol[2] == 1:  # 成功求解
                    point = sol[0]
                    # 检查是否已经找到这个点
                    is_new = True
                    for existing in intersections:
                        if np.linalg.norm(point - existing) < 1e-3:
                            is_new = False
                            break
                    if is_new:
                        intersections.append(point)
            except:
                pass

    return intersections


# ======================== 计算A点的真实参数 ========================
print("\n" + "=" * 80)
print("1. A点的真实椭圆参数（基准）")
print("=" * 80)

receivers = [R1, R2, R3]
bistatic_ranges_A = []
for i, R in enumerate(receivers):
    br = calculate_bistatic_range(p_A, T, R)
    bistatic_ranges_A.append(br)
    print(f"接收机R{i+1}: 双基地距离 = {br:.6f} m")

# ======================== 计算B点的真实参数（无误差）========================
print("\n" + "=" * 80)
print("2. B点的真实椭圆参数（无测量误差）")
print("=" * 80)

bistatic_ranges_B_true = []
dopplers_B_true = []
phases_B_true = []

for i, R in enumerate(receivers):
    br = calculate_bistatic_range(p_B_true, T, R)
    bistatic_ranges_B_true.append(br)

    fd = calculate_doppler_shift(p_A, p_B_true, T, R, T_CPI)
    dopplers_B_true.append(fd)

    phase = calculate_phase_difference(p_A, p_B_true, T, R)
    phases_B_true.append(phase)

    delta_R = br - bistatic_ranges_A[i]

    print(f"\n接收机R{i+1}:")
    print(f"  双基地距离: {br:.6f} m")
    print(f"  距离变化 ΔR: {delta_R:.6f} m")
    print(f"  多普勒频移: {fd:.6f} Hz")
    print(f"  相位差: {phase:.6f} rad")

# ======================== 计算测量误差 ========================
print("\n" + "=" * 80)
print("3. 测量误差分析")
print("=" * 80)

sigma_fd = measurement_error_doppler(SNR_linear, T_CPI, lambda_wave)
sigma_phi = measurement_error_phase(SNR_linear)

print(f"多普勒测量误差标准差: σ_fd = {sigma_fd:.6f} Hz")
print(f"相位差测量误差标准差: σ_φ = {sigma_phi:.6f} rad")

# ======================== 添加测量误差（1个标准差）========================
print("\n" + "=" * 80)
print("4. B点的测量椭圆参数（含测量误差，+1σ，相位弧度）")
print("=" * 80)

bistatic_ranges_B_measured = []
dopplers_B_measured = []
phases_B_measured = []

np.random.seed(42)  # 固定随机种子以便复现

for i, R in enumerate(receivers):
    # 添加1个标准差的误差（可以是正或负）
    error_sign = 1 if i % 2 == 0 else -1  # 交替正负以展示效果

    fd_measured = dopplers_B_true[i] + error_sign * sigma_fd
    phase_measured = (phases_B_true[i] + error_sign * sigma_phi) % (2 * np.pi)

    # 从多普勒和相位反推距离变化（含误差）
    delta_R_from_doppler = fd_measured * lambda_wave * T_CPI
    delta_R_from_phase = (phase_measured / (2 * np.pi)) * lambda_wave

    # 使用多普勒计算的距离变化（更可靠）
    delta_R_measured = delta_R_from_doppler
    br_measured = bistatic_ranges_A[i] + delta_R_measured

    bistatic_ranges_B_measured.append(br_measured)
    dopplers_B_measured.append(fd_measured)
    phases_B_measured.append(phase_measured)

    print(f"\n接收机R{i+1}:")
    print(f"  真实多普勒: {dopplers_B_true[i]:.6f} Hz")
    print(
        f"  测量多普勒: {fd_measured:.6f} Hz (误差: {error_sign:+d}σ = {error_sign * sigma_fd:+.6f} Hz)"
    )
    print(f"  真实相位差: {phases_B_true[i]:.6f} rad")
    print(
        f"  测量相位差: {phase_measured:.6f} rad (误差: {error_sign:+d}σ = {error_sign * sigma_phi:+.6f} rad)"
    )
    print(f"  真实双基地距离: {bistatic_ranges_B_true[i]:.6f} m")
    print(f"  测量双基地距离: {br_measured:.6f} m")

# ======================== 计算三角形顶点 ========================
print("\n" + "=" * 80)
print("5. 计算定位三角形的顶点（两两椭圆交点）")
print("=" * 80)

# 椭圆1和椭圆2的交点
intersections_12 = find_ellipse_intersections(
    bistatic_ranges_B_measured[0], bistatic_ranges_B_measured[1], T, R1, R2
)

# 椭圆2和椭圆3的交点
intersections_23 = find_ellipse_intersections(
    bistatic_ranges_B_measured[1], bistatic_ranges_B_measured[2], T, R2, R3
)

# 椭圆1和椭圆3的交点
intersections_13 = find_ellipse_intersections(
    bistatic_ranges_B_measured[0], bistatic_ranges_B_measured[2], T, R1, R3
)

print(f"椭圆R1-R2交点: {len(intersections_12)} 个")
for p in intersections_12:
    print(f"  {p}")

print(f"椭圆R2-R3交点: {len(intersections_23)} 个")
for p in intersections_23:
    print(f"  {p}")

print(f"椭圆R1-R3交点: {len(intersections_13)} 个")
for p in intersections_13:
    print(f"  {p}")

# 选择最接近真实B点的交点作为三角形顶点
triangle_vertices = []
for intersections in [intersections_12, intersections_23, intersections_13]:
    if len(intersections) > 0:
        # 选择离B点真实位置最近的交点
        closest = min(intersections, key=lambda p: np.linalg.norm(p - p_B_true))
        triangle_vertices.append(closest)

if len(triangle_vertices) == 3:
    print(f"\n定位三角形的三个顶点:")
    for i, v in enumerate(triangle_vertices):
        dist_to_true = np.linalg.norm(v - p_B_true)
        print(f"  顶点{i+1}: ({v[0]:.4f}, {v[1]:.4f}), 距真实B点: {dist_to_true:.4f} m")

    # 计算三角形面积
    v1, v2, v3 = triangle_vertices
    area = 0.5 * abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    )
    print(f"\n定位三角形面积: {area:.6f} m²")

# ======================== 可视化 ========================
print("\n" + "=" * 80)
print("6. 生成可视化图像")
print("=" * 80)


def plot_bistatic_ellipse(ax, transmitter, receiver, bistatic_range, **kwargs):
    """绘制双基地椭圆"""
    # 焦点
    f1, f2 = transmitter, receiver
    c = np.linalg.norm(f2 - f1) / 2  # 焦距的一半
    a = bistatic_range / 2  # 半长轴

    if a > c:
        b = np.sqrt(a**2 - c**2)  # 半短轴

        # 椭圆中心
        center = (f1 + f2) / 2

        # 旋转角度
        angle = np.degrees(np.arctan2(f2[1] - f1[1], f2[0] - f1[0]))

        # 绘制椭圆
        ellipse = EllipsePatch(center, 2 * a, 2 * b, angle=angle, **kwargs)
        ax.add_patch(ellipse)

        return True
    return False


# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# ======================== 子图1: A点的椭圆 ========================
ax1 = axes[0]
ax1.set_title("A点的真实椭圆\n(三个椭圆交于A点)", fontsize=14, fontweight="bold")

# 绘制发射机和接收机
ax1.plot(T[0], T[1], "r^", markersize=15, label="发射机T")
for i, R in enumerate(receivers):
    ax1.plot(R[0], R[1], "bs", markersize=12, label=f"R{i+1}")
    ax1.text(R[0], R[1] + 10, f"R{i+1}", ha="center", fontsize=10, fontweight="bold")

# 绘制A点
ax1.plot(p_A[0], p_A[1], "go", markersize=15, label="目标A", zorder=10)
ax1.text(p_A[0], p_A[1] - 10, "A", ha="center", fontsize=12, fontweight="bold")

# 绘制三个椭圆
colors = ["red", "blue", "green"]
for i, R in enumerate(receivers):
    plot_bistatic_ellipse(
        ax1,
        T,
        R,
        bistatic_ranges_A[i],
        fill=False,
        edgecolor=colors[i],
        linewidth=2,
        linestyle="-",
        alpha=0.7,
        label=f"椭圆R{i+1}",
    )

ax1.set_xlabel("X (m)", fontsize=12)
ax1.set_ylabel("Y (m)", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", fontsize=9)
ax1.set_xlim(-50, 350)
ax1.set_ylim(-50, 200)
ax1.set_aspect("equal")

# ======================== 子图2: B点的真实椭圆 ========================
ax2 = axes[1]
title2 = "B点的真实椭圆（无测量误差）\n(三个椭圆交于真实B点)"
ax2.set_title(title2, fontsize=14, fontweight="bold")

# 绘制发射机和接收机
ax2.plot(T[0], T[1], "r^", markersize=15, label="发射机T")
for i, R in enumerate(receivers):
    ax2.plot(R[0], R[1], "bs", markersize=12)
    ax2.text(R[0], R[1] + 10, f"R{i+1}", ha="center", fontsize=10, fontweight="bold")

# 绘制A点和真实B点
ax2.plot(p_A[0], p_A[1], "go", markersize=12, label="目标A", zorder=10, alpha=0.5)
ax2.plot(p_B_true[0], p_B_true[1], "mo", markersize=15, label="真实B点", zorder=10)
ax2.text(
    p_B_true[0], p_B_true[1] - 10, "B", ha="center", fontsize=12, fontweight="bold"
)

# 绘制A→B的位移向量
ax2.arrow(
    p_A[0],
    p_A[1],
    p_B_true[0] - p_A[0],
    p_B_true[1] - p_A[1],
    head_width=2,
    head_length=2,
    fc="purple",
    ec="purple",
    linewidth=2,
    alpha=0.6,
)

# 绘制三个椭圆
for i, R in enumerate(receivers):
    plot_bistatic_ellipse(
        ax2,
        T,
        R,
        bistatic_ranges_B_true[i],
        fill=False,
        edgecolor=colors[i],
        linewidth=2,
        linestyle="-",
        alpha=0.7,
        label=f"椭圆R{i+1}",
    )

# 添加真实椭圆变化信息
info_text = "真实椭圆变化:\n"
for i in range(3):
    info_text += (
        f"R{i+1}: Δf={dopplers_B_true[i]:.3f}Hz, Δφ={phases_B_true[i]:.2f}rad\n"
    )

ax2.text(
    0.02,
    0.98,
    info_text,
    transform=ax2.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

ax2.set_xlabel("X (m)", fontsize=12)
ax2.set_ylabel("Y (m)", fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_xlim(-50, 350)
ax2.set_ylim(-50, 200)
ax2.set_aspect("equal")

# ======================== 子图3: B点的测量椭圆（含误差）========================
ax3 = axes[2]
title3 = (
    f"B点的测量椭圆（含测量误差，相位弧度）\n(SNR={SNR_dB}dB, 三个椭圆形成定位三角形)"
)
ax3.set_title(title3, fontsize=14, fontweight="bold")

# 绘制发射机和接收机
ax3.plot(T[0], T[1], "r^", markersize=15, label="发射机T")
for i, R in enumerate(receivers):
    ax3.plot(R[0], R[1], "bs", markersize=12)
    ax3.text(R[0], R[1] + 10, f"R{i+1}", ha="center", fontsize=10, fontweight="bold")

# 绘制A点和真实B点
ax3.plot(p_A[0], p_A[1], "go", markersize=12, label="目标A", zorder=10, alpha=0.5)
ax3.plot(p_B_true[0], p_B_true[1], "mo", markersize=15, label="真实B点", zorder=11)
ax3.text(
    p_B_true[0] + 2,
    p_B_true[1] + 2,
    "B(真实)",
    ha="left",
    fontsize=10,
    fontweight="bold",
)

# 绘制三个测量椭圆
for i, R in enumerate(receivers):
    plot_bistatic_ellipse(
        ax3,
        T,
        R,
        bistatic_ranges_B_measured[i],
        fill=False,
        edgecolor=colors[i],
        linewidth=2,
        linestyle="--",
        alpha=0.7,
        label=f"测量椭圆R{i+1}",
    )

# 绘制定位三角形
if len(triangle_vertices) == 3:
    triangle = plt.Polygon(
        triangle_vertices,
        fill=True,
        facecolor="yellow",
        edgecolor="black",
        linewidth=2,
        alpha=0.3,
        label="定位不确定区域",
    )
    ax3.add_patch(triangle)

    # 标注三角形顶点
    for i, v in enumerate(triangle_vertices):
        ax3.plot(v[0], v[1], "kx", markersize=12, markeredgewidth=3)
        ax3.text(
            v[0] + 2, v[1] + 2, f"P{i+1}", ha="left", fontsize=9, fontweight="bold"
        )

# 添加测量椭圆变化信息
info_text = "测量椭圆变化:\n"
info_text += f"σ_fd={sigma_fd:.4f}Hz, σ_φ={sigma_phi:.3f}rad\n\n"
for i in range(3):
    error_sign = 1 if i % 2 == 0 else -1
    info_text += f"R{i+1}: Δf={dopplers_B_measured[i]:.3f}Hz({error_sign:+d}σ)\n"
    info_text += f"    Δφ={phases_B_measured[i]:.2f}rad({error_sign:+d}σ)\n"

ax3.text(
    0.02,
    0.98,
    info_text,
    transform=ax3.transAxes,
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
)

ax3.set_xlabel("X (m)", fontsize=12)
ax3.set_ylabel("Y (m)", fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper right", fontsize=9)
ax3.set_xlim(-50, 350)
ax3.set_ylim(-50, 200)
ax3.set_aspect("equal")

plt.tight_layout()
plt.savefig("ellipse_positioning_verification.png", dpi=300, bbox_inches="tight")
print("可视化图像已保存为 'ellipse_positioning_verification.png'")
plt.show()

# ======================== 局部放大图 ========================
fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))

zoom_range = 1  # 放大范围

for idx, (ax, title_suffix) in enumerate(
    zip(axes2, ["A点附近", "B点真实椭圆", "B点测量椭圆（定位三角形）"])
):

    ax.set_title(f"{title_suffix} - 局部放大图", fontsize=14, fontweight="bold")

    # 根据不同子图选择中心点
    if idx == 0:
        center = p_A
        bistatic_ranges = bistatic_ranges_A
        target_point = p_A
        target_label = "A"
        target_color = "go"
    elif idx == 1:
        center = p_B_true
        bistatic_ranges = bistatic_ranges_B_true
        target_point = p_B_true
        target_label = "B(真实)"
        target_color = "mo"
    else:
        center = p_B_true
        bistatic_ranges = bistatic_ranges_B_measured
        target_point = p_B_true
        target_label = "B(真实)"
        target_color = "mo"

    # 绘制目标点
    ax.plot(p_A[0], p_A[1], "go", markersize=10, label="A点", alpha=0.5, zorder=10)
    if idx > 0:
        ax.plot(
            p_B_true[0], p_B_true[1], "mo", markersize=12, label="真实B点", zorder=11
        )
        ax.text(
            p_B_true[0] + 0.5,
            p_B_true[1] + 0.5,
            "B",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

        # A→B的位移向量
        ax.arrow(
            p_A[0],
            p_A[1],
            p_B_true[0] - p_A[0],
            p_B_true[1] - p_A[1],
            head_width=0.3,
            head_length=0.3,
            fc="purple",
            ec="purple",
            linewidth=2,
            alpha=0.6,
        )

    # 绘制三个椭圆
    linestyle = "--" if idx == 2 else "-"
    for i, R in enumerate(receivers):
        plot_bistatic_ellipse(
            ax,
            T,
            R,
            bistatic_ranges[i],
            fill=False,
            edgecolor=colors[i],
            linewidth=2,
            linestyle=linestyle,
            alpha=0.7,
            label=f"椭圆R{i+1}",
        )

    # 如果是第三个子图，绘制定位三角形
    if idx == 2 and len(triangle_vertices) == 3:
        triangle = plt.Polygon(
            triangle_vertices,
            fill=True,
            facecolor="yellow",
            edgecolor="black",
            linewidth=2,
            alpha=0.4,
            label="定位区域",
        )
        ax.add_patch(triangle)

        for i, v in enumerate(triangle_vertices):
            ax.plot(v[0], v[1], "kx", markersize=10, markeredgewidth=2)

    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(center[0] - zoom_range, center[0] + zoom_range)
    ax.set_ylim(center[1] - zoom_range, center[1] + zoom_range)
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("ellipse_positioning_verification_zoom.png", dpi=300, bbox_inches="tight")
print("局部放大图已保存为 'ellipse_positioning_verification_zoom.png'")
plt.show()

print("\n" + "=" * 80)
print("程序执行完成！")
print("=" * 80)
