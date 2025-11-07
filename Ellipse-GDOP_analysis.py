import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ==================== 理论严格的椭圆GDOP类 ====================
class TheoreticalEllipseGDOP:
    """
    基于Fisher信息矩阵理论的椭圆定位GDOP计算器

    完整理论下界公式：
    PEB = σ_ρ × GDOP_traditional × γ_eccentricity
    """

    def __init__(self, transmitter, receivers, target, wavelength, target_velocity):
        """
        参数:
            transmitter: 发射机位置 (x, y)
            receivers: 接收机位置列表 [(x1,y1), (x2,y2), ...]
            target: 目标位置 (x, y)
            wavelength: 雷达波长 (m)
            target_velocity: 目标速度 (m/s)
        """
        self.T = np.array(transmitter)
        self.receivers = [np.array(r) for r in receivers]
        self.target = np.array(target)
        self.N = len(receivers)
        self.wavelength = wavelength
        self.v = target_velocity

        # 计算基础几何参数
        self.G = self._compute_geometry_matrix()
        self.M = self.G.T @ self.G  # 传统信息矩阵

    def _compute_geometry_matrix(self):
        """计算传统几何矩阵G (假设圆形等高线)"""
        x, y = self.target
        x_t, y_t = self.T

        G = []
        for receiver in self.receivers:
            x_r, y_r = receiver

            R_T = np.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
            R_R = np.sqrt((x - x_r) ** 2 + (y - y_r) ** 2)

            dRho_dx = (x - x_t) / R_T + (x - x_r) / R_R
            dRho_dy = (y - y_t) / R_T + (y - y_r) / R_R

            G.append([dRho_dx, dRho_dy])

        return np.array(G)

    def compute_ellipse_parameters(self, receiver_idx):
        """计算第i个椭圆的参数"""
        receiver = self.receivers[receiver_idx]

        # 焦距的一半
        c = np.linalg.norm(receiver - self.T) / 2

        # 双基地距离
        R_T = np.linalg.norm(self.target - self.T)
        R_R = np.linalg.norm(self.target - receiver)
        rho = R_T + R_R
        a = rho / 2  # 半长轴

        # 离心率
        if a > 1e-10:
            e = c / a
            e = min(e, 0.9999)
        else:
            e = 0.0

        # 主轴方向
        baseline = receiver - self.T
        phi = np.arctan2(baseline[1], baseline[0])

        return e, phi, e**2

    def compute_enhanced_FIM_correct(self, sigma_measurement):
        """计算增强的Fisher信息矩阵"""
        FIM_total = np.zeros((2, 2))
        FIM_details = []

        for i in range(self.N):
            e_i, phi_i, e_i_squared = self.compute_ellipse_parameters(i)

            sin_phi = np.sin(phi_i)
            cos_phi = np.cos(phi_i)

            FIM_i = (4 / sigma_measurement**2) * np.array(
                [
                    [1 - e_i_squared * sin_phi**2, e_i_squared * cos_phi * sin_phi],
                    [e_i_squared * cos_phi * sin_phi, 1 - e_i_squared * cos_phi**2],
                ]
            )

            FIM_total += FIM_i

            FIM_details.append(
                {
                    "receiver_idx": i,
                    "eccentricity": e_i,
                    "e_squared": e_i_squared,
                    "phi_deg": np.rad2deg(phi_i),
                    "phi_rad": phi_i,
                    "FIM": FIM_i,
                }
            )

        return FIM_total, FIM_details

    def compute_theoretical_enhancement_factor_correct(self):
        """计算理论增强因子"""
        eccentricities = []
        phis = []
        e_squareds = []

        for i in range(self.N):
            e_i, phi_i, e_i_squared = self.compute_ellipse_parameters(i)
            eccentricities.append(e_i)
            phis.append(phi_i)
            e_squareds.append(e_i_squared)

        # 检查是否均匀分布
        phi_sorted = sorted(phis)
        phi_diffs = np.diff(phi_sorted)
        expected_diff = 2 * np.pi / self.N
        is_uniform = np.allclose(phi_diffs, expected_diff, atol=0.1)

        # 检查离心率是否相同
        e_mean = np.mean(eccentricities)
        is_same_e = np.allclose(eccentricities, e_mean, rtol=0.1)

        if is_uniform and is_same_e:
            e_squared_mean = np.mean(e_squareds)
            gamma_eccentricity = 1.0 / np.sqrt(1 - e_squared_mean / 2)
            method = "闭式解（均匀分布）"
        else:
            A = sum(1 - e_sq * np.sin(phi) ** 2 for e_sq, phi in zip(e_squareds, phis))
            B = sum(
                e_sq * np.cos(phi) * np.sin(phi) for e_sq, phi in zip(e_squareds, phis)
            )
            C = sum(1 - e_sq * np.cos(phi) ** 2 for e_sq, phi in zip(e_squareds, phis))

            gamma_squared = (self.N / 2.0) * (A + C) / (A * C - B**2)
            gamma_eccentricity = np.sqrt(gamma_squared)
            method = "数值计算（非均匀）"

        return {
            "gamma": gamma_eccentricity,
            "eccentricities": eccentricities,
            "e_squareds": e_squareds,
            "phis_deg": [np.rad2deg(p) for p in phis],
            "phis_rad": phis,
            "e_mean": np.mean(eccentricities),
            "is_uniform": is_uniform,
            "is_same_e": is_same_e,
            "method": method,
        }

    def compute_traditional_GDOP(self):
        """计算传统GDOP"""
        try:
            M_inv = np.linalg.inv(self.M)
            GDOP = np.sqrt(np.trace(M_inv))
            return GDOP
        except:
            return np.inf

    def compute_E_GDOP(self):
        """计算增强GDOP"""
        GDOP_trad = self.compute_traditional_GDOP()
        enhancement_info = self.compute_theoretical_enhancement_factor_correct()
        gamma = enhancement_info["gamma"]
        E_GDOP = GDOP_trad * gamma

        result = {
            "E_GDOP": E_GDOP,
            "GDOP_traditional": GDOP_trad,
            "gamma_eccentricity": gamma,
            "eccentricities": enhancement_info["eccentricities"],
            "e_squareds": enhancement_info["e_squareds"],
            "phis_deg": enhancement_info["phis_deg"],
            "phis_rad": enhancement_info["phis_rad"],
            "e_mean": enhancement_info["e_mean"],
            "method": enhancement_info["method"],
            "is_uniform": enhancement_info["is_uniform"],
            "is_same_e": enhancement_info["is_same_e"],
        }

        sigma_temp = 1.0
        FIM_total, FIM_details = self.compute_enhanced_FIM_correct(sigma_temp)

        try:
            FIM_inv = np.linalg.inv(FIM_total)
            eigenvalues = np.linalg.eigvalsh(FIM_inv)
            result["eigenvalues"] = eigenvalues
            result["FIM_details"] = FIM_details
        except:
            result["eigenvalues"] = None
            result["FIM_details"] = None

        return result

    def compute_theoretical_PEB_bounds(self, sigma_measurement):
        """计算理论PEB下界"""
        egdop_info = self.compute_E_GDOP()

        PEB_traditional = sigma_measurement * egdop_info["GDOP_traditional"]
        PEB_enhanced = sigma_measurement * egdop_info["E_GDOP"]

        FIM_total, FIM_details = self.compute_enhanced_FIM_correct(sigma_measurement)
        try:
            Sigma_total = np.linalg.inv(FIM_total)
            PEB_strict = np.sqrt(np.trace(Sigma_total))
            sigma_x = np.sqrt(Sigma_total[0, 0])
            sigma_y = np.sqrt(Sigma_total[1, 1])
            rho_xy = (
                Sigma_total[0, 1] / (sigma_x * sigma_y) if sigma_x * sigma_y > 0 else 0
            )
        except:
            PEB_strict = PEB_enhanced
            sigma_x = PEB_enhanced / np.sqrt(2)
            sigma_y = PEB_enhanced / np.sqrt(2)
            rho_xy = 0

        enhancement_percent = (PEB_enhanced - PEB_traditional) / PEB_traditional * 100

        return {
            "sigma_measurement": sigma_measurement,
            "PEB_traditional": PEB_traditional,
            "PEB_enhanced": PEB_enhanced,
            "PEB_strict": PEB_strict,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho_xy": rho_xy,
            "GDOP_traditional": egdop_info["GDOP_traditional"],
            "E_GDOP": egdop_info["E_GDOP"],
            "gamma_eccentricity": egdop_info["gamma_eccentricity"],
            "enhancement_percent": enhancement_percent,
            "eccentricities": egdop_info["eccentricities"],
            "e_mean": egdop_info["e_mean"],
            "method": egdop_info["method"],
            "FIM_total": FIM_total,
            "FIM_details": FIM_details,
        }


# ==================== 辅助函数 ====================
def calculate_range_measurement_std(SNR_linear, lambda_wave, T_CPI):
    """计算距离测量误差标准差"""
    sigma_d_phi = lambda_wave / (2 * np.pi * np.sqrt(2 * SNR_linear))
    sigma_d_v = lambda_wave / np.sqrt(2 * SNR_linear)
    sigma_d = np.sqrt(sigma_d_phi**2 + sigma_d_v**2)
    return sigma_d, sigma_d_phi, sigma_d_v


# ==================== 系统参数配置 ====================
print("=" * 80)
print("多基地雷达定位系统 - E-GDOP理论分析")
print("=" * 80)

# 位置参数
T = [0, 0]
R1 = [200, 0]
R2 = [100, 100]
R3 = [300, 150]
p_A = [80.03, 50.03]

# 系统参数
lambda_wave = 0.05
T_CPI = 0.01
v_target = 30

# SNR参数
SNR_dB = np.array([-10, -5, 0, 5, 10])
SNR_linear = 10 ** (SNR_dB / 10)

print(f"\n📍 系统配置:")
print(f"  发射机 T: {T}")
print(f"  接收机 R1: {R1}, R2: {R2}, R3: {R3}")
print(f"  目标位置: {p_A}")
print(f"\n⚙️  系统参数: λ={lambda_wave}m, CPI={T_CPI}s, v={v_target}m/s")
print("=" * 80)

# ==================== 创建理论计算器 ====================
theoretical_gdop = TheoreticalEllipseGDOP(
    transmitter=T,
    receivers=[R1, R2, R3],
    target=p_A,
    wavelength=lambda_wave,
    target_velocity=v_target,
)

# ==================== 计算E-GDOP ====================
egdop_result = theoretical_gdop.compute_E_GDOP()

print(f"\n📊 E-GDOP分析 ({egdop_result['method']}):")
print(f"  传统GDOP  = {egdop_result['GDOP_traditional']:.6f}")
print(f"  增强因子γ = {egdop_result['gamma_eccentricity']:.4f}")
print(f"  E-GDOP    = {egdop_result['E_GDOP']:.6f}")

# ==================== 计算不同SNR下的PEB ====================
results = []
for snr_db, snr_lin in zip(SNR_dB, SNR_linear):
    sigma_d, sigma_d_phi, sigma_d_v = calculate_range_measurement_std(
        snr_lin, lambda_wave, T_CPI
    )
    bounds = theoretical_gdop.compute_theoretical_PEB_bounds(sigma_d)
    results.append(bounds)

# 提取数据
sigma_d_values = [r["sigma_measurement"] for r in results]
PEB_trad_values = [r["PEB_traditional"] for r in results]
PEB_enhanced_values = [r["PEB_enhanced"] for r in results]
PEB_strict_values = [r["PEB_strict"] for r in results]
sigma_x_values = [r["sigma_x"] for r in results]
sigma_y_values = [r["sigma_y"] for r in results]
enhancement_values = [r["enhancement_percent"] for r in results]

# 计算sigma分量
sigma_d_phi_values = [
    calculate_range_measurement_std(10 ** (s / 10), lambda_wave, T_CPI)[1]
    for s in SNR_dB
]
sigma_d_v_values = [
    calculate_range_measurement_std(10 ** (s / 10), lambda_wave, T_CPI)[2]
    for s in SNR_dB
]

# ==================== 优化后的可视化布局 ====================
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(
    3, 4, hspace=0.35, wspace=0.35, left=0.06, right=0.98, top=0.94, bottom=0.06
)

# ========== First row: Core PEB comparison ==========

# Subplot 1: Three PEB bounds (zoomed, span 2 cols)
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.semilogy(
    SNR_dB,
    PEB_trad_values,
    "g-o",
    linewidth=3,
    markersize=10,
    label="Traditional PEB = σ_ρ × GDOP",
    markeredgewidth=2,
    markeredgecolor="darkgreen",
)
ax1.semilogy(
    SNR_dB,
    PEB_enhanced_values,
    "b-^",
    linewidth=4,
    markersize=12,
    label="Enhanced PEB = σ_ρ × E-GDOP",
    markeredgewidth=2.5,
    markeredgecolor="darkblue",
)
ax1.set_xlabel("SNR (dB)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Position error bound PEB (m)", fontsize=13, fontweight="bold")
ax1.set_title(
    "Complete theoretical bound: PEB = σ_ρ × E-GDOP", fontsize=14, fontweight="bold"
)
ax1.legend(fontsize=11, framealpha=0.95, loc="upper right")
ax1.grid(True, alpha=0.3, linestyle="--")

# Subplot 2: Measurement noise decomposition
ax2 = fig.add_subplot(gs[0, 2])
ax2.semilogy(
    SNR_dB,
    sigma_d_values,
    "k-o",
    linewidth=3,
    markersize=9,
    label="σ_ρ (total)",
    markeredgewidth=2,
    markeredgecolor="black",
)
ax2.semilogy(
    SNR_dB,
    sigma_d_phi_values,
    "b--^",
    linewidth=2,
    markersize=7,
    label="σ_φ (phase)",
    alpha=0.7,
)
ax2.semilogy(
    SNR_dB,
    sigma_d_v_values,
    "r--s",
    linewidth=2,
    markersize=7,
    label="σ_v (velocity)",
    alpha=0.7,
)
ax2.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Standard deviation (m)", fontsize=12, fontweight="bold")
ax2.set_title("Component 1: Measurement noise σ_ρ", fontsize=13, fontweight="bold")
ax2.legend(fontsize=9.5, framealpha=0.9, loc="upper right")
ax2.grid(True, alpha=0.3, linestyle="--")

# Subplot 3: GDOP comparison (bar)
ax3 = fig.add_subplot(gs[0, 3])
x_pos = [0, 1]
gdop_values = [egdop_result["GDOP_traditional"], egdop_result["E_GDOP"]]
colors = ["green", "blue"]
bars = ax3.bar(
    x_pos,
    gdop_values,
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=2.5,
    width=0.5,
)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(
    ["GDOP\n(Traditional)", "E-GDOP\n(Enhanced)"], fontsize=11, fontweight="bold"
)
ax3.set_ylabel("GDOP value", fontsize=12, fontweight="bold")
ax3.set_title("Components 2 & 3: GDOP × γ", fontsize=13, fontweight="bold")
ax3.grid(True, alpha=0.3, axis="y", linestyle="--")

for bar, val in zip(bars, gdop_values):
    height = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        height * 1.05,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# 添加增强因子箭头
y_mid = (gdop_values[0] + gdop_values[1]) / 2
ax3.annotate(
    "",
    xy=(1, gdop_values[1] * 0.95),
    xytext=(0, gdop_values[0] * 1.05),
    arrowprops=dict(arrowstyle="->", lw=2.5, color="red"),
)
ax3.text(
    0.5,
    y_mid,
    f'×{egdop_result["gamma_eccentricity"]:.3f}',
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="red",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7, pad=0.5),
)

# ========== Second row: Ellipse parameter analysis ==========

# Subplot 4: Directional positioning error
ax4 = fig.add_subplot(gs[1, 0])
ax4.semilogy(
    SNR_dB, sigma_x_values, "b-o", linewidth=2.5, markersize=8, label="σ_x (x-axis)"
)
ax4.semilogy(
    SNR_dB, sigma_y_values, "r-s", linewidth=2.5, markersize=8, label="σ_y (y-axis)"
)
ax4.semilogy(SNR_dB, PEB_enhanced_values, "k--", linewidth=2, alpha=0.5, label="PEB")
ax4.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Standard deviation (m)", fontsize=12, fontweight="bold")
ax4.set_title("Directional positioning accuracy", fontsize=13, fontweight="bold")
ax4.legend(fontsize=10, framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle="--")

# Subplot 5: Eccentricity distribution
ax5 = fig.add_subplot(gs[1, 1])
ellipse_indices = range(1, len(egdop_result["eccentricities"]) + 1)
bars = ax5.bar(
    ellipse_indices,
    egdop_result["eccentricities"],
    color="orange",
    alpha=0.7,
    edgecolor="black",
    linewidth=2,
)
ax5.axhline(
    y=egdop_result["e_mean"],
    color="red",
    linestyle="--",
    linewidth=2.5,
    label=f'Mean: {egdop_result["e_mean"]:.3f}',
)
ax5.set_xlabel("Ellipse index", fontsize=12, fontweight="bold")
ax5.set_ylabel("Eccentricity e", fontsize=12, fontweight="bold")
ax5.set_title("Eccentricity distribution", fontsize=13, fontweight="bold")
ax5.set_ylim([0, 1])
ax5.legend(fontsize=10, framealpha=0.9)
ax5.grid(True, alpha=0.3, axis="y", linestyle="--")
for bar, val in zip(bars, egdop_result["eccentricities"]):
    height = bar.get_height()
    ax5.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Subplot 6: Principal axis directions (polar)
ax6 = fig.add_subplot(gs[1, 2], projection="polar")
theta = np.array(egdop_result["phis_rad"])
r = np.ones(len(theta))
colors_polar = plt.cm.hsv(np.linspace(0, 0.8, len(theta)))
bars_polar = ax6.bar(
    theta,
    r,
    width=0.4,
    bottom=0.0,
    color=colors_polar,
    alpha=0.75,
    edgecolor="black",
    linewidth=2,
)
ax6.set_theta_zero_location("E")
ax6.set_theta_direction(1)
ax6.set_title("Principal axis directions", fontsize=13, fontweight="bold", pad=20)
ax6.set_ylim([0, 1.2])

for t, phi_deg in zip(theta, egdop_result["phis_deg"]):
    ax6.text(
        t,
        1.1,
        f"{phi_deg:.0f}°",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
    )

# Subplot 7: Improvement percentage
ax7 = fig.add_subplot(gs[1, 3])
ax7.plot(
    SNR_dB,
    enhancement_values,
    "b-o",
    linewidth=3,
    markersize=11,
    markeredgewidth=2,
    markeredgecolor="darkblue",
)
ax7.axhline(y=0, color="k", linestyle="--", linewidth=1.5, alpha=0.5)
ax7.fill_between(SNR_dB, 0, enhancement_values, alpha=0.25, color="blue")
ax7.set_xlabel("SNR (dB)", fontsize=12, fontweight="bold")
ax7.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
ax7.set_title("E-GDOP improvement magnitude", fontsize=13, fontweight="bold")
ax7.grid(True, alpha=0.3, linestyle="--")

avg_enhancement = np.mean(enhancement_values)
ax7.axhline(
    y=avg_enhancement,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Mean: {avg_enhancement:.1f}%",
)
ax7.legend(fontsize=10, framealpha=0.9)

# ========== Third row: Theory curves and formulas ==========

# Subplot 8: Enhancement factor curve (zoomed)
ax8 = fig.add_subplot(gs[2, 0:2])
e_range = np.linspace(0, 0.99, 200)
gamma_range = 1.0 / np.sqrt(1 - e_range**2 / 2)

ax8.plot(e_range, gamma_range, "b-", linewidth=4, label="Theory: γ = 1/√(1-e²/2)")
ax8.scatter(
    egdop_result["eccentricities"],
    [egdop_result["gamma_eccentricity"]] * len(egdop_result["eccentricities"]),
    color="red",
    s=250,
    marker="*",
    zorder=5,
    label=f'Current: γ={egdop_result["gamma_eccentricity"]:.3f}',
    edgecolors="darkred",
    linewidth=2.5,
)
ax8.axhline(
    y=1, color="k", linestyle="--", linewidth=1.5, alpha=0.5, label="Circle (γ=1)"
)
ax8.axhline(
    y=np.sqrt(2),
    color="gray",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"Limit √2≈{np.sqrt(2):.3f}",
)
ax8.set_xlabel("Eccentricity e", fontsize=13, fontweight="bold")
ax8.set_ylabel("Enhancement factor γ", fontsize=13, fontweight="bold")
ax8.set_title("Theoretical enhancement factor", fontsize=14, fontweight="bold")
ax8.legend(fontsize=11, framealpha=0.9, loc="upper left")
ax8.grid(True, alpha=0.3, linestyle="--")
ax8.set_xlim([0, 1])
ax8.set_ylim([0.95, 1.45])

# Subplot 9: Fisher Information Matrix
ax9 = fig.add_subplot(gs[2, 2])
mid_idx = len(results) // 2
FIM_display = results[mid_idx]["FIM_total"]
im = ax9.imshow(FIM_display, cmap="YlOrRd", aspect="auto")
ax9.set_xticks([0, 1])
ax9.set_yticks([0, 1])
ax9.set_xticklabels(["x", "y"], fontsize=12, fontweight="bold")
ax9.set_yticklabels(["x", "y"], fontsize=12, fontweight="bold")
ax9.set_title(f"FIM matrix (SNR={SNR_dB[mid_idx]}dB)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax9, shrink=0.8)

for i in range(2):
    for j in range(2):
        text = ax9.text(
            j,
            i,
            f"{FIM_display[i, j]:.1e}",
            ha="center",
            va="center",
            color="black",
            fontsize=10,
            fontweight="bold",
        )

# Subplot 10: Formula summary
ax10 = fig.add_subplot(gs[2, 3])
ax10.axis("off")

formula_text = f"""
Complete theoretical bound
━━━━━━━━━━━━━━━
PEB = σ_ρ × E-GDOP

Where:
E-GDOP = GDOP × γ

━━━━━━━━━━━━━━━
Component 1: Measurement noise
σ_ρ = √(σ_φ² + σ_v²)

Component 2: Traditional GDOP
GDOP = √(tr(M⁻¹))

Component 3: Enhancement factor
γ = 1/√(1 - e²/2)

━━━━━━━━━━━━━━━
Current system:
GDOP = {egdop_result['GDOP_traditional']:.4f}
γ    = {egdop_result['gamma_eccentricity']:.3f}
E-GDOP = {egdop_result['E_GDOP']:.4f}
"""

ax10.text(
    0.1,
    0.5,
    formula_text,
    fontsize=10,
    verticalalignment="center",
    fontfamily="monospace",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3, pad=0.8),
)

plt.suptitle(
    "Multistatic radar E-GDOP theory - PEB = σ_ρ × E-GDOP",
    fontsize=17,
    fontweight="bold",
    y=0.98,
)

plt.savefig("E_GDOP_Optimized_Layout.png", dpi=300, bbox_inches="tight")
print(f"\n✅ 优化布局图表已保存")
plt.show()

# ==================== 打印详细结果 ====================
print(f"\n{'='*80}")
print("理论下界详细结果")
print("=" * 80)

print(f"\n{'SNR':>6} {'σ_ρ':>10} {'PEB_trad':>12} {'PEB_enh':>12} {'增强':>8}")
print(f"{'(dB)':>6} {'(m)':>10} {'(m)':>12} {'(m)':>12} {'(%)':>8}")
print("-" * 60)
for snr_db, r in zip(SNR_dB, results):
    print(
        f"{snr_db:>6} {r['sigma_measurement']:>10.6f} "
        f"{r['PEB_traditional']:>12.6f} {r['PEB_enhanced']:>12.6f} "
        f"{r['enhancement_percent']:>8.1f}"
    )

print(f"\n📊 E-GDOP components:")
print(f"  GDOP (traditional)  = {egdop_result['GDOP_traditional']:.6f}")
print(f"  γ (enhancement)     = {egdop_result['gamma_eccentricity']:.4f}")
print(f"  E-GDOP              = {egdop_result['E_GDOP']:.6f}")
print(f"  mean eccentricity   = {egdop_result['e_mean']:.4f}")

print("\n" + "=" * 80)
