import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as EllipsePatch
from scipy.optimize import fsolve
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ======================== 系统参数配置 ========================
class SystemConfig:
    """系统配置参数"""
    def __init__(self):
        # 基站位置
        self.T = np.array([0, 0])      # 发射机
        self.R1 = np.array([200, 0])   # 接收机1
        self.R2 = np.array([100, 100]) # 接收机2
        self.R3 = np.array([300, 150]) # 接收机3
        self.receivers = [self.R1, self.R2, self.R3]
        
        # 射频参数
        self.lambda_wave = 0.05  # 波长 (m)
        self.f_c = 3e8 / self.lambda_wave  # 载频 (Hz)
        self.T_CPI = 0.01  # 相干处理时间 (s)
        
        # 噪声参数
        self.SNR_dB = 5
        self.SNR_linear = 10 ** (self.SNR_dB / 10)
        
        # 计算测量误差标准差
 
        print("=" * 90)
        print("系统配置参数")
        print("=" * 90)
        print(f"发射机位置:          {self.T}")
        print(f"接收机位置:          R1={self.R1}, R2={self.R2}, R3={self.R3}")
        print(f"载频:                {self.f_c/1e9:.2f} GHz")
        print(f"波长:                {self.lambda_wave*1000:.1f} mm")
        print(f"相干处理时间:        {self.T_CPI*1000:.1f} ms")
        print(f"SNR:                 {self.SNR_dB} dB")
        print(f"距离测量标准差 σ_ρ:  {self.sigma_d:.6f} m")
        print(f"  ├─ 相位贡献 σ_d_φ: {self.sigma_d_phi:.6f} m")
        print(f"  └─ 速度贡献 σ_d_v: {self.sigma_d_v:.6f} m")
        print("=" * 90)


# ======================== CRLB理论计算器（完整版）========================
class EllipticCRLB:
    """
    椭圆定位的CRLB计算器（完整理论版本）
    
    核心公式：
    PEB = σ_ρ × √(trace(FIM^(-1)))
    
    其中 FIM = 雅可比矩阵^T @ 协方差矩阵^(-1) @ 雅可比矩阵
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.T = config.T
        self.receivers = config.receivers
        self.N = len(self.receivers)
        
    def evaluate_at_state(self, state):
        """
        在特定状态点评估CRLB
        
        参数:
            state: [x_A, y_A, v_x, v_y] 目标状态（位置+速度）
            
        注意：这不是"先验估计"，而是"评估点"
              - 场景A：想分析某条航迹的定位性能 → 代入该航迹参数
              - 场景B：不知道目标在哪 → 扫描整个区域
              - 场景C：验证算法性能 → 用真实值或估计值
        """
        x_A, y_A, v_x, v_y = state
        p_A = np.array([x_A, y_A])
        v_A = np.array([v_x, v_y])
        
        # 计算完整的Fisher信息矩阵
        FIM = self._compute_fisher_information_matrix(p_A, v_A)
        
        try:
            # CRLB = FIM^(-1)
            CRLB_matrix = np.linalg.inv(FIM)
            
            # 位置误差下界（PEB）
            PEB = np.sqrt(CRLB_matrix[0, 0] + CRLB_matrix[1, 1])
            
            # 速度误差下界（VEB）
            VEB = np.sqrt(CRLB_matrix[2, 2] + CRLB_matrix[3, 3])
            
            # GDOP分解
            GDOP_position = PEB / self.config.sigma_d
            GDOP_velocity = VEB / self.config.sigma_d
            
            return {
                'PEB': PEB,
                'VEB': VEB,
                'GDOP_position': GDOP_position,
                'GDOP_velocity': GDOP_velocity,
                'CRLB_matrix': CRLB_matrix,
                'FIM': FIM,
                'valid': True
            }
        except np.linalg.LinAlgError:
            return {
                'PEB': np.inf,
                'VEB': np.inf,
                'GDOP_position': np.inf,
                'GDOP_velocity': np.inf,
                'valid': False
            }
    
    def _compute_fisher_information_matrix(self, p_A, v_A):
        """
        计算Fisher信息矩阵（完整版：椭圆距离 + 多普勒）
        
        状态向量: θ = [x_A, y_A, v_x, v_y]
        
        测量向量: z = [ρ_1, ..., ρ_N, f_d1, ..., f_dN]
        
        FIM = G^T @ R^(-1) @ G
        """
        # 构建雅可比矩阵 G (2N × 4)
        G = np.zeros((2 * self.N, 4))
        
        for i, receiver in enumerate(self.receivers):
            # 距离计算
            R_T = np.linalg.norm(p_A - self.T)
            R_R = np.linalg.norm(p_A - receiver)
            
            # 单位方向向量
            u_T = (p_A - self.T) / R_T if R_T > 1e-10 else np.zeros(2)
            u_R = (p_A - receiver) / R_R if R_R > 1e-10 else np.zeros(2)
            
            # ===== 椭圆距离 ρ 对位置的偏导 =====
            # ∂ρ_i/∂x = (x-x_T)/R_T + (x-x_R)/R_R
            # ∂ρ_i/∂y = (y-y_T)/R_T + (y-y_R)/R_R
            dRho_dp = u_T + u_R  # [∂ρ/∂x, ∂ρ/∂y]
            
            # ===== 椭圆距离 ρ 对速度的偏导 =====
            # 在单个时刻，ρ不依赖于速度
            dRho_dv = np.zeros(2)
            
            # 填入雅可比矩阵（椭圆距离行）
            G[i, 0:2] = dRho_dp
            G[i, 2:4] = dRho_dv
            
            # ===== 多普勒频率 f_d 对位置的偏导 =====
            # f_d = (v·u_T + v·u_R) / λ
            # ∂f_d/∂p = ∂(u_T + u_R)/∂p · v / λ
            
            # ∂u_T/∂p = (I - u_T⊗u_T) / R_T
            # ∂u_R/∂p = (I - u_R⊗u_R) / R_R
            I = np.eye(2)
            dU_T_dp = (I - np.outer(u_T, u_T)) / R_T if R_T > 1e-10 else np.zeros((2, 2))
            dU_R_dp = (I - np.outer(u_R, u_R)) / R_R if R_R > 1e-10 else np.zeros((2, 2))
            
            dFd_dp = ((dU_T_dp + dU_R_dp) @ v_A) / self.config.lambda_wave
            
            # ===== 多普勒频率 f_d 对速度的偏导 =====
            # f_d = (v·u_T + v·u_R) / λ
            # ∂f_d/∂v = (u_T + u_R) / λ
            dFd_dv = (u_T + u_R) / self.config.lambda_wave
            
            # 填入雅可比矩阵（多普勒行）
            G[self.N + i, 0:2] = dFd_dp
            G[self.N + i, 2:4] = dFd_dv
        
        # 构建协方差矩阵 R (2N × 2N)
        # 假设测量独立，R为对角矩阵
        R = np.zeros((2 * self.N, 2 * self.N))
        
        # 椭圆距离测量方差
        sigma_rho_squared = self.config.sigma_d ** 2
        
        # 多普勒频率测量方差
        # σ_fd = 1 / (2π × T_CPI × √(2×SNR))
        sigma_fd = 1 / (2 * np.pi * self.config.T_CPI * np.sqrt(2 * self.config.SNR_linear))
        sigma_fd_squared = sigma_fd ** 2
        
        # 填充协方差矩阵
        for i in range(self.N):
            R[i, i] = sigma_rho_squared          # 椭圆距离
            R[self.N + i, self.N + i] = sigma_fd_squared  # 多普勒
        
        # Fisher信息矩阵
        R_inv = np.linalg.inv(R)
        FIM = G.T @ R_inv @ G
        
        return FIM
    
    def scan_region(self, x_range, y_range, v_x_range, v_y_range, grid_size=20):
        """
        场景B：扫描整个区域的定位性能
        
        用于回答："在所有可能的位置和速度下，定位性能如何分布？"
        """
        print("\n" + "=" * 90)
        print("场景B：区域扫描模式")
        print("=" * 90)
        print(f"位置范围: x ∈ [{x_range[0]}, {x_range[1]}], y ∈ [{y_range[0]}, {y_range[1]}]")
        print(f"速度范围: v_x ∈ [{v_x_range[0]}, {v_x_range[1]}], v_y ∈ [{v_y_range[0]}, {v_y_range[1]}]")
        print(f"网格尺寸: {grid_size} × {grid_size}")
        
        x_vals = np.linspace(x_range[0], x_range[1], grid_size)
        y_vals = np.linspace(y_range[0], y_range[1], grid_size)
        
        # 固定速度取中间值
        v_x_mid = (v_x_range[0] + v_x_range[1]) / 2
        v_y_mid = (v_y_range[0] + v_y_range[1]) / 2
        
        PEB_map = np.zeros((grid_size, grid_size))
        GDOP_map = np.zeros((grid_size, grid_size))
        
        print(f"正在扫描... (固定速度 v=[{v_x_mid:.1f}, {v_y_mid:.1f}] m/s)")
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                state = [x, y, v_x_mid, v_y_mid]
                result = self.evaluate_at_state(state)
                PEB_map[j, i] = result['PEB']
                GDOP_map[j, i] = result['GDOP_position']
        
        print(f"✓ 扫描完成")
        print(f"  最小PEB: {np.min(PEB_map[np.isfinite(PEB_map)]):.4f} m")
        print(f"  最大PEB: {np.max(PEB_map[np.isfinite(PEB_map)]):.4f} m")
        print(f"  平均PEB: {np.mean(PEB_map[np.isfinite(PEB_map)]):.4f} m")
        
        return {
            'x_vals': x_vals,
            'y_vals': y_vals,
            'PEB_map': PEB_map,
            'GDOP_map': GDOP_map
        }
    
    def monte_carlo_analysis(self, state_distribution, n_samples=1000):
        """
        场景C：蒙特卡洛分析（完全未知情况）
        
        用于回答："如果目标可能在任何位置以任何速度运动，
                  定位性能的统计分布是什么？"
        """
        print("\n" + "=" * 90)
        print("场景C：蒙特卡洛分析模式")
        print("=" * 90)
        print(f"样本数量: {n_samples}")
        print(f"状态分布: {state_distribution}")
        
        PEB_samples = []
        GDOP_samples = []
        
        for i in range(n_samples):
            # 根据分布随机采样状态
            if state_distribution == 'uniform':
                x_A = np.random.uniform(0, 300)
                y_A = np.random.uniform(0, 200)
                v_x = np.random.uniform(-30, 30)
                v_y = np.random.uniform(-30, 30)
            else:
                raise ValueError(f"未知分布: {state_distribution}")
            
            state = [x_A, y_A, v_x, v_y]
            result = self.evaluate_at_state(state)
            
            if result['valid']:
                PEB_samples.append(result['PEB'])
                GDOP_samples.append(result['GDOP_position'])
            
            if (i + 1) % 200 == 0:
                print(f"  完成 {i+1}/{n_samples}...")
        
        PEB_samples = np.array(PEB_samples)
        GDOP_samples = np.array(GDOP_samples)
        
        print(f"✓ 分析完成")
        print(f"  平均PEB:    {np.mean(PEB_samples):.4f} m")
        print(f"  中位数PEB:  {np.median(PEB_samples):.4f} m")
        print(f"  90%分位数:  {np.percentile(PEB_samples, 90):.4f} m")
        print(f"  最大PEB:    {np.max(PEB_samples):.4f} m")
        
        return {
            'PEB_samples': PEB_samples,
            'GDOP_samples': GDOP_samples,
            'mean_PEB': np.mean(PEB_samples),
            'median_PEB': np.median(PEB_samples),
            'percentile_90': np.percentile(PEB_samples, 90),
            'max_PEB': np.max(PEB_samples)
        }


# ======================== 蒙特卡洛验证器 ========================
class MonteCarloValidator:
    """蒙特卡洛模拟验证CRLB"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def calculate_bistatic_range(self, point, transmitter, receiver):
        R_T = np.linalg.norm(point - transmitter)
        R_R = np.linalg.norm(point - receiver)
        return R_T + R_R
    
    def calculate_doppler_shift(self, p1, p2, transmitter, receiver):
        R1_T = np.linalg.norm(p1 - transmitter)
        R1_R = np.linalg.norm(p1 - receiver)
        R2_T = np.linalg.norm(p2 - transmitter)
        R2_R = np.linalg.norm(p2 - receiver)
        delta_R = (R2_T + R2_R) - (R1_T + R1_R)
        return delta_R / (self.config.lambda_wave * self.config.T_CPI)
    
    def find_ellipse_intersections(self, br1, br2, T, R1, R2):
        """求两个椭圆的交点"""
        def equations(p):
            x, y = p
            point = np.array([x, y])
            eq1 = self.calculate_bistatic_range(point, T, R1) - br1
            eq2 = self.calculate_bistatic_range(point, T, R2) - br2
            return [eq1, eq2]
        
        intersections = []
        for x0 in np.linspace(0, 300, 5):
            for y0 in np.linspace(0, 200, 5):
                try:
                    sol = fsolve(equations, [x0, y0], full_output=True)
                    if sol[2] == 1:
                        point = sol[0]
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
    
    def run_simulation(self, p_A, p_B_true, n_trials=2000):
        """
        运行蒙特卡洛模拟
        
        这里的p_A和p_B_true是"真实场景"，用于生成模拟数据
        但在实际定位中，我们不知道真实值，需要估计
        """
        print("\n" + "=" * 90)
        print("蒙特卡洛模拟验证")
        print("=" * 90)
        print(f"目标A位置: {p_A}")
        print(f"目标B真实位置: {p_B_true}")
        print(f"真实位移: {np.linalg.norm(p_B_true - p_A):.6f} m")
        print(f"模拟次数: {n_trials}")
        
        # A点的真实参数
        bistatic_ranges_A = [
            self.calculate_bistatic_range(p_A, self.config.T, R)
            for R in self.config.receivers
        ]
        
        positioning_errors = []
        vertex_errors = []
        
        np.random.seed(42)
        
        for trial in range(n_trials):
            # B点的真实参数
            bistatic_ranges_B_true = [
                self.calculate_bistatic_range(p_B_true, self.config.T, R)
                for R in self.config.receivers
            ]
            dopplers_B_true = [
                self.calculate_doppler_shift(p_A, p_B_true, self.config.T, R)
                for R in self.config.receivers
            ]
            
            # 添加测量噪声
            sigma_fd = 1 / (2 * np.pi * self.config.T_CPI * np.sqrt(2 * self.config.SNR_linear))
            bistatic_ranges_B_measured = []
            
            for i, R in enumerate(self.config.receivers):
                # 多普勒噪声
                fd_noise = np.random.normal(0, sigma_fd)
                fd_measured = dopplers_B_true[i] + fd_noise
                
                # 从多普勒反推距离
                delta_R_measured = fd_measured * self.config.lambda_wave * self.config.T_CPI
                br_measured = bistatic_ranges_A[i] + delta_R_measured
                bistatic_ranges_B_measured.append(br_measured)
            
            # 求椭圆交点
            intersections_12 = self.find_ellipse_intersections(
                bistatic_ranges_B_measured[0], bistatic_ranges_B_measured[1],
                self.config.T, self.config.R1, self.config.R2
            )
            intersections_23 = self.find_ellipse_intersections(
                bistatic_ranges_B_measured[1], bistatic_ranges_B_measured[2],
                self.config.T, self.config.R2, self.config.R3
            )
            intersections_13 = self.find_ellipse_intersections(
                bistatic_ranges_B_measured[0], bistatic_ranges_B_measured[2],
                self.config.T, self.config.R1, self.config.R3
            )
            
            # 选择最接近真实B点的交点
            triangle_vertices = []
            for intersections in [intersections_12, intersections_23, intersections_13]:
                if len(intersections) > 0:
                    closest = min(intersections, key=lambda p: np.linalg.norm(p - p_B_true))
                    triangle_vertices.append(closest)
            
            # 统计
            if len(triangle_vertices) == 3:
                for v in triangle_vertices:
                    vertex_errors.append(np.linalg.norm(v - p_B_true))
                
                center = np.mean(triangle_vertices, axis=0)
                positioning_errors.append(np.linalg.norm(center - p_B_true))
            
            if (trial + 1) % 400 == 0:
                print(f"  完成 {trial+1}/{n_trials}...")
        
        # 统计分析
        rms_positioning_error = np.sqrt(np.mean(np.array(positioning_errors) ** 2))
        mean_positioning_error = np.mean(positioning_errors)
        median_positioning_error = np.median(positioning_errors)
        
        print(f"✓ 模拟完成")
        print(f"  RMS定位误差: {rms_positioning_error:.6f} m")
        print(f"  平均误差:     {mean_positioning_error:.6f} m")
        print(f"  中位数误差:   {median_positioning_error:.6f} m")
        
        return {
            'positioning_errors': positioning_errors,
            'vertex_errors': vertex_errors,
            'rms_error': rms_positioning_error,
            'mean_error': mean_positioning_error,
            'median_error': median_positioning_error
        }


# ======================== 主程序 ========================
def main():
    # 初始化系统配置
    config = SystemConfig()
    
    # 创建CRLB计算器
    crlb_calculator = EllipticCRLB(config)
    
    # ======================== 场景A：单点评估 ========================
    print("\n" + "=" * 90)
    print("场景A：单点评估模式")
    print("=" * 90)
    print("评估特定目标状态下的定位性能")
    print("（这不是'先验估计'，而是'我想分析这个场景'）")
    
    # 评估点：目标B的真实状态
    p_A = np.array([40, 30])
    p_B_true = np.array([40.05, 30.05])
    v_A = (p_B_true - p_A) / config.T_CPI  # 从位移反推速度
    
    state_B = [p_B_true[0], p_B_true[1], v_A[0], v_A[1]]
    
    print(f"\n📍 评估点状态:")
    print(f"  位置: ({state_B[0]:.3f}, {state_B[1]:.3f}) m")
    print(f"  速度: ({state_B[2]:.3f}, {state_B[3]:.3f}) m/s")
    print(f"  速率: {np.linalg.norm(v_A):.3f} m/s")
    
    result_B = crlb_calculator.evaluate_at_state(state_B)
    
    print(f"\n🎯 CRLB理论下界:")
    print(f"  PEB (位置误差下界): {result_B['PEB']:.6f} m")
    print(f"  VEB (速度误差下界): {result_B['VEB']:.6f} m/s")
    print(f"  GDOP (位置):         {result_B['GDOP_position']:.6f}")
    print(f"  GDOP (速度):         {result_B['GDOP_velocity']:.6f}")
    
    print(f"\n📐 理论公式:")
    print(f"  PEB = σ_ρ × GDOP_position")
    print(f"      = {config.sigma_d:.6f} × {result_B['GDOP_position']:.6f}")
    print(f"      = {result_B['PEB']:.6f} m")
    
    # ======================== 场景B：区域扫描 ========================
    scan_result = crlb_calculator.scan_region(
        x_range=[0, 300],
        y_range=[0, 200],
        v_x_range=[-30, 30],
        v_y_range=[-30, 30],
        grid_size=30
    )
    
    # ======================== 场景C：蒙特卡洛分析 ========================
    mc_result = crlb_calculator.monte_carlo_analysis(
        state_distribution='uniform',
        n_samples=1000
    )
    
    # ======================== 蒙特卡洛验证 ========================
    validator = MonteCarloValidator(config)
    validation_result = validator.run_simulation(p_A, p_B_true, n_trials=2000)
    
    # ======================== 对比分析 ========================
    print("\n" + "=" * 90)
    print("理论预测 vs 实际统计")
    print("=" * 90)
    
    theoretical_PEB = result_B['PEB']
    actual_RMS = validation_result['rms_error']
    ratio = actual_RMS / theoretical_PEB
    
    print(f"\n📊 对比结果:")
    print(f"  理论PEB (CRLB):      {theoretical_PEB:.6f} m")
    print(f"  实际RMS误差:         {actual_RMS:.6f} m")
    print(f"  验证比值:            {ratio:.3f}×")
    
    print(f"\n💡 验证结论:")
    if 0.9 <= ratio <= 1.3:
        print(f"  ✅ 验证成功！实际误差与理论下界高度吻合")
        print(f"     CRLB提供了准确的性能下界预测")
    elif ratio < 0.9:
        print(f"  ⚠️  实际误差小于理论下界 - 不符合理论")
        print(f"     可能原因：")
        print(f"     1. 统计样本不足")
        print(f"     2. 估计器达到了超效率（理论上不可能）")
        print(f"     3. CRLB计算有误")
    elif ratio > 1.3:
        print(f"  ⚠️  实际误差显著大于理论下界")
        print(f"     可能原因：")
        print(f"     1. 估计器不是有效的（未达到CRLB）")
        print(f"     2. 存在未建模的误差源")
        print(f"     3. 数值求解不稳定")
    
    # ======================== 可视化 ========================
    visualize_results(config, crlb_calculator, scan_result, mc_result, 
                     validation_result, result_B, p_A, p_B_true)


def visualize_results(config, crlb_calculator, scan_result, mc_result, 
                     validation_result, result_B, p_A, p_B_true):
    """综合可视化"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    # 子图1: PEB热力图
    ax1 = fig.add_subplot(gs[0, 0])
    X, Y = np.meshgrid(scan_result['x_vals'], scan_result['y_vals'])
    PEB_map = scan_result['PEB_map']
    PEB_map_plot = np.clip(PEB_map, 0, np.percentile(PEB_map[np.isfinite(PEB_map)], 95))
    
    im1 = ax1.contourf(X, Y, PEB_map_plot, levels=20, cmap='RdYlGn_r')
    plt.colorbar(im1, ax=ax1, label='PEB (m)')
    
    # 绘制基站
    ax1.plot(*config.T, 'r^', markersize=15, label='发射机', markeredgecolor='black', markeredgewidth=2)
    for i, R in enumerate(config.receivers):
        ax1.plot(*R, 'bs', markersize=12, label=f'接收机{i+1}' if i == 0 else '', 
                markeredgecolor='black', markeredgewidth=1.5)
    
    # 标记评估点
    ax1.plot(*p_B_true, 'k*', markersize=20, label='评估点B', markeredgecolor='yellow', markeredgewidth=2)
    
    ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax1.set_title('PEB热力图（区域扫描）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 子图2: GDOP热力图
    ax2 = fig.add_subplot(gs[0, 1])
    GDOP_map = scan_result['GDOP_map']
    GDOP_map_plot = np.clip(GDOP_map, 0, np.percentile(GDOP_map[np.isfinite(GDOP_map)], 95))
    
    im2 = ax2.contourf(X, Y, GDOP_map_plot, levels=20, cmap='viridis')
    plt.colorbar(im2, ax=ax2, label='GDOP')
    
    ax2.plot(*config.T, 'r^', markersize=15, markeredgecolor='black', markeredgewidth=2)
    for R in config.receivers:
        ax2.plot(*R, 'bs', markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(*p_B_true, 'k*', markersize=20, markeredgecolor='yellow', markeredgewidth=2)
    
    ax2.set_xlabel('X (m)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
    ax2.set_title('GDOP热力图', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 子图3: 蒙特卡洛PEB分布
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mc_result['PEB_samples'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(mc_result['mean_PEB'], color='red', linestyle='--', linewidth=2.5, 
               label=f"均值={mc_result['mean_PEB']:.3f}m")
    ax3.axvline(mc_result['median_PEB'], color='green', linestyle=':', linewidth=2.5,
               label=f"中位数={mc_result['median_PEB']:.3f}m")
    ax3.axvline(mc_result['percentile_90'], color='orange', linestyle='-.', linewidth=2.5,
               label=f"90%={mc_result['percentile_90']:.3f}m")
    
    ax3.set_xlabel('PEB (m)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('频次', fontsize=12, fontweight='bold')
    ax3.set_title('PEB统计分布（随机采样）', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 验证误差分布
    ax4 = fig.add_subplot(gs[1, 0])
    positioning_errors = validation_result['positioning_errors']
    ax4.hist(positioning_errors, bins=60, color='lightcoral', edgecolor='black', alpha=0.7)
    
    theoretical_PEB = result_B['PEB']
    actual_RMS = validation_result['rms_error']
    
    ax4.axvline(actual_RMS, color='darkred', linestyle='-', linewidth=3,
               label=f"实际RMS={actual_RMS:.4f}m")
    ax4.axvline(theoretical_PEB, color='magenta', linestyle='-.', linewidth=3,
               label=f"理论PEB={theoretical_PEB:.4f}m")
    
    ax4.set_xlabel('定位误差 (m)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('频次', fontsize=12, fontweight='bold')
    ax4.set_title('定位误差分布 vs 理论下界', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 子图5: CDF对比
    ax5 = fig.add_subplot(gs[1, 1])
    sorted_errors = np.sort(positioning_errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    ax5.plot(sorted_errors, cdf, linewidth=3, color='darkred', label='实际CDF')
    ax5.axvline(theoretical_PEB, color='magenta', linestyle='-.', linewidth=3, 
               label=f'理论PEB', alpha=0.9)
    ax5.axvline(actual_RMS, color='orange', linestyle='--', linewidth=2.5, label=f'RMS误差')
    
    percentiles = [50, 68, 90, 95]
    for p in percentiles:
        val = np.percentile(positioning_errors, p)
        ax5.axhline(y=p/100, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax5.plot(val, p/100, 'ro', markersize=6)
        ax5.text(val*1.02, p/100, f'{p}%', fontsize=9, va='center')
    
    ax5.set_xlabel('定位误差 (m)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('累积概率', fontsize=12, fontweight='bold')
    ax5.set_title('累积分布函数', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 理论对比
    ax6 = fig.add_subplot(gs[1, 2])
    categories = ['理论PEB\n(CRLB)', '实际RMS\n(蒙特卡洛)']
    values = [theoretical_PEB, actual_RMS]
    colors_bar = ['magenta', 'darkred']
    bars = ax6.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2.5)
    
    ax6.set_ylabel('定位误差 (m)', fontsize=12, fontweight='bold')
    ax6.set_title('理论 vs 实际', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2.0, height * 1.05,
                f"{val:.4f}m", ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    ratio = actual_RMS / theoretical_PEB
    ratio_text = f"验证比值:\n{ratio:.3f}×"
    status = '✅' if 0.9 <= ratio <= 1.3 else '⚠️'
    ax6.text(0.98, 0.97, f"{status}\n{ratio_text}", transform=ax6.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', 
                     facecolor='lightgreen' if 0.9 <= ratio <= 1.3 else 'yellow', 
                     alpha=0.7))
    
    plt.suptitle(f'椭圆定位CRLB理论分析与验证 (SNR={config.SNR_dB}dB)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('ellipse_positioning_CRLB_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n✅ 综合分析图已保存为 'ellipse_positioning_CRLB_comprehensive.png'")
    plt.show()


if __name__ == "__main__":
    main()