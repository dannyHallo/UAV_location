import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class BistaticDifferentialCRLB:
    def __init__(self, SNR_dB=20):
        # Station positions (unit: m)
        self.T = np.array([0, 0])  # Transmitter
        self.R1 = np.array([200, 0])  # Receiver 1
        self.R2 = np.array([100, 100])  # Receiver 2
        self.R3 = np.array([300, 150])  # Receiver 3
        self.receivers = [self.R1, self.R2, self.R3]
        
        # All stations (for computing convex hull)
        self.all_stations = np.array([self.T, self.R1, self.R2, self.R3])
        
        # RF parameters
        self.lambda_wave = 0.05  # Wavelength (m)
        self.T_CPI = 0.01  # CPI duration (s)
        
        # SNR
        self.SNR_dB = SNR_dB
        self.SNR_linear = 10 ** (SNR_dB / 10)
        
        # Noise standard deviations derived from SNR
        # Phase noise
        self.sigma_phi = 1 / np.sqrt(2 * self.SNR_linear)
        
        # Range measurement noise
        self.sigma_d = self.sigma_phi * self.lambda_wave / (2 * np.pi)
        
        # Differential range noise (two independent CPIs)
        self.sigma_Delta_rho = np.sqrt(2) * self.sigma_d
        
        # Doppler frequency noise
        self.sigma_fd = 1 / (self.T_CPI * np.sqrt(2 * self.SNR_linear))
        
        print(f"=== System Parameters ===")
        print(f"SNR: {SNR_dB} dB")
        print(f"Wavelength: {self.lambda_wave} m")
        print(f"CPI duration: {self.T_CPI} s")
        print(f"\n=== Noise Standard Deviations ===")
        print(f"Phase noise: {self.sigma_phi:.6f} rad")
        print(f"Single range measurement: {self.sigma_d:.6f} m")
        print(f"Differential range: {self.sigma_Delta_rho:.6f} m")
        print(f"Doppler: {self.sigma_fd:.6f} Hz\n")
        
    def get_scaled_polygon(self, scale_factor=0.8):
        """
        计算四个站点围成的多边形，并按比例缩小
        
        Args:
            scale_factor: 缩放比例，默认0.8（缩小到80%）
            
        Returns:
            scaled_vertices: 缩放后的多边形顶点
        """
        # 计算凸包
        hull = ConvexHull(self.all_stations)
        vertices = self.all_stations[hull.vertices]
        
        # 计算中心点
        center = np.mean(vertices, axis=0)
        
        # 向中心缩放
        scaled_vertices = center + scale_factor * (vertices - center)
        
        return scaled_vertices
    
    def point_in_polygon(self, point, polygon):
        """
        判断点是否在多边形内（射线法）
        
        Args:
            point: 待测点 [x, y]
            polygon: 多边形顶点数组 [[x1,y1], [x2,y2], ...]
            
        Returns:
            bool: True if inside
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
            
        return inside
    
    def is_too_close_to_stations(self, point, min_distance=1.0):
        """
        检查点是否距离任何站点太近（避免除零）
        
        Args:
            point: 待测点
            min_distance: 最小允许距离（米）
            
        Returns:
            bool: True if too close
        """
        for station in self.all_stations:
            if np.linalg.norm(point - station) < min_distance:
                return True
        return False
        
    def compute_jacobian(self, p_A, p_B):
        """
        Compute Jacobian matrix G = ∂z/∂θ
        
        Parameter θ = [x_A, y_A, x_B, y_B]
        Measurement z = [Δρ_1, Δρ_2, Δρ_3, f_d1, f_d2, f_d3]
        """
        # 检查是否太接近站点
        if self.is_too_close_to_stations(p_A) or self.is_too_close_to_stations(p_B):
            raise ValueError("Point too close to stations")
            
        G = np.zeros((6, 4))
        
        # Trajectory midpoint (for computing Doppler direction vectors)
        p_c = (p_A + p_B) / 2
        
        # For 3 receivers
        for i, R_i in enumerate(self.receivers):
            # ===== Distances at point A =====
            r_TA = np.linalg.norm(p_A - self.T)
            r_iA = np.linalg.norm(p_A - R_i)
            
            # ===== Distances at point B =====
            r_TB = np.linalg.norm(p_B - self.T)
            r_iB = np.linalg.norm(p_B - R_i)
            
            # ===== Distances and directions at midpoint =====
            r_Tc = np.linalg.norm(p_c - self.T)
            r_ic = np.linalg.norm(p_c - R_i)
            
            # 检查除零
            if r_TA < 1e-6 or r_iA < 1e-6 or r_TB < 1e-6 or r_iB < 1e-6 or \
               r_Tc < 1e-6 or r_ic < 1e-6:
                raise ValueError("Distance too small, division by zero")
            
            u_T = (p_c - self.T) / r_Tc  # Transmitter direction unit vector
            u_i = (p_c - R_i) / r_ic      # Receiver direction unit vector
            
            # ===== Jacobian of differential range sum =====
            grad_rho_A = -((p_A - self.T) / r_TA + (p_A - R_i) / r_iA)
            grad_rho_B = +((p_B - self.T) / r_TB + (p_B - R_i) / r_iB)
            
            G[i, 0:2] = grad_rho_A
            G[i, 2:4] = grad_rho_B
            
            # ===== Jacobian of Doppler =====
            w_i = u_T + u_i
            
            I = np.eye(2)
            H_i = (I - np.outer(u_T, u_T)) / r_Tc + \
                  (I - np.outer(u_i, u_i)) / r_ic
            
            grad_fd_A = (1 / (self.lambda_wave * self.T_CPI)) * \
                        (-w_i - 0.5 * H_i @ (p_B - p_A))
            
            grad_fd_B = (1 / (self.lambda_wave * self.T_CPI)) * \
                        (+w_i - 0.5 * H_i @ (p_B - p_A))
            
            G[3 + i, 0:2] = grad_fd_A
            G[3 + i, 2:4] = grad_fd_B
            
        return G
    
    def compute_crlb(self, p_A, p_B):
        """
        Compute CRLB
        """
        try:
            G = self.compute_jacobian(p_A, p_B)
        except ValueError:
            return np.inf, np.inf, np.inf, None
        
        # Noise covariance matrix R (6x6)
        R = np.diag([self.sigma_Delta_rho**2] * 3 + [self.sigma_fd**2] * 3)
        
        # Fisher information matrix F = G^T R^{-1} G
        F = G.T @ np.linalg.inv(R) @ G
        
        # CRLB matrix Σ = F^{-1}
        try:
            Sigma = np.linalg.inv(F)
        except np.linalg.LinAlgError:
            return np.inf, np.inf, np.inf, None
        
        # Extract sub-blocks
        Sigma_AA = Sigma[0:2, 0:2]
        Sigma_BB = Sigma[2:4, 2:4]
        Sigma_AB = Sigma[0:2, 2:4]
        Sigma_BA = Sigma[2:4, 0:2]
        
        # Position error bounds (PEB)
        PEB_A = np.sqrt(np.trace(Sigma_AA))
        PEB_B = np.sqrt(np.trace(Sigma_BB))
        
        # Velocity error bound (derived metric)
        Sigma_vv = (1 / self.T_CPI**2) * (Sigma_AA + Sigma_BB - Sigma_AB - Sigma_BA)
        VEB = np.sqrt(np.trace(Sigma_vv))
        
        return PEB_A, PEB_B, VEB, Sigma
    
    def test_grid_average(self, grid_step=5, velocity_ms=30.0, scale_factor=0.8):
        """
        在缩小后的四边形网格内计算平均CRLB
        
        Args:
            grid_step: 网格步长（米）
            velocity_ms: 目标速度（m/s）
            scale_factor: 四边形缩放比例
            
        Returns:
            平均PEB_A, 平均PEB_B, 平均VEB, 有效点数
        """
        # 获取缩放后的多边形
        polygon = self.get_scaled_polygon(scale_factor)
        
        print(f"=== Grid Average CRLB Test ===")
        print(f"Polygon scale factor: {scale_factor} (80% of original)")
        print(f"Grid step: {grid_step} m")
        print(f"Velocity: {velocity_ms} m/s\n")
        
        # 确定网格范围
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)
        
        # 计算位移向量
        displacement_m = velocity_ms * self.T_CPI
        angle = np.pi / 4
        displacement_vec = displacement_m * np.array([np.cos(angle), np.sin(angle)])
        
        # 存储有效结果
        PEB_A_list = []
        PEB_B_list = []
        VEB_list = []
        
        # 遍历网格
        x_coords = np.arange(x_min, x_max + grid_step, grid_step)
        y_coords = np.arange(y_min, y_max + grid_step, grid_step)
        
        total_points = len(x_coords) * len(y_coords)
        valid_points = 0
        
        print(f"Scanning {total_points} grid points...")
        
        for i, x in enumerate(x_coords):
            if i % 5 == 0:
                print(f"Progress: {i}/{len(x_coords)} columns")
                
            for y in y_coords:
                p_A = np.array([x, y])
                
                # 检查是否在多边形内
                if not self.point_in_polygon(p_A, polygon):
                    continue
                
                # 检查是否太接近站点
                if self.is_too_close_to_stations(p_A):
                    continue
                
                p_B = p_A + displacement_vec
                
                # 检查B点也在范围内
                if not self.point_in_polygon(p_B, polygon):
                    continue
                
                if self.is_too_close_to_stations(p_B):
                    continue
                
                # 计算CRLB
                PEB_A, PEB_B, VEB, _ = self.compute_crlb(p_A, p_B)
                
                # 只保留有限值
                if np.isfinite(PEB_A) and np.isfinite(PEB_B) and np.isfinite(VEB):
                    PEB_A_list.append(PEB_A)
                    PEB_B_list.append(PEB_B)
                    VEB_list.append(VEB)
                    valid_points += 1
        
        print(f"\nValid points: {valid_points}/{total_points}")
        
        if valid_points == 0:
            print("No valid points found!")
            return None, None, None, 0
        
        # 计算平均值
        avg_PEB_A = np.mean(PEB_A_list)
        avg_PEB_B = np.mean(PEB_B_list)
        avg_VEB = np.mean(VEB_list)
        
        # 计算标准差
        std_PEB_A = np.std(PEB_A_list)
        std_PEB_B = np.std(PEB_B_list)
        std_VEB = np.std(VEB_list)
        
        print(f"\n=== Average CRLB Results ===")
        print(f"Average PEB at point A: {avg_PEB_A:.6f} ± {std_PEB_A:.6f} m")
        print(f"Average PEB at point B: {avg_PEB_B:.6f} ± {std_PEB_B:.6f} m")
        print(f"Average VEB: {avg_VEB:.6f} ± {std_VEB:.6f} m/s")
        
        print(f"\n=== Range Statistics ===")
        print(f"PEB_A range: [{np.min(PEB_A_list):.6f}, {np.max(PEB_A_list):.6f}] m")
        print(f"PEB_B range: [{np.min(PEB_B_list):.6f}, {np.max(PEB_B_list):.6f}] m")
        print(f"VEB range: [{np.min(VEB_list):.6f}, {np.max(VEB_list):.6f}] m/s")
        
        return avg_PEB_A, avg_PEB_B, avg_VEB, valid_points
    
    def plot_error_bounds(self, grid_step=10, velocity_ms=30.0, scale_factor=0.8):
        """
        绘制误差界热图（仅在缩小的四边形内）
        """
        # 获取缩放后的多边形
        polygon = self.get_scaled_polygon(scale_factor)
        
        # 确定网格范围
        x_min, y_min = np.min(polygon, axis=0) - 10
        x_max, y_max = np.max(polygon, axis=0) + 10
        
        # 计算位移向量
        displacement_m = velocity_ms * self.T_CPI
        angle = np.pi / 4
        displacement_vec = displacement_m * np.array([np.cos(angle), np.sin(angle)])
        
        x = np.arange(x_min, x_max + grid_step, grid_step)
        y = np.arange(y_min, y_max + grid_step, grid_step)
        X, Y = np.meshgrid(x, y)
        
        PEB_A_map = np.full_like(X, np.nan, dtype=float)
        PEB_B_map = np.full_like(X, np.nan, dtype=float)
        VEB_map = np.full_like(X, np.nan, dtype=float)
        
        print("Computing CRLB grid...")
        for i in range(len(y)):
            if i % 5 == 0:
                print(f"Progress: {i}/{len(y)}")
            for j in range(len(x)):
                p_A = np.array([X[i, j], Y[i, j]])
                
                # 只计算多边形内的点
                if not self.point_in_polygon(p_A, polygon):
                    continue
                
                if self.is_too_close_to_stations(p_A):
                    continue
                
                p_B = p_A + displacement_vec
                
                if not self.point_in_polygon(p_B, polygon):
                    continue
                
                if self.is_too_close_to_stations(p_B):
                    continue
                
                PEB_A, PEB_B, VEB, _ = self.compute_crlb(p_A, p_B)
                PEB_A_map[i, j] = PEB_A if np.isfinite(PEB_A) else np.nan
                PEB_B_map[i, j] = PEB_B if np.isfinite(PEB_B) else np.nan
                VEB_map[i, j] = VEB if np.isfinite(VEB) else np.nan
        
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        titles = ['PEB at Point A (m)', 
                  'PEB at Point B (m)', 
                  'Velocity Error Bound (m/s)']
        data = [PEB_A_map, PEB_B_map, VEB_map]
        
        for ax, title, d in zip(axes, titles, data):
            # 只对有效数据计算百分位数
            valid_data = d[~np.isnan(d)]
            if len(valid_data) > 0:
                vmin = np.percentile(valid_data, 5)
                vmax = np.percentile(valid_data, 95)
            else:
                vmin, vmax = 0, 1
            
            im = ax.contourf(X, Y, d, levels=20, cmap='jet', vmin=vmin, vmax=vmax)
            
            # 绘制多边形边界
            polygon_closed = np.vstack([polygon, polygon[0]])
            ax.plot(polygon_closed[:, 0], polygon_closed[:, 1], 
                   'w--', linewidth=2, label='Flight zone (80%)')
            
            # 绘制站点
            ax.plot(self.T[0], self.T[1], 'k^', markersize=15, 
                   markeredgewidth=2, markerfacecolor='white', label='Tx')
            for idx, R in enumerate(self.receivers):
                ax.plot(R[0], R[1], 'ks', markersize=12, 
                       markeredgewidth=2, markerfacecolor='white')
            
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'Differential Bistatic Radar CRLB (SNR={self.SNR_dB}dB, velocity={velocity_ms}m/s)', 
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()


# ===== 测试代码 =====

crlb = BistaticDifferentialCRLB(SNR_dB=30)

# 计算网格平均CRLB（在80%缩小的四边形内）
avg_PEB_A, avg_PEB_B, avg_VEB, valid_points = crlb.test_grid_average(
    grid_step=3, 
    velocity_ms=30.0, 
    scale_factor=0.8
)

print("\n" + "="*60 + "\n")

# 绘制误差界分布图
crlb.plot_error_bounds(grid_step=10, velocity_ms=30.0, scale_factor=0.8)