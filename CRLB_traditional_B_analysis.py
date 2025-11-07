import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义位置参数
T = np.array([0, 0])
R1 = np.array([200, 0])
R2 = np.array([100, 100])
R3 = np.array([300, 150])
p_A = np.array([40.1, 30.1])       # 目标位置

# 系统参数
lambda_wave = 0.05             # 波长 (m)
T_CPI = 0.01                   # 相干处理时间 (s)
v_max = 30                     # 最大速度 (m/s)
d_max = v_max * T_CPI          # 最大位移 0.3m

# SNR参数
SNR_dB = np.array([-10, -5, 0, 5, 10])
SNR_linear = 10 ** (SNR_dB / 10)

# 接收机位置列表
receivers = [R1, R2, R3]
num_receivers = len(receivers)

print("="*70)
print("多基地雷达定位系统参数")
print("="*70)
print(f"发射机位置 T: {T}")
print(f"接收机1位置 R1: {R1}")
print(f"接收机2位置 R2: {R2}")
print(f"接收机3位置 R3: {R3}")
print(f"目标A位置 p_A: {p_A}")
print(f"波长 λ: {lambda_wave} m")
print(f"相干处理时间 T_CPI: {T_CPI} s")
print(f"最大速度: {v_max} m/s")
print(f"AB最大位移: {d_max} m")
print("="*70)

def calculate_distance(p1, p2):
    """计算两点之间的距离"""
    return np.linalg.norm(p1 - p2)

def calculate_bistatic_range(target, transmitter, receiver):
    """计算双基地距离"""
    R_T = calculate_distance(target, transmitter)
    R_R = calculate_distance(target, receiver)
    return R_T + R_R

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
        R_T = np.sqrt((x - x_t)**2 + (y - y_t)**2)
        # 计算接收机到目标的距离
        R_R = np.sqrt((x - x_r)**2 + (y - y_r)**2)
        
        # 计算对x的偏导数
        dRho_dx = (x - x_t) / R_T + (x - x_r) / R_R
        # 计算对y的偏导数
        dRho_dy = (y - y_t) / R_T + (y - y_r) / R_R
        
        G.append([dRho_dx, dRho_dy])
    
    return np.array(G)

def calculate_range_measurement_std(SNR_linear, lambda_wave, T_CPI):
    """
    计算距离测量误差标准差 σ_d
    
    根据雷达测距理论：
    1. 相位测量误差: σ_φ ≈ 1/√(2·SNR) (弧度)
       对应距离误差: σ_d,φ = λ/(2π) · σ_φ = λ/(4π√(2·SNR))
    
    2. 多普勒测量误差: σ_fd ≈ 1/(T_CPI·√(2·SNR)) (Hz)
       对应速度误差: σ_v = λ · σ_fd
       在时间T_CPI内对应距离误差: σ_d,v = σ_v · T_CPI = λ/(√(2·SNR))
    
    3. 组合距离测量误差（假设独立）:
       σ²_d = σ²_d,φ + σ²_d,v
    """
    # 相位测量对应的距离误差
    sigma_d_phi = lambda_wave / (2 * np.pi * np.sqrt(2 * SNR_linear))
    
    # 多普勒测量对应的距离误差
    sigma_d_v = lambda_wave / np.sqrt(2 * SNR_linear)
    
    # 组合误差
    sigma_d = np.sqrt(sigma_d_phi**2 + sigma_d_v**2)
    
    return sigma_d, sigma_d_phi, sigma_d_v

def calculate_GDOP(G):
    """
    计算几何精度衰减因子 GDOP
    GDOP = √(trace((G^T G)^(-1)))
    """
    GTG = G.T @ G
    try:
        GTG_inv = np.linalg.inv(GTG)
        GDOP = np.sqrt(np.trace(GTG_inv))
        return GDOP, GTG, GTG_inv
    except:
        return None, GTG, None

def calculate_PEB(sigma_d, GDOP):
    """
    计算位置误差界 PEB (Position Error Bound)
    PEB = σ_d · GDOP
    """
    return sigma_d * GDOP

# 计算几何矩阵
print("\n几何矩阵 G (雅可比矩阵):")
print("-"*70)
G = calculate_geometry_matrix(p_A, T, receivers)
print(f"形状: {G.shape} ({num_receivers}个接收机 × 2维坐标)")
print("\n几何矩阵 G =")
print(G)
print("\n各接收机对应的梯度:")
for i, receiver in enumerate(receivers):
    rho_i = calculate_bistatic_range(p_A, T, receiver)
    print(f"R{i+1} {receiver}: ρ{i+1} = {rho_i:.4f} m, "
          f"∇ρ{i+1} = [{G[i,0]:.6f}, {G[i,1]:.6f}]")

# 计算 G^T G 和 GDOP
print("\n" + "="*70)
print("几何特性分析")
print("="*70)
GDOP, GTG, GTG_inv = calculate_GDOP(G)
print("\nG^T G =")
print(GTG)
print("\n(G^T G)^(-1) =")
print(GTG_inv)
print(f"\nGDOP = √(trace((G^T G)^(-1))) = {GDOP:.6f}")

# 计算不同SNR下的PEB
print("\n" + "="*70)
print("不同SNR下的定位精度分析")
print("="*70)

results = []
for snr_db, snr_lin in zip(SNR_dB, SNR_linear):
    print(f"\nSNR = {snr_db} dB (线性值: {snr_lin:.4f})")
    print("-"*70)
    
    # 计算距离测量误差
    sigma_d, sigma_d_phi, sigma_d_v = calculate_range_measurement_std(
        snr_lin, lambda_wave, T_CPI)
    
    print(f"距离测量误差分析:")
    print(f"  相位测量贡献:     σ_d,φ = {sigma_d_phi:.6f} m")
    print(f"  多普勒测量贡献:   σ_d,v = {sigma_d_v:.6f} m")
    print(f"  总距离测量误差:   σ_d   = {sigma_d:.6f} m")
    
    # 计算Fisher信息矩阵
    Fisher_Info = (1 / sigma_d**2) * GTG
    print(f"\nFisher信息矩阵 I = (1/σ²_d)·G^T G =")
    print(Fisher_Info)
    
    # 计算CRLB矩阵
    CRLB_matrix = sigma_d**2 * GTG_inv
    print(f"\nCRLB矩阵 Σ_CRLB = σ²_d·(G^T G)^(-1) =")
    print(CRLB_matrix)
    
    # 提取x和y方向的标准差
    sigma_x = np.sqrt(CRLB_matrix[0, 0])
    sigma_y = np.sqrt(CRLB_matrix[1, 1])
    
    # 计算位置误差界 PEB
    PEB = calculate_PEB(sigma_d, GDOP)
    
    print(f"\n定位精度指标:")
    print(f"  σ_x (x方向标准差):        {sigma_x:.6f} m")
    print(f"  σ_y (y方向标准差):        {sigma_y:.6f} m")
    print(f"  PEB = σ_d × GDOP:        {PEB:.6f} m  ← 这是理论下界!")
    print(f"  验证: √(σ²_x + σ²_y) =   {np.sqrt(sigma_x**2 + sigma_y**2):.6f} m")
    
    results.append({
        'SNR_dB': snr_db,
        'SNR_linear': snr_lin,
        'sigma_d': sigma_d,
        'sigma_d_phi': sigma_d_phi,
        'sigma_d_v': sigma_d_v,
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'PEB': PEB
    })

# 绘制结果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

SNR_values = [r['SNR_dB'] for r in results]
sigma_d_values = [r['sigma_d'] for r in results]
sigma_d_phi_values = [r['sigma_d_phi'] for r in results]
sigma_d_v_values = [r['sigma_d_v'] for r in results]
PEB_values = [r['PEB'] for r in results]
sigma_x_values = [r['sigma_x'] for r in results]
sigma_y_values = [r['sigma_y'] for r in results]

# 子图1: 距离测量误差分量
axes[0, 0].semilogy(SNR_values, sigma_d_phi_values, 'b-o', 
                    linewidth=2, markersize=8, label='相位测量 σ_d,φ')
axes[0, 0].semilogy(SNR_values, sigma_d_v_values, 'r-s', 
                    linewidth=2, markersize=8, label='多普勒测量 σ_d,v')
axes[0, 0].semilogy(SNR_values, sigma_d_values, 'k-^', 
                    linewidth=2.5, markersize=8, label='总测量误差 σ_d')
axes[0, 0].set_xlabel('SNR (dB)', fontsize=12)
axes[0, 0].set_ylabel('距离测量误差 (m)', fontsize=12)
axes[0, 0].set_title('距离测量误差分析', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2: PEB (主要结果!)
axes[0, 1].semilogy(SNR_values, PEB_values, 'g-^', 
                    linewidth=3, markersize=10)
axes[0, 1].set_xlabel('SNR (dB)', fontsize=12)
axes[0, 1].set_ylabel('位置误差界 PEB (m)', fontsize=12)
axes[0, 1].set_title('位置误差界 (CRLB)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
for i, (snr, peb) in enumerate(zip(SNR_values, PEB_values)):
    axes[0, 1].text(snr, peb*1.1, f'{peb:.3f}m', 
                    ha='center', fontsize=9)

# 子图3: x和y方向误差对比
axes[1, 0].semilogy(SNR_values, sigma_x_values, 'b-o', 
                    linewidth=2, markersize=8, label='σ_x (x方向)')
axes[1, 0].semilogy(SNR_values, sigma_y_values, 'r-s', 
                    linewidth=2, markersize=8, label='σ_y (y方向)')
axes[1, 0].set_xlabel('SNR (dB)', fontsize=12)
axes[1, 0].set_ylabel('定位误差标准差 (m)', fontsize=12)
axes[1, 0].set_title('各方向定位精度', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 子图4: PEB vs σ_d 的关系
axes[1, 1].loglog(sigma_d_values, PEB_values, 'mo-', 
                  linewidth=2, markersize=8)
axes[1, 1].set_xlabel('距离测量误差 σ_d (m)', fontsize=12)
axes[1, 1].set_ylabel('位置误差界 PEB (m)', fontsize=12)
axes[1, 1].set_title(f'PEB = σ_d × GDOP (GDOP={GDOP:.3f})', 
                     fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, which='both')
# 添加理论直线
sigma_d_theory = np.array([min(sigma_d_values), max(sigma_d_values)])
PEB_theory = sigma_d_theory * GDOP
axes[1, 1].plot(sigma_d_theory, PEB_theory, 'k--', 
                linewidth=2, alpha=0.5, label=f'理论: PEB={GDOP:.3f}σ_d')
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('CRLB_PEB_analysis.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为 'CRLB_PEB_analysis.png'")
plt.show()

# 总结表格
print("\n" + "="*90)
print("结果汇总表")
print("="*90)
print(f"{'SNR(dB)':<8} {'σ_d,φ(m)':<12} {'σ_d,v(m)':<12} {'σ_d(m)':<12} "
      f"{'PEB(m)':<12} {'σ_x(m)':<12} {'σ_y(m)':<12}")
print("-"*90)
for r in results:
    print(f"{r['SNR_dB']:<8} {r['sigma_d_phi']:<12.6f} {r['sigma_d_v']:<12.6f} "
          f"{r['sigma_d']:<12.6f} {r['PEB']:<12.6f} "
          f"{r['sigma_x']:<12.6f} {r['sigma_y']:<12.6f}")
print("="*90)
print(f"\n注: GDOP = {GDOP:.6f} (与SNR无关,仅由几何构型决定)")
print(f"    PEB = σ_d × GDOP  (位置误差界,CRLB的标量表示)")