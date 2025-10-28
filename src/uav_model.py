# ============================================================
# Physics-Informed UAV Locator  ——  **Robust GN 改进版 + 噪声感知预处理**
# ------------------------------------------------------------
#  0) NoiseAwareDenoiser                       → 卡尔曼滤波/RTS平滑去噪
#  1) Huber-IRLS + Tikhonov + 单步 Gauss–Newton  → Δp_lin
#  2) Residual MLP (SE-ResMLP)                  → Δp_corr
#  3) 可选 LSTM                                 → 时序滤波
# ============================================================

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 全局常量 -----------------
DIM = 2         # 空间维度（2 or 3）
LAMBDA = 0.06   # 载波波长 (m)
C0 = 3.0e8      # 光速 (m/s)
FC = C0 / LAMBDA

# ============================================================
# 0.  噪声感知预处理层  ——  卡尔曼滤波 / RTS 平滑
# ============================================================

class NoiseAwareDenoiser(nn.Module):
    """
    对 3 相位差 + 3 多普勒 做卡尔曼滤波 / RTS 平滑
    已知:
        σ_φ = sqrt(1 / SNR)
        σ_f = sqrt(1 / (SNR·T_CPI²))
    用一阶随机游走 x_k = x_{k-1} + w_k
    process_var 可人工给一个很小的常数 (代表相邻帧 '真实值' 变化缓慢)

    输入:
        feats_noisy: (B,S,6) 或 (B,6)  [w12,w13,w14, v12,v13,v14]
        snr_db: scalar or (B,) or (B,S)  信噪比(dB)
    输出:
        denoised: (B,S,6) 或 (B,6)  去噪后的测量值
    """

    def __init__(self, t_cpi: float, process_var: float = 1e-4):
        super().__init__()
        self.register_buffer("t_cpi", torch.tensor(t_cpi, dtype=torch.float32))
        self.process_var = process_var

    @staticmethod
    def _kf_step(m_prev, P_prev, z, R, Q):
        """单步卡尔曼滤波更新
        Args:
            m_prev: (B, 6) 先验均值
            P_prev: (B, 6, 6) 先验协方差
            z: (B, 6) 观测值
            R: (B, 6, 6) 观测噪声协方差
            Q: (6, 6) 过程噪声协方差
        Returns:
            m_post: (B, 6) 后验均值
            P_post: (B, 6, 6) 后验协方差
        """
        device = z.device
        dtype = z.dtype
        B = z.shape[0]

        # 预测
        m_pred = m_prev
        P_pred = P_prev + Q.unsqueeze(0)  # (B, 6, 6)

        # Kalman 增益
        S = P_pred + R  # (B, 6, 6)
        K = P_pred @ torch.linalg.inv(S)  # (B, 6, 6)

        # 更新
        innov = (z - m_pred).unsqueeze(-1)  # (B, 6, 1)
        m_post = m_pred + (K @ innov).squeeze(-1)  # (B, 6)

        I = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        P_post = (I - K) @ P_pred  # (B, 6, 6)

        return m_post, P_post

    def forward(
        self,
        feats_noisy: torch.Tensor,   # (B,S,6) 或 (B,6)
        snr_db: Optional[torch.Tensor] = None  # scalar or (B,) or (B,S)
    ) -> torch.Tensor:
        """
        前向传播：对输入特征进行去噪
        """
        single_frame = feats_noisy.dim() == 2
        if single_frame:
            feats_noisy = feats_noisy.unsqueeze(1)  # (B,1,6)

        B, S, _ = feats_noisy.shape
        device = feats_noisy.device
        dtype = feats_noisy.dtype

        # 处理 SNR
        if snr_db is None:
            # 默认使用固定 SNR = 10 dB
            snr_db = torch.tensor(10.0, device=device, dtype=dtype)

        if snr_db.dim() == 0:  # scalar
            snr_db = snr_db.expand(B, S)
        elif snr_db.dim() == 1:  # (B,)
            snr_db = snr_db.unsqueeze(1).expand(B, S)
        else:  # (B,S)
            pass

        # 观测噪声方差
        snr_linear = 10 ** (snr_db / 10.0)
        sigma_phi = torch.sqrt(1.0 / snr_linear)                    # (B,S)
        sigma_f   = torch.sqrt(1.0 / (snr_linear * self.t_cpi**2))  # (B,S)

        # 构造观测协方差矩阵 R (对角阵)
        R_full = torch.zeros(B, S, 6, 6, device=device, dtype=dtype)
        diag = torch.cat([
            sigma_phi.unsqueeze(-1).repeat(1, 1, 3),
            sigma_f.unsqueeze(-1).repeat(1, 1, 3)
        ], dim=-1) ** 2  # (B,S,6)

        # 填充对角线
        for i in range(6):
            R_full[:, :, i, i] = diag[:, :, i]

        # 过程噪声 Q (假设状态缓慢变化)
        Q = torch.eye(6, device=device, dtype=dtype) * self.process_var

        # 初始化
        m = torch.zeros(B, 6, device=device, dtype=dtype)
        P = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) * 1.0

        m_hist = []
        P_hist = []

        # ---------- 前向卡尔曼滤波 ----------
        for k in range(S):
            m, P = self._kf_step(m, P, feats_noisy[:, k], R_full[:, k], Q)
            m_hist.append(m)
            P_hist.append(P)

        m_hist = torch.stack(m_hist, dim=1)   # (B,S,6)
        P_hist = torch.stack(P_hist, dim=1)   # (B,S,6,6)

        # ---------- RTS 平滑器 (只在时序模式下使用) ----------
        if S > 1:
            m_s = m_hist[:, -1].clone()
            P_s = P_hist[:, -1].clone()
            smoothed = [m_s]

            for k in reversed(range(S - 1)):
                P_pred = P_hist[:, k] + Q.unsqueeze(0)
                G = P_hist[:, k] @ torch.linalg.inv(P_pred)
                m_s = m_hist[:, k] + (G @ (m_s - m_hist[:, k]).unsqueeze(-1)).squeeze(-1)
                P_s = P_hist[:, k] + G @ (P_s - P_pred) @ G.transpose(-1, -2)
                smoothed.append(m_s)

            smoothed = torch.stack(list(reversed(smoothed)), dim=1)  # (B,S,6)
        else:
            # 单帧只用滤波结果
            smoothed = m_hist

        if single_frame:
            smoothed = smoothed.squeeze(1)  # (B,6)

        return smoothed

# ============================================================
# 1.  物理显式层   ——  Huber-IRLS + Tikhonov + GN
# ============================================================

def _huber_weight(r: torch.Tensor, delta: float = 0.3) -> torch.Tensor:
    """
    Huber 权:
        w = 1                    , |r| <= δ
          = δ / (|r| + ε)        , |r| >  δ
    """
    abs_r = r.abs()
    w = torch.where(abs_r <= delta,
                    torch.ones_like(r),
                    delta / (abs_r + 1e-12))
    return w


class LeastSquaresLayer(nn.Module):
    """
    Robust Least-Squares 物理层
    输入:
        p_prev : (B,Dim)      上一时刻 UAV 坐标 (若无填 0)
        feats  : (B,6)        [φ1,φ2,φ3,   fd1,fd2,fd3]
    常量:
        T      : (Dim,)       发射机坐标
        R      : (3,Dim)      三只接收机坐标
    输出:
        Δp_lin : (B,Dim)      粗位移 (一次 GN)
    """

    def __init__(
        self,
        T: torch.Tensor,
        R: torch.Tensor,
        huber_delta: float = 0.3,
        alpha: float = 1e-3,
    ):
        super().__init__()
        self.register_buffer("T", T.float())
        self.register_buffer("R", R.float())
        self.delta = huber_delta
        self.alpha = alpha

    # ---------- 内部工具 ---------- #
    @staticmethod
    def _unit(vec: torch.Tensor) -> torch.Tensor:
        return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _gn_step(self, J: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        单步 Gauss–Newton with Huber-IRLS & Tikhonov
        J : (B,3,Dim), y : (B,3)
        return (B,Dim)
        """
        # 1) Huber 权
        w = _huber_weight(-y, self.delta)           # 初始残差取 0 → r = -y
        WJ = w.unsqueeze(-1) * J                    # (B,3,Dim)
        Wy = w * y                                  # (B,3)

        # 2) 正则化解  (JᵀJ+αI)⁻¹ Jᵀy
        JTJ = WJ.transpose(1, 2) @ WJ               # (B,Dim,Dim)
        JTJ_reg = JTJ + self.alpha * torch.eye(DIM, device=J.device)
        JTy = WJ.transpose(1, 2) @ Wy.unsqueeze(-1)  # (B,Dim,1)

        # Cholesky 求解
        L = torch.linalg.cholesky(JTJ_reg)          # (B,Dim,Dim)
        sol = torch.cholesky_solve(JTy, L).squeeze(-1)  # (B,Dim)
        return sol

    # ---------- 前向 ---------- #
    def forward(self, p_prev: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        返回 Δp_lin  (B,Dim)
        """
        phi = feats[:, :3]     # 相位差 rad
        fD  = feats[:, 3:]     # 多普勒 Hz

        # ---------- 构造 J ----------
        u_AT = self._unit(p_prev - self.T)                  # (B,Dim)
        u_AR = self._unit(p_prev.unsqueeze(1) - self.R)     # (B,3,Dim)
        J = u_AT.unsqueeze(1) + u_AR                        # (B,3,Dim)

        # ---------- 右端 ----------
        d = LAMBDA * phi / (4.0 * math.pi)                  # (B,3)
        # 如需速度同法构 b = -λ fD

        # ---------- 解 Δp ----------
        delta_p = self._gn_step(J, d)                       # (B,Dim)
        return delta_p


# ============================================================
# 2.  SE-ResMLP Block
# ============================================================

class SEResBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int,
                 drop: float = 0.1, se_ratio: float = 0.25):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.d1   = nn.Dropout(drop)

        se_hid = max(8, int(hidden_dim * se_ratio))
        self.se_r = nn.Linear(hidden_dim, se_hid, bias=False)
        self.se_a = nn.SiLU()
        self.se_e = nn.Linear(se_hid, hidden_dim, bias=False)

        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.d2  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        y = self.d1(self.norm(self.act(self.fc1(x))))
        # SE
        w = self.se_e(self.se_a(self.se_r(y.mean(0, keepdim=True)))).sigmoid()
        y = y * w
        y = self.d2(self.fc2(y))
        return res + y


# ============================================================
# 3.  误差补偿 MLP
# ============================================================

class ResidualMLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        embed: int = 64,
        depth: int = 4,
        hid_ratio: float = 2.5,
        drop: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(dim_in, embed, bias=False)
        hidden = int(embed * hid_ratio)
        self.blocks = nn.Sequential(
            *[SEResBlock(embed, hidden, drop) for _ in range(depth)]
        )
        self.head = nn.Linear(embed, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.embed(x))
        x = self.blocks(x)
        return self.head(x)


# ============================================================
# 4.  单帧 PINN 模型 (带去噪)
# ============================================================

class PinUavModel(nn.Module):
    """
    输入 :
        feats  (B,6)    三相位差 + 三多普勒 (带噪声)
        p_prev (B,Dim)  上一帧坐标 (无则 None)
        snr_db (scalar/B,)  信噪比(dB)
    输出 :
        p_hat  (B,Dim)  当前位置估计
    """

    def __init__(self, T: torch.Tensor, R: torch.Tensor,
                 t_cpi: float = 0.01, enable_denoising: bool = True):
        super().__init__()
        self.enable_denoising = enable_denoising
        if enable_denoising:
            self.denoiser = NoiseAwareDenoiser(t_cpi=t_cpi, process_var=1e-4)
        self.ls = LeastSquaresLayer(T, R)
        self.refine = ResidualMLP(dim_in=6 + DIM)

    def forward(
        self,
        feats: torch.Tensor,
        p_prev: Optional[torch.Tensor] = None,
        snr_db: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if p_prev is None:
            p_prev = torch.zeros(feats.size(0), DIM, device=feats.device)

        # 0) 去噪 (如果启用)
        if self.enable_denoising:
            feats_clean = self.denoiser(feats, snr_db)
        else:
            feats_clean = feats

        # 1) 物理层
        Δp_lin = self.ls(p_prev, feats_clean)

        # 2) MLP 校正
        inp = torch.cat([feats_clean, Δp_lin], dim=-1)
        Δp_corr = self.refine(inp)

        p_hat = p_prev + Δp_lin + Δp_corr
        return p_hat


# ============================================================
# 5.  时序版本  (去噪 + LSTM + PINN)
# ============================================================

class PinUavSeqModel(nn.Module):
    """
    输入 : feats (B,Seq,6)  带噪声的特征
           snr_db (scalar/B,/B,Seq)  信噪比
    输出 : p_hat (B,Seq,Dim)
    """

    def __init__(
        self,
        T: torch.Tensor,
        R: torch.Tensor,
        t_cpi: float = 0.01,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        drop: float = 0.1,
        enable_denoising: bool = True,
    ):
        super().__init__()
        self.enable_denoising = enable_denoising
        if enable_denoising:
            self.denoiser = NoiseAwareDenoiser(t_cpi=t_cpi, process_var=1e-4)
        self.ls_layer = LeastSquaresLayer(T, R)
        self.embed = nn.Linear(6 + DIM, 96, bias=False)
        self.lstm = nn.LSTM(
            input_size=96,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=drop if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, DIM)

    def forward(
        self,
        feats: torch.Tensor,
        p_prev0: Optional[torch.Tensor] = None,
        snr_db: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, _ = feats.shape
        if p_prev0 is None:
            p_prev0 = torch.zeros(B, DIM, device=feats.device)

        # 0) 去噪 (RTS 平滑，利用整个序列信息)
        if self.enable_denoising:
            feats_clean = self.denoiser(feats, snr_db)  # (B,S,6)
        else:
            feats_clean = feats

        # 1) 物理层游程 (Δp_lin) & LSTM 输入
        p_prev = p_prev0
        lstm_in = []
        Δp_ls_all = []
        for t in range(S):
            Δp_ls_t = self.ls_layer(p_prev, feats_clean[:, t])
            Δp_ls_all.append(Δp_ls_t)
            lstm_in.append(torch.cat([feats_clean[:, t], Δp_ls_t], dim=-1))
            p_prev = p_prev + Δp_ls_t
        lstm_in = torch.stack(lstm_in, dim=1)      # (B,S,6+Dim)
        Δp_ls_all = torch.stack(Δp_ls_all, dim=1)  # (B,S,Dim)

        # 2) LSTM 校正
        x = F.gelu(self.embed(lstm_in))
        x, _ = self.lstm(x)
        Δp_corr = self.head(x)                     # (B,S,Dim)

        p_hat = p_prev0.unsqueeze(1) + Δp_ls_all.cumsum(dim=1) + Δp_corr
        return p_hat