"""
Physics-Informed UAV Locator - 渐进式改进版
基于旧模型，温和改进：
1. 保留单步 GN（稳定）
2. 可选自适应 Huber delta
3. 增强 MLP 输入（几何特征）
4. 改进输出约束
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 全局常量 -----------------
DIM = 2
LAMBDA = 0.06
C0 = 3.0e8
FC = C0 / LAMBDA


# ============================================================
# 1. 改进的物理层（保留单步GN + 自适应Huber）
# ============================================================

def _adaptive_huber_weight(r: torch.Tensor, k: float = 1.5) -> torch.Tensor:
    """
    自适应 Huber 权重：
    delta = k * median(|r|)
    """
    delta = k * r.abs().median(dim=-1, keepdim=True)[0].clamp(min=0.1)
    abs_r = r.abs()
    w = torch.where(
        abs_r <= delta,
        torch.ones_like(r),
        delta / (abs_r + 1e-12)
    )
    return w


class ImprovedLeastSquaresLayer(nn.Module):
    """
    改进的物理层：
    - 保留单步 GN（稳定性）
    - 自适应 Huber 权重
    - 增加步长约束
    """

    def __init__(
        self,
        T: torch.Tensor,
        R: torch.Tensor,
        huber_k: float = 1.5,
        alpha: float = 1e-3,
        max_step: float = 10.0,  # 🔧 新增：最大步长约束
    ):
        super().__init__()
        self.register_buffer("T", T.float())
        self.register_buffer("R", R.float())
        self.huber_k = huber_k
        self.alpha = alpha
        self.max_step = max_step

    @staticmethod
    def _unit(vec: torch.Tensor) -> torch.Tensor:
        return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _gn_step(self, J: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        单步 GN + 自适应 Huber
        J : (B,3,Dim), y : (B,3)
        return (B,Dim)
        """
        # 🔧 改进1：自适应 Huber 权重
        w = _adaptive_huber_weight(-y, self.huber_k)
        WJ = w.unsqueeze(-1) * J
        Wy = w * y

        # 正则化解
        JTJ = WJ.transpose(1, 2) @ WJ
        JTJ_reg = JTJ + self.alpha * torch.eye(DIM, device=J.device)
        JTy = WJ.transpose(1, 2) @ Wy.unsqueeze(-1)

        try:
            L = torch.linalg.cholesky(JTJ_reg)
            sol = torch.cholesky_solve(JTy, L).squeeze(-1)
        except:
            # 🔧 改进2：失败时用 lstsq
            sol = torch.linalg.lstsq(JTJ_reg, JTy).solution.squeeze(-1)

        # 🔧 改进3：步长约束
        step_norm = sol.norm(dim=-1, keepdim=True) + 1e-8
        scale = torch.clamp(self.max_step / step_norm, max=1.0)
        sol = sol * scale

        return sol

    def forward(self, p_prev: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        返回 Δp_lin (B,Dim)
        """
        phi = feats[:, :3]

        # 构造 Jacobian（保留旧模型的简单方案）
        u_AT = self._unit(p_prev - self.T)
        u_AR = self._unit(p_prev.unsqueeze(1) - self.R)
        J = u_AT.unsqueeze(1) + u_AR  # (B,3,Dim)

        # 右端
        d = LAMBDA * phi / (4.0 * math.pi)

        # 解 Δp
        delta_p = self._gn_step(J, d)
        return delta_p


# ============================================================
# 2. 几何特征提取器
# ============================================================

class GeometricFeatureExtractor(nn.Module):
    """提取几何特征增强 MLP"""

    def __init__(self, T: torch.Tensor, R: torch.Tensor):
        super().__init__()
        self.register_buffer("T", T.float())
        self.register_buffer("R", R.float())

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """
        返回几何特征:
        - dist_T: 到发射机距离
        - dist_R: 到各接收机距离 (3个)
        - angle_T: 与发射机的角度（2D情况）
        """
        dist_T = (p - self.T).norm(dim=-1, keepdim=True)
        dist_R = (p.unsqueeze(1) - self.R).norm(dim=-1)

        if DIM == 2:
            vec_T = p - self.T
            angle_T = torch.atan2(vec_T[:, 1], vec_T[:, 0]).unsqueeze(-1)
            # 🔧 归一化特征
            dist_T_norm = dist_T / 100.0  # 假设范围 0-100m
            dist_R_norm = dist_R / 100.0
            geo_feats = torch.cat([dist_T_norm, dist_R_norm, 
                                   torch.sin(angle_T), torch.cos(angle_T)], dim=-1)
        else:
            geo_feats = torch.cat([dist_T, dist_R], dim=-1)

        return geo_feats


# ============================================================
# 3. SE-ResMLP Block（保留不变）
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
        w = self.se_e(self.se_a(self.se_r(y.mean(0, keepdim=True)))).sigmoid()
        y = y * w
        y = self.d2(self.fc2(y))
        return res + y


# ============================================================
# 4. 增强的残差 MLP
# ============================================================

class EnhancedResidualMLP(nn.Module):
    """
    增强的 MLP：
    输入 = [原始特征(6), Δp_lin(Dim), 几何特征(6)]
    """

    def __init__(
        self,
        T: torch.Tensor,
        R: torch.Tensor,
        embed: int = 64,
        depth: int = 4,
        hid_ratio: float = 2.5,
        drop: float = 0.1,
    ):
        super().__init__()
        self.geo_extractor = GeometricFeatureExtractor(T, R)

        # 输入维度：feats(6) + Δp_lin(Dim) + geo(6)
        dim_in = 6 + DIM + 6

        self.embed = nn.Linear(dim_in, embed, bias=False)
        hidden = int(embed * hid_ratio)
        self.blocks = nn.Sequential(
            *[SEResBlock(embed, hidden, drop) for _ in range(depth)]
        )
        self.head = nn.Linear(embed, DIM)

    def forward(
        self,
        feats: torch.Tensor,
        delta_p_lin: torch.Tensor,
        p_current: torch.Tensor
    ) -> torch.Tensor:
        """返回 Δp_corr"""
        geo_feats = self.geo_extractor(p_current)

        x = torch.cat([feats, delta_p_lin, geo_feats], dim=-1)
        x = F.gelu(self.embed(x))
        x = self.blocks(x)

        delta_p_corr = self.head(x)
        return delta_p_corr


# ============================================================
# 5. 单帧模型（增强版）
# ============================================================

class PinUavModel(nn.Module):
    """
    增强的单帧模型
    输入: feats (B,6), p_prev (B,Dim)
    输出: p_hat (B,Dim)
    """

    def __init__(self, T: torch.Tensor, R: torch.Tensor, max_pos: float = 150.0):
        super().__init__()
        self.ls = ImprovedLeastSquaresLayer(T, R, max_step=10.0)
        self.refine = EnhancedResidualMLP(T, R)
        self.max_pos = max_pos

    def forward(
        self,
        feats: torch.Tensor,
        p_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if p_prev is None:
            # 🔧 改进：使用更好的初始猜测
            p_prev = self._get_initial_guess(feats)

        delta_p_lin = self.ls(p_prev, feats)
        p_current = p_prev + delta_p_lin

        delta_p_corr = self.refine(feats, delta_p_lin, p_current)

        p_hat = p_current + delta_p_corr

        # 🔧 温和约束（避免过度限制）
        p_hat = torch.tanh(p_hat / self.max_pos) * self.max_pos

        return p_hat

    def _get_initial_guess(self, feats: torch.Tensor) -> torch.Tensor:
        """
        基于相位差的粗估计
        假设目标在 30-80m 范围内
        """
        B = feats.size(0)
        # 简单策略：随机初始化在合理范围内
        r = torch.rand(B, 1, device=feats.device) * 50 + 30  # 30-80m
        theta = torch.rand(B, 1, device=feats.device) * 2 * math.pi
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.cat([x, y], dim=1)


# ============================================================
# 6. 时序模型（保留原版结构 + 使用改进物理层）
# ============================================================

class PinUavSeqModel(nn.Module):
    """
    时序模型
    输入: feats (B,Seq,6)
    输出: p_hat (B,Seq,Dim)
    """

    def __init__(
        self,
        T: torch.Tensor,
        R: torch.Tensor,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        drop: float = 0.1,
    ):
        super().__init__()
        self.ls_layer = ImprovedLeastSquaresLayer(T, R, max_step=10.0)
        self.geo_extractor = GeometricFeatureExtractor(T, R)

        dim_in = 6 + DIM + 6
        self.embed = nn.Linear(dim_in, 96, bias=False)
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
        p_prev0: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, S, _ = feats.shape
        if p_prev0 is None:
            p_prev0 = torch.zeros(B, DIM, device=feats.device)

        # 物理层 + 几何特征
        p_prev = p_prev0
        lstm_in = []
        delta_p_ls_all = []

        for t in range(S):
            delta_p_ls_t = self.ls_layer(p_prev, feats[:, t])
            p_current = p_prev + delta_p_ls_t

            geo_feats = self.geo_extractor(p_current)

            lstm_in_t = torch.cat([feats[:, t], delta_p_ls_t, geo_feats], dim=-1)
            lstm_in.append(lstm_in_t)
            delta_p_ls_all.append(delta_p_ls_t)

            p_prev = p_current

        lstm_in = torch.stack(lstm_in, dim=1)
        delta_p_ls_all = torch.stack(delta_p_ls_all, dim=1)

        # LSTM 校正
        x = F.gelu(self.embed(lstm_in))
        x, _ = self.lstm(x)
        delta_p_corr = self.head(x)

        p_hat = p_prev0.unsqueeze(1) + delta_p_ls_all.cumsum(dim=1) + delta_p_corr

        return p_hat