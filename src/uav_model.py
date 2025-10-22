# ============================================================
# Physics-Informed UAV Locator —— Range-&-Doppler GN
# + State-Space LSTM for Process-Noise Compensation
# ============================================================

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 全局常量 -----------------
DIM     = 2                # 平面定位
FC      = 6e9              # 6 GHz 载波
LAMBDA  = 3.0e8 / FC       # ≈ 0.05 m 波长


# ============================================================
# 1.  Robust Gauss-Newton (距离 + 多普勒 6×2)
# ============================================================

def _huber_weight(r: torch.Tensor, delta: float = 0.3) -> torch.Tensor:
    """Huber-IRLS 权重"""
    abs_r = r.abs()
    return torch.where(abs_r <= delta,
                       torch.ones_like(r),
                       delta / (abs_r + 1e-12))


class LeastSquaresLayer(nn.Module):
    """
    单步 Gauss-Newton：由 3 距离 + 3 多普勒 解 2-D 位置增量
    输入:
        p_prev : (B,2)  当前先验
        feats  : (B,6)  [φ1,φ2,φ3,  fD1,fD2,fD3]
    输出:
        Δp_lin : (B,2)  GN 线性化增量
    """

    def __init__(self,
                 T: torch.Tensor,             # (2,) 发射机坐标
                 R: torch.Tensor,             # (3,2) 接收机坐标
                 huber_delta: float = 0.3,
                 alpha: float = 1e-3):
        super().__init__()
        self.register_buffer("T", T.float())
        self.register_buffer("R", R.float())
        self.delta = huber_delta
        self.alpha = alpha

    # ---------- 工具 ---------- #
    @staticmethod
    def _unit(v: torch.Tensor) -> torch.Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # ---------- GN 内部 ---------- #
    def _gn_step(self, J: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w  = _huber_weight(-y, self.delta)            # (B,6)
        WJ = w.unsqueeze(-1) * J                      # (B,6,2)
        Wy = w * y                                    # (B,6)

        JTJ = WJ.transpose(1, 2) @ WJ                # (B,2,2)
        JTJ = JTJ + self.alpha * torch.eye(DIM, device=J.device)
        JTy = WJ.transpose(1, 2) @ Wy.unsqueeze(-1)  # (B,2,1)

        L   = torch.linalg.cholesky(JTJ)
        Δp  = torch.cholesky_solve(JTy, L).squeeze(-1)  # (B,2)
        return Δp

    # ---------- 前向 ---------- #
    def forward(self,
                p_prev: torch.Tensor,      # (B,2)
                feats: torch.Tensor) -> torch.Tensor:
        phi = feats[:, :3]                         # (B,3) 相位差 rad
        fD  = feats[:, 3:]                         # (B,3) 多普勒 Hz

        # 距离/速度方向向量
        u_AT = self._unit(p_prev - self.T)                 # (B,2)
        u_AR = self._unit(p_prev.unsqueeze(1) - self.R)    # (B,3,2)

        # Jacobian (距离 & 速度)
        J_r = u_AT.unsqueeze(1) + u_AR                     # (B,3,2)
        J   = torch.cat([J_r, J_r.clone()], dim=1)         # (B,6,2)

        # 残差
        d =  LAMBDA * phi / (4 * math.pi)                  # (B,3) m
        v = -LAMBDA * fD / 2.0                             # (B,3) m/s
        y = torch.cat([d, v], dim=1)                       # (B,6)

        return self._gn_step(J, y)                         # (B,2)


# ============================================================
# 2.  误差补偿 MLP  (SE-ResBlock)
# ============================================================

class SEResBlock(nn.Module):
    def __init__(self, emb: int, hidden: int,
                 drop: float = 0.1, se_ratio: float = 0.25):
        super().__init__()
        self.fc1  = nn.Linear(emb, hidden, bias=False)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(hidden)
        self.d1   = nn.Dropout(drop)

        se_hid = max(8, int(hidden * se_ratio))
        self.se_r = nn.Linear(hidden, se_hid, bias=False)
        self.se_a = nn.SiLU()
        self.se_e = nn.Linear(se_hid, hidden, bias=False)

        self.fc2 = nn.Linear(hidden, emb, bias=False)
        self.d2  = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.d1(self.norm(self.act(self.fc1(x))))
        w = self.se_e(self.se_a(self.se_r(y.mean(0, keepdim=True)))).sigmoid()
        y = y * w
        y = self.d2(self.fc2(y))
        return x + y


class ResidualMLP(nn.Module):
    """单帧误差补偿网络 (可在静态数据集使用)"""
    def __init__(self,
                 dim_in: int,
                 embed: int = 64,
                 depth: int = 4,
                 hid_ratio: float = 2.5,
                 drop: float = 0.1):
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
# 3.  单帧 PINN (仅供离线测试)
# ============================================================

class PinUavModel(nn.Module):
    """
    feats : (B,6)   [3 φ, 3 fD]
    p_prev: (B,2)
    return: (B,2)
    """
    def __init__(self, T: torch.Tensor, R: torch.Tensor):
        super().__init__()
        self.ls     = LeastSquaresLayer(T, R)
        self.refine = ResidualMLP(dim_in=6 + DIM)

    def forward(self,
                feats: torch.Tensor,
                p_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        if p_prev is None:
            p_prev = torch.zeros(feats.size(0), DIM, device=feats.device)

        Δp_ls  = self.ls(p_prev, feats)                          # 物理增量
        Δp_cor = self.refine(torch.cat([feats, Δp_ls], dim=-1))  # 学习增量
        return p_prev + Δp_ls + Δp_cor


# ============================================================
# 4.  时序版：State-Space LSTM
# ============================================================

class PinUavSeqModel(nn.Module):
    """
    输入:
        feats : (B,S,6)  —— 3 相位差 + 3 多普勒
        p0    : (B,2)    —— 初始先验 (可 None → 0)
    输出:
        traj  : (B,S,2)  —— 估计轨迹
    流程:
        1. GN 得到 Δp_GN,t   (物理模型)
        2. LSTM 输出 Δp_proc,t (残余过程噪声)
        3. p_hat_t = p0 + Σ(Δp_GN + Δp_proc)
    """

    def __init__(self,
                 T: torch.Tensor,
                 R: torch.Tensor,
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 drop: float = 0.1):
        super().__init__()
        self.gn   = LeastSquaresLayer(T, R)

        self.embed = nn.Linear(6 + DIM, 96, bias=False)
        self.lstm  = nn.LSTM(
            input_size=96,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=drop if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(lstm_hidden, DIM)

    # ---------- 前向 ---------- #
    def forward(self,
                feats: torch.Tensor,               # (B,S,6)
                p0: Optional[torch.Tensor] = None  # (B,2)
                ) -> torch.Tensor:
        B, S, _ = feats.shape
        if p0 is None:
            p0 = torch.zeros(B, DIM, device=feats.device)

        # ---- 1) GN 物理增量序列 ----
        p_prev = p0
        dp_gn_seq, lstm_in = [], []
        for t in range(S):
            dp = self.gn(p_prev, feats[:, t])                   # (B,2)
            dp_gn_seq.append(dp)
            lstm_in.append(torch.cat([feats[:, t], dp], dim=-1))
            p_prev = p_prev + dp                                # 物理游程

        Δp_GN   = torch.stack(dp_gn_seq, dim=1)                 # (B,S,2)
        lstm_in = torch.stack(lstm_in,  dim=1)                  # (B,S,8)

        # ---- 2) LSTM 过程噪声增量 ----
        x, _ = self.lstm(F.gelu(self.embed(lstm_in)))
        Δp_proc = self.head(x)                                  # (B,S,2)

        # ---- 3) 积分得到轨迹 ----
        traj = (
            p0.unsqueeze(1)
            + torch.cumsum(Δp_GN,   dim=1)
            + torch.cumsum(Δp_proc, dim=1)
        )                                                       # (B,S,2)
        return traj