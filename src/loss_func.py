import torch
import torch.nn as nn
import numpy as np


class CartesianTestLoss(nn.Module):
    """
    混合坐标系损失函数：

    网络输出仍为 [r, sin(theta), cos(theta)]，
        r = 极坐标半径 (可归一化)

    主损失始终在笛卡尔坐标 (x,y) 上计算：
        L_main = ‖(x_pred, y_pred) - (x_true, y_true)‖²
    并附加三角恒等式约束：
        L_trig = (sin² + cos² - 1)²

    total_loss = L_main + λ·L_trig
    """

    def __init__(
        self,
        distance_scale: float = 200.0,      # r 归一化系数
        normalize_distance: bool = True,    # 是否对 r 归一化
        trig_constraint_weight: float = 0.01,
    ):
        super(CartesianTestLoss, self).__init__()
        self.distance_scale = float(distance_scale)
        self.normalize_distance = bool(normalize_distance)
        self.trig_constraint_weight = float(trig_constraint_weight)
        self.epsilon = 1e-10  # 避免 sqrt(0)

    # ---------- 工具函数 ---------- #
    @staticmethod
    def mse_loss(pred, target):
        return torch.mean((pred - target) ** 2)

    def safe_sqrt(self, x):
        return torch.sqrt(torch.clamp(x, min=self.epsilon))

    def normalize_dist(self, distance):
        """r  →  r_hat"""
        if self.normalize_distance:
            return distance / self.distance_scale
        return distance

    def denormalize_dist(self, distance_hat):
        """r_hat  →  r"""
        if self.normalize_distance:
            return distance_hat * self.distance_scale
        return distance_hat

    @staticmethod
    def trig_identity_loss(sin_values, cos_values):
        """(sin² + cos² − 1)²"""
        return torch.mean((sin_values**2 + cos_values**2 - 1.0) ** 2)

    # ---------- 评估用欧氏误差（米） ---------- #
    def _compute_cartesian_error(self, outputs, targets):
        r_hat, s_hat, c_hat = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        r,     s,     c     = targets[:, 0], targets[:, 1], targets[:, 2]

        r_hat = self.denormalize_dist(r_hat)
        r     = self.denormalize_dist(r)

        x_pred = r_hat * c_hat
        y_pred = r_hat * s_hat
        x_true = r     * c
        y_true = r     * s

        euclidean_dist = self.safe_sqrt((x_pred - x_true) ** 2 +
                                        (y_pred - y_true) ** 2)
        return torch.mean(euclidean_dist)

    # ---------- 前向：训练 / 推断主损失 ---------- #
    def forward(self, outputs, targets):
        r_hat, s_hat, c_hat = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        r,     s,     c     = targets[:, 0], targets[:, 1], targets[:, 2]

        # 反归一化
        r_hat_real = self.denormalize_dist(r_hat)
        r_real     = self.denormalize_dist(r)

        # 极坐标 -> 笛卡尔
        x_pred = r_hat_real * c_hat
        y_pred = r_hat_real * s_hat
        x_true = r_real     * c
        y_true = r_real     * s

        # ① 主损失：笛卡尔平方误差
        cartesian_loss = torch.mean((x_pred - x_true) ** 2 +
                                    (y_pred - y_true) ** 2)

        # ② 三角恒等式约束
        trig_loss = self.trig_identity_loss(s_hat, c_hat)

        total_loss = cartesian_loss + self.trig_constraint_weight * trig_loss
        return total_loss