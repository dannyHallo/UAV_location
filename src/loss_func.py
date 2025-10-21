import torch
import torch.nn as nn
import torch.nn.functional as F


class CartesianTestLoss(nn.Module):
    """
    适用于PINN模型的损失函数：
    - 模型输出：极坐标格式 [r_pred, sin(θ_pred), cos(θ_pred)]
    - 标签格式：极坐标 [r_true, sin(θ_true), cos(θ_true)]
    - 计算笛卡尔坐标上的MSE损失
    """

    def __init__(
        self,
        distance_scale: float = 1.0,
        normalize_distance: bool = False,
        trig_constraint_weight: float = 0.01,
        reduction: str = "mean",
        debug: bool = False,
    ):
        super().__init__()
        self.distance_scale = distance_scale
        self.normalize_distance = normalize_distance
        self.trig_constraint_weight = trig_constraint_weight
        self.reduction = reduction
        self.debug = debug
        self.epsilon = 1e-10

    def _denormalize_dist(self, distance):
        """反归一化距离"""
        if self.normalize_distance:
            return distance * self.distance_scale
        return distance

    @staticmethod
    def _trig_identity_loss(sin_values, cos_values):
        """三角恒等式约束损失"""
        return torch.mean((sin_values**2 + cos_values**2 - 1.0) ** 2)

    def _safe_sqrt(self, x):
        """安全的平方根计算"""
        return torch.sqrt(torch.clamp(x, min=self.epsilon))

    def forward(self, outputs, targets):
        """
        Args:
            outputs: (B, 3) 模型输出 [r_pred, sin(θ_pred), cos(θ_pred)]
            targets: (B, 3) 标签 [r_true, sin(θ_true), cos(θ_true)]
        """
        # 解析预测和真实值
        r_pred, sin_pred, cos_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        r_true, sin_true, cos_true = targets[:, 0], targets[:, 1], targets[:, 2]

        # 反归一化距离
        r_pred_real = self._denormalize_dist(r_pred)
        r_true_real = self._denormalize_dist(r_true)

        # 转换为笛卡尔坐标
        x_pred = r_pred_real * cos_pred
        y_pred = r_pred_real * sin_pred
        x_true = r_true_real * cos_true
        y_true = r_true_real * sin_true

        # 主损失：笛卡尔坐标MSE
        cartesian_loss = torch.mean((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)

        # 三角恒等式约束
        trig_loss = self._trig_identity_loss(sin_pred, cos_pred)

        # 总损失
        total_loss = cartesian_loss + self.trig_constraint_weight * trig_loss

        # # 调试信息
        # if self.debug:
        #     print(f"Cartesian Loss: {cartesian_loss.item():.6f}")
        #     print(f"Trig Loss: {trig_loss.item():.6f}")
        #     print(f"Total Loss: {total_loss.item():.6f}")
        #     print(
        #         f"Pred range: r=[{r_pred.min().item():.3f}, {r_pred.max().item():.3f}]"
        #     )
        #     print(
        #         f"True range: r=[{r_true.min().item():.3f}, {r_true.max().item():.3f}]"
        #     )

        return total_loss


class PINNLoss(nn.Module):
    """
    专门为PINN模型设计的损失函数：
    - 支持多种损失组合
    - 自适应权重调整
    - 数值稳定性优化
    """

    def __init__(
        self,
        cartesian_weight: float = 1.0,
        trig_weight: float = 0.01,
        smooth_weight: float = 0.001,
        debug: bool = False,
    ):
        super().__init__()
        self.cartesian_weight = cartesian_weight
        self.trig_weight = trig_weight
        self.smooth_weight = smooth_weight
        self.debug = debug
        self.epsilon = 1e-10

        # 自适应权重（可选）
        self.use_adaptive_weights = False
        self.weight_decay = 0.99

    def _cartesian_loss(self, outputs, targets):
        """笛卡尔坐标MSE损失"""
        r_pred, sin_pred, cos_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        r_true, sin_true, cos_true = targets[:, 0], targets[:, 1], targets[:, 2]

        # 转换为笛卡尔坐标
        x_pred = r_pred * cos_pred
        y_pred = r_pred * sin_pred
        x_true = r_true * cos_true
        y_true = r_true * sin_true

        return torch.mean((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)

    def _trig_constraint_loss(self, outputs):
        """三角恒等式约束损失"""
        _, sin_pred, cos_pred = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        return torch.mean((sin_pred**2 + cos_pred**2 - 1.0) ** 2)

    def _smoothness_loss(self, outputs):
        """平滑性约束损失（用于序列模型）"""
        if outputs.dim() == 2:  # 单帧模型
            return torch.tensor(0.0, device=outputs.device)

        # 序列模型的平滑性约束
        r_pred = outputs[:, :, 0]  # (B, S)
        diff = torch.diff(r_pred, dim=1)
        return torch.mean(diff**2)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: (B, 3) or (B, S, 3) 模型输出
            targets: (B, 3) or (B, S, 3) 标签
        """
        # 计算各项损失
        cartesian_loss = self._cartesian_loss(outputs, targets)
        trig_loss = self._trig_constraint_loss(outputs)
        smooth_loss = self._smoothness_loss(outputs)

        # 总损失
        total_loss = (
            self.cartesian_weight * cartesian_loss
            + self.trig_weight * trig_loss
            + self.smooth_weight * smooth_loss
        )

        # 调试信息
        if self.debug:
            print(f"=== Loss Components ===")
            print(f"Cartesian Loss: {cartesian_loss.item():.6f}")
            print(f"Trig Loss: {trig_loss.item():.6f}")
            print(f"Smooth Loss: {smooth_loss.item():.6f}")
            print(f"Total Loss: {total_loss.item():.6f}")
            print(
                f"Weights: cart={self.cartesian_weight}, trig={self.trig_weight}, smooth={self.smooth_weight}"
            )

        return total_loss


class PolarLabelCartesianLoss(nn.Module):
    """
    • pred   : (B, 2)  —— 网络直接输出的笛卡尔坐标  (x̂ , ŷ)
    • target : (B, 3)  —— 仍保持极坐标标签 [ r , sinθ , cosθ ]

    主损失  L_cart = ‖(x̂ , ŷ) - (x , y)‖²
              其中  x = r·cosθ ,  y = r·sinθ

    可选附加  L_trig = (sin² + cos² - 1)²
    total_loss = L_cart + λ · L_trig
    """

    def __init__(
        self,
        distance_scale: float = 1.0,  # 若 r 已是米，则保持 1.0
        normalize_distance: bool = False,  # 标签 r 是否做过归一化
        trig_constraint_weight: float = 0.01,
        reduction: str = "mean",  # 'mean' / 'sum' / 'none'
    ):
        super().__init__()
        self.distance_scale = distance_scale
        self.normalize_distance = normalize_distance
        self.trig_w = trig_constraint_weight
        self.reduction = reduction
        self.eps = 1e-10  # 防止数值为 0 做 sqrt

    # ---------- 辅助 ---------- #
    def _denorm_r(self, r_hat):
        return r_hat * self.distance_scale if self.normalize_distance else r_hat

    @staticmethod
    def _trig_identity_loss(sin_val, cos_val):
        return (sin_val**2 + cos_val**2 - 1.0) ** 2

    # ---------- 前向 ---------- #
    def forward(self, pred_xy: torch.Tensor, target_polar: torch.Tensor):
        """
        pred_xy      : (B, 2)
        target_polar : (B, 3)   [r , sinθ , cosθ]
        """
        # ---- 解析 target ----
        r_hat, sin_theta, cos_theta = (
            target_polar[:, 0],
            target_polar[:, 1],
            target_polar[:, 2],
        )
        r = self._denorm_r(r_hat)

        x_true = r * cos_theta
        y_true = r * sin_theta
        true_xy = torch.stack([x_true, y_true], dim=-1)  # (B,2)

        # ---- 主损失：笛卡尔 L2 ----
        cart_err = (pred_xy - true_xy) ** 2  # (B,2)
        cart_loss = cart_err.sum(dim=-1)  # (B,)

        # ---- 附加三角恒等式约束 ----
        trig_loss = self._trig_identity_loss(sin_theta, cos_theta)  # (B,)

        total = cart_loss + self.trig_w * trig_loss

        if self.reduction == "mean":
            return total.mean()
        if self.reduction == "sum":
            return total.sum()
        return total  # 'none'
