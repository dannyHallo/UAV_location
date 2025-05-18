import torch
import torch.nn as nn
import numpy as np


class CombinedPolarLossWithPhysicalConstraint(nn.Module):
    """
    综合极坐标损失函数，同时计算:
    1. 距离(ρ)的MSE损失
    2. 角度(θ)的L2损失（考虑角度的周期性）
    3. 基于余弦定理的物理约束损失

    可以通过权重参数控制各种损失的相对重要性
    """

    def __init__(
        self, is_training, distance_weight=1.0, angle_weight=1.0, physical_weight=1.0
    ):
        super(CombinedPolarLossWithPhysicalConstraint, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.physical_weight = physical_weight
        self.is_training = is_training
        self.mse = nn.MSELoss(reduction="mean")

    def angular_difference(self, theta_true, theta_pred):
        """
        计算角度差异，返回范围在(-π, π]的最小有符号差值

        参数:
        theta_true: 真实角度，张量
        theta_pred: 预测角度，张量

        返回:
        角度差异张量
        """
        return (theta_pred - theta_true + np.pi) % (2 * np.pi) - np.pi

    def l2_angle_loss(self, theta_true, theta_pred):
        """
        计算角度的L2损失: Δ²，范围在[0, π²]

        参数:
        theta_true: 真实角度，张量
        theta_pred: 预测角度，张量

        返回:
        角度L2损失
        """
        d = self.angular_difference(theta_true, theta_pred)
        return torch.mean(d * d)

    def forward(self, outputs, targets, extra_infos):
        """
        计算综合损失

        参数:
        outputs: 模型输出，形状为 [batch_size, 2]，其中第一列是距离(ρ_pred)，第二列是角度(θ_pred)
        targets: 目标值，形状为 [batch_size, 2]，其中第一列是距离(ρ_real)，第二列是角度(θ_real)
        extra_infos: 包含 三个接收机距离和相对位置信息
                    d21, d31, d41 - UAV到三个接收机的实际距离
                    (ρ1,θ1), (ρ2,θ2), (ρ3,θ3) - 三个接收机相对于发射机的位置
                    形状为 [batch_size, 9]

        返回:
        训练模式: 综合损失值 = distance_weight * 距离MSE + angle_weight * 角度L2损失 + physical_weight * 物理约束损失
        评估模式: 笛卡尔坐标下的预测位置与真实位置之间的MSE损失
        """
        # 提取距离和角度维度
        distance_outputs = outputs[:, 0]  # ρ_pred
        distance_targets = targets[:, 0]  # ρ_real

        angle_outputs = outputs[:, 1]  # θ_pred
        angle_targets = targets[:, 1]  # θ_real

        # 计算距离MSE损失
        distance_loss = self.mse(distance_outputs, distance_targets)

        # 计算角度L2损失
        angle_loss = self.l2_angle_loss(angle_targets, angle_outputs)

        # 提取extra_infos中的信息
        # 接收机到UAV的实际距离
        d21_real = extra_infos[:, 0]  # O2到UAV的实际距离
        d31_real = extra_infos[:, 1]  # O3到UAV的实际距离
        d41_real = extra_infos[:, 2]  # O4到UAV的实际距离

        # 接收机相对于发射机O1的位置（极坐标）
        rho2 = extra_infos[:, 3]  # O2到O1的距离 ρ2
        theta2 = extra_infos[:, 4]  # O2相对于O1的角度 θ2

        rho3 = extra_infos[:, 5]  # O3到O1的距离 ρ3
        theta3 = extra_infos[:, 6]  # O3相对于O1的角度 θ3

        rho4 = extra_infos[:, 7]  # O4到O1的距离 ρ4
        theta4 = extra_infos[:, 8]  # O4相对于O1的角度 θ4

        # 计算基于余弦定理的物理约束损失

        # 计算UAV到O2的预测距离（基于余弦定理）
        # 在三角形O1AO2中，利用ρ_pred(UAV到O1)、ρ2(O2到O1)和(θ_pred-θ2)
        # AO2² = O1A² + O1O2² - 2*O1A*O1O2*cos(θ_pred-θ2)
        d21_pred = torch.sqrt(
            distance_outputs**2
            + rho2**2
            - 2
            * distance_outputs
            * rho2
            * torch.cos(self.angular_difference(theta2, angle_outputs))
        )

        # 计算UAV到O3的预测距离（基于余弦定理）
        d31_pred = torch.sqrt(
            distance_outputs**2
            + rho3**2
            - 2
            * distance_outputs
            * rho3
            * torch.cos(self.angular_difference(theta3, angle_outputs))
        )

        # 计算UAV到O4的预测距离（基于余弦定理）
        d41_pred = torch.sqrt(
            distance_outputs**2
            + rho4**2
            - 2
            * distance_outputs
            * rho4
            * torch.cos(self.angular_difference(theta4, angle_outputs))
        )

        # 计算物理约束损失（预测距离与实际距离的MSE）
        physical_loss_21 = self.mse(d21_pred, d21_real)
        physical_loss_31 = self.mse(d31_pred, d31_real)
        physical_loss_41 = self.mse(d41_pred, d41_real)

        # 合并物理约束损失
        physical_loss = (physical_loss_21 + physical_loss_31 + physical_loss_41) / 3.0

        # 训练模式：返回五部分组成的加权综合损失
        if self.is_training:
            combined_loss = (
                self.distance_weight * distance_loss
                + self.angle_weight * angle_loss
                + self.physical_weight * physical_loss
            )
            return combined_loss
        else:
            # 评估模式：将极坐标转换为笛卡尔坐标，计算位置损失

            # 预测位置的笛卡尔坐标
            x_pred = distance_outputs * torch.cos(angle_outputs)
            y_pred = distance_outputs * torch.sin(angle_outputs)

            # 真实位置的笛卡尔坐标
            x_real = distance_targets * torch.cos(angle_targets)
            y_real = distance_targets * torch.sin(angle_targets)

            # 计算笛卡尔坐标下的位置MSE损失
            position_loss_x = self.mse(x_pred, x_real)
            position_loss_y = self.mse(y_pred, y_real)
            position_loss = (position_loss_x + position_loss_y) / 2.0

            # 也可以直接计算欧几里得距离的均值
            euclidean_distance = torch.sqrt(
                (x_pred - x_real) ** 2 + (y_pred - y_real) ** 2
            )
            mean_euclidean_distance = torch.mean(euclidean_distance)

            return mean_euclidean_distance # 直接返回位置损失
