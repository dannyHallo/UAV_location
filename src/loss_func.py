import torch
import torch.nn as nn


class CartesianTestLoss(nn.Module):
    """
    混合坐标系损失函数：
    - 训练模式(model.train())：处理 [距离, sin(θ), cos(θ)] 格式输入，使用MSE损失
    并添加sin²(θ) + cos²(θ) = 1 的三角约束
    - 测试模式(model.eval())：将输出转换为笛卡尔坐标 [ρcosθ, ρsinθ] 计算欧几里得误差

    可通过normalize_distance开关控制是否进行归一化
    使用标准的PyTorch train/eval机制切换模式
    """

    def __init__(
        self,
        distance_scale=200.0,  # 距离归一化系数
        normalize_distance=True,  # 是否对距离进行归一化
        trig_constraint_weight=0.5,
    ):
        super(CartesianTestLoss, self).__init__()
        self.distance_scale = distance_scale
        self.normalize_distance = normalize_distance
        self.trig_constraint_weight = trig_constraint_weight
        self.epsilon = 1e-10

        # 默认设置为训练模式
        self.train()

    def mse_loss(self, pred, target):
        """计算均方误差损失"""
        return torch.mean((pred - target) ** 2)

    def safe_sqrt(self, x):
        """安全的平方根计算，确保输入非负"""
        return torch.sqrt(torch.clamp(x, min=self.epsilon))

    def normalize_dist(self, distance):
        """归一化距离值以提高数值稳定性"""
        if self.normalize_distance:
            return distance / self.distance_scale
        else:
            return distance

    def trig_identity_loss(self, sin_values, cos_values):
        sum_of_squares = sin_values**2 + cos_values**2
        trig_loss = torch.mean((sum_of_squares - 1.0) ** 2)
        return trig_loss

    def compute_cartesian_error(self, outputs, targets):
        """
        计算笛卡尔坐标系中的欧几里得误差
        将极坐标输出转换为笛卡尔坐标后计算误差
        """
        # 提取各个维度
        distance_outputs = outputs[:, 0]  # ρ_pred
        sin_outputs = outputs[:, 1]  # sin(θ)_pred
        cos_outputs = outputs[:, 2]  # cos(θ)_pred

        distance_targets = targets[:, 0]  # ρ_real
        sin_targets = targets[:, 1]  # sin(θ)_real
        cos_targets = targets[:, 2]  # cos(θ)_real

        # 计算笛卡尔坐标
        x_pred = distance_outputs * cos_outputs  # ρcosθ
        y_pred = distance_outputs * sin_outputs  # ρsinθ

        x_true = distance_targets * cos_targets
        y_true = distance_targets * sin_targets

        # 计算每个样本的欧几里得距离
        squared_distances = (x_pred - x_true) ** 2 + (y_pred - y_true) ** 2
        euclidean_distances = self.safe_sqrt(squared_distances)

        # 返回平均欧几里得距离
        return torch.mean(euclidean_distances)

    def forward(self, outputs, targets):
        """
        计算损失，支持训练和测试两种模式

        参数:
        - outputs: 模型输出 [batch_size, 3] (ρ, sin(θ), cos(θ))
        - targets: 目标值 [batch_size, 3] (ρ, sin(θ), cos(θ))

        返回:
        - 训练模式: 总MSE损失
        - 评估模式: 笛卡尔坐标系中的欧几里得距离均值
        """
        # 提取各个维度
        distance_outputs = outputs[:, 0]  # ρ_pred
        sin_outputs = outputs[:, 1]  # sin(θ)_pred``
        cos_outputs = outputs[:, 2]  # cos(θ)_pred

        distance_targets = targets[:, 0]  # ρ_real
        sin_targets = targets[:, 1]  # sin(θ)_real
        cos_targets = targets[:, 2]  # cos(θ)_real

        # 评估模式：返回笛卡尔坐标系中的欧几里得误差
        if not self.training:
            return self.compute_cartesian_error(outputs, targets)

        # 训练模式：计算MSE损失

        # 归一化距离（如果启用）
        norm_distance_outputs = self.normalize_dist(distance_outputs)
        norm_distance_targets = self.normalize_dist(distance_targets)

        # 计算各部分Loss
        distance_loss = self.mse_loss(norm_distance_outputs, norm_distance_targets)
        sin_loss = self.mse_loss(sin_outputs, sin_targets)
        cos_loss = self.mse_loss(cos_outputs, cos_targets)

        trig_loss = self.trig_identity_loss(sin_outputs, cos_outputs)

        # 计算总损失 - 无权重区分，简单相加
        total_loss = (
            distance_loss
            + sin_loss
            + cos_loss
            + self.trig_constraint_weight * trig_loss
        )

        return total_loss
