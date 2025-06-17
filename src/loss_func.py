import torch
import torch.nn as nn


class CartesianTestLoss(nn.Module):
    """
    混合坐标系损失函数：
    - 训练模式(model.train())：处理 [距离, sin(θ), cos(θ)] 格式输入，使用MSE损失
    并添加sin²(θ) + cos²(θ) = 1 的三角约束
    - 测试模式(model.eval())：将输出转换为笛卡尔坐标 [ρcosθ, ρsinθ] 计算欧几里得误差

    使用当前batch的最大值进行距离归一化
    """

    def __init__(
        self,
        normalize_distance=True,  # 是否对距离进行归一化
        trig_constraint_weight=0.5,
    ):
        super(CartesianTestLoss, self).__init__()
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

    def normalize_dist(self, distance_pred, distance_target):
        """
        使用当前batch的最大值进行距离归一化

        Args:
            distance_pred: 预测的距离值
            distance_target: 真实的距离值

        Returns:
            归一化后的预测值和真实值，范围在[0,1]
        """
        if not self.normalize_distance:
            return distance_pred, distance_target

        # 合并预测值和真实值以计算全局最大值
        combined_distances = torch.cat([distance_pred, distance_target])
        max_distance = torch.max(combined_distances) + self.epsilon

        # 使用最大值进行归一化
        norm_pred = distance_pred / max_distance
        norm_target = distance_target / max_distance

        return norm_pred, norm_target

    def trig_identity_loss(self, sin_values, cos_values):
        """三角恒等式约束损失: sin²(θ) + cos²(θ) = 1"""
        sum_of_squares = sin_values**2 + cos_values**2
        trig_loss = torch.mean((sum_of_squares - 1.0) ** 2)
        return trig_loss

    def _compute_cartesian_error(self, outputs, targets):
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
        distance_outputs = outputs[:, 0]  # ρ_pred
        sin_outputs = outputs[:, 1]  # sin(θ)_pred
        cos_outputs = outputs[:, 2]  # cos(θ)_pred

        distance_targets = targets[:, 0]  # ρ_real
        sin_targets = targets[:, 1]  # sin(θ)_real
        cos_targets = targets[:, 2]  # cos(θ)_real

        # 测试模式：直接计算笛卡尔误差
        # if not self.training:
        #     return self._compute_cartesian_error(outputs, targets)

        # 使用batch最大值归一化距离
        norm_distance_outputs, norm_distance_targets = self.normalize_dist(
            distance_outputs, distance_targets
        )

        # 计算各部分Loss
        distance_loss = self.mse_loss(norm_distance_outputs, norm_distance_targets)
        sin_loss = self.mse_loss(sin_outputs, sin_targets)
        cos_loss = self.mse_loss(cos_outputs, cos_targets)

        # 三角恒等式约束
        trig_loss = self.trig_identity_loss(sin_outputs, cos_outputs)

        # 计算总损失
        total_loss = (
            distance_loss
            + sin_loss
            + cos_loss
            + self.trig_constraint_weight * trig_loss
        )

        return total_loss


# 使用示例
def create_loss_function():
    """创建使用batch最大值归一化的损失函数"""
    return CartesianTestLoss(normalize_distance=True, trig_constraint_weight=0.5)


# 测试函数
def test_max_normalization():
    """测试batch最大值归一化的效果"""
    # 创建测试数据
    batch_size = 32
    outputs = torch.randn(batch_size, 3)
    outputs[:, 0] = torch.abs(outputs[:, 0]) * 100 + 10  # 距离值: [10, 110]
    targets = torch.randn(batch_size, 3)
    targets[:, 0] = torch.abs(targets[:, 0]) * 80 + 20  # 距离值: [20, 100]

    loss_fn = create_loss_function()

    # 获取归一化前的距离
    dist_pred = outputs[:, 0]
    dist_target = targets[:, 0]
    print(f"归一化前 - 预测距离范围: [{dist_pred.min():.2f}, {dist_pred.max():.2f}]")
    print(
        f"归一化前 - 真实距离范围: [{dist_target.min():.2f}, {dist_target.max():.2f}]"
    )

    # 手动测试归一化
    norm_pred, norm_target = loss_fn.normalize_dist(dist_pred, dist_target)
    print(f"归一化后 - 预测距离范围: [{norm_pred.min():.3f}, {norm_pred.max():.3f}]")
    print(
        f"归一化后 - 真实距离范围: [{norm_target.min():.3f}, {norm_target.max():.3f}]"
    )

    # 计算损失
    loss = loss_fn(outputs, targets)
    print(f"总损失: {loss.item():.6f}")


if __name__ == "__main__":
    test_max_normalization()
