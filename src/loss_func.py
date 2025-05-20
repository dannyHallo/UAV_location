import torch
import torch.nn as nn
import numpy as np


class CombinedPolarLossWithPhysicalConstraint(nn.Module):
    """
    优化版综合极坐标损失函数，同时计算:
    1. 距离(ρ)的MSE损失
    2. 角度(θ)的MSE损失（考虑角度的周期性）
    3. 基于余弦定理的物理约束损失
    
    特点:
    - 所有损失分量归一化到[0,1]范围
    - 距离值归一化以提高数值稳定性
    - 优化的物理约束计算方式
    - 详细的诊断信息输出
    """

    def __init__(
        self, 
        is_training=True, 
        distance_weight=1.0, 
        angle_weight=0.5, 
        physical_weight=0.01,
        distance_scale=1000.0,  # 距离归一化尺度
        verbose=False  # 是否输出详细诊断信息
    ):
        super(CombinedPolarLossWithPhysicalConstraint, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.physical_weight = physical_weight
        self.is_training = is_training
        self.distance_scale = distance_scale
        self.verbose = verbose
        self.mse = nn.MSELoss(reduction="mean")
        self.epsilon = 1e-10  # 数值稳定性常数
        
        # 用于损失归一化的经验值
        self.distance_loss_scale = 1.0
        self.angle_loss_scale = (np.pi**2) / 3.0  # π²/3是角度差异的理论最大MSE
        self.physical_loss_scale = 1.0
        
        # 保存上一批次的损失值，用于动态调整归一化因子
        self.last_distance_loss = 1.0
        self.last_angle_loss = 1.0
        self.last_physical_loss = 1.0
        
        # 平滑因子，控制归一化尺度的更新速度
        self.smoothing_factor = 0.9

    def angular_difference(self, theta_true, theta_pred):
        """
        计算角度差异，返回范围在(-π, π]的最小有符号差值
        """
        return (theta_pred - theta_true + np.pi) % (2 * np.pi) - np.pi

    def mse_angle_loss(self, theta_true, theta_pred):
        """
        计算角度的MSE损失，考虑角度的周期性
        """
        angle_diff = self.angular_difference(theta_true, theta_pred)
        return self.mse(angle_diff, torch.zeros_like(angle_diff))

    def safe_sqrt(self, x):
        """
        安全的平方根计算，确保输入非负
        """
        return torch.sqrt(torch.clamp(x, min=self.epsilon))

    def normalize_distance(self, distance):
        """
        归一化距离值以提高数值稳定性
        """
        return distance / self.distance_scale
        
    def normalize_loss_to_01(self, loss_value, scale_factor):
        """
        将损失值归一化到[0,1]范围
        """
        # 使用2/(1+e^(-x))-1将[0,∞)映射到[0,1)
        normalized = 2.0 / (1.0 + torch.exp(-loss_value / scale_factor)) - 1.0
        
        # 确保值在[0,1]范围内
        normalized = torch.clamp(normalized, min=0.0, max=1.0)
        return normalized

    def update_normalization_scales(self, distance_loss, angle_loss, physical_loss):
        """
        动态更新归一化尺度因子
        """
        if not torch.isnan(distance_loss) and not torch.isinf(distance_loss):
            self.distance_loss_scale = self.smoothing_factor * self.distance_loss_scale + \
                                      (1 - self.smoothing_factor) * max(distance_loss.item(), self.epsilon)
            self.last_distance_loss = distance_loss.item()
        
        if not torch.isnan(angle_loss) and not torch.isinf(angle_loss):
            self.angle_loss_scale = self.smoothing_factor * self.angle_loss_scale + \
                                   (1 - self.smoothing_factor) * max(angle_loss.item(), self.epsilon)
            self.last_angle_loss = angle_loss.item()
        
        if not torch.isnan(physical_loss) and not torch.isinf(physical_loss):
            self.physical_loss_scale = self.smoothing_factor * self.physical_loss_scale + \
                                      (1 - self.smoothing_factor) * max(physical_loss.item(), self.epsilon)
            self.last_physical_loss = physical_loss.item()

    def forward(self, outputs, targets, extra_infos):
        """
        计算综合损失

        参数:
        outputs: 模型输出，形状为 [batch_size, 2]，其中第一列是距离(ρ_pred)，第二列是角度(θ_pred)
        targets: 目标值，形状为 [batch_size, 2]，其中第一列是距离(ρ_real)，第二列是角度(θ_real)
        extra_infos: 包含三个接收机距离和相对位置信息，形状为 [batch_size, 9]

        返回:
        训练模式: 综合损失值 (归一化到[0,1])
        评估模式: 欧几里得距离的均值 (归一化到[0,1])
        """
        # 提取距离和角度维度
        distance_outputs = outputs[:, 0]  # ρ_pred
        distance_targets = targets[:, 0]  # ρ_real
        angle_outputs = outputs[:, 1]  # θ_pred
        angle_targets = targets[:, 1]  # θ_real
        
        # 确保距离为正值
        distance_outputs = torch.abs(distance_outputs) + self.epsilon
        distance_targets = torch.abs(distance_targets) + self.epsilon

        # 计算原始距离MSE损失
        distance_loss = self.mse(distance_outputs, distance_targets)

        # 计算角度MSE损失
        angle_loss = self.mse_angle_loss(angle_targets, angle_outputs)

        # 提取extra_infos中的信息
        d21_real = extra_infos[:, 0]  # O2到UAV的实际距离
        d31_real = extra_infos[:, 1]  # O3到UAV的实际距离
        d41_real = extra_infos[:, 2]  # O4到UAV的实际距离
        
        rho2 = extra_infos[:, 3]  # O2到O1的距离 ρ2
        theta2 = extra_infos[:, 4]  # O2相对于O1的角度 θ2
        rho3 = extra_infos[:, 5]  # O3到O1的距离 ρ3
        theta3 = extra_infos[:, 6]  # O3相对于O1的角度 θ3
        rho4 = extra_infos[:, 7]  # O4到O1的距离 ρ4
        theta4 = extra_infos[:, 8]  # O4相对于O1的角度 θ4

        # 距离归一化处理
        norm_distance_outputs = self.normalize_distance(distance_outputs)
        norm_distance_targets = self.normalize_distance(distance_targets)
        norm_d21_real = self.normalize_distance(d21_real)
        norm_d31_real = self.normalize_distance(d31_real)
        norm_d41_real = self.normalize_distance(d41_real)
        norm_rho2 = self.normalize_distance(rho2)
        norm_rho3 = self.normalize_distance(rho3)
        norm_rho4 = self.normalize_distance(rho4)
        
        # 计算物理约束损失
        try:
            # 计算预测距离（基于余弦定理）
            cosine_term_2 = torch.cos(self.angular_difference(theta2, angle_outputs))
            cosine_term_2 = torch.clamp(cosine_term_2, min=-1.0, max=1.0)  # 确保余弦值在[-1,1]范围内
            
            d21_pred_squared = (
                norm_distance_outputs**2 + 
                norm_rho2**2 - 
                2 * norm_distance_outputs * norm_rho2 * cosine_term_2
            )
            d21_pred = self.safe_sqrt(d21_pred_squared)
            
            cosine_term_3 = torch.cos(self.angular_difference(theta3, angle_outputs))
            cosine_term_3 = torch.clamp(cosine_term_3, min=-1.0, max=1.0)
            
            d31_pred_squared = (
                norm_distance_outputs**2 + 
                norm_rho3**2 - 
                2 * norm_distance_outputs * norm_rho3 * cosine_term_3
            )
            d31_pred = self.safe_sqrt(d31_pred_squared)
            
            cosine_term_4 = torch.cos(self.angular_difference(theta4, angle_outputs))
            cosine_term_4 = torch.clamp(cosine_term_4, min=-1.0, max=1.0)
            
            d41_pred_squared = (
                norm_distance_outputs**2 + 
                norm_rho4**2 - 
                2 * norm_distance_outputs * norm_rho4 * cosine_term_4
            )
            d41_pred = self.safe_sqrt(d41_pred_squared)
            
            # 计算物理约束损失（预测距离与实际距离的MSE）
            physical_loss_21 = self.mse(d21_pred, norm_d21_real)
            physical_loss_31 = self.mse(d31_pred, norm_d31_real)
            physical_loss_41 = self.mse(d41_pred, norm_d41_real)
            
            # 合并物理约束损失
            physical_loss = (physical_loss_21 + physical_loss_31 + physical_loss_41) / 3.0
            
            # 输出诊断信息
            if self.verbose and self.is_training:
                with torch.no_grad():
                    print(f"物理约束损失分量: O2: {physical_loss_21.item():.4f}, "
                          f"O3: {physical_loss_31.item():.4f}, O4: {physical_loss_41.item():.4f}")
                    
                    # 检查余弦值是否正常
                    print(f"余弦值范围: O2: [{cosine_term_2.min().item():.4f}, {cosine_term_2.max().item():.4f}], "
                          f"O3: [{cosine_term_3.min().item():.4f}, {cosine_term_3.max().item():.4f}], "
                          f"O4: [{cosine_term_4.min().item():.4f}, {cosine_term_4.max().item():.4f}]")
                    
                    # 检查距离值
                    print(f"距离(ρ) - Real: [{norm_distance_targets.min().item():.4f}, {norm_distance_targets.max().item():.4f}], "
                          f"Predicate: [{norm_distance_outputs.min().item():.4f}, {norm_distance_outputs.max().item():.4f}]")
        
        except Exception as e:
            print(f"物理约束计算出错: {e}")
            # 如果物理约束计算出错，使用一个小的常数作为物理约束损失
            physical_loss = torch.tensor(0.1, device=distance_outputs.device)

        # 更新归一化尺度
        self.update_normalization_scales(distance_loss, angle_loss, physical_loss)
        
        # 归一化各个损失组件到[0,1]范围
        norm_distance_loss = self.normalize_loss_to_01(distance_loss, self.distance_loss_scale)
        norm_angle_loss = self.normalize_loss_to_01(angle_loss, self.angle_loss_scale)
        norm_physical_loss = self.normalize_loss_to_01(physical_loss, self.physical_loss_scale)

        # 训练模式：返回加权综合损失
        if self.is_training:
            combined_loss = (
                self.distance_weight * norm_distance_loss
                + self.angle_weight * norm_angle_loss
                + self.physical_weight * norm_physical_loss
            )
            
            # 确保组合损失也在[0,1]范围内
            total_weight = self.distance_weight + self.angle_weight + self.physical_weight
            combined_loss = combined_loss / total_weight if total_weight > 0 else combined_loss
            
            # 检查损失值是否合理
            if torch.isnan(combined_loss) or torch.isinf(combined_loss):
                print("警告: 损失函数包含NaN或Inf值！")
                print(f"归一化前 - 距离损失: {distance_loss.item()}, 角度损失: {angle_loss.item()}, 物理约束损失: {physical_loss.item()}")
                print(f"归一化后 - 距离损失: {norm_distance_loss.item()}, 角度损失: {norm_angle_loss.item()}, 物理约束损失: {norm_physical_loss.item()}")
                
                # 使用非NaN部分构建损失
                valid_loss = 0.0
                valid_weight = 0.0
                
                if not torch.isnan(norm_distance_loss) and not torch.isinf(norm_distance_loss):
                    valid_loss += self.distance_weight * norm_distance_loss
                    valid_weight += self.distance_weight
                
                if not torch.isnan(norm_angle_loss) and not torch.isinf(norm_angle_loss):
                    valid_loss += self.angle_weight * norm_angle_loss
                    valid_weight += self.angle_weight
                
                if not torch.isnan(norm_physical_loss) and not torch.isinf(norm_physical_loss):
                    valid_loss += self.physical_weight * norm_physical_loss
                    valid_weight += self.physical_weight
                
                if valid_weight > 0:
                    combined_loss = valid_loss / valid_weight
                else:
                    return torch.tensor(0.5, device=distance_loss.device, requires_grad=True)
            
            # 详细诊断输出
            if self.verbose:
                print(f"原始损失: 距离={distance_loss.item():.4f}, 角度={angle_loss.item():.4f}, "
                      f"物理约束={physical_loss.item():.4f}")
                print(f"归一化损失[0-1]: 距离={norm_distance_loss.item():.4f}, 角度={norm_angle_loss.item():.4f}, "
                      f"物理约束={norm_physical_loss.item():.4f}, 总损失={combined_loss.item():.4f}")
                print(f"归一化尺度: 距离={self.distance_loss_scale:.4f}, 角度={self.angle_loss_scale:.4f}, "
                      f"物理约束={self.physical_loss_scale:.4f}")
                
            return combined_loss
        else:
            # 评估模式：将极坐标转换为笛卡尔坐标，计算欧几里得距离
            try:
                # 预测位置的笛卡尔坐标
                x_pred = distance_outputs * torch.cos(angle_outputs)
                y_pred = distance_outputs * torch.sin(angle_outputs)

                # 真实位置的笛卡尔坐标
                x_real = distance_targets * torch.cos(angle_targets)
                y_real = distance_targets * torch.sin(angle_targets)

                # 计算欧几里得距离
                squared_distances = (x_pred - x_real)**2 + (y_pred - y_real)**2
                euclidean_distances = self.safe_sqrt(squared_distances)
                mean_euclidean_distance = torch.mean(euclidean_distances)
                
                # 归一化欧几里得距离到[0,1]范围
                normalized_distance = self.normalize_loss_to_01(mean_euclidean_distance, self.distance_scale)
                
                # 检查结果
                if torch.isnan(normalized_distance) or torch.isinf(normalized_distance):
                    print("警告: 欧几里得距离计算出现NaN或Inf!")
                    return torch.tensor(0.5, device=distance_outputs.device)
                
                return normalized_distance
                
            except Exception as e:
                print(f"欧几里得距离计算出错: {e}")
                return torch.tensor(0.5, device=distance_outputs.device)