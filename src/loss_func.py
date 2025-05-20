import torch
import torch.nn as nn
import numpy as np


class CombinedPolarLossWithPhysicalConstraint(nn.Module):
    """
    优化版极坐标损失函数，结合了多项改进：
    1. 使用更鲁棒的Huber损失代替MSE，减少异常值影响
    2. 增强物理约束权重和有效性
    3. 提供基于阈值的异常值过滤
    4. 自适应距离归一化
    
    特点:
    - 更强大的异常值处理能力
    - 物理约束权重优化，提高位置预测的物理合理性
    - 自动调整distance_scale，适应不同尺度的数据
    - 更好地平衡距离和角度损失
    - 评估模式直接返回原始欧几里得距离(不归一化)
    """

    def __init__(
        self, 
        is_training=True, 
        distance_weight=1.0, 
        angle_weight=0.7,  # 增加角度权重
        physical_weight=0.3,  # 大幅增加物理约束权重
        outlier_threshold=2.5,  # 异常值阈值 (均值 + 2.5*标准差)
        distance_scale=200.0,  # 更合适的初始距离归一化系数
        adaptive_scale=True,   # 启用自适应归一化
        huber_delta_distance=1.0,  # 距离Huber损失delta参数
        huber_delta_angle=0.2,     # 角度Huber损失delta参数
        huber_delta_physical=0.1,  # 物理约束Huber损失delta参数
        verbose=False
    ):
        super(CombinedPolarLossWithPhysicalConstraint, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.physical_weight = physical_weight
        self.outlier_threshold = outlier_threshold
        self.is_training = is_training
        self.distance_scale = distance_scale
        self.adaptive_scale = adaptive_scale
        self.huber_delta_distance = huber_delta_distance
        self.huber_delta_angle = huber_delta_angle
        self.huber_delta_physical = huber_delta_physical
        self.verbose = verbose
        self.epsilon = 1e-10
        
        # 用于损失归一化的经验值
        self.distance_loss_scale = 1.0
        self.angle_loss_scale = (np.pi**2) / 3.0
        self.physical_loss_scale = 1.0
        
        # 平滑因子，控制归一化尺度的更新速度
        self.smoothing_factor = 0.9
        
        # 自适应scale参数
        self.scale_history = []
        self.scale_update_freq = 20
        
        # 异常值统计
        self.outlier_count = 0
        self.total_points = 0
        
        # 批次计数器
        self.batch_count = 0

    def angular_difference(self, theta_true, theta_pred):
        """计算角度差异，范围在(-π, π]"""
        return (theta_pred - theta_true + np.pi) % (2 * np.pi) - np.pi

    def huber_loss(self, pred, target, delta=None):
        """
        Huber损失：对大误差不像MSE那样敏感
        对于|x| <= delta，返回0.5*x^2
        对于|x| > delta，返回delta*(|x| - 0.5*delta)
        """
        if delta is None:
            delta = self.huber_delta_distance
            
        abs_diff = torch.abs(pred - target)
        quadratic = torch.min(abs_diff, torch.tensor(delta, device=pred.device))
        linear = abs_diff - quadratic
        return torch.mean(0.5 * quadratic.pow(2) + delta * linear)

    def angle_loss(self, theta_true, theta_pred):
        """计算角度的Huber损失，考虑周期性"""
        angle_diff = self.angular_difference(theta_true, theta_pred)
        return self.huber_loss(angle_diff, torch.zeros_like(angle_diff), delta=self.huber_delta_angle)

    def safe_sqrt(self, x):
        """安全的平方根计算，确保输入非负"""
        return torch.sqrt(torch.clamp(x, min=self.epsilon))

    def normalize_distance(self, distance):
        """归一化距离值以提高数值稳定性，支持自适应缩放"""
        if self.adaptive_scale and self.is_training:
            # 记录当前批次的距离范围并更新scale
            with torch.no_grad():
                current_max = torch.max(distance).item()
                self.scale_history.append(current_max)
                
                # 定期更新scale值
                if len(self.scale_history) >= self.scale_update_freq:
                    # 使用近期最大值的中位数作为新的scale基准
                    # 乘以1.5确保大多数值归一化后小于1
                    new_scale = np.median(self.scale_history) * 1.5
                    # 平滑更新
                    self.distance_scale = 0.9 * self.distance_scale + 0.1 * new_scale
                    self.scale_history = []
                    if self.verbose:
                        print(f"更新distance_scale为: {self.distance_scale:.2f}")
        
        # 返回归一化后的距离
        return distance / self.distance_scale
        
    def normalize_loss_to_01(self, loss_value, scale_factor):
        """将损失值归一化到[0,1]范围"""
        normalized = 2.0 / (1.0 + torch.exp(-loss_value / scale_factor)) - 1.0
        return torch.clamp(normalized, min=0.0, max=1.0)

    def update_normalization_scales(self, distance_loss, angle_loss, physical_loss):
        """动态更新归一化尺度因子"""
        if not torch.isnan(distance_loss) and not torch.isinf(distance_loss):
            self.distance_loss_scale = self.smoothing_factor * self.distance_loss_scale + \
                                      (1 - self.smoothing_factor) * max(distance_loss.item(), self.epsilon)
        
        if not torch.isnan(angle_loss) and not torch.isinf(angle_loss):
            self.angle_loss_scale = self.smoothing_factor * self.angle_loss_scale + \
                                   (1 - self.smoothing_factor) * max(angle_loss.item(), self.epsilon)
        
        if not torch.isnan(physical_loss) and not torch.isinf(physical_loss):
            self.physical_loss_scale = self.smoothing_factor * self.physical_loss_scale + \
                                      (1 - self.smoothing_factor) * max(physical_loss.item(), self.epsilon)

    def detect_outliers(self, distance_outputs, angle_outputs, distance_targets, angle_targets):
        """
        检测异常值：计算每个点的误差，并返回掩码表示正常值
        """
        # 转换为笛卡尔坐标
        x_pred = distance_outputs * torch.cos(angle_outputs)
        y_pred = distance_outputs * torch.sin(angle_outputs)
        x_true = distance_targets * torch.cos(angle_targets)
        y_true = distance_targets * torch.sin(angle_targets)
        
        # 计算每个点的欧几里得距离误差
        squared_distances = (x_pred - x_true)**2 + (y_pred - y_true)**2
        euclidean_distances = self.safe_sqrt(squared_distances)
        
        # 计算标准差和均值
        mean_error = torch.mean(euclidean_distances)
        std_error = torch.std(euclidean_distances)
        
        # 使用Z-score识别异常值
        threshold = mean_error + self.outlier_threshold * std_error
        inlier_mask = euclidean_distances <= threshold
        
        # 确保至少保留50%的样本
        if torch.sum(inlier_mask) < 0.5 * len(inlier_mask):
            # 如果过滤太多点，则基于排序选择前50%的点
            _, indices = torch.sort(euclidean_distances)
            keep_count = max(int(0.5 * len(euclidean_distances)), 1)
            keep_indices = indices[:keep_count]
            inlier_mask = torch.zeros_like(inlier_mask)
            inlier_mask[keep_indices] = True
        
        # 更新统计
        self.outlier_count += torch.sum(~inlier_mask).item()
        self.total_points += len(inlier_mask)
        
        if self.verbose and torch.sum(~inlier_mask) > 0:
            outlier_percent = 100 * torch.sum(~inlier_mask).item() / len(inlier_mask)
            print(f"检测到 {torch.sum(~inlier_mask).item()} 个异常值，占比 {outlier_percent:.1f}%")
            print(f"均值={mean_error.item():.2f}，阈值={threshold.item():.2f}")
        
        return inlier_mask

    def compute_euclidean_error(self, distance_outputs, angle_outputs, distance_targets, angle_targets):
        """计算欧几里得误差，用于评估模式"""
        # 转换为笛卡尔坐标
        x_pred = distance_outputs * torch.cos(angle_outputs)
        y_pred = distance_outputs * torch.sin(angle_outputs)
        x_true = distance_targets * torch.cos(angle_targets)
        y_true = distance_targets * torch.sin(angle_targets)
        
        # 计算欧几里得距离
        squared_distances = (x_pred - x_true)**2 + (y_pred - y_true)**2
        euclidean_distances = self.safe_sqrt(squared_distances)
        mean_euclidean_distance = torch.mean(euclidean_distances)
        
        return mean_euclidean_distance, euclidean_distances

    def forward(self, outputs, targets, extra_infos):
        """
        计算优化版综合损失
        
        参数:
        - outputs: 模型输出 [batch_size, 2] (ρ, θ)
        - targets: 目标值 [batch_size, 2] (ρ, θ)
        - extra_infos: 额外信息 [batch_size, 9]
        
        返回:
        - 训练模式: 综合损失值 (归一化到[0,1])
        - 评估模式: 欧几里得距离均值 (原始值，不归一化)
        """
        self.batch_count += 1
        
        # 提取距离和角度维度
        distance_outputs = outputs[:, 0]  # ρ_pred
        distance_targets = targets[:, 0]  # ρ_real
        angle_outputs = outputs[:, 1]  # θ_pred
        angle_targets = targets[:, 1]  # θ_real
        
        # 确保距离为正值
        distance_outputs = torch.abs(distance_outputs) + self.epsilon
        distance_targets = torch.abs(distance_targets) + self.epsilon

        # 评估模式：直接返回欧几里得距离
        if not self.is_training:
            try:
                mean_error, _ = self.compute_euclidean_error(
                    distance_outputs, angle_outputs, distance_targets, angle_targets
                )
                
                # 检查结果
                if torch.isnan(mean_error) or torch.isinf(mean_error):
                    print("警告: 欧几里得距离计算出现NaN或Inf!")
                    return torch.tensor(100.0, device=distance_outputs.device)
                
                # 直接返回原始欧几里得距离，方便解释
                return mean_error
                
            except Exception as e:
                print(f"欧几里得距离计算出错: {e}")
                return torch.tensor(100.0, device=distance_outputs.device)
        
        # 训练模式继续...
        
        # 检测异常值
        inlier_mask = self.detect_outliers(distance_outputs, angle_outputs, distance_targets, angle_targets)

        # 计算距离Loss (使用Huber损失)
        distance_loss = self.huber_loss(
            distance_outputs[inlier_mask], 
            distance_targets[inlier_mask], 
            delta=self.huber_delta_distance
        )

        # 计算角度Loss
        angle_loss = self.angle_loss(angle_targets[inlier_mask], angle_outputs[inlier_mask])

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
            cosine_term_2 = torch.clamp(cosine_term_2, min=-1.0, max=1.0)
            
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
            
            # 使用Huber损失替代MSE，并应用异常值掩码
            physical_loss_21 = self.huber_loss(
                d21_pred[inlier_mask], 
                norm_d21_real[inlier_mask], 
                delta=self.huber_delta_physical
            )
            physical_loss_31 = self.huber_loss(
                d31_pred[inlier_mask], 
                norm_d31_real[inlier_mask], 
                delta=self.huber_delta_physical
            )
            physical_loss_41 = self.huber_loss(
                d41_pred[inlier_mask], 
                norm_d41_real[inlier_mask], 
                delta=self.huber_delta_physical
            )
            
            # 合并物理约束损失
            physical_loss = (physical_loss_21 + physical_loss_31 + physical_loss_41) / 3.0
            
            # 输出诊断信息
            if self.verbose and (self.batch_count % 10 == 0):
                with torch.no_grad():
                    print(f"Batch {self.batch_count} - 物理约束损失: O2: {physical_loss_21.item():.4f}, "
                        f"O3: {physical_loss_31.item():.4f}, O4: {physical_loss_41.item():.4f}")
                    
                    # 检查余弦值是否正常
                    print(f"余弦值范围: O2: [{cosine_term_2.min().item():.4f}, {cosine_term_2.max().item():.4f}], "
                        f"O3: [{cosine_term_3.min().item():.4f}, {cosine_term_3.max().item():.4f}], "
                        f"O4: [{cosine_term_4.min().item():.4f}, {cosine_term_4.max().item():.4f}]")
                    
                    # 检查距离归一化
                    print(f"归一化距离值: 预测: [{norm_distance_outputs.min().item():.4f}, {norm_distance_outputs.max().item():.4f}], "
                          f"真实: [{norm_distance_targets.min().item():.4f}, {norm_distance_targets.max().item():.4f}]")
                    print(f"当前distance_scale: {self.distance_scale:.2f}")
        
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

        # 计算加权综合损失
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
        if self.verbose and (self.batch_count % 10 == 0):
            # 计算当前欧几里得误差，用于监控
            mean_error, _ = self.compute_euclidean_error(
                distance_outputs, angle_outputs, distance_targets, angle_targets
            )
            
            print(f"Batch {self.batch_count} - 原始损失: 距离={distance_loss.item():.4f}, 角度={angle_loss.item():.4f}, "
                f"物理约束={physical_loss.item():.4f}, 欧几里得误差={mean_error.item():.4f}")
            print(f"归一化损失: 距离={norm_distance_loss.item():.4f}, 角度={norm_angle_loss.item():.4f}, "
                f"物理约束={norm_physical_loss.item():.4f}, 总损失={combined_loss.item():.4f}")
            print(f"归一化尺度: 距离={self.distance_loss_scale:.4f}, 角度={self.angle_loss_scale:.4f}, "
                f"物理约束={self.physical_loss_scale:.4f}")
            
            if self.total_points > 0:
                outlier_percent = 100.0 * self.outlier_count / self.total_points
                print(f"异常值统计: 累计检测到 {self.outlier_count}/{self.total_points} 个异常值 ({outlier_percent:.2f}%)")
            
        return combined_loss