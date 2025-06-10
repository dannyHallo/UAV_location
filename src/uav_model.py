import torch.nn as nn
from src.kan import KAN
import torch
import torch.nn.functional as F


class KANModel(nn.Module):
    def __init__(self):
        super(KANModel, self).__init__()
        # 假设输入是 6 维，输出是 2 维，hidden layers 依然根据原模型的架构
        self.model = KAN(width=[12, 10, 8, 6, 4], grid=300, k=3, seed=42)

    def forward(self, x):
        return self.model(x)


class DnnModule1(nn.Module):
    def __init__(self):
        super(DnnModule1, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)
        self.fc7 = nn.Linear(32, 3)
        # Activation function can be assigned as a member variable
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.activation(self.bn4(self.fc4(x)))
        x = self.activation(self.bn5(self.fc5(x)))
        x = self.activation(self.bn6(self.fc6(x)))
        x = self.fc7(x)  # Usually no batch norm just before the final layer
        return x

class EnhancedDualExpertModel(nn.Module):
    """增强型双专家模型，专门处理轨迹突起问题"""
    def __init__(self, input_dim=6, dropout_rate=0.15):
        super(EnhancedDualExpertModel, self).__init__()
        
        # 空间感知特征增强器
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.BatchNorm1d(input_dim*2),
            nn.LeakyReLU(0.1),
        )
        
        # 空间位置编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.Tanh(),
            nn.Linear(16, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Tanh(),
        )
        
        # 计算合并后的特征维度
        combined_dim = input_dim + input_dim*2 + input_dim  # 原始 + 变换 + 空间编码
        
        # 特征合并 - 修正输入输出维度
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_dim, input_dim*3),  # 修正输入维度
            nn.BatchNorm1d(input_dim*3),
            nn.LeakyReLU(0.1),
        )
        
        enhanced_dim = input_dim*3
        
        # 专家选择网络
        self.expert_selector = nn.Sequential(
            nn.Linear(enhanced_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        
        # 直线段专家网络
        self.straight_expert = nn.Sequential(
            nn.Linear(enhanced_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )
        
        # 曲线段专家网络
        self.curve_expert = nn.Sequential(
            nn.Linear(enhanced_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.1),
        )
        
        # 集成层
        self.ensemble = nn.Sequential(
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
        )
        
        # 输出层
        self.distance_head = nn.Linear(32, 1)
        self.angle_head = nn.Linear(32, 2)
        
        # 异常检测与平滑层 - 修正输入维度
        self.outlier_detector = nn.Sequential(
            nn.Linear(32 + 3, 32),  # 集成特征 + 基本输出
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        self.smoothing_net = nn.Sequential(
            nn.Linear(32 + 3, 32),  # 集成特征 + 基本输出
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 3),
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # 特征增强 - 明确计算维度
        base_features = self.feature_transform(x)  # 6 → 12
        spatial_encoding = self.spatial_encoder(x)  # 6 → 6
        
        # 连接所有特征 - 明确维度
        combined = torch.cat([x, base_features, spatial_encoding], dim=1)  # 6 + 12 + 6 = 24
        enhanced_x = self.feature_combiner(combined)  # 24 → 18
        
        # 专家选择
        expert_weights = self.expert_selector(enhanced_x)
        
        # 各专家的输出
        straight_out = self.straight_expert(enhanced_x)
        curve_out = self.curve_expert(enhanced_x)
        
        # 按权重组合专家输出
        weighted_straight = straight_out * expert_weights[:, 0:1]
        weighted_curve = curve_out * expert_weights[:, 1:2]
        
        # 组合专家输出
        combined_features = weighted_straight + weighted_curve
        
        # 集成处理
        ensemble_features = self.ensemble(combined_features)
        
        # 基本输出
        distance = torch.abs(self.distance_head(ensemble_features))
        angle_components = self.angle_head(ensemble_features)
        basic_output = torch.cat([distance, angle_components], dim=1)
        
        # 异常检测与平滑
        combined_for_outlier = torch.cat([ensemble_features, basic_output], dim=1)
        outlier_score = self.outlier_detector(combined_for_outlier)
        smooth_correction = self.smoothing_net(combined_for_outlier)
        
        # 应用平滑修正
        final_output = basic_output * (1 - outlier_score) + smooth_correction * outlier_score
        
        return final_output


class LSTMModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate
        )
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: [BS, seq_len, hidden_dim]

        # Use the output from the last timestep
        final_output = lstm_out[:, -1, :]  # shape: [BS, hidden_dim]

        # Pass the last outputs through a final fully connected layer to get the desired output_dim
        final_output = self.final_fc(final_output)  # shape: [BS, output_dim]

        return final_output


class TransformerModule(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, nhead, num_layers, dropout_rate):
        super(TransformerModule, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Reshape input to (seq_len, batch_size, input_dim)
        # x = x.permute(1, 0, 2)
        x = self.input_fc(x)
        x = self.transformer_encoder(x)
        x = self.output_fc(x)
        # Take the last time step output
        x = x[-1, :, :]  # shape: [batch_size, output_dim]
        return x


class UavModel(nn.Module):
    def __init__(self):
        super(UavModel, self).__init__()
        # self.kan = KANModel()
        self.dnn1 = DnnModule1()
        # self.dual_positioning = EnhancedDualExpertModel()
        # self.transformer = TransformerModule(input_dim=2,output_dim=2,d_model=128,nhead=8,num_layers=2,dropout_rate=0.2)
        # self.lstm = LSTMModule(input_dim=6, output_dim=2,
        #                        hidden_dim=128, num_layers=2, dropout_rate=0.2)

        # self.dnn2 = DnnModule2(dropout_rate=0.35)

    def forward(self, x):
        batch_size, features_len = x.size()

        # reshape input to discard x temporarily for the first module
        # x = x.view(-1, features_len)

        # x = self.kan(x)
        x = self.dnn1(x)
        # x = self.dual_positioning(x)

        # x = x.view(seq_len,batch_size,2)
        # x = self.transformer(x)
        # print(x.shape)

        # reshape x to original shape (restoring seq)
        # x = x.view(batch_size, seq_len, 6)
        # x = self.lstm(x)

        # lstm already returns the last hidden state of the sequences, no need to reshape

        # print(x.shape)

        # print(x.shape)
        # x = x.permute(1, 0, 2)

        # x = self.dnn2(x)

        return x
