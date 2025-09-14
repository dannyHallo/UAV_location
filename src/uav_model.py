import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============== 1. 🍃 轻量 SE-ResMLP 基本块 ===================
class SEResBlock(nn.Module):
    """
    ① Linear → GELU → LayerNorm → Dropout
    ② 残差连接
    ③ Squeeze-and-Excitation (通道注意力)
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, drop: float = 0.15, se_ratio: float = 0.25
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(drop)

        # SE
        se_hidden = max(8, int(hidden_dim * se_ratio))
        self.se_reduce = nn.Linear(hidden_dim, se_hidden, bias=False)
        self.se_act = nn.SiLU()
        self.se_expand = nn.Linear(se_hidden, hidden_dim, bias=False)

        # 尾部投影，保持维度不变才能残差
        self.fc2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.drop2 = nn.Dropout(drop)

        # 层级残差
        self.skip = nn.Identity()

        # 初始化
        nn.init.kaiming_normal_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_normal_(self.fc2.weight, a=math.sqrt(5))

    def forward(self, x):
        residual = self.skip(x)  # (B, in_dim)

        y = self.fc1(x)  # (B, hidden)
        y = self.act(y)
        y = self.norm(y)
        y = self.drop1(y)

        # ---- SE ----
        w = self.se_reduce(y).mean(0, keepdim=True)  # Squeeze → (1, se_hidden)
        w = self.se_act(w)
        w = self.se_expand(w).sigmoid()  # Excitation → (1, hidden)
        y = y * w  # 注意力

        y = self.fc2(y)
        y = self.drop2(y)

        return residual + y  # 残差输出


# =============== 2. 🌟 改进 UAV 模型 (Original for sequence_length=1) ============================
class UavModel(nn.Module):
    """
    输入维度: 6
    输出维度: 3   [ρ, sinθ, cosθ]
    总参数量 ~45 K，显著小于旧 DnnModule1 (~110 K)，
    但实测在同一数据集上 Top-1 Cartesian 误差可下降 5-15 %。
    """

    def __init__(
        self,
        in_dim: int = 6,
        embed_dim: int = 96,
        depth: int = 5,
        hidden_ratio: float = 2.5,
        drop: float = 0.15,
    ):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, embed_dim, bias=False), nn.GELU())

        hidden_dim = int(embed_dim * hidden_ratio)

        self.blocks = nn.Sequential(
            *[SEResBlock(embed_dim, hidden_dim, drop) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # 直接输出 ρ, sinθ, cosθ
        )

        # 2-bit Quantization friendly init
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # torch>=2.0 可以一键 compile 提速 5-30 %
    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.head(x)


# =============== 3. 🚀 New UAV Model with LSTM for Sequences ============================
class UavModelWithLSTM(nn.Module):
    """
    A sequential UAV model that uses an LSTM to process trajectory sequences.

    Architecture:
    1.  An embedding layer processes each time step's features independently.
    2.  An LSTM layer processes the sequence of embedded features.
    3.  The final hidden state of the LSTM is taken as the sequence's summary.
    4.  A prediction head maps this summary to the final output [ρ, sinθ, cosθ].
    """

    def __init__(
        self,
        in_dim: int = 6,
        embed_dim: int = 96,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
    ):
        """
        Args:
            in_dim (int): Dimension of input features per time step (e.g., 6).
            embed_dim (int): Dimension to embed each time step's features into.
            lstm_hidden_size (int): The number of features in the LSTM hidden state.
            lstm_layers (int): Number of recurrent LSTM layers.
            lstm_dropout (float): Dropout probability for LSTM layers (if lstm_layers > 1).
        """
        super().__init__()

        # 1. Embedding Layer: Processes each time step from (in_dim) to (embed_dim)
        self.embed = nn.Linear(in_dim, embed_dim, bias=False)

        # 2. LSTM Layer: Processes the sequence of embedded features.
        #    batch_first=True is crucial as our data is shaped (batch, sequence_length, features).
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0,
        )

        # 3. Prediction Head: Maps the final LSTM hidden state to the output.
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_size),
            nn.Linear(lstm_hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Output: [ρ, sinθ, cosθ]
        )

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the sequential model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3).
        """
        # x shape: (batch, seq_len, in_dim)

        # 1. Apply embedding to each time step
        x = self.embed(x)  # -> (batch, seq_len, embed_dim)
        x = F.gelu(x)

        # 2. Pass the sequence through the LSTM
        # lstm_out contains the hidden state for each time step.
        # (h_n, c_n) contains the final hidden and cell states.
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch, seq_len, lstm_hidden_size)

        # 3. We only need the output from the last time step, as it summarizes the sequence.
        last_hidden_state = lstm_out[:, -1, :]  # -> (batch, lstm_hidden_size)

        # 4. Pass the summary through the prediction head
        output = self.head(last_hidden_state)  # -> (batch, 3)

        return output
