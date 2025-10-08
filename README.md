# 无人机定位系统 (UAV Location System)

基于物理信息神经网络(PINN)和多普勒频移技术的无人机轨迹定位系统，结合物理约束和深度学习进行高精度位置预测。

## 项目概述

本项目实现了一个创新的无人机定位系统，通过分析无人机飞行轨迹中的相位差和多普勒频移信息，使用物理信息神经网络(PINN)模型预测无人机的位置坐标。系统采用 **Huber-IRLS + Tikhonov 正则化 + Gauss-Newton 单步迭代** 的鲁棒物理层，结合 SE-ResMLP 深度学习残差校正，实现了厘米级到毫米级的定位精度。

## 主要特性

- 🚁 **多轨迹生成**：支持生成大量无人机飞行轨迹用于训练
- 📡 **相位差与多普勒分析**：基于相位差 (φ) 和多普勒频移 (fD) 信息进行位置预测
- 🧠 **鲁棒物理信息神经网络**：Huber-IRLS + Tikhonov 正则化的鲁棒 Gauss-Newton 物理层
- 🎯 **SE-ResMLP 残差网络**：Squeeze-and-Excitation + 残差连接的深度学习校正层
- ⚡ **并行处理**：多进程并行生成训练数据
- 📊 **交互式可视化**：Tkinter GUI 界面，支持轨迹生成、模型加载和预测可视化
- 🔧 **模块化设计**：清晰的代码结构，易于扩展和维护
- 🎭 **时序建模支持**：可选 LSTM 层进行时序信息融合和平滑
- 🏆 **高精度定位**：厘米级粗解 + 毫米级精修的两阶段定位

## 系统架构

```
无人机定位系统 (鲁棒 PINN 架构)
├── 物理显式层 (LeastSquaresLayer)
│   ├── Huber-IRLS 鲁棒加权
│   ├── Tikhonov 正则化
│   ├── Gauss-Newton 单步迭代
│   └── 输出：厘米级粗位移 Δp_lin
├── 深度学习层
│   ├── SE-ResMLP 残差网络
│   │   ├── SEResBlock (SE注意力 + 残差连接)
│   │   ├── LayerNorm + GELU 激活
│   │   └── Dropout 正则化
│   ├── LSTM 时序建模 (可选)
│   └── 输出：毫米级位置校正 Δp_corr
├── 数据生成模块
│   ├── trajectory_generator.py - 轨迹生成器
│   ├── detecting_region_info_generator.py - 检测区域生成
│   ├── doppler_info.py - 多普勒信息计算
│   ├── get_phi_info.py - 相位差计算
│   ├── w_and_doppler_generator.py - W和多普勒生成
│   └── features_and_labels_generator.py - 特征标签提取
└── 训练与评估
    ├── stage_1_dataset_gen.py - 数据集生成
    ├── lab.ipynb - PINN 模型训练
    ├── visualization.py - 交互式可视化工具
    └── filter.py - 平滑滤波器
```

## PINN模型架构

### 核心设计理念

系统采用两阶段鲁棒定位策略：
1. **物理粗解 (Robust Gauss-Newton)**：使用 Huber-IRLS + Tikhonov 正则化的鲁棒最小二乘法获得厘米级粗位置
2. **深度学习精修 (SE-ResMLP)**：使用 SE-ResMLP 残差网络学习毫米级校正

### 模型组件详解

#### 1. 物理显式层 (LeastSquaresLayer)
**输入**：
- `p_prev`: (B, Dim) - 上一时刻 UAV 坐标
- `feats`: (B, 6) - [φ₁, φ₂, φ₃, fD₁, fD₂, fD₃]

**常量**：
- `T`: (Dim,) - 发射机坐标
- `R`: (3, Dim) - 三只接收机坐标

**核心算法**：
1. **Huber 鲁棒权重**：
   - w = 1，当 |r| ≤ δ
   - w = δ / (|r| + ε)，当 |r| > δ

2. **正则化求解**：
   - 构建 Jacobian 矩阵 J = u_AT + u_AR
   - Tikhonov 正则化：(JᵀWJ + αI)⁻¹ JᵀWy
   - Cholesky 分解求解线性系统

**输出**：Δp_lin (B, Dim) - 厘米级粗位移

#### 2. SE-ResMLP 残差网络
**SEResBlock 结构**：
```
输入 x
  ↓
Linear(in_dim → hidden_dim)
  ↓
GELU + LayerNorm + Dropout
  ↓
SE模块: GlobalAvg → Linear → SiLU → Linear → Sigmoid
  ↓
通道重新加权
  ↓
Linear(hidden_dim → in_dim) + Dropout
  ↓
残差连接 (x + 输出)
```

**ResidualMLP**：
- 输入维度：6 + Dim (特征 + 物理粗解)
- 嵌入维度：64
- 深度：4 层 SEResBlock
- 隐藏层比例：2.5
- Dropout：0.1

**输出**：Δp_corr (B, Dim) - 毫米级位置校正

#### 3. 时序模型 (PinUavSeqModel)
**架构**：
1. 物理层逐帧计算 Δp_lin
2. 拼接特征和粗解：[feats_t, Δp_lin_t]
3. 线性嵌入到 96 维
4. LSTM (hidden=128, layers=2) 进行时序平滑
5. 线性层输出 Δp_corr

**输入**：(B, S, 6) - 批次 × 序列长度 × 特征维度
**输出**：(B, S, Dim) - 批次 × 序列长度 × 位置维度

**适用场景**：sequence_length > 1

## 环境配置

### 系统要求
- Python 3.9+
- PyTorch 1.12.1+ (CUDA 11.3 或更高版本推荐)
- CUDA 11.3+ (GPU 加速推荐)
- 其他依赖：见 requirements.txt

### 安装步骤

```bash
# 1. 创建虚拟环境
conda create -n "UAV_location" python=3.9.19 -y
conda activate UAV_location

# 2. 查看 CUDA 版本 (Linux)
nvcc --version

# 3. 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.3 示例：
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 4. 安装其他依赖
pip install -r requirements.txt
```

**依赖清单** (requirements.txt)：
- matplotlib - 绘图和可视化
- numpy<2 - 数值计算
- scikit_learn - 机器学习工具
- ipykernel - Jupyter notebook 支持
- sympy - 符号数学
- PyYAML - 配置文件解析
- tqdm - 进度条
- imageio - 图像处理
- pandas - 数据处理
- shapely - 几何计算
- ipywidgets - 交互式组件
- scipy - 科学计算 (用于平滑滤波)

## 快速开始

### 1. 生成训练数据集

```bash
python stage_1_dataset_gen.py
```

这将生成训练集和测试集：
- `dataset/stage_1_dataset_train.pt` - 训练集
- `dataset/stage_1_dataset_test.pt` - 测试集

**数据集生成过程**：
1. 生成检测区域配置 (detecting_region_info)
2. 多进程并行生成飞行轨迹
3. 计算相位差 (φ) 和多普勒频移 (fD)
4. 提取特征向量和标签
5. 保存为 PyTorch 张量格式

**数据集统计**：
- 检测区域数：1
- 每区域轨迹数：20,000
- 训练/测试比例：85% / 15%
- 序列长度：5 帧
- 时间间隔：0.01s

### 2. 训练 PINN 模型

```bash
# 启动 Jupyter Notebook
jupyter notebook lab.ipynb
```

在 notebook 中：
1. 加载数据集
2. 初始化 PinUavModel 或 PinUavSeqModel
3. 配置训练参数
4. 运行训练循环
5. 保存模型到 `models/` 目录

**训练超参数** (config.py)：
- Epochs: 500
- Batch size: 4096
- Learning rate: 2e-4
- Optimizer: Adam
- 损失函数：CartesianTestLoss (MSE + 三角约束)

### 3. 可视化与预测

```bash
# 启动交互式可视化工具
python visualization.py
```

**功能特性**：
- **生成轨迹**：输入 seed 和轨迹数量，生成测试轨迹
- **加载模型**：从 `models/` 目录选择训练好的模型
- **预测**：点击 "Predict" 按钮进行位置预测
- **可视化对比**：真实轨迹 vs 预测轨迹
- **自适应平滑**：使用 Savitzky-Golay 滤波平滑预测结果

**GUI 界面**：
- Seed 输入：控制随机轨迹生成
- Lines 输入：生成的轨迹数量
- SeqLen 输入：序列长度 (1=单帧，>1=时序)
- Trajectory Selector：选择要可视化的轨迹
- Matplotlib 画布：实时显示轨迹和预测结果

## 项目结构

```
UAV_location/
├── src/                                # 源代码目录
│   ├── config.py                       # 全局配置参数
│   ├── uav_model.py                   # 鲁棒 PINN 模型实现
│   │   ├── LeastSquaresLayer          # 物理显式层 (Huber-IRLS + Tikhonov + GN)
│   │   ├── SEResBlock                 # SE 注意力残差块
│   │   ├── ResidualMLP                # 残差 MLP 网络
│   │   ├── PinUavModel                # 单帧 PINN 模型
│   │   └── PinUavSeqModel             # 时序 PINN 模型 (带 LSTM)
│   ├── trajectory_dataset.py          # PyTorch Dataset 类
│   ├── traj_data_loader.py            # 数据加载器
│   ├── loss_func.py                   # 损失函数 (CartesianTestLoss + 约束)
│   ├── trajectory_generator.py        # 轨迹生成器
│   ├── detecting_region_info.py       # 检测区域信息类
│   ├── detecting_region_info_generator.py  # 检测区域生成器
│   ├── doppler_info.py                # 多普勒信息计算
│   ├── get_phi_info.py                # 相位差计算
│   ├── w_and_doppler_generator.py     # W 和多普勒生成
│   ├── features_and_labels_generator.py  # 特征标签提取
│   ├── extra_info_generator.py        # 额外信息生成
│   ├── coord_dataset.py               # 坐标数据集
│   ├── line_gen_util.py               # 轨迹生成工具
│   ├── seed_utils.py                  # 随机种子工具
│   ├── debug_utils.py                 # 调试工具
│   └── kan/                           # KAN (Kolmogorov-Arnold Network) 实验模块
├── dataset/                            # 数据集目录
│   ├── stage_1_dataset_train.pt       # 训练集
│   └── stage_1_dataset_test.pt        # 测试集
├── models/                             # 训练好的模型
├── results/                            # 实验结果
├── graphs/                             # 项目图表
├── stage_1_dataset_gen.py             # 数据集生成主脚本
├── lab.ipynb                          # PINN 模型训练 notebook
├── visualization.py                   # 交互式 Tkinter 可视化工具
├── filter.py                          # 平滑滤波器 (Savitzky-Golay 自适应)
├── cal.py                             # 计算工具
├── demo_straight_lines.py             # 直线轨迹演示
├── straight_line_trajectory_generator.py  # 直线轨迹生成器
├── straight_line_visualizer.py        # 直线轨迹可视化
├── test_straight_lines.py            # 直线轨迹测试
├── test_visualization.py             # 可视化测试
├── test_loss.py                       # 损失函数测试
├── check_dependencies.py              # 依赖检查脚本
├── uav_model_vis.html                 # HTML 可视化页面
├── requirements.txt                   # Python 依赖列表
├── repomix.config.json                # Repomix 配置
└── README.md                          # 项目文档
```

## 模型使用指南

### 1. 单帧模型 (PinUavModel)

**适用场景**：sequence_length = 1

**前向传播**：
```python
p_hat = model(feats, p_prev)
# feats: (B, 6) - [φ₁, φ₂, φ₃, fD₁, fD₂, fD₃]
# p_prev: (B, 2) - 上一时刻位置 (可选，默认为零)
# p_hat: (B, 2) - 当前位置预测
```

**计算流程**：
1. 物理层：`Δp_lin = LeastSquaresLayer(p_prev, feats)`
2. 拼接特征：`inp = [feats, Δp_lin]`
3. 残差网络：`Δp_corr = ResidualMLP(inp)`
4. 最终位置：`p_hat = p_prev + Δp_lin + Δp_corr`

**模型参数**：~50K

### 2. 时序模型 (PinUavSeqModel)

**适用场景**：sequence_length > 1

**前向传播**：
```python
p_hat = model(feats, p_prev0)
# feats: (B, S, 6) - 序列特征
# p_prev0: (B, 2) - 初始位置 (可选，默认为零)
# p_hat: (B, S, 2) - 序列位置预测
```

**计算流程**：
1. 逐帧计算物理粗解：`Δp_lin_t = LeastSquaresLayer(p_prev, feats_t)`
2. 拼接并嵌入：`x = Linear([feats_t, Δp_lin_t])`
3. LSTM 时序平滑：`x, _ = LSTM(x)`
4. 输出校正：`Δp_corr = Linear(x)`
5. 最终位置：`p_hat = p_prev0 + cumsum(Δp_lin) + Δp_corr`

**模型参数**：~100K

### 3. 损失函数 (CartesianTestLoss)

**组成**：
- **主损失**：笛卡尔坐标 MSE
  ```
  L_main = MSE(x_pred, x_true) + MSE(y_pred, y_true)
  ```

- **约束损失**：三角恒等式约束
  ```
  L_trig = MSE(sin²θ + cos²θ, 1)
  ```

- **总损失**：
  ```
  L_total = L_main + λ·L_trig
  ```

**特点**：保持物理约束，防止极坐标转换时的数值不稳定

## 配置参数

主要配置参数在 `src/config.py` 中：

```python
# ==================== 数据集配置 ====================
num_detecting_regions = 1                    # 检测区域数量
num_lines_to_generate_per_region = 20000     # 每个区域生成的轨迹数
sequence_length = 5                          # 序列长度（>1 启用 LSTM）
train_test_split_ratio = 0.85                # 训练/测试分割比例
time_interval = 0.01                         # 采样时间间隔 (s)

# ==================== 训练配置 ====================
epoch = 500                                   # 训练轮数
batch_size = 4096                            # 批次大小
learning_rate = 2e-4                         # 学习率
optimizer = "Adam"                           # 优化器 ["Adam" | "LBFGS"]

# ==================== 物理参数 ====================
fc = 6e9                                     # 载波频率 (Hz) = 6 GHz
c = 3e8                                      # 光速 (m/s)
v = 30                                       # UAV 速度 (m/s)
lambda_wave = c / fc = 0.05                  # 载波波长 (m) = 5 cm

# ==================== 随机种子 ====================
region_seed = 42                             # 区域生成种子
line_seed_train = 2024                       # 训练集轨迹种子
line_seed_test = 2026                        # 测试集轨迹种子
visualize_line_seed = 0                      # 可视化轨迹种子

# ==================== 数据集路径 ====================
stage_1_dataset_train_path = "dataset/stage_1_dataset_train.pt"
stage_1_dataset_test_path = "dataset/stage_1_dataset_test.pt"
```

## 实验结果

### 性能指标
- **物理粗解精度**：厘米级 (cm)
- **深度学习精修精度**：毫米级 (mm)
- **总体定位精度**：亚厘米级 (sub-cm)
- **训练时间**：~2-4 小时 (500 epochs, batch_size=4096, GPU)
- **推理速度**：实时 (< 1ms per frame on GPU)
- **模型大小**：
  - PinUavModel: ~50K 参数 (~200 KB)
  - PinUavSeqModel: ~100K 参数 (~400 KB)

### 鲁棒性改进
相比传统最小二乘法，本系统的鲁棒 PINN 具有：
- ✅ **抗野值能力**：Huber 权重降低异常值影响
- ✅ **数值稳定性**：Tikhonov 正则化防止病态矩阵
- ✅ **约束保持**：三角恒等式约束确保物理一致性
- ✅ **自适应校正**：SE-ResMLP 学习系统性偏差

### 结果保存
- `models/` - 保存训练好的 .pth 模型文件
- `results/` - 保存实验结果和损失曲线
- `dataset/` - 保存生成的数据集 (.pt 文件)

## 可视化功能

系统提供完整的交互式可视化工具 (`visualization.py`)：

### Tkinter GUI 界面
- **轨迹生成区域**：输入 seed 和轨迹数量，实时生成测试轨迹
- **模型加载区域**：从文件对话框选择 .pth 模型文件
- **序列长度配置**：切换单帧 (seq_len=1) 或时序 (seq_len>1) 模式
- **轨迹选择器**：下拉菜单选择要可视化的轨迹
- **预测按钮**：点击运行模型预测
- **Matplotlib 画布**：实时显示轨迹、检测区域和预测结果

### 可视化内容
1. **检测区域边界**：
   - 外边界 (黑色实线)
   - 内生成区域 (绿色虚线)

2. **轨迹可视化**：
   - 真实轨迹 (蓝色实线 + 圆点)
   - 关键点 (紫色散点)

3. **预测对比**：
   - 真实位置 (蓝色圆点连线)
   - 预测位置 (红色叉号虚线)
   - 误差连线 (灰色细线)

4. **自适应平滑**：
   - Savitzky-Golay 滤波平滑预测轨迹
   - 自适应窗口大小 (min=50, max=100)
   - 多项式阶数 = 3

### HTML 可视化
- `uav_model_vis.html` - Web 端交互式可视化界面

## 开发说明

### 扩展模型架构
1. 在 `src/uav_model.py` 中继承 `nn.Module`
2. 实现物理层和深度学习层的组合
3. 遵循接口：`forward(feats, p_prev=None) -> p_hat`
4. 在 `lab.ipynb` 中集成新模型

### 修改物理约束
1. 编辑 `src/uav_model.py` 中的 `LeastSquaresLayer`
2. 调整 Jacobian 矩阵构造：`J = u_AT + u_AR`
3. 调整权重函数：修改 `_huber_weight` 的 δ 参数
4. 调整正则化参数：修改 `alpha` (Tikhonov)
5. 重新训练模型

### 调整网络架构
1. **修改 SEResBlock**：
   - `hidden_dim`: 隐藏层维度
   - `se_ratio`: SE 注意力压缩比例
   - `drop`: Dropout 概率

2. **修改 ResidualMLP**：
   - `embed`: 嵌入维度 (default=64)
   - `depth`: SEResBlock 层数 (default=4)
   - `hid_ratio`: 隐藏层扩展比例 (default=2.5)

3. **修改 LSTM**：
   - `lstm_hidden`: LSTM 隐藏维度 (default=128)
   - `lstm_layers`: LSTM 层数 (default=2)

### 调试工具
- `check_dependencies.py` - 检查依赖是否正确安装
- `test_loss.py` - 测试损失函数计算
- `test_visualization.py` - 测试可视化功能
- `src/debug_utils.py` - 调试工具函数

## 技术特点与创新

### 物理信息融合
- **显式物理建模**：
  - Gauss-Newton 单步迭代求解非线性方程
  - Jacobian 矩阵显式构造：J = u_AT + u_AR
  - 相位差转位移：d = λφ / (4π)

- **隐式学习**：
  - SE-ResMLP 学习系统性偏差和非线性残差
  - 数据驱动校正物理模型的局限性

- **约束保持**：
  - Tikhonov 正则化保持数值稳定性
  - 三角恒等式约束确保极坐标一致性

### 鲁棒性设计
- **Huber-IRLS 权重**：
  - 自动降权异常测量值
  - 迭代重加权最小二乘 (Iteratively Reweighted Least Squares)
  - δ 参数控制鲁棒性强度

- **正则化策略**：
  - Tikhonov 正则化 (L2)：防止病态矩阵
  - Dropout (0.1)：防止过拟合
  - LayerNorm：稳定训练

- **多尺度定位**：
  - **粗解 (厘米级)**：物理层提供稳定基准
  - **精解 (毫米级)**：深度学习提供高精度校正
  - **时序平滑 (可选)**：LSTM 消除时序噪声

### SE-ResMLP 架构优势
- **Squeeze-and-Excitation**：通道级自适应重加权
- **残差连接**：缓解梯度消失，加速收敛
- **GELU 激活**：平滑非线性，性能优于 ReLU
- **LayerNorm**：归一化特征分布

### 时序建模 (LSTM)
- **逐帧物理计算**：保持物理约束一致性
- **时序信息融合**：利用历史信息平滑轨迹
- **cumsum 位移累积**：从增量恢复绝对位置
- **适用长序列**：有效处理 sequence_length > 1 的场景

## Git 历史与版本演进

### 最近提交记录
```
226647a - adapt env setup to remote ubuntu machine
1319494 - A_to_B_fixed_visualization
793e5a9 - fixed_coord_knowedA_toB
44b48b4 - t_r_coordinates_se_lstm_filter
870d20e - seattention_mlp_lstm_SGfilter
72cfea3 - feat: add lstm back with a more elegant interface
```

### 分支信息
- **当前分支**：`lstm`
- **主分支**：`main`

### 演进路径
1. **初始版本**：基础 PINN 架构
2. **SE 注意力**：引入 Squeeze-and-Excitation 机制
3. **LSTM 集成**：添加时序建模能力
4. **鲁棒改进**：Huber-IRLS + Tikhonov 鲁棒物理层
5. **可视化增强**：Tkinter GUI + 自适应平滑
6. **环境适配**：支持远程 Ubuntu 服务器部署

## 常见问题 (FAQ)

### Q1: 如何选择 sequence_length？
- **seq_len = 1**：单帧独立预测，适合实时场景
- **seq_len > 1**：时序平滑，适合轨迹跟踪，推荐 seq_len = 5-10

### Q2: 模型加载时出现 missing keys 怎么办？
`visualization.py` 已自动处理 state_dict 前缀剥离 (`base_model.`, `model.`, `module.`)，并使用 `strict=False` 宽容加载。

### Q3: 如何提高定位精度？
1. 增加训练轮数 (epoch)
2. 调大批次大小 (batch_size)，充分利用 GPU
3. 调整物理层正则化参数 (alpha, huber_delta)
4. 增加 SE-ResMLP 深度 (depth) 和嵌入维度 (embed)

### Q4: 如何处理不同的检测区域配置？
修改 `src/detecting_region_info_generator.py` 中的区域生成逻辑，调整发射机 (T) 和接收机 (R) 的位置。

### Q5: 可以用于 3D 定位吗？
当前实现为 2D (DIM=2)。要扩展到 3D，需修改：
- `src/uav_model.py` 中 `DIM = 3`
- 调整 Jacobian 矩阵构造
- 增加第 4 个接收机 (3D 需要至少 4 个接收机)

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库：[GitHub]
- 邮箱：[联系邮箱]

## 致谢

本项目采用创新的鲁棒 PINN 架构，结合了：
- 经典控制理论 (Gauss-Newton)
- 鲁棒统计学 (Huber, IRLS)
- 深度学习 (SE-ResMLP, LSTM)
- 物理信息神经网络 (PINN)

感谢开源社区提供的优秀工具和库。

---

**注意**：本项目采用创新的鲁棒 PINN 架构，结合了物理约束、鲁棒优化和深度学习技术。建议在使用前仔细阅读代码和配置说明，理解物理建模和网络架构的设计理念。