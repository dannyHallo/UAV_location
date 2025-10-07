# 无人机定位系统 (UAV Location System)

基于物理信息神经网络(PINN)和多普勒频移技术的无人机轨迹定位系统，结合物理约束和深度学习进行高精度位置预测。

## 项目概述

本项目实现了一个创新的无人机定位系统，通过分析无人机飞行轨迹中的多普勒频移信息，使用物理信息神经网络(PINN)模型预测无人机的位置坐标。系统结合了物理显式建模和深度学习技术，实现了厘米级到毫米级的定位精度。

## 主要特性

- 🚁 **多轨迹生成**：支持生成大量无人机飞行轨迹用于训练
- 📡 **多普勒分析**：基于多普勒频移信息进行位置预测
- 🧠 **物理信息神经网络**：结合物理约束和深度学习的PINN架构
- ⚡ **并行处理**：多进程并行生成训练数据
- 📊 **可视化工具**：提供轨迹和结果的可视化界面
- 🔧 **模块化设计**：清晰的代码结构，易于扩展和维护
- 🎯 **高精度定位**：厘米级粗解 + 毫米级精修的两阶段定位

## 系统架构

```
无人机定位系统 (PINN架构)
├── 物理显式层
│   ├── 最小二乘层 (LeastSquaresLayer)
│   └── 物理约束建模
├── 深度学习层
│   ├── SE-ResMLP残差网络
│   ├── LSTM时序建模 (可选)
│   └── 注意力机制
├── 数据生成模块
│   ├── 轨迹生成器 (trajectory_generator.py)
│   ├── 多普勒信息计算 (doppler_info.py)
│   └── 特征提取 (features_and_labels_generator.py)
└── 训练与评估
    ├── 数据集生成 (stage_1_dataset_gen.py)
    ├── PINN模型训练 (lab.ipynb)
    └── 结果可视化 (visualization.py)
```

## PINN模型架构

### 核心设计理念

系统采用两阶段定位策略：
1. **物理粗解**：使用最小二乘法获得厘米级粗位置
2. **深度学习精修**：使用SE-ResMLP网络学习毫米级残差

### 模型组件

#### 1. 物理显式层 (LeastSquaresLayer)
- 基于相位差和多普勒信息的线性最小二乘解
- 输入：相位差 φ₁,φ₂,φ₃ 和多普勒频移 fD₁,fD₂,fD₃
- 输出：厘米级粗位置估计 Δp_ls

#### 2. SE-ResMLP残差网络
- **SE-ResBlock**：Squeeze-and-Excitation + 残差连接
- **ResidualMLP**：多层SE-ResBlock堆叠
- 输入：特征向量 + 物理粗解
- 输出：毫米级位置校正 Δp_corr

#### 3. 时序建模 (PinUavSeqModel)
- LSTM网络进行时序平滑
- 物理层逐帧显式计算
- 适用于序列长度 > 1 的场景

## 环境配置

### 系统要求
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (推荐)

### 安装步骤

```bash
# 创建虚拟环境
conda create -n "UAV_location" python=3.9 -y
conda activate UAV_location

# 安装PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 生成数据集

```bash
python stage_1_dataset_gen.py
```

这将生成训练集和测试集：
- `dataset/stage_1_dataset_train.pt` - 训练集
- `dataset/stage_1_dataset_test.pt` - 测试集

### 2. 运行PINN训练

```bash
# 运行Jupyter notebook进行PINN模型训练
jupyter notebook lab.ipynb
```

### 3. 可视化结果

```bash
python visualization.py
```

## 项目结构

```
UAV_location/
├── src/                          # 源代码目录
│   ├── config.py                 # 配置文件
│   ├── uav_model.py             # PINN模型实现
│   │   ├── LeastSquaresLayer    # 物理显式层
│   │   ├── SEResBlock           # SE-ResMLP块
│   │   ├── ResidualMLP          # 残差网络
│   │   ├── PinUavModel          # 单帧PINN模型
│   │   └── PinUavSeqModel       # 时序PINN模型
│   ├── trajectory_dataset.py     # 数据集类
│   ├── loss_func.py             # 损失函数
│   ├── trajectory_generator.py  # 轨迹生成器
│   ├── doppler_info.py          # 多普勒信息计算
│   └── ...                      # 其他模块
├── dataset/                      # 数据集目录
├── models/                       # 训练好的模型
├── results/                      # 实验结果
├── graphs/                       # 项目图表
├── stage_1_dataset_gen.py       # 数据集生成脚本
├── lab.ipynb                    # PINN模型训练notebook
├── visualization.py             # 可视化工具
└── requirements.txt             # 依赖列表
```

## PINN模型详解

### 1. 单帧模型 (PinUavModel)
- **物理层**：LeastSquaresLayer 提供厘米级粗解
- **残差网络**：ResidualMLP 学习毫米级校正
- **输入**：6维特征向量 [φ₁,φ₂,φ₃,fD₁,fD₂,fD₃]
- **输出**：2D位置坐标 (x,y)
- **参数量**：~50K

### 2. 时序模型 (PinUavSeqModel)
- **物理层**：逐帧最小二乘计算
- **LSTM层**：时序信息融合
- **输入**：序列特征 (B,S,6)
- **输出**：序列位置 (B,S,2)
- **适用场景**：sequence_length > 1

### 3. 损失函数 (CartesianTestLoss)
- **主损失**：笛卡尔坐标MSE损失
- **约束损失**：三角恒等式约束
- **总损失**：L_main + λ·L_trig

## 配置参数

主要配置参数在 `src/config.py` 中：

```python
# 数据集配置
num_detecting_regions = 1                    # 检测区域数量
num_lines_to_generate_per_region = 20000     # 每个区域生成的轨迹数
sequence_length = 5                          # 序列长度
train_test_split_ratio = 0.85                # 训练测试分割比例

# 训练配置
epoch = 500                                   # 训练轮数
batch_size = 4096                            # 批次大小
learning_rate = 2e-4                         # 学习率
optimizer = "Adam"                           # 优化器

# 物理参数
fc = 6e9                                     # 载波频率 (Hz)
c = 3e8                                      # 光速 (m/s)
lambda_wave = 0.06                           # 载波波长 (m)
```

## 实验结果

### 性能指标
- **粗解精度**：厘米级 (cm)
- **精解精度**：毫米级 (mm)
- **训练时间**：~2小时 (500 epochs, 4096 batch size)
- **推理速度**：实时 (< 1ms per frame)

### 结果保存
- `models/` - 保存训练好的PINN模型
- `results/` - 保存实验结果和损失曲线
- `trajectories/` - 保存轨迹可视化结果

## 可视化功能

系统提供完整的可视化工具：
- **轨迹生成可视化**：检测区域和飞行轨迹
- **模型预测对比**：真实轨迹 vs 预测轨迹
- **损失曲线**：训练和测试损失变化
- **误差分析**：位置误差分布和统计

## 开发说明

### 添加新模型
1. 在 `src/uav_model.py` 中继承 `nn.Module`
2. 实现物理层和深度学习层的组合
3. 在 `lab.ipynb` 中集成新模型

### 修改物理约束
1. 编辑 `src/uav_model.py` 中的 `LeastSquaresLayer`
2. 调整物理方程和约束条件
3. 重新训练模型

### 调整网络架构
1. 修改 `SEResBlock` 和 `ResidualMLP` 参数
2. 调整LSTM层数和隐藏维度
3. 优化SE注意力机制

## 技术特点

### 物理信息融合
- **显式物理建模**：最小二乘法求解线性方程组
- **隐式学习**：神经网络学习非线性残差
- **约束保持**：保持物理定律的约束

### 多尺度定位
- **粗解**：物理层提供稳定基准
- **精解**：深度学习提供高精度校正
- **鲁棒性**：物理约束提高模型稳定性

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库：[GitHub链接]
- 邮箱：[联系邮箱]

---

**注意**：本项目采用创新的PINN架构，结合了物理约束和深度学习技术。建议在使用前仔细阅读代码和配置说明，理解物理建模和网络架构的设计理念。