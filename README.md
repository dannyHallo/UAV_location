# 无人机定位系统 (UAV Location System)

基于多普勒频移和机器学习技术的无人机轨迹定位系统，使用多种深度学习模型进行位置预测。

## 项目概述

本项目实现了一个完整的无人机定位系统，通过分析无人机飞行轨迹中的多普勒频移信息，使用机器学习模型预测无人机的位置坐标。系统支持多种模型架构，包括MLP、Transformer和KAN（Kolmogorov-Arnold Networks）。

## 主要特性

- 🚁 **多轨迹生成**：支持生成大量无人机飞行轨迹用于训练
- 📡 **多普勒分析**：基于多普勒频移信息进行位置预测
- 🤖 **多模型支持**：MLP、Transformer、KAN等多种深度学习模型
- ⚡ **并行处理**：多进程并行生成训练数据
- 📊 **可视化工具**：提供轨迹和结果的可视化界面
- 🔧 **模块化设计**：清晰的代码结构，易于扩展和维护

## 系统架构

```
无人机定位系统
├── 数据生成模块
│   ├── 轨迹生成器 (trajectory_generator.py)
│   ├── 多普勒信息计算 (doppler_info.py)
│   └── 特征提取 (features_and_labels_generator.py)
├── 模型模块
│   ├── MLP模型 (uav_model.py)
│   ├── Transformer模型 (transformer_model.py)
│   └── KAN模型 (kan/)
└── 训练与评估
    ├── 数据集生成 (stage_1_dataset_gen.py)
    ├── 模型训练
    └── 结果可视化 (visualization.py)
```

## 环境配置

### 系统要求
- Python 3.9.19
- PyTorch (需要单独安装)
- 其他依赖见 requirements.txt

### 安装步骤

```bash
# 创建虚拟环境
conda create -n "UAV_location" python=3.9.19 -y
conda activate UAV_location

# 安装PyTorch (根据您的CUDA版本选择)
pip install torch torchvision torchaudio

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

### 2. 运行实验

```bash
# 运行Jupyter notebook进行实验
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
│   ├── trajectory_dataset.py     # 数据集类
│   ├── uav_model.py             # MLP模型
│   ├── transformer_model.py     # Transformer模型
│   ├── kan/                     # KAN模型实现
│   ├── trajectory_generator.py  # 轨迹生成器
│   ├── doppler_info.py          # 多普勒信息计算
│   └── ...                      # 其他模块
├── dataset/                      # 数据集目录
├── models/                       # 训练好的模型
├── results/                      # 实验结果
├── graphs/                       # 项目图表
├── stage_1_dataset_gen.py       # 数据集生成脚本
├── visualization.py             # 可视化工具
└── requirements.txt             # 依赖列表
```

## 模型说明

### 1. MLP模型 (UavModel)
- 基于SE-ResMLP架构
- 输入维度：6（phi角、w值、多普勒信息）
- 输出维度：3（ρ, sinθ, cosθ）
- 参数量：~45K

### 2. Transformer模型
- 基于PyTorch TransformerEncoder
- 支持序列数据处理
- 可配置层数和注意力头数

### 3. KAN模型
- 基于Kolmogorov-Arnold Networks
- 支持符号回归
- 可解释性强

## 配置参数

主要配置参数在 `src/config.py` 中：

```python
# 数据集配置
num_detecting_regions = 1                    # 检测区域数量
num_lines_to_generate_per_region = 500       # 每个区域生成的轨迹数
train_test_split_ratio = 0.85                # 训练测试分割比例

# 训练配置
epoch = 500                                   # 训练轮数
batch_size = 4096                            # 批次大小
learning_rate = 2e-4                         # 学习率

# 物理参数
fc = 6e9                                     # 载波频率 (Hz)
c = 3e8                                      # 光速 (m/s)
v = 30                                       # 无人机速度 (m/s)
```

## 实验结果

项目包含多个实验结果的保存目录，每个实验都有时间戳标识：
- `models/` - 保存训练好的模型
- `results/` - 保存实验结果和指标

## 可视化功能

系统提供完整的可视化工具：
- 轨迹生成可视化
- 模型预测结果对比
- 误差分析图表
- 交互式GUI界面

## 开发说明

### 添加新模型
1. 在 `src/` 目录下创建新的模型文件
2. 继承 `nn.Module` 并实现 `forward` 方法
3. 在训练脚本中导入并使用

### 修改数据生成
1. 编辑 `src/trajectory_generator.py` 修改轨迹生成逻辑
2. 编辑 `src/features_and_labels_generator.py` 修改特征提取
3. 重新运行 `stage_1_dataset_gen.py` 生成新数据集

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库：[GitHub链接]
- 邮箱：[联系邮箱]

---

**注意**：本项目仍在持续开发中，部分功能可能尚未完善。建议在使用前仔细阅读代码和配置说明。
