# UAV定位模型 - 噪声感知改进说明

## 改进概述

已成功集成噪声感知预处理层到 UAV 定位模型中。主要改进包括：

### 1. 新增 `NoiseAwareDenoiser` 类 (src/uav_model.py:27-146)

**功能**:
- 对 3 个相位差测量 (w12, w13, w14) 和 3 个多普勒测量 (v12, v13, v14) 进行去噪
- 使用卡尔曼滤波 + RTS 平滑算法
- 支持单帧和时序两种模式

**关键参数**:
- `t_cpi`: 相干处理时间间隔 (默认: config.T_CPI = 0.01s)
- `process_var`: 过程噪声方差 (默认: 1e-4)
- `snr_db`: 信噪比 (dB)，可以是标量、(B,) 或 (B,S)

**噪声模型**:
```python
σ_φ = sqrt(1 / SNR)           # 相位差噪声标准差
σ_f = sqrt(1 / (SNR * T_CPI²)) # 多普勒噪声标准差
```

### 2. 更新后的模型类

#### PinUavModel (单帧模型)
```python
model = uav_model.PinUavModel(
    T=T,  # 发射机位置
    R=R,  # 接收机位置 (3, 2)
    t_cpi=config.T_CPI,  # 新增
    enable_denoising=True  # 新增：开启去噪
)

# 前向传播
output = model(
    feats,  # (B, 6) 带噪声特征
    p_prev=None,  # (B, 2) 可选
    snr_db=torch.tensor(config.SNR_dB)  # 新增：SNR
)
```

#### PinUavSeqModel (时序模型)
```python
model = uav_model.PinUavSeqModel(
    T=T,
    R=R,
    t_cpi=config.T_CPI,  # 新增
    lstm_hidden=128,
    lstm_layers=2,
    drop=0.1,
    enable_denoising=True  # 新增：开启去噪
)

# 前向传播
output = model(
    feats,  # (B, S, 6) 带噪声特征
    p_prev0=None,  # (B, 2) 可选
    snr_db=torch.tensor(config.SNR_dB)  # 新增：SNR
)
```

### 3. 配置文件更新 (src/config.py)

新增噪声相关参数：
```python
# NOISE PARAMETERS
SNR_dB = 10  # Signal-to-Noise Ratio in dB
T_CPI = time_interval  # Coherent Processing Interval (0.01s)
add_noise = True  # Enable/disable noise addition
```

### 4. 数据生成更新

**stage_1_dataset_gen.py** 和 **w_and_doppler_generator.py** 已更新：
- `generate_w_and_doppler()` 现在支持添加高斯白噪声
- 噪声由 `add_noise`, `snr_db`, `t_cpi` 参数控制
- w12, w13, w14 互相独立
- v12, v13, v14 互相独立

## 使用方法

### 方法1: 使用新模型 (推荐)

直接使用更新后的模型，它会自动进行去噪：

```python
import torch
import src.config as config
import src.uav_model as uav_model

# 初始化模型
if config.sequence_length > 1:
    base_model = uav_model.PinUavSeqModel(
        T=T, R=R,
        t_cpi=config.T_CPI,
        lstm_hidden=128,
        lstm_layers=2,
        enable_denoising=True  # 启用去噪
    )
else:
    base_model = uav_model.PinUavModel(
        T=T, R=R,
        t_cpi=config.T_CPI,
        enable_denoising=True  # 启用去噪
    )

# 使用
output = base_model(
    feats,
    snr_db=torch.tensor(config.SNR_dB)
)
```

### 方法2: 禁用去噪

如果想要使用原始模型（不去噪）：

```python
base_model = uav_model.PinUavModel(
    T=T, R=R,
    enable_denoising=False  # 禁用去噪
)
```

### 方法3: 单独使用去噪器

可以单独使用去噪器进行特征预处理：

```python
denoiser = uav_model.NoiseAwareDenoiser(
    t_cpi=config.T_CPI,
    process_var=1e-4
)

feats_clean = denoiser(
    feats_noisy,  # (B, S, 6) 或 (B, 6)
    snr_db=torch.tensor(config.SNR_dB)
)
```

## 关键改进点

1. **物理约束**:
   - 基于已知的噪声统计特性（SNR、T_CPI）
   - 利用物理模型构建观测协方差矩阵

2. **时序信息利用**:
   - 单帧：卡尔曼滤波
   - 时序：卡尔曼滤波 + RTS 平滑（双向处理，利用整个序列）

3. **端到端可训练**:
   - 去噪层可与定位模型联合训练
   - 支持反向传播

4. **灵活性**:
   - 可通过 `enable_denoising` 开关控制
   - 可调整 `process_var` 参数平衡过程噪声和观测噪声

## 测试结果

运行 `python test_denoising_model.py` 查看详细测试结果。

关键发现：
- ✅ 去噪器成功运行，输入输出维度正确
- ✅ 单帧和时序模式都正常工作
- ✅ RTS平滑利用整个序列信息
- ⚠️ 去噪效果取决于 `process_var` 参数，需要调优

## 下一步工作

1. **调优 process_var**:
   - 当前默认值 1e-4 可能不是最优
   - 建议通过交叉验证找到最佳值
   - 不同场景可能需要不同的值

2. **重新生成数据集**:
   ```bash
   python stage_1_dataset_gen.py
   ```
   这将生成带噪声的新数据集

3. **训练模型**:
   - 使用 lab.ipynb 训练模型
   - 比较开启/关闭去噪的性能差异

4. **评估去噪性能**:
   - 在不同 SNR 下测试
   - 比较 CRLB 理论极限

## 注意事项

1. **兼容性**:
   - 如果使用 ModelAdapter 包装器转换为极坐标输出，需要相应修改
   - 新模型直接输出笛卡尔坐标

2. **SNR 参数**:
   - 训练时如果不传 `snr_db`，默认使用 10 dB
   - 建议根据实际情况传入准确的 SNR

3. **计算开销**:
   - 去噪增加了一定计算开销
   - 时序模式的 RTS 平滑需要额外的反向遍历

## 文件清单

修改的文件：
- ✅ `src/uav_model.py` - 添加 NoiseAwareDenoiser，更新模型
- ✅ `src/config.py` - 添加噪声参数
- ✅ `src/w_and_doppler_generator.py` - 添加加噪功能
- ✅ `stage_1_dataset_gen.py` - 传递噪声参数

新增文件：
- ✅ `test_noise.py` - 测试加噪功能
- ✅ `test_denoising_model.py` - 测试去噪模型
- ✅ `MODEL_IMPROVEMENTS.md` - 本文档
