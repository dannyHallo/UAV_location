#!/usr/bin/env python3
"""
测试带去噪功能的 UAV 定位模型
"""
import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import src.config as config
import src.uav_model as uav_model
from src.detecting_region_info_generator import generate_detecting_region_infos


def test_denoiser():
    """测试去噪器单独功能"""
    print("=" * 60)
    print("1. 测试 NoiseAwareDenoiser")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 创建去噪器
    denoiser = uav_model.NoiseAwareDenoiser(t_cpi=config.T_CPI).to(device)

    # 测试单帧
    print("1.1 单帧测试:")
    B = 4
    feats_single = torch.randn(B, 6, device=device)
    snr_db = torch.tensor(config.SNR_dB, device=device)

    feats_clean_single = denoiser(feats_single, snr_db)
    print(f"  输入形状: {feats_single.shape}")
    print(f"  输出形状: {feats_clean_single.shape}")
    print(f"  输入样例: {feats_single[0][:3].cpu().numpy()}")
    print(f"  输出样例: {feats_clean_single[0][:3].cpu().numpy()}\n")

    # 测试时序
    print("1.2 时序测试 (RTS平滑):")
    S = 5
    feats_seq = torch.randn(B, S, 6, device=device)
    feats_clean_seq = denoiser(feats_seq, snr_db)
    print(f"  输入形状: {feats_seq.shape}")
    print(f"  输出形状: {feats_clean_seq.shape}")
    print(f"  输入第一帧: {feats_seq[0, 0, :3].cpu().numpy()}")
    print(f"  输出第一帧: {feats_clean_seq[0, 0, :3].cpu().numpy()}\n")


def test_pin_uav_model():
    """测试单帧PINN模型"""
    print("=" * 60)
    print("2. 测试 PinUavModel (单帧)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成检测区域
    region_infos = generate_detecting_region_infos(
        num_configurations=1, seed=config.region_seed
    )
    region_info = region_infos[0]

    # 获取 T 和 R
    T = torch.tensor(region_info.transmittor_position, dtype=torch.float32, device=device)
    R = torch.stack([
        torch.tensor(region_info.receiver_position_1, dtype=torch.float32, device=device),
        torch.tensor(region_info.receiver_position_2, dtype=torch.float32, device=device),
        torch.tensor(region_info.receiver_position_3, dtype=torch.float32, device=device)
    ], dim=0)

    print(f"T (发射机): {T.cpu().numpy()}")
    print(f"R (接收机):\n{R.cpu().numpy()}\n")

    # 创建模型
    model_with_denoise = uav_model.PinUavModel(
        T=T, R=R, t_cpi=config.T_CPI, enable_denoising=True
    ).to(device)

    model_without_denoise = uav_model.PinUavModel(
        T=T, R=R, t_cpi=config.T_CPI, enable_denoising=False
    ).to(device)

    print("2.1 模型结构:")
    print(f"  带去噪: {sum(p.numel() for p in model_with_denoise.parameters())} 参数")
    print(f"  不带去噪: {sum(p.numel() for p in model_without_denoise.parameters())} 参数\n")

    # 测试前向传播
    print("2.2 前向传播测试:")
    B = 4
    feats = torch.randn(B, 6, device=device)
    snr_db = torch.tensor(config.SNR_dB, device=device)

    with torch.no_grad():
        output_with = model_with_denoise(feats, snr_db=snr_db)
        output_without = model_without_denoise(feats, snr_db=snr_db)

    print(f"  输入形状: {feats.shape}")
    print(f"  输出形状 (带去噪): {output_with.shape}")
    print(f"  输出形状 (不带去噪): {output_without.shape}")
    print(f"  输出样例 (带去噪): {output_with[0].cpu().numpy()}")
    print(f"  输出样例 (不带去噪): {output_without[0].cpu().numpy()}\n")


def test_pin_uav_seq_model():
    """测试时序PINN模型"""
    print("=" * 60)
    print("3. 测试 PinUavSeqModel (时序)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成检测区域
    region_infos = generate_detecting_region_infos(
        num_configurations=1, seed=config.region_seed
    )
    region_info = region_infos[0]

    T = torch.tensor(region_info.transmittor_position, dtype=torch.float32, device=device)
    R = torch.stack([
        torch.tensor(region_info.receiver_position_1, dtype=torch.float32, device=device),
        torch.tensor(region_info.receiver_position_2, dtype=torch.float32, device=device),
        torch.tensor(region_info.receiver_position_3, dtype=torch.float32, device=device)
    ], dim=0)

    # 创建模型
    model_with_denoise = uav_model.PinUavSeqModel(
        T=T, R=R, t_cpi=config.T_CPI,
        lstm_hidden=128, lstm_layers=2,
        enable_denoising=True
    ).to(device)

    model_without_denoise = uav_model.PinUavSeqModel(
        T=T, R=R, t_cpi=config.T_CPI,
        lstm_hidden=128, lstm_layers=2,
        enable_denoising=False
    ).to(device)

    print("3.1 模型结构:")
    print(f"  带去噪: {sum(p.numel() for p in model_with_denoise.parameters())} 参数")
    print(f"  不带去噪: {sum(p.numel() for p in model_without_denoise.parameters())} 参数\n")

    # 测试前向传播
    print("3.2 前向传播测试:")
    B = 4
    S = config.sequence_length
    feats = torch.randn(B, S, 6, device=device)
    snr_db = torch.tensor(config.SNR_dB, device=device)

    with torch.no_grad():
        output_with = model_with_denoise(feats, snr_db=snr_db)
        output_without = model_without_denoise(feats, snr_db=snr_db)

    print(f"  输入形状: {feats.shape}")
    print(f"  输出形状 (带去噪): {output_with.shape}")
    print(f"  输出形状 (不带去噪): {output_without.shape}")
    print(f"  输出样例 (带去噪，最后一帧): {output_with[0, -1].cpu().numpy()}")
    print(f"  输出样例 (不带去噪，最后一帧): {output_without[0, -1].cpu().numpy()}\n")


def test_denoising_effectiveness():
    """测试去噪效果"""
    print("=" * 60)
    print("4. 测试去噪效果")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建去噪器
    denoiser = uav_model.NoiseAwareDenoiser(t_cpi=config.T_CPI).to(device)

    # 创建干净信号
    B = 100
    S = 10
    feats_clean = torch.randn(B, S, 6, device=device)

    # 添加噪声
    snr_db = config.SNR_dB
    snr_linear = 10 ** (snr_db / 10)
    sigma_phi = np.sqrt(1 / snr_linear)
    sigma_f = np.sqrt(1 / (snr_linear * config.T_CPI**2))

    noise = torch.randn_like(feats_clean)
    noise[:, :, :3] *= sigma_phi  # 相位差噪声
    noise[:, :, 3:] *= sigma_f     # 多普勒噪声
    feats_noisy = feats_clean + noise

    # 去噪
    with torch.no_grad():
        feats_denoised = denoiser(feats_noisy, torch.tensor(snr_db, device=device))

    # 计算MSE
    mse_noisy = torch.mean((feats_noisy - feats_clean) ** 2, dim=[0, 1])
    mse_denoised = torch.mean((feats_denoised - feats_clean) ** 2, dim=[0, 1])

    print(f"SNR = {snr_db} dB")
    print(f"理论 σ_φ = {sigma_phi:.6f}, σ_f = {sigma_f:.6f}\n")
    print("MSE (相位差 w12, w13, w14):")
    print(f"  加噪前: {mse_noisy[:3].cpu().numpy()}")
    print(f"  去噪后: {mse_denoised[:3].cpu().numpy()}")
    print(f"  改善比: {(mse_noisy[:3] / mse_denoised[:3]).cpu().numpy()}\n")

    print("MSE (多普勒 v12, v13, v14):")
    print(f"  加噪前: {mse_noisy[3:].cpu().numpy()}")
    print(f"  去噪后: {mse_denoised[3:].cpu().numpy()}")
    print(f"  改善比: {(mse_noisy[3:] / mse_denoised[3:]).cpu().numpy()}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("测试带去噪功能的 UAV 定位模型")
    print("=" * 60 + "\n")

    try:
        test_denoiser()
        test_pin_uav_model()
        test_pin_uav_seq_model()
        test_denoising_effectiveness()

        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
