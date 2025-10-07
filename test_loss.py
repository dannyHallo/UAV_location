#!/usr/bin/env python3
"""
测试损失函数的脚本
"""
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.loss_func import CartesianTestLoss, PINNLoss


def test_loss_functions():
    """测试损失函数"""
    print("=== 测试损失函数 ===")

    # 创建测试数据
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟模型输出和标签
    outputs = torch.randn(batch_size, 3, device=device)
    targets = torch.randn(batch_size, 3, device=device)

    # 确保sin和cos值在合理范围内
    outputs[:, 1] = torch.tanh(outputs[:, 1])  # sin值
    outputs[:, 2] = torch.tanh(outputs[:, 2])  # cos值
    targets[:, 1] = torch.tanh(targets[:, 1])  # sin值
    targets[:, 2] = torch.tanh(targets[:, 2])  # cos值

    # 确保距离值为正
    outputs[:, 0] = torch.abs(outputs[:, 0])
    targets[:, 0] = torch.abs(targets[:, 0])

    print(f"设备: {device}")
    print(f"输出形状: {outputs.shape}")
    print(f"标签形状: {targets.shape}")
    print()

    # 测试 CartesianTestLoss
    print("--- 测试 CartesianTestLoss ---")
    criterion1 = CartesianTestLoss(debug=True)
    loss1 = criterion1(outputs, targets)
    print(f"损失值: {loss1.item():.6f}")
    print()

    # 测试 PINNLoss
    print("--- 测试 PINNLoss ---")
    criterion2 = PINNLoss(debug=True)
    loss2 = criterion2(outputs, targets)
    print(f"损失值: {loss2.item():.6f}")
    print()

    # 测试序列模型
    print("--- 测试序列模型损失 ---")
    seq_outputs = torch.randn(batch_size, 5, 3, device=device)  # (B, S, 3)
    seq_targets = torch.randn(batch_size, 5, 3, device=device)  # (B, S, 3)

    # 确保sin和cos值在合理范围内
    seq_outputs[:, :, 1] = torch.tanh(seq_outputs[:, :, 1])
    seq_outputs[:, :, 2] = torch.tanh(seq_outputs[:, :, 2])
    seq_targets[:, :, 1] = torch.tanh(seq_targets[:, :, 1])
    seq_targets[:, :, 2] = torch.tanh(seq_targets[:, :, 2])

    # 确保距离值为正
    seq_outputs[:, :, 0] = torch.abs(seq_outputs[:, :, 0])
    seq_targets[:, :, 0] = torch.abs(seq_targets[:, :, 0])

    criterion3 = PINNLoss(debug=True, smooth_weight=0.01)
    loss3 = criterion3(seq_outputs, seq_targets)
    print(f"序列损失值: {loss3.item():.6f}")
    print()

    print("=== 测试完成 ===")


if __name__ == "__main__":
    test_loss_functions()
