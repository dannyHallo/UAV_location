#!/usr/bin/env python3
"""
测试 visualization.py 的模型加载功能
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.uav_model import PinUavModel, PinUavSeqModel


def test_model_creation():
    """测试PINN模型创建"""
    print("=== 测试PINN模型创建 ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 测试参数
    T = torch.tensor([0.0, 0.0], device=device)
    R = torch.tensor([[100.0, 0.0], [50.0, 86.6], [-50.0, 86.6]], device=device)

    print(f"T (发射机): {T}")
    print(f"R (接收机): {R}")

    try:
        # 测试单帧模型
        print("\n--- 测试单帧模型 ---")
        model1 = PinUavModel(T, R).to(device)
        print(f"单帧模型创建成功: {type(model1)}")

        # 测试前向传播
        feats = torch.randn(4, 6, device=device)
        output1 = model1(feats)
        print(f"单帧模型输出形状: {output1.shape}")

        # 测试序列模型
        print("\n--- 测试序列模型 ---")
        model2 = PinUavSeqModel(T, R).to(device)
        print(f"序列模型创建成功: {type(model2)}")

        # 测试前向传播
        feats_seq = torch.randn(4, 5, 6, device=device)  # (B, S, 6)
        output2 = model2(feats_seq)
        print(f"序列模型输出形状: {output2.shape}")

        print("\n✅ 所有测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()


def test_model_loading():
    """测试模型加载"""
    print("\n=== 测试模型加载 ===")

    # 查找可用的模型文件
    models_dir = "./models"
    if not os.path.exists(models_dir):
        print("❌ models 目录不存在")
        return

    # 查找最新的模型文件
    model_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith(".pth"):
                model_files.append(os.path.join(root, file))

    if not model_files:
        print("❌ 没有找到 .pth 模型文件")
        return

    # 选择最新的模型文件
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"尝试加载模型: {latest_model}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建模型
        T = torch.tensor([0.0, 0.0], device=device)
        R = torch.tensor([[100.0, 0.0], [50.0, 86.6], [-50.0, 86.6]], device=device)
        model = PinUavModel(T, R).to(device)

        # 尝试加载权重
        checkpoint = torch.load(latest_model, map_location=device, weights_only=True)
        print(f"检查点键: {list(checkpoint.keys())[:5]}...")  # 显示前5个键

        # 检查模型状态字典的键
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint.keys())

        print(f"模型键数量: {len(model_keys)}")
        print(f"检查点键数量: {len(checkpoint_keys)}")

        # 检查匹配的键
        matching_keys = model_keys.intersection(checkpoint_keys)
        print(f"匹配的键数量: {len(matching_keys)}")

        if len(matching_keys) == 0:
            print("❌ 模型权重不兼容 - 没有匹配的键")
            print("这可能是旧模型架构的权重文件")
        else:
            print("✅ 找到匹配的键，尝试加载...")
            model.load_state_dict(checkpoint, strict=False)
            print("✅ 模型加载成功！")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_model_creation()
    test_model_loading()
