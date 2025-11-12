"""
分析 w 和分母分布的脚本
用于检测生成数据过程中的数值稳定性问题
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import src.config as config
from src.detecting_region_info_generator import generate_detecting_region_infos
import src.doppler_info as doppler_info_module
from src.trajectory_generator import generate_lines
from src.get_phi_info import get_angle_phi
from src.w_and_doppler_generator import generate_w_and_doppler, _get_angle_theta


def analyze_denominator_distribution(
    detecting_region_info, coords_a_list, coords_b_list, phis_1234_list
):
    """
    分析训练数据中分母的分布

    参数：
        detecting_region_info: 检测区域信息
        coords_a_list: 起始坐标列表
        coords_b_list: 结束坐标列表
        phis_1234_list: 相位信息列表
    """
    all_denominators = []

    for coords_a, coords_b, phis_1234 in zip(
        coords_a_list, coords_b_list, phis_1234_list
    ):
        # 对于每个轨迹中的每个点
        for i in range(len(coords_a)):
            coord_a = coords_a[i]
            coord_b = coords_b[i]
            [phi1, phi2, phi3, phi4] = phis_1234[i]

            [theta1, theta2, theta3, theta4] = _get_angle_theta(
                detecting_region_info, coord_a, coord_b
            )

            # 计算三个通道的分母
            denom1 = theta1 * np.cos(phi1) + np.sin(phi1)
            denom2 = theta2 * np.cos(phi2) + np.sin(phi2)
            denom3 = theta3 * np.cos(phi3) + np.sin(phi3)
            denom4 = theta4 * np.cos(phi4) + np.sin(phi4)

            all_denominators.extend([denom1, denom2, denom3, denom4])

    all_denominators = np.array(all_denominators)

    # 统计信息
    print("=" * 50)
    print("分母统计信息：")
    print(f"最小值: {all_denominators.min():.6f}")
    print(f"最大值: {all_denominators.max():.6f}")
    print(f"平均值: {all_denominators.mean():.6f}")
    print(f"中位数: {np.median(all_denominators):.6f}")
    print(f"标准差: {all_denominators.std():.6f}")
    print(f"接近零的比例 (|denom| < 0.1): {np.mean(np.abs(all_denominators) < 0.1):.2%}")
    print(f"接近零的比例 (|denom| < 0.01): {np.mean(np.abs(all_denominators) < 0.01):.2%}")
    print(f"接近零的比例 (|denom| < 0.001): {np.mean(np.abs(all_denominators) < 0.001):.2%}")
    print("=" * 50)

    # 绘制直方图
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(all_denominators, bins=100, edgecolor="black", alpha=0.7)
    plt.axvline(x=0, color="r", linestyle="--", linewidth=2, label="零点")
    plt.axvline(x=0.1, color="orange", linestyle="--", label="阈值 ±0.1")
    plt.axvline(x=-0.1, color="orange", linestyle="--")
    plt.xlabel("分母值")
    plt.ylabel("频数")
    plt.title("分母分布")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.hist(
        np.log10(np.abs(all_denominators) + 1e-10),
        bins=100,
        edgecolor="black",
        alpha=0.7,
    )
    plt.xlabel("log10(|分母|)")
    plt.ylabel("频数")
    plt.title("分母绝对值分布（对数尺度）")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    # 只显示接近零的部分
    near_zero = all_denominators[np.abs(all_denominators) < 0.5]
    if len(near_zero) > 0:
        plt.hist(near_zero, bins=100, edgecolor="black", alpha=0.7)
        plt.axvline(x=0, color="r", linestyle="--", linewidth=2, label="零点")
        plt.axvline(x=0.1, color="orange", linestyle="--", label="阈值 ±0.1")
        plt.axvline(x=-0.1, color="orange", linestyle="--")
        plt.xlabel("分母值")
        plt.ylabel("频数")
        plt.title("分母分布（|分母| < 0.5）")
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("denominator_distribution.png", dpi=150, bbox_inches="tight")
    print(f"分母分布图已保存至 denominator_distribution.png")

    return all_denominators


def analyze_w_distribution(w_values):
    """
    检查 w 是否有极端值

    参数：
        w_values: w 值数组，形状为 (n, 3)
    """
    w_array = np.concatenate(w_values, axis=0)

    print("\n" + "=" * 50)
    print("W 值统计（每个通道）：")
    for i in range(w_array.shape[1]):
        w_channel = w_array[:, i]
        print(f"\n通道 {i}:")
        print(f"  最小值: {w_channel.min():.6e}")
        print(f"  最大值: {w_channel.max():.6e}")
        print(f"  平均值: {w_channel.mean():.6e}")
        print(f"  中位数: {np.median(w_channel):.6e}")
        print(f"  标准差: {w_channel.std():.6e}")
        print(f"  99%分位数: {np.percentile(w_channel, 99):.6e}")
        print(f"  1%分位数: {np.percentile(w_channel, 1):.6e}")

        # 检查极端值
        mean_val = w_channel.mean()
        std_val = w_channel.std()
        if std_val > 0:
            extreme_ratio = np.sum(np.abs(w_channel) > mean_val + 10 * std_val) / len(
                w_channel
            )
            print(f"  极端值比例 (>μ+10σ): {extreme_ratio:.2%}")

            # 检查 NaN 和 Inf
            nan_ratio = np.sum(np.isnan(w_channel)) / len(w_channel)
            inf_ratio = np.sum(np.isinf(w_channel)) / len(w_channel)
            print(f"  NaN 比例: {nan_ratio:.2%}")
            print(f"  Inf 比例: {inf_ratio:.2%}")

    print("=" * 50)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for i in range(min(3, w_array.shape[1])):
        # 第一行：完整分布
        axes[0, i].hist(w_array[:, i], bins=100, edgecolor="black", alpha=0.7)
        axes[0, i].set_xlabel(f"w{i+1}")
        axes[0, i].set_ylabel("频数")
        axes[0, i].set_title(f"通道 {i+1} 完整分布")
        axes[0, i].grid(True, alpha=0.3)

        # 第二行：对数尺度分布（绝对值）
        w_abs = np.abs(w_array[:, i])
        w_abs_nonzero = w_abs[w_abs > 0]
        if len(w_abs_nonzero) > 0:
            axes[1, i].hist(
                np.log10(w_abs_nonzero), bins=100, edgecolor="black", alpha=0.7
            )
            axes[1, i].set_xlabel(f"log10(|w{i+1}|)")
            axes[1, i].set_ylabel("频数")
            axes[1, i].set_title(f"通道 {i+1} 绝对值分布（对数尺度）")
            axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("w_distribution.png", dpi=150, bbox_inches="tight")
    print(f"\nW 值分布图已保存至 w_distribution.png")


def analyze_doppler_distribution(doppler_values):
    """
    检查 doppler 是否有极端值

    参数：
        doppler_values: doppler 值数组列表
    """
    doppler_array = np.concatenate(doppler_values, axis=0)

    print("\n" + "=" * 50)
    print("Doppler 值统计（每个通道）：")
    for i in range(doppler_array.shape[1]):
        doppler_channel = doppler_array[:, i]
        print(f"\n通道 {i}:")
        print(f"  最小值: {doppler_channel.min():.6e}")
        print(f"  最大值: {doppler_channel.max():.6e}")
        print(f"  平均值: {doppler_channel.mean():.6e}")
        print(f"  中位数: {np.median(doppler_channel):.6e}")
        print(f"  标准差: {doppler_channel.std():.6e}")

        # 检查 NaN 和 Inf
        nan_ratio = np.sum(np.isnan(doppler_channel)) / len(doppler_channel)
        inf_ratio = np.sum(np.isinf(doppler_channel)) / len(doppler_channel)
        print(f"  NaN 比例: {nan_ratio:.2%}")
        print(f"  Inf 比例: {inf_ratio:.2%}")

    print("=" * 50)


def main():
    """主函数：生成样本数据并进行分析"""
    print("开始分析 w 和分母分布...")
    # print(f"配置: SNR_dB = {config.SNR_dB}")

    # 生成检测区域
    doppler_info = doppler_info_module.DopplerInfo(
        config.c, config.fc, config.time_interval
    )
    detecting_region_infos = generate_detecting_region_infos(
        num_configurations=1, seed=config.region_seed
    )
    detecting_region_info = detecting_region_infos[0]

    # 生成轨迹数据（使用较小的样本量以加快分析）
    num_samples = 100  # 可以调整样本数量
    print(f"\n生成 {num_samples} 条轨迹用于分析...")

    lines_a, lines_b = generate_lines(
        detecting_region_info,
        num_lines=num_samples,
        time_interval=config.time_interval,
        seed=config.line_seed_train,
    )

    if not lines_a:
        print("错误：未能生成轨迹数据")
        return

    # 准备数据列表
    coords_a_list = []
    coords_b_list = []
    phis_1234_list = []
    w_list = []
    doppler_list = []

    print("处理轨迹数据...")
    for i in range(len(lines_a)):
        coords_a = lines_a[i]
        coords_b = lines_b[i]

        if coords_a.shape[0] == 0:
            continue

        # 计算相位
        phis1234 = np.array(
            [
                get_angle_phi(detecting_region_info, ca, cb)
                for ca, cb in zip(coords_a, coords_b)
            ]
        )

        # 生成 w 和 doppler
        w, doppler = generate_w_and_doppler(
            detecting_region_info, doppler_info, coords_a, coords_b, phis1234
        )

        coords_a_list.append(coords_a)
        coords_b_list.append(coords_b)
        phis_1234_list.append(phis1234)
        w_list.append(w)
        doppler_list.append(doppler)

    print(f"成功处理 {len(coords_a_list)} 条轨迹\n")

    # 分析分母分布
    print("\n【分析分母分布】")
    all_denoms = analyze_denominator_distribution(
        detecting_region_info, coords_a_list, coords_b_list, phis_1234_list
    )

    # 分析 w 值分布
    print("\n【分析 W 值分布】")
    analyze_w_distribution(w_list)

    # 分析 doppler 值分布
    print("\n【分析 Doppler 值分布】")
    analyze_doppler_distribution(doppler_list)

    print("\n分析完成！")
    print("生成的图像文件:")
    print("  - denominator_distribution.png")
    print("  - w_distribution.png")


if __name__ == "__main__":
    main()
