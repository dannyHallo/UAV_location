"""
打印生成的数据集中的相位差和多普勒信息的前几行
"""
import torch
import numpy as np
import src.config as config


def print_data_info(dataset_path, num_rows=10):
    """
    从数据集文件中加载数据并打印相位差和多普勒的前几行

    参数：
        dataset_path: 数据集文件路径
        num_rows: 打印的行数，默认为10
    """
    print(f"\n{'='*80}")
    print(f"加载数据集: {dataset_path}")
    print(f"{'='*80}\n")

    # 加载数据集
    dataset_dict = torch.load(dataset_path)

    features = dataset_dict["features"].numpy()
    labels = dataset_dict["labels"].numpy()
    extra_infos = dataset_dict["extra_infos"].numpy()
    trajectory_lengths = dataset_dict["trajectory_lengths"]

    print(f"数据集总体信息：")
    print(f"  - 总数据点数: {len(features)}")
    print(f"  - 总轨迹数: {len(trajectory_lengths)}")
    print(f"  - 特征维度: {features.shape[1]}")
    print(f"  - 标签维度: {labels.shape[1]}")
    print(f"  - 额外信息维度: {extra_infos.shape[1]}")

    # 根据特征维度解析
    # features = [w (3维), doppler (3维)]，共6维
    w_dim = 3
    doppler_dim = 3

    w = features[:, :w_dim]  # 前3列是w（相位差相关）
    doppler = features[:, w_dim:w_dim+doppler_dim]  # 后3列是doppler

    print(f"\n{'-'*80}")
    print(f"前 {min(num_rows, len(features))} 行数据：")
    print(f"{'-'*80}\n")

    # 打印表头
    print(f"{'索引':<6} | {'W12':<15} {'W13':<15} {'W14':<15} | {'D12':<15} {'D13':<15} {'D14':<15}")
    print(f"{'-'*6}-+-{'-'*47}-+-{'-'*47}")

    # 打印数据
    for i in range(min(num_rows, len(features))):
        print(f"{i:<6} | {w[i,0]:>14.6f} {w[i,1]:>14.6f} {w[i,2]:>14.6f} | "
              f"{doppler[i,0]:>14.6f} {doppler[i,1]:>14.6f} {doppler[i,2]:>14.6f}")

    # 打印统计信息
    print(f"\n{'-'*80}")
    print(f"统计信息：")
    print(f"{'-'*80}\n")

    print("W (相位差相关) 统计：")
    print(f"  W12 - 均值: {np.mean(w[:,0]):>12.6f}, 标准差: {np.std(w[:,0]):>12.6f}, "
          f"最小值: {np.min(w[:,0]):>12.6f}, 最大值: {np.max(w[:,0]):>12.6f}")
    print(f"  W13 - 均值: {np.mean(w[:,1]):>12.6f}, 标准差: {np.std(w[:,1]):>12.6f}, "
          f"最小值: {np.min(w[:,1]):>12.6f}, 最大值: {np.max(w[:,1]):>12.6f}")
    print(f"  W14 - 均值: {np.mean(w[:,2]):>12.6f}, 标准差: {np.std(w[:,2]):>12.6f}, "
          f"最小值: {np.min(w[:,2]):>12.6f}, 最大值: {np.max(w[:,2]):>12.6f}")

    print("\nDoppler (多普勒频移) 统计：")
    print(f"  D12 - 均值: {np.mean(doppler[:,0]):>12.6f}, 标准差: {np.std(doppler[:,0]):>12.6f}, "
          f"最小值: {np.min(doppler[:,0]):>12.6f}, 最大值: {np.max(doppler[:,0]):>12.6f}")
    print(f"  D13 - 均值: {np.mean(doppler[:,1]):>12.6f}, 标准差: {np.std(doppler[:,1]):>12.6f}, "
          f"最小值: {np.min(doppler[:,1]):>12.6f}, 最大值: {np.max(doppler[:,1]):>12.6f}")
    print(f"  D14 - 均值: {np.mean(doppler[:,2]):>12.6f}, 标准差: {np.std(doppler[:,2]):>12.6f}, "
          f"最小值: {np.min(doppler[:,2]):>12.6f}, 最大值: {np.max(doppler[:,2]):>12.6f}")

    print(f"\n{'-'*80}")
    print(f"前 {min(5, len(trajectory_lengths))} 个轨迹的长度: {trajectory_lengths[:5]}")
    print(f"{'-'*80}\n")


def main():
    """主函数"""
    print("\n" + "="*80)
    print(" "*25 + "相位差和多普勒数据查看工具")
    print("="*80)

    # 打印训练集数据
    if config.stage_1_dataset_train_path:
        try:
            print_data_info(config.stage_1_dataset_train_path, num_rows=10)
        except FileNotFoundError:
            print(f"\n警告: 训练集文件不存在: {config.stage_1_dataset_train_path}")
        except Exception as e:
            print(f"\n错误: 加载训练集时出错: {e}")

    # 打印测试集数据
    if config.stage_1_dataset_test_path:
        try:
            print_data_info(config.stage_1_dataset_test_path, num_rows=10)
        except FileNotFoundError:
            print(f"\n警告: 测试集文件不存在: {config.stage_1_dataset_test_path}")
        except Exception as e:
            print(f"\n错误: 加载测试集时出错: {e}")

    print("\n" + "="*80)
    print(" "*30 + "数据查看完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
