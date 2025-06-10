import torch
import os
import time
import numpy as np

import src.trajectory_generator as trajectory_generator
from src.trajectory_dataset import TrajectoryDataset
from src.w_and_doppler_generator import extract_coords_from_lines
import src.get_phi_info as get_phi_info
import src.w_and_doppler_generator as w_and_doppler_generator
import src.features_and_labels_generator as features_and_labels_generator
from src.detecting_region_info_generator import generate_detecting_region_infos
from src.extra_info_generator import get_extra_infos
import src.config as config
import src.doppler_info as doppler_info_module


def construct_dataset(
    detecting_region_info,
    lines_to_generate_per_region,
    doppler_info,
    seed,
):
    """
    Constructs the feature and label dataset for a single detecting region.
    This function remains unchanged from the original version.
    """
    lines_a, lines_b = trajectory_generator.generate_lines(
        detecting_region_info=detecting_region_info,
        num_lines=lines_to_generate_per_region,
        time_interval=config.time_interval,
        seed=seed,
    )

    coords_a = extract_coords_from_lines(lines_a)
    coords_b = extract_coords_from_lines(lines_b)

    phis_1234 = []
    for ca, cb in zip(coords_a, coords_b):
        phi1, phi2, phi3, phi4 = get_phi_info.get_angle_phi(
            detecting_region_info, ca, cb
        )
        phis_1234.append([phi1, phi2, phi3, phi4])

    w, doppler = w_and_doppler_generator.generate_w_and_doppler(
        detecting_region_info=detecting_region_info,
        doppler_info=doppler_info,
        coords_a=coords_a,
        coords_b=coords_b,
        phis_1234=phis_1234,
    )
    labels = features_and_labels_generator.get_labels(
        detecting_region_info=detecting_region_info, coords_b=coords_b
    )

    features = features_and_labels_generator.get_features(
        phis1234=phis_1234,
        w=w,
        doppler=doppler,
    )

    extra_infos = get_extra_infos(
        detecting_region_info=detecting_region_info, coords_a=coords_b
    )
    return features, labels, extra_infos


def _process_one_region(args):
    """
    A helper function to unpack arguments and call construct_dataset.
    This helps keep the main loop in get_trajectory_dataset clean.
    """
    (
        idx,
        detecting_region_info,
        lines_to_generate_per_region,
        doppler_info,
        base_seed,
    ) = args
    # Each region uses a different seed for variety in data generation.
    seed = base_seed + idx
    trajectory_dataset = construct_dataset(
        detecting_region_info, lines_to_generate_per_region, doppler_info, seed
    )
    return trajectory_dataset


def get_trajectory_dataset(
    detecting_region_nums,
    lines_to_generate_per_region,
    doppler_info,
    seed,
):
    """
    Generates the stage_1 features and labels sequentially (single-threaded).
    The ProcessPoolExecutor has been replaced with a standard for-loop.
    """
    detecting_region_infos = generate_detecting_region_infos(detecting_region_nums)
    tasks = [
        (idx, info, lines_to_generate_per_region, doppler_info, seed)
        for idx, info in enumerate(detecting_region_infos)
    ]

    results = []
    # Loop through each task sequentially instead of using a process pool.
    for task in tasks:
        result = _process_one_region(task)
        results.append(result)

    # Unzip the results from each region's dataset generation.
    features_list, labels_list, extra_info_list = zip(*results)

    # Concatenate the results from all regions into single numpy arrays.
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    extra_infos = np.concatenate(extra_info_list, axis=0)

    trajectory_dataset = TrajectoryDataset(features, labels, extra_infos)

    return trajectory_dataset


def load_dataset(path) -> TrajectoryDataset:
    """
    Loads a TrajectoryDataset from the specified path.
    """
    dataset = torch.load(path)
    print(f"从 {path} 加载了数据集，包含 {len(dataset)} 个样本")
    return dataset


def _save_dataset(path, trajectory_dataset):
    """
    Saves a TrajectoryDataset to the specified path.
    """
    torch.save(trajectory_dataset, path)
    print(f"数据集已保存至: {path}")


def _load_or_generate_trajectory_dataset(
    save_path,
    detecting_region_nums,
    lines_to_generate_per_region,
    seed=42,
) -> TrajectoryDataset:
    """
    Loads a dataset from a cache file if it exists, otherwise generates it.
    The num_workers parameter has been removed.
    """
    if os.path.exists(save_path):
        print(f"→ 找到缓存文件，开始加载：'{save_path}'")
        trajectory_dataset = load_dataset(save_path)
        return trajectory_dataset
    else:
        print(f"→ 缓存文件不存在，开始生成：'{save_path}'")
        dop_info = doppler_info_module.DopplerInfo(
            config.c, config.fc, config.time_interval
        )
        trajectory_dataset = get_trajectory_dataset(
            detecting_region_nums=detecting_region_nums,
            lines_to_generate_per_region=lines_to_generate_per_region,
            doppler_info=dop_info,
            seed=seed,
        )
        _save_dataset(save_path, trajectory_dataset)
        return trajectory_dataset


def main():
    """
    Main execution function.
    """
    os.makedirs("cache", exist_ok=True)

    presets = [
        (
            "训练集",
            config.stage_1_train_set_path,
            config.train_detecting_region_nums,
            config.train_num_of_lines_to_generate_per_region,
            config.train_seed,
        ),
        (
            "测试集",
            config.stage_1_test_set_path,
            config.test_detecting_region_nums,
            config.test_num_of_lines_to_generate_per_region,
            config.test_seed,
        ),
    ]

    print("========== stage_1 数据集 生成/加载 ==========")
    for name, path, region_nums, lines_per_region, seed in presets:
        print(f"\n--- 处理{name} ---")
        print(f"检查缓存路径：{path}")

        start = time.time()
        print("以单线程模式运行")

        _load_or_generate_trajectory_dataset(
            save_path=path,
            detecting_region_nums=region_nums,
            lines_to_generate_per_region=lines_per_region,
            seed=seed,
        )
        print(f"耗时：{time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()
