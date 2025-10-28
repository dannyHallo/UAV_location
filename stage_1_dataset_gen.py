import os
import time
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import src.config as config
from src.trajectory_dataset import TrajectoryDataset
from src.detecting_region_info_generator import generate_detecting_region_infos
import src.doppler_info as doppler_info_module
from src.trajectory_generator import generate_lines
from src.features_and_labels_generator import (
    get_features,
    get_labels,
    _extract_coords_from_lines,
)
from src.get_phi_info import get_angle_phi
from src.w_and_doppler_generator import generate_w_and_doppler
from src.extra_info_generator import get_extra_infos
from src.seed_utils import spawn_child_seeds


def construct_dataset(
    detecting_region_info, num_of_lines_to_generate, doppler_info, seed
):
    """
    为单个区域构建数据集。
    修改：现在返回每个轨迹的数据列表，而不是拼接后的大数组。
    """
    lines_a, lines_b = generate_lines(
        detecting_region_info,
        num_lines=num_of_lines_to_generate,
        time_interval=config.time_interval,
        seed=seed,
    )

    if not lines_a:
        return [], [], [], []

    list_features, list_labels, list_extra_infos = [], [], []
    trajectory_lengths = []

    for i in range(len(lines_a)):
        coords_a = lines_a[i]
        coords_b = lines_b[i]

        if coords_a.shape[0] == 0:
            continue

        phis1234 = np.array(
            [
                get_angle_phi(detecting_region_info, ca, cb)
                for ca, cb in zip(coords_a, coords_b)
            ]
        )
        w, doppler = generate_w_and_doppler(
            detecting_region_info,
            doppler_info,
            coords_a,
            coords_b,
            phis1234,
            add_noise=config.add_noise,
            snr_db=config.SNR_dB,
            t_cpi=config.T_CPI,
        )
        features = get_features(phis1234, w, doppler)
        labels = get_labels(detecting_region_info, coords_b)
        extra_infos = get_extra_infos(detecting_region_info, coords_a)

        list_features.append(features)
        list_labels.append(labels)
        list_extra_infos.append(extra_infos)
        trajectory_lengths.append(len(features))

    return list_features, list_labels, list_extra_infos, trajectory_lengths


def construct_dataset_worker(args):
    """
    多进程工作单元的包装器。
    """
    num_lines, seed, detecting_region_info, doppler_info = args
    try:
        return construct_dataset(detecting_region_info, num_lines, doppler_info, seed)
    except Exception as e:
        print(f"Error in worker process with seed {seed}: {e}")
        return [], [], [], []


def construct_dataset_parallel(
    detecting_region_info, num_total_lines: int, doppler_info, base_entropy
):
    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes (multi-threaded)...")

    seeds = spawn_child_seeds(base_entropy, num_workers)
    lines_per_worker = [num_total_lines // num_workers] * num_workers
    for i in range(num_total_lines % num_workers):
        lines_per_worker[i] += 1

    tasks = [
        (lines, seed, detecting_region_info, doppler_info)
        for lines, seed in zip(lines_per_worker, seeds)
    ]

    all_features_list, all_labels_list, all_extra_infos_list = [], [], []
    all_lengths = []

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(construct_dataset_worker, tasks),
                total=len(tasks),
                desc=f"Generating {num_total_lines} lines in parallel",
            )
        )

    for f_list, l_list, e_list, lengths in results:
        if f_list:
            all_features_list.extend(f_list)
            all_labels_list.extend(l_list)
            all_extra_infos_list.extend(e_list)
            all_lengths.extend(lengths)

    if not all_features_list:
        return np.array([]), np.array([]), np.array([]), []

    # 现在进行拼接
    final_features = np.concatenate(all_features_list, axis=0)
    final_labels = np.concatenate(all_labels_list, axis=0)
    final_extra_infos = np.concatenate(all_extra_infos_list, axis=0)

    return final_features, final_labels, final_extra_infos, all_lengths


def generate_and_save_dataset(
    dataset_path, num_regions, lines_per_region, region_seed, line_seed
):
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}. Skipping generation.")
        return

    print(f"\n--- Generating dataset for: {os.path.basename(dataset_path)} ---")
    start_time = time.time()

    doppler_info = doppler_info_module.DopplerInfo(
        config.c, config.fc, config.time_interval
    )
    detecting_region_infos = generate_detecting_region_infos(
        num_configurations=num_regions, seed=region_seed
    )
    region_seeds = spawn_child_seeds(line_seed, num_regions)

    full_features, full_labels, full_extra_infos, full_lengths = [], [], [], []

    for i, (region_info, region_entropy) in enumerate(
        zip(detecting_region_infos, region_seeds)
    ):
        print(f"Processing region {i + 1}/{num_regions} ...")
        features, labels, extra_infos, lengths = construct_dataset_parallel(
            region_info,
            int(lines_per_region),
            doppler_info,
            base_entropy=region_entropy,
        )
        if len(features) > 0:
            full_features.append(features)
            full_labels.append(labels)
            full_extra_infos.append(extra_infos)
            full_lengths.extend(lengths)

    if not full_features:
        print("Warning: No data was generated.")
        return

    final_features = np.concatenate(full_features, axis=0)
    final_labels = np.concatenate(full_labels, axis=0)
    final_extra_infos = np.concatenate(full_extra_infos, axis=0)

    # 不再保存 TrajectoryDataset 对象，而是保存一个包含所有信息的字典
    dataset_dict = {
        "features": torch.tensor(final_features, dtype=torch.float32),
        "labels": torch.tensor(final_labels, dtype=torch.float32),
        "extra_infos": torch.tensor(final_extra_infos, dtype=torch.float32),
        "trajectory_lengths": full_lengths,
    }

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save(dataset_dict, dataset_path)

    end_time = time.time()
    print(f"Successfully generated and saved dataset to {dataset_path}")
    print(f"Total generated data points: {len(final_features)}")
    print(f"Total number of trajectories: {len(full_lengths)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


def load_dataset(path):
    """从指定路径加载数据集字典。"""
    print(f"Loading dataset from {path}...")
    # 我们信任自己生成的文件
    return torch.load(path)


def main():
    print("Starting Parallel Dataset Generation")

    os.makedirs("dataset", exist_ok=True)

    generate_and_save_dataset(
        dataset_path=config.stage_1_dataset_train_path,
        num_regions=config.num_detecting_regions,
        lines_per_region=config.num_lines_to_generate_per_region
        * config.train_test_split_ratio,
        region_seed=config.region_seed,
        line_seed=config.line_seed_train,
    )

    generate_and_save_dataset(
        dataset_path=config.stage_1_dataset_test_path,
        num_regions=config.num_detecting_regions,
        lines_per_region=config.num_lines_to_generate_per_region
        * (1 - config.train_test_split_ratio),
        region_seed=config.region_seed,
        line_seed=config.line_seed_test,
    )

    print("\nAll datasets generated.")


if __name__ == "__main__":
    main()
