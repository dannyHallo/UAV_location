import torch
import os
import multiprocessing
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
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
    # 生成两条轨迹
    lines_a, lines_b = trajectory_generator.generate_lines(
        detecting_region_info=detecting_region_info,
        num_lines=lines_to_generate_per_region,
        time_interval=config.time_interval,
        seed=seed,
    )

    coords_a = extract_coords_from_lines(lines_a)
    coords_b = extract_coords_from_lines(lines_b)

    # 计算 phi1..4
    phis_1234 = []
    for ca, cb in zip(coords_a, coords_b):
        phi1, phi2, phi3, phi4 = get_phi_info.get_angle_phi(
            detecting_region_info, ca, cb
        )
        phis_1234.append([phi1, phi2, phi3, phi4])

    # 计算 w 和 doppler
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

    # 生成特征和标签
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
    (
        idx,
        detecting_region_info,
        lines_to_generate_per_region,
        doppler_info,
        base_seed,
    ) = args
    # 每个 region 用不同 seed
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
    num_workers=None,
):
    """
    并行生成 stage_1 的 features 和 labels。
    num_workers=None 时自动取 cpu_count()-1。
    """
    if num_workers is None:
        cpu_cnt = multiprocessing.cpu_count() or 1
        num_workers = max(1, cpu_cnt - 1)

    detecting_region_infos = generate_detecting_region_infos(detecting_region_nums)
    tasks = [
        (idx, info, lines_to_generate_per_region, doppler_info, seed)
        for idx, info in enumerate(detecting_region_infos)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_one_region, tasks))

    features_list, labels_list, extra_info_list = zip(*results)
    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    extra_infos = np.concatenate(extra_info_list, axis=0)
    trajectory_dataset = TrajectoryDataset(features, labels, extra_infos)

    return trajectory_dataset


def load_dataset(path) -> TrajectoryDataset:
    dataset = torch.load(path)
    print(f"从 {path} 加载了数据集，包含 {len(dataset)} 个样本")
    return dataset


def _save_dataset(path, trajectory_dataset):
    torch.save(trajectory_dataset, path)
    print(f"数据集已保存至: {path}")


def _load_or_generate_trajectory_dataset(
    save_path,
    detecting_region_nums,
    lines_to_generate_per_region,
    seed=42,
    num_workers=None,
) -> TrajectoryDataset:
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
            num_workers=num_workers,
        )
        _save_dataset(save_path, trajectory_dataset)
        return trajectory_dataset


def _get_best_worker_count() -> tuple[int, int]:
    """
    Returns: (num_workers, cpu_count)
    """
    cpu_cnt = multiprocessing.cpu_count() or 1
    return max(1, cpu_cnt - 1), cpu_cnt


def main():
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
        num_workers, cpu_cnt = _get_best_worker_count()
        print(f"检测到 {cpu_cnt} 核心，使用 {num_workers} 个进程并行")

        _load_or_generate_trajectory_dataset(
            save_path=path,
            detecting_region_nums=region_nums,
            lines_to_generate_per_region=lines_per_region,
            seed=seed,
            num_workers=num_workers,
        )
        print(f"耗时：{time.time() - start:.2f} 秒")


if __name__ == "__main__":
    main()
