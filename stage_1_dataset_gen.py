# 并行化前需要的 imports
import multiprocessing
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import src.trajectory_generator as trajectory_generator
from src.w_and_doppler_generator import extract_coords_from_lines
import src.get_phi_info as get_phi_info
import src.w_and_doppler_generator as w_and_doppler_generator
import src.features_and_labels_generator as features_and_labels_generator
from src.detecting_region_info_generator import generate_detecting_region_infos  # 假设这个函数在这里
import src.Config as config
import src.doppler_info as doppler_info

# 1) 将对单个 detecting_region_info 的处理抽出来
def _process_one_region(args):
    idx, detecting_region_info, lines_to_generate_per_region, doppler_info, base_seed = args

    # 可选：为每个 region 使用不同 seed，避免所有子进程随机数相同
    seed = base_seed + idx

    # 生成两条轨迹
    lines_a, lines_b = trajectory_generator.generate_lines(
        detecting_region_info=detecting_region_info,
        time_interval=config.time_interval,
        num_lines=lines_to_generate_per_region,
        seed=seed
    )

    # 提取坐标
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
    w, doppler = w_and_doppler_generator.generateWAndDoppler(
        detecting_region_info=detecting_region_info,
        doppler_info=doppler_info,
        coords_a=coords_a,
        coords_b=coords_b,
        phis_1234=phis_1234,
    )

    # 生成特征和标签
    feature_stage1, phis_label_stage1 = features_and_labels_generator.generateFeaturesAndLabelsStage1(
        phis1234=phis_1234,
        w=w,
        doppler=doppler,
        detecting_region_info=detecting_region_info
    )

    return feature_stage1, phis_label_stage1


# 2) 并行版的主函数
def get_features_and_labels(detecting_region_nums,
                            lines_to_generate_per_region,
                            doppler_info,
                            seed,
                            num_workers=4):
    # 生成所有检测区的信息
    detecting_region_infos = generate_detecting_region_infos(detecting_region_nums)

    # 为每个 region 构造一个任务元组 (idx, region_info, lines_per_region, seed)
    tasks = [
        (idx, info, lines_to_generate_per_region, doppler_info, seed)
        for idx, info in enumerate(detecting_region_infos)
    ]

    # 启动进程池并行执行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_one_region, tasks))

    # 将所有子任务的输出拆分并合并
    features_list, labels_list = zip(*results)
    features_stage1 = np.concatenate(features_list, axis=0)
    phis_labels_stage1 = np.concatenate(labels_list, axis=0)

    return features_stage1, phis_labels_stage1

if __name__ == "__main__":
    print("开始生成 stage1 数据...")

    doppler_info = doppler_info.DopplerInfo(config.c, config.fc, config.time_interval)
    
    start_time = time.time()

    cpu_cnt = multiprocessing.cpu_count() or 1
    # CPU 密集型：留 1 核给系统；IO 密集型可以直接用 cpu_cnt * 2
    num_workers = max(1, cpu_cnt - 1)
    print(f"检测到 {cpu_cnt} 个 CPU 核心")
    print(f"使用 {num_workers} 个进程进行数据生成")
    
    # in benchmark, it can speed up to 7x

    features, labels = get_features_and_labels(
        detecting_region_nums=config.train_detecting_region_nums,
        lines_to_generate_per_region=config.train_num_of_lines_to_generate_per_region,
        doppler_info=doppler_info,
        seed=42,
        num_workers=num_workers
    )
    
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    
    print("数据生成完成，耗时：", time.time() - start_time, "秒")
