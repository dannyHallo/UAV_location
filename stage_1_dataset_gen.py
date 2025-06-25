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


def construct_dataset(
    detecting_region_info, num_of_lines_to_generate, doppler_info, seed
):
    """
    Constructs a dataset for a single detecting region.
    This function contains the core logic that will be executed by each worker.
    """
    # 1. Generate trajectories (lines) for the given region
    lines_a, lines_b = generate_lines(
        detecting_region_info,
        num_lines=num_of_lines_to_generate,
        time_interval=config.time_interval,
        seed=seed,
    )

    if not lines_a:
        return None, None, None

    # 2. Extract individual coordinates from the trajectories
    coords_a = _extract_coords_from_lines(lines_a)
    coords_b = _extract_coords_from_lines(lines_b)

    if coords_a.shape[0] == 0:
        return None, None, None

    # 3. Calculate phi angles for each coordinate pair
    phis1234 = np.array(
        [
            get_angle_phi(detecting_region_info, ca, cb)
            for ca, cb in zip(coords_a, coords_b)
        ]
    )

    # 4. Generate w and doppler values
    w, doppler = generate_w_and_doppler(
        detecting_region_info, doppler_info, coords_a, coords_b, phis1234
    )

    # 5. Assemble final features and labels
    features = get_features(phis1234, w, doppler)
    labels = get_labels(detecting_region_info, coords_b)
    extra_infos = get_extra_infos(detecting_region_info, coords_a)

    return features, labels, extra_infos


def construct_dataset_worker(args):
    """
    A wrapper function for the multiprocessing Pool. It unpacks arguments
    and calls the main dataset construction logic.
    """
    num_lines, seed, detecting_region_info, doppler_info = args
    try:
        return construct_dataset(detecting_region_info, num_lines, doppler_info, seed)
    except Exception as e:
        print(f"Error in worker process with seed {seed}: {e}")
        return None, None, None


def construct_dataset_parallel(
    detecting_region_info, num_total_lines, doppler_info, base_seed
):
    """
    Generates a dataset in parallel by distributing the line generation
    across multiple CPU cores.
    """
    # Determine the number of worker processes, leaving one core free.
    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} worker processes (multi-threaded)...")

    # Divide the total number of lines among the workers
    lines_per_worker = [num_total_lines // num_workers] * num_workers
    for i in range(num_total_lines % num_workers):
        lines_per_worker[i] += 1

    # Generate a unique seed for each worker
    seeds = [base_seed + i for i in range(num_workers)]
    tasks = [
        (lines, seed, detecting_region_info, doppler_info)
        for lines, seed in zip(lines_per_worker, seeds)
    ]

    all_features, all_labels, all_extra_infos = [], [], []

    with Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(construct_dataset_worker, tasks),
                total=len(tasks),
                desc=f"Generating {num_total_lines} lines in parallel",
            )
        )

    for features, labels, extra_infos in results:
        if features is not None and len(features) > 0:
            all_features.append(features)
            all_labels.append(labels)
            all_extra_infos.append(extra_infos)

    if not all_features:
        return np.array([]), np.array([]), np.array([])

    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    final_extra_infos = np.concatenate(all_extra_infos, axis=0)

    return final_features, final_labels, final_extra_infos


def generate_and_save_dataset(
    dataset_path, num_regions, lines_per_region, region_seed, line_seed
):
    """
    Main function to orchestrate the generation and saving of a dataset
    using parallel processing.
    """
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

    full_features, full_labels, full_extra_infos = [], [], []

    for i, region_info in enumerate(detecting_region_infos):
        print(f"Processing region {i+1}/{num_regions}...")
        line_seed = line_seed + num_regions + i

        # Directly call the parallel constructor
        features, labels, extra_infos = construct_dataset_parallel(
            region_info, lines_per_region, doppler_info, line_seed
        )

        if len(features) > 0:
            full_features.append(features)
            full_labels.append(labels)
            full_extra_infos.append(extra_infos)

    if not full_features:
        print("Warning: No data was generated.")
        return

    final_features = np.concatenate(full_features, axis=0)
    final_labels = np.concatenate(full_labels, axis=0)
    final_extra_infos = np.concatenate(full_extra_infos, axis=0)

    dataset = TrajectoryDataset(final_features, final_labels, final_extra_infos)
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    torch.save(dataset, dataset_path)

    end_time = time.time()
    print(f"Successfully generated and saved dataset to {dataset_path}")
    print(f"Total generated data points: {len(final_features)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")


def load_dataset(path):
    """Loads a dataset from the specified path."""
    print(f"Loading dataset from {path}...")
    # FIX: Explicitly set weights_only=False to silence the FutureWarning.
    # This is safe because we are loading a trusted file that we generated ourselves,
    # and we need to load the full TrajectoryDataset object, not just tensors.
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    print("Starting Parallel Dataset Generation")

    os.makedirs("dataset", exist_ok=True)

    generate_and_save_dataset(
        dataset_path=config.stage_1_dataset_path,
        num_regions=config.num_detecting_regions,
        lines_per_region=config.num_lines_to_generate_per_region,
        region_seed=config.region_seed,
        line_seed=config.line_seed,
    )

    print("\nAll datasets generated.")
