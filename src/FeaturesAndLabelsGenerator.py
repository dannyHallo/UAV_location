import numpy as np


def get_d_phi(detecting_region_info, coord_a):
    ref = detecting_region_info.transmittor_position
    phi = np.arctan2(coord_a[1] - ref[1], coord_a[0] - ref[0])
    # change range from -pi to pi to 0 to 2pi
    if phi < 0:
        phi += 2 * np.pi
    d = np.sqrt((coord_a[0] - ref[0]) ** 2 + (coord_a[1] - ref[1]) ** 2)
    return [d, phi]


def get_labels(detecting_region_info, lines_a):
    labels = []
    for line in lines_a:
        for coord_a in line:
            labels.append(get_d_phi(detecting_region_info, coord_a))
    return np.array(labels)


def get_labels_coords(lines_a):
    labels = []
    for line in lines_a:
        for coord_a in line:
            labels.append(coord_a)
    return np.array(labels)


def generateFeaturesAndLabels(
    detecting_region_info,
    lines_a,
    w,
    doppler,
    num_of_lines_to_generate,
    step_count_per_line,
):
    features = np.concatenate((w, doppler), axis=1)
    # labels = get_labels(detecting_region_info, lines_a)
    labels = get_labels_coords(lines_a)
    reshaped_features = features.reshape(
        num_of_lines_to_generate, step_count_per_line, 6
    )
    reshaped_labels = labels.reshape(num_of_lines_to_generate, step_count_per_line, 2)

    # drop the sequence of the label
    reshaped_labels = reshaped_labels[:, -1, :]

    return [reshaped_features, reshaped_labels]


# 定义计算距离的函数
def calculate_distances(train_label, detecting_region_info):
    # 提取接收器位置
    receiver_1 = detecting_region_info.receiver_position_1
    receiver_2 = detecting_region_info.receiver_position_2
    receiver_3 = detecting_region_info.receiver_position_3

    # 计算 train_label 到 receiver_1, receiver_2, receiver_3 的欧几里得距离
    distances_to_r1 = np.sqrt(
        (train_label[:, 0] - receiver_1[0]) ** 2
        + (train_label[:, 1] - receiver_1[1]) ** 2
    )
    distances_to_r2 = np.sqrt(
        (train_label[:, 0] - receiver_2[0]) ** 2
        + (train_label[:, 1] - receiver_2[1]) ** 2
    )
    distances_to_r3 = np.sqrt(
        (train_label[:, 0] - receiver_3[0]) ** 2
        + (train_label[:, 1] - receiver_3[1]) ** 2
    )

    # 将结果组合成 (n, 3) 形状的数组
    distances = np.stack((distances_to_r1, distances_to_r2, distances_to_r3), axis=1)

    return distances
