from src.detecting_region_info import DetectingRegionInfo

# from src.features_and_labels_generator import get_d_phi
import numpy as np


def _get_distance(ref, coord_a):
    distance = np.sqrt((coord_a[0] - ref[0]) ** 2 + (coord_a[1] - ref[1]) ** 2)
    return distance


# extra里面有 三个距离和相对位置 d21,d31,d41,（ρ1，θ1），（ρ2，θ2），（ρ3，θ3）
def get_extra_infos(detecting_region_info: DetectingRegionInfo, coords_a):
    ro1, beta1 = detecting_region_info.get_ro_beta_t_r1()
    ro2, beta2 = detecting_region_info.get_ro_beta_t_r2()
    ro3, beta3 = detecting_region_info.get_ro_beta_t_r3()
    extra_infos = []
    for coord_a in coords_a:
        extra_infos.append(
            [
                _get_distance(detecting_region_info.receiver_position_1, coord_a),
                _get_distance(detecting_region_info.receiver_position_2, coord_a),
                _get_distance(detecting_region_info.receiver_position_3, coord_a),
                ro1,
                beta1,
                ro2,
                beta2,
                ro3,
                beta3,
            ]
        )

    return extra_infos
