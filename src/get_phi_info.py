import numpy as np

# 输入：区域，一组AB --> coor_a,coor_b
# 输出φ角的大小


def _calculate_angle_phi(coord_a, coord_b, origin):
    # Function to calculate the angle between vectors OA and AB for each pair of points

    vector_OA = coord_a - origin
    vector_AB = coord_b - coord_a
    dot_product = np.dot(vector_OA, vector_AB)
    norm_OA = np.linalg.norm(vector_OA)
    norm_AB = np.linalg.norm(vector_AB)
    # Avoid division by zero in case one of the vectors is zero
    if norm_OA == 0 or norm_AB == 0:
        angle = 0
    else:
        # Clip the cosine value to avoid numerical errors beyond the range [-1, 1]
        cos_angle = dot_product / (norm_OA * norm_AB)
        cos_angle_clipped = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle_clipped)
    return angle


def get_angle_phi(detecting_region_info, coord_A, coord_B):
    # Call the function and pass the arrays
    phi1 = _calculate_angle_phi(coord_A, coord_B, detecting_region_info.v1)
    phi2 = _calculate_angle_phi(coord_A, coord_B, detecting_region_info.v2)
    phi3 = _calculate_angle_phi(coord_A, coord_B, detecting_region_info.v3)
    phi4 = _calculate_angle_phi(coord_A, coord_B, detecting_region_info.v4)

    return [phi1, phi2, phi3, phi4]
