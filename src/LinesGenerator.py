import numpy as np
import src.LineGenUtils as line_utils

def get_random_sample_in_detecting_region(detecting_region_info, rng=None):
    """
    Return a random point within the detecting region.
    Optionally use a user-provided random number generator (rng).
    """
    if rng is None:
        rng = np.random

    v1 = detecting_region_info.v1
    v2 = detecting_region_info.v2
    v3 = detecting_region_info.v3
    v4 = detecting_region_info.v4

    tri_size_1 = line_utils.getTriangleSize(v1, v2, v3)
    tri_size_2 = line_utils.getTriangleSize(v2, v3, v4)

    # Randomly choose which triangle to sample from,
    # weighted by respective triangle areas
    if rng.rand() < tri_size_1 / (tri_size_1 + tri_size_2):
        # Triangle 1 (v1, v2, v3)
        base_point = v1
        edge0 = v2 - v1
        edge1 = v3 - v1
    else:
        # Triangle 2 (v2, v3, v4)
        base_point = v2
        edge0 = v3 - v2
        edge1 = v4 - v2

    # Generate random coordinates in [0, 1]
    x, y = rng.rand(2)
    # Ensure the point is inside the triangle
    if x + y > 1:
        x = 1 - x
        y = 1 - y

    return base_point + edge0 * x + edge1 * y


def generate_bspline_control_points(detecting_region_info,num_control_points,rng=None):
    '''Generate random control points within the detecting region'''
    if rng is None:
        rng = np.random
    
    control_points = []
    for _ in range(num_control_points):
        point = get_random_sample_in_detecting_region(detecting_region_info,rng=rng)
        control_points.append(point)
    return np.array(control_points)

def random_angle_change(original_angle, angle_change_tolerance, rng=None):
    """
    Return a new angle by randomly perturbing the original angle
    within the provided tolerance. Optionally use rng.
    """
    if rng is None:
        rng = np.random

    ang = original_angle + (rng.rand() - 0.5) * 2 * angle_change_tolerance
    # Ensure the angle is within [0, 2π]
    if ang < 0:
        ang += 2 * np.pi
    elif ang > 2 * np.pi:
        ang -= 2 * np.pi
    return ang


def random_walk(
    angle_change_per_step, length_per_step, last_angle, last_coords_a, rng=None
):
    """
    Single step of a random walk.
    """
    if rng is None:
        rng = np.random

    # New A position
    this_coords_a = (
        last_coords_a
        + np.array([np.cos(last_angle), np.sin(last_angle)]) * length_per_step
    )
    # Random angle for next step
    this_angle = random_angle_change(last_angle, angle_change_per_step, rng=rng)
    return [this_angle, this_coords_a]


def get_coords_b(a_b_distance, coords_a, angle):
    return coords_a + np.array([np.cos(angle), np.sin(angle)]) * a_b_distance


def try_create_line_in_bounding_box(
    a_b_distance,
    detecting_region_info,
    step_count_per_line,
    length_per_step,
    angle_change_limit_per_step,
    rng=None,
):
    """
    Attempt to create a single random line (A-trajectory and B-trajectory).
    Return line_a, line_b, and a boolean indicating if the line is valid.
    """
    if rng is None:
        rng = np.random

    # Start from a random point inside the boundary
    _angle = rng.rand() * 2 * np.pi
    _coords_a = get_random_sample_in_detecting_region(detecting_region_info, rng=rng)
    _coords_b = get_coords_b(a_b_distance, _coords_a, _angle)

    in_boundary = line_utils.pointInRegion(
        detecting_region_info, _coords_a
    ) and line_utils.pointInRegion(detecting_region_info, _coords_b)
    if not in_boundary:
        return [None, None, False]

    line_a = [_coords_a]
    line_b = [_coords_b]

    # Randomly walk until we fill step_count_per_line points or exit boundary
    while True:
        [_angle, _coords_a] = random_walk(
            angle_change_limit_per_step, length_per_step, _angle, _coords_a, rng=rng
        )
        _coords_b = get_coords_b(a_b_distance, _coords_a, _angle)

        in_boundary = line_utils.pointInRegion(
            detecting_region_info, _coords_a
        ) and line_utils.pointInRegion(detecting_region_info, _coords_b)
        if not in_boundary:
            return [None, None, False]

        line_a.append(_coords_a)
        line_b.append(_coords_b)

        if len(line_a) == step_count_per_line:
            return [line_a, line_b, True]


def generateLines(
    detecting_region_info,
    a_b_distance,
    num_of_lines_to_generate,
    step_count_per_line,
    length_per_step,
    angle_change_limit_per_step,
    seed=None,
):
    """
    Generate multiple random lines (A-trajectory, B-trajectory).
    Optionally use a fixed random seed.
    """
    if seed is not None:
        # You can also use np.random.default_rng(seed) if you like reproducibility via a Generator
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    valid_line_count = 0
    lines_a = []
    lines_b = []

    while valid_line_count < num_of_lines_to_generate:
        [line_a, line_b, valid] = try_create_line_in_bounding_box(
            a_b_distance,
            detecting_region_info,
            step_count_per_line,
            length_per_step,
            angle_change_limit_per_step,
            rng=rng,
        )

        if valid:
            valid_line_count += 1
            lines_a.append(line_a)
            lines_b.append(line_b)

    lines_a = np.array(lines_a)
    lines_b = np.array(lines_b)

    return [lines_a, lines_b]


def reparseSingleLineAsLines(
    line_a, line_b, num_of_lines_to_generate, step_count_per_line, seed=None
):
    """
    Take a single long line (A, B) with multiple points,
    and split it into num_of_lines_to_generate smaller lines
    each of length step_count_per_line.

    Optionally use a random seed if you happen to add randomness
    in the future (this version doesn't actually have random calls).
    """
    if seed is not None:
        # Currently we do not have any random function calls here,
        # but let's seed in case you want to add random logic in the future.
        np.random.seed(seed)

    lines_a = []
    lines_b = []

    point_size_of_line_a = len(line_a)
    print("point_size_of_line_a: ", point_size_of_line_a)

    assert (
        num_of_lines_to_generate + (step_count_per_line - 1)
    ) <= point_size_of_line_a, "Not enough points to carve out the requested lines."

    cursor = 0
    while cursor < num_of_lines_to_generate:
        lines_a.append(line_a[cursor : cursor + step_count_per_line])
        lines_b.append(line_b[cursor : cursor + step_count_per_line])
        cursor += 1

    return [lines_a, lines_b]
