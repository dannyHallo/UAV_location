import numpy as np
import random
from typing import List
from src.detecting_region_info import DetectingRegionInfo


class TrajectoryGenerator:
    def __init__(self, velocity=20.0, max_acceleration=25.0, name=None):
        """初始化轨迹生成器"""
        self.velocity = velocity
        self.max_acceleration = max_acceleration
        self.name = name
        self.trajectory = []
        self.timestamps = []
        self.velocities = []
        self.accelerations = []
        self.time = 0.0
        self.key_points = []
        self.key_times = []

    def _add_straight_line(self, start_point, end_point, duration):
        """添加匀速直线段"""
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        if len(self.key_points) == 0:
            self.key_points.append(start_point)
            self.key_times.append(self.time)
        self.key_points.append(end_point)
        self.key_times.append(self.time + duration)

        num_samples = max(int(duration * 10), 2)
        times = np.linspace(0, duration, num_samples)

        for t in times:
            alpha = t / duration
            position = (1 - alpha) * start_point + alpha * end_point
            self.trajectory.append(position)
            self.timestamps.append(self.time + t)

        velocity_vector = (end_point - start_point) / duration
        speed = np.linalg.norm(velocity_vector)
        if speed > 0:
            velocity_unit = velocity_vector / speed
        else:
            velocity_unit = np.array([1.0, 0.0])

        self.last_velocity_direction = velocity_unit

        for _ in range(num_samples):
            self.velocities.append(velocity_unit * self.velocity)
            self.accelerations.append(np.zeros(2))

        self.time += duration
        return velocity_unit

    def _add_curved_transition(self, start_point, end_point, start_direction, end_direction, control_point_offset_factor=0.7):
        """添加确保速度连续性的平滑弯曲过渡"""
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        start_direction = np.array(start_direction)
        end_direction = np.array(end_direction)

        if np.allclose(start_point, end_point):
            return None

        if len(self.key_points) == 0:
            self.key_points.append(start_point)
            self.key_times.append(self.time)

        if np.linalg.norm(start_direction) > 0:
            start_direction = start_direction / np.linalg.norm(start_direction)
        else:
            start_direction = np.array([1.0, 0.0])

        if np.linalg.norm(end_direction) > 0:
            end_direction = end_direction / np.linalg.norm(end_direction)
        else:
            end_direction = np.array([1.0, 0.0])

        direct_distance = np.linalg.norm(end_point - start_point)

        control_distance_1 = direct_distance * control_point_offset_factor
        control_point_1 = start_point + start_direction * control_distance_1

        control_distance_2 = direct_distance * control_point_offset_factor
        control_point_2 = end_point - end_direction * control_distance_2

        curve_length = (np.linalg.norm(control_point_1 - start_point) +
                        np.linalg.norm(control_point_2 - control_point_1) +
                        np.linalg.norm(end_point - control_point_2))

        duration = curve_length / self.velocity

        self.key_points.append(end_point)
        self.key_times.append(self.time + duration)

        num_samples = max(int(duration * 30), 30)
        t_values = np.linspace(0, 1, num_samples)

        positions = []
        velocities = []
        accelerations = []

        for t in t_values:
            b0 = (1-t)**3
            b1 = 3*(1-t)**2*t
            b2 = 3*(1-t)*t**2
            b3 = t**3

            position = b0 * start_point + b1 * control_point_1 + b2 * control_point_2 + b3 * end_point
            positions.append(position)

            v0 = 3*(1-t)**2
            v1 = 6*(1-t)*t
            v2 = 3*t**2

            velocity_dir = v0 * (control_point_1 - start_point) + \
                          v1 * (control_point_2 - control_point_1) + \
                          v2 * (end_point - control_point_2)

            speed = np.linalg.norm(velocity_dir)
            if speed > 1e-10:
                velocity_dir = velocity_dir / speed
                velocities.append(velocity_dir * self.velocity)
            else:
                velocities.append(np.array([0.0, 0.0]))

            a0 = 6*(1-t)
            a1 = 6*t

            acceleration = a0 * (control_point_2 - 2*control_point_1 + start_point) + \
                          a1 * (end_point - 2*control_point_2 + control_point_1)

            accel_magnitude = self.velocity**2 * np.linalg.norm(acceleration) / (speed**2) if speed > 1e-10 else 0

            if accel_magnitude > self.max_acceleration:
                accel_magnitude = self.max_acceleration

            if speed > 1e-10:
                normal_vector = np.array([-velocity_dir[1], velocity_dir[0]])

                if np.dot(normal_vector, acceleration) < 0:
                    normal_vector = -normal_vector

                accelerations.append(normal_vector * accel_magnitude)
            else:
                accelerations.append(np.array([0.0, 0.0]))

        for i in range(num_samples):
            self.trajectory.append(positions[i])
            self.timestamps.append(self.time + duration * t_values[i])
            self.velocities.append(velocities[i])
            self.accelerations.append(accelerations[i])

        self.last_velocity_direction = velocities[-1] / self.velocity if velocities else end_direction

        self.time += duration

        return self.last_velocity_direction


def generate_trajectories_in_region(detecting_region_info, num_trajectories, seed, inner_scale_factor=0.7):
    """
    Generates a list of Trajectory objects within a scaled inner region.

    Args:
        detecting_region_info: The object containing the outer boundary info.
        num_trajectories: The number of trajectories to generate.
        seed: The base random seed for reproducibility.
        inner_scale_factor: Factor to scale the inner generation area.

    Returns:
        A tuple containing:
        - list: A list of generated TrajectoryGenerator objects.
        - list: The vertices of the outer boundary.
        - list: The vertices of the inner generation boundary.
    """
    outer_vertices = _get_quad_vertices_from_detecting_region_info(detecting_region_info)
    generation_boundary = _get_inner_quad_vertices(outer_vertices, inner_scale_factor)
    
    # Use the seed-incrementing generator
    trajectories = _generate_multiple_trajectories(num_trajectories, generation_boundary, seed)
    
    return trajectories, outer_vertices, generation_boundary


def _get_inner_quad_vertices(quad_vertices, scale_factor):
    """
    Calculates the vertices of an inner quadrilateral, scaled towards the centroid.
    """
    centroid = np.mean(quad_vertices, axis=0)
    inner_vertices = [centroid + (vertex - centroid) * scale_factor for vertex in quad_vertices]
    return inner_vertices


def generate_lines(detecting_region_info, num_lines=10, time_interval=0.5, seed=42, inner_scale_factor=0.7):
    """
    Generates pairs of lines (trajectories) within a scaled-down inner region.
    This function remains for other parts of the project that may depend on it.
    """
    # This now uses the same underlying logic as the visualizer
    trajectories, _, _ = generate_trajectories_in_region(
        detecting_region_info, num_lines, seed, inner_scale_factor
    )
    
    all_lines_a = []
    all_lines_b = []
    
    for trajectory in trajectories:
        line_a, line_b = _generate_lines_from_trajectory(trajectory, time_interval)
        if len(line_a) > 0:
            all_lines_a.append(line_a)
            all_lines_b.append(line_b)
    
    return all_lines_a, all_lines_b


def part_lines(step_count_per_line, long_lines):
    """
    将多条长线分别划分成短线段
    """
    all_short_lines = []
    for long_line in long_lines:
        short_lines = _part_line(step_count_per_line, long_line)
        all_short_lines.extend(short_lines)
    return all_short_lines


def _get_quad_vertices_from_detecting_region_info(
    info: DetectingRegionInfo
) -> List[np.ndarray]:
    """从一个 DetectingRegionInfo 对象中提取四边形的 4 个顶点坐标"""
    return [
        info.transmittor_position,
        info.receiver_position_1,
        info.receiver_position_2,
        info.receiver_position_3,
    ]


def _is_point_inside_quadrilateral(point, quad_vertices):
    """检查点是否在四边形内"""
    x, y = point
    n = len(quad_vertices)
    inside = False
    p1x, p1y = quad_vertices[0]
    for i in range(n + 1):
        p2x, p2y = quad_vertices[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def _is_bezier_curve_inside_quadrilateral(p0, p1, p2, p3, quad_vertices, num_checks=20):
    """检查三次贝塞尔曲线是否在四边形内"""
    for i in range(num_checks + 1):
        t = i / num_checks
        point = (1-t)**3 * np.array(p0) + \
                3*(1-t)**2*t * np.array(p1) + \
                3*(1-t)*t**2 * np.array(p2) + \
                t**3 * np.array(p3)
        if not _is_point_inside_quadrilateral(point.tolist(), quad_vertices):
            return False
    return True


def _generate_random_point_inside_quadrilateral(quad_vertices):
    """在四边形内生成随机点"""
    min_x = min(v[0] for v in quad_vertices)
    max_x = max(v[0] for v in quad_vertices)
    min_y = min(v[1] for v in quad_vertices)
    max_y = max(v[1] for v in quad_vertices)

    for _ in range(100):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        point = [x, y]
        if _is_point_inside_quadrilateral(point, quad_vertices):
            return point
    return np.mean(quad_vertices, axis=0).tolist()


def _generate_trajectory_with_abcd_in_quadrilateral(boundary_vertices, max_tries=100):
    """生成一条符合要求的轨迹"""
    for _ in range(max_tries):
        velocity = random.uniform(15.0, 25.0)
        max_acceleration = random.uniform(20.0, 30.0)
        generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)
        points = [_generate_random_point_inside_quadrilateral(boundary_vertices) for _ in range(4)]

        ab_vector = np.array(points[1]) - np.array(points[0])
        cd_vector = np.array(points[3]) - np.array(points[2])
        if np.linalg.norm(ab_vector) < 5.0 or np.linalg.norm(cd_vector) < 5.0:
            continue

        ab_direction = ab_vector / np.linalg.norm(ab_vector)
        cd_direction = cd_vector / np.linalg.norm(cd_vector)
        bc_vector = np.array(points[2]) - np.array(points[1])
        if np.linalg.norm(bc_vector) < 10.0:
            continue

        control_point_1 = np.array(points[1]) + ab_direction * (np.linalg.norm(bc_vector) * 0.5)
        control_point_2 = np.array(points[2]) - cd_direction * (np.linalg.norm(bc_vector) * 0.5)

        if not _is_point_inside_quadrilateral(control_point_1.tolist(), boundary_vertices) or \
           not _is_point_inside_quadrilateral(control_point_2.tolist(), boundary_vertices) or \
           not _is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(),
                                                 control_point_2.tolist(), points[2], boundary_vertices):
            continue

        try:
            b_vel_dir = generator._add_straight_line(points[0], points[1], np.linalg.norm(ab_vector) / velocity)
            generator._add_curved_transition(points[1], points[2], b_vel_dir, cd_direction)
            generator._add_straight_line(points[2], points[3], np.linalg.norm(cd_vector) / velocity)
            return generator
        except Exception:
            continue
    return None


def _generate_multiple_trajectories(num_trajectories, generation_boundary, base_seed):
    """生成多条轨迹, 每次生成使用递增的种子"""
    trajectories = []
    for i in range(num_trajectories):
        current_seed = base_seed + i
        random.seed(current_seed)
        np.random.seed(current_seed)
        
        trajectory = _generate_trajectory_with_abcd_in_quadrilateral(generation_boundary)
        if trajectory:
            trajectory.name = f"Trajectory {i+1}"
            trajectories.append(trajectory)
    return trajectories


def _sample_trajectory_at_time_interval(trajectory, time_interval):
    """按固定时间间隔采样轨迹点，使用线性插值保证点位置准确"""
    if not trajectory.trajectory:
        return np.array([])
        
    trajectory_points = np.array(trajectory.trajectory)
    timestamps = np.array(trajectory.timestamps)
    
    if not timestamps.any():
        return np.array([])
        
    total_time = timestamps[-1]
    
    sample_times = np.arange(0, total_time, time_interval)
    if not sample_times.size or sample_times[-1] < total_time:
        sample_times = np.append(sample_times, total_time)
    
    sampled_points = []
    
    for t in sample_times:
        # Find the interval t is in
        next_idx = np.searchsorted(timestamps, t)
        
        if next_idx == 0:
            # Before the first timestamp
            sampled_points.append(trajectory_points[0])
        elif next_idx == len(timestamps):
            # After the last timestamp
            sampled_points.append(trajectory_points[-1])
        else:
            # Between two timestamps, perform linear interpolation
            prev_idx = next_idx - 1
            prev_time = timestamps[prev_idx]
            next_time = timestamps[next_idx]
            
            # Handle the case where t is exactly on a timestamp
            if next_time == prev_time:
                 alpha = 0
            else:
                 alpha = (t - prev_time) / (next_time - prev_time)

            prev_point = trajectory_points[prev_idx]
            next_point = trajectory_points[next_idx]
            
            interpolated_point = prev_point * (1 - alpha) + next_point * alpha
            sampled_points.append(interpolated_point)
    
    return np.array(sampled_points)


def _generate_lines_from_trajectory(trajectory_generator, time_interval):
    """从轨迹生成器按时间间隔t生成相邻点的line_a和line_b"""
    sampled_points = _sample_trajectory_at_time_interval(trajectory_generator, time_interval)
    if len(sampled_points) < 2:
        return np.array([]), np.array([])
    return sampled_points[:-1], sampled_points[1:]


def _part_line(step_count_per_line, long_line):
    """将一条长线分成多条短线段"""
    if len(long_line) < step_count_per_line:
        return []
    num_lines = len(long_line) - step_count_per_line + 1
    return [long_line[i:i+step_count_per_line] for i in range(num_lines)]
