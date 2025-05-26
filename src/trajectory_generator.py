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


def generate_lines(detecting_region_info, num_lines=10, time_interval=0.5, seed=42):
    """生成指定数量的线对
    
    Args:
        detecting_region_info: 检测区域信息
        num_lines: 生成的轨迹条数
        time_interval: 轨迹采样时间间隔，单位为秒
        seed: 随机种子
    
    Returns:
        tuple: (lines_a, lines_b)，两个list，每个list中包含多条轨迹
        轨迹是相邻时间点的位置坐标
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 生成轨迹
    trajectories = _generate_multiple_trajectories(num_lines, detecting_region_info)
    
    # 初始化存储所有轨迹的列表
    all_lines_a = []
    all_lines_b = []
    
    # 为每条轨迹生成对应的线对
    for trajectory in trajectories:
        line_a, line_b = _generate_lines_from_trajectory(trajectory, time_interval)
        if len(line_a) > 0:
            all_lines_a.append(line_a)
            all_lines_b.append(line_b)
    
    return all_lines_a, all_lines_b

def part_lines(step_count_per_line, long_lines):
    """
    将多条长线分别划分成短线段
    
    参数:
        step_count_per_line (int): 每条短线段包含的点数
        long_lines (list): 包含多条长线的列表，每条长线是一个形状为(n, 2)的numpy数组
    
    返回:
        list: 包含所有短线段的列表，每条短线段是一个numpy数组，形状为(step_count_per_line, 2)
    """
    all_short_lines = []
    
    # 处理每条长线
    for long_line in long_lines:
        # 调用part_line函数将当前长线划分成短线段
        short_lines = _part_line(step_count_per_line, long_line)
        
        # 将当前长线生成的短线段添加到总列表中
        all_short_lines.extend(short_lines)
    
    return all_short_lines


def _get_quad_vertices_from_detecting_region_info(
    info: DetectingRegionInfo
) -> List[np.ndarray]:
    """从一个 DetectingRegionInfo 对象中提取四边形的 4 个顶点坐标（v1,v2,v3,v4）"""
    # 如果你需要按画图时的顺序（v1,v2,v4,v3），可把下面 v3、v4 对换：
    return [
        info.transmittor_position,   # v1
        info.receiver_position_1,    # v2
        info.receiver_position_2,    # v3
        info.receiver_position_3,    # v4
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
        b0 = (1-t)**3
        b1 = 3*(1-t)**2*t
        b2 = 3*(1-t)*t**2
        b3 = t**3

        point = [
            b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
            b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        ]

        if not _is_point_inside_quadrilateral(point, quad_vertices):
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

    centroid = np.mean(quad_vertices, axis=0)
    return centroid.tolist()

def _generate_trajectory_with_abcd_in_quadrilateral(quad_vertices, max_tries=100):
    """生成一条符合要求的轨迹，包含ABCD四个点，且BC之间为明显的曲线，确保速度方向连续"""
    for attempt in range(max_tries):
        velocity = random.uniform(15.0, 25.0)
        max_acceleration = random.uniform(20.0, 30.0)

        generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)

        points = []
        for _ in range(4):
            points.append(_generate_random_point_inside_quadrilateral(quad_vertices))

        if not all(_is_point_inside_quadrilateral(p, quad_vertices) for p in points):
            continue

        ab_vector = np.array(points[1]) - np.array(points[0])
        cd_vector = np.array(points[3]) - np.array(points[2])

        if np.linalg.norm(ab_vector) < 5.0 or np.linalg.norm(cd_vector) < 5.0:
            continue

        ab_direction = ab_vector / np.linalg.norm(ab_vector)
        cd_direction = cd_vector / np.linalg.norm(cd_vector)

        bc_vector = np.array(points[2]) - np.array(points[1])
        if np.linalg.norm(bc_vector) < 10.0:
            continue

        bc_direction = bc_vector / np.linalg.norm(bc_vector)

        control_distance_1 = np.linalg.norm(bc_vector) * 0.5
        control_point_1 = np.array(points[1]) + ab_direction * control_distance_1

        control_distance_2 = np.linalg.norm(bc_vector) * 0.5
        control_point_2 = np.array(points[2]) - cd_direction * control_distance_2

        if not _is_point_inside_quadrilateral(control_point_1.tolist(), quad_vertices) or \
           not _is_point_inside_quadrilateral(control_point_2.tolist(), quad_vertices) or \
           not _is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(),
                                                 control_point_2.tolist(), points[2], quad_vertices):
            continue

        try:
            b_velocity_direction = generator._add_straight_line(points[0], points[1], 
                                                             np.linalg.norm(ab_vector) / velocity)
            
            c_velocity_direction = generator._add_curved_transition(points[1], points[2], 
                                                                b_velocity_direction, cd_direction)
            
            generator._add_straight_line(points[2], points[3], np.linalg.norm(cd_vector) / velocity)
            
            return generator
        except Exception as e:
            continue

    return None

def _generate_multiple_trajectories(num_trajectories, detecting_region_info):
    """生成多条轨迹"""
    # 提取四边形顶点
    quad_vertices = _get_quad_vertices_from_detecting_region_info(detecting_region_info)
    
    trajectories = []
    
    for i in range(num_trajectories):
        trajectory = _generate_trajectory_with_abcd_in_quadrilateral(quad_vertices)
        if trajectory:
            trajectory.name = f"轨迹{i+1}"
            trajectories.append(trajectory)
    
    return trajectories

def _sample_trajectory_at_time_interval(trajectory, time_interval):
    """按固定时间间隔采样轨迹点，使用线性插值保证点位置准确"""
    trajectory_points = np.array(trajectory.trajectory)
    timestamps = np.array(trajectory.timestamps)
    
    total_time = timestamps[-1]
    
    # 计算采样时间点
    sample_times = np.arange(0, total_time, time_interval)
    if sample_times[-1] < total_time:
        sample_times = np.append(sample_times, total_time)
    
    sampled_points = []
    
    # 对每个采样时间点，计算插值位置
    for t in sample_times:
        # 如果采样时间恰好是轨迹中的时间点
        if t in timestamps:
            idx = np.where(timestamps == t)[0][0]
            sampled_points.append(trajectory_points[idx])
            continue
            
        # 找到采样时间点所在的时间区间
        next_idx = np.searchsorted(timestamps, t)
        if next_idx == 0:
            # 采样时间小于第一个时间点
            sampled_points.append(trajectory_points[0])
        elif next_idx == len(timestamps):
            # 采样时间大于最后一个时间点
            sampled_points.append(trajectory_points[-1])
        else:
            # 在两个时间点之间插值
            prev_idx = next_idx - 1
            prev_time = timestamps[prev_idx]
            next_time = timestamps[next_idx]
            prev_point = trajectory_points[prev_idx]
            next_point = trajectory_points[next_idx]
            
            # 线性插值计算位置
            alpha = (t - prev_time) / (next_time - prev_time)
            interpolated_point = prev_point * (1 - alpha) + next_point * alpha
            sampled_points.append(interpolated_point)
    
    return np.array(sampled_points)

def _generate_lines_from_trajectory(trajectory_generator, time_interval):
    """从轨迹生成器按时间间隔t生成相邻点的line_a和line_b
    
    生成格式:
    line_a: 第1到第n-1个点
    line_b: 第2到第n个点
    
    返回:
    两个numpy数组，对应相邻时间点的位置
    """
    # 按时间间隔采样轨迹
    sampled_points = _sample_trajectory_at_time_interval(trajectory_generator, time_interval)
    
    if len(sampled_points) < 2:
        return np.array([]), np.array([])
    
    # line_a是第1到第n-1个点
    line_a = sampled_points[:-1]
    
    # line_b是第2到第n个点
    line_b = sampled_points[1:]
    
    return line_a, line_b

def _part_line(step_count_per_line, long_line):
    """
    将一条长线分成多条短线段
    
    参数:
        step_count_per_line (int): 每条短线段包含的点数
        long_line (numpy.ndarray): 长线上所有点的坐标，形状为(n, 2)，其中n是点的数量
    
    返回:
        list: 包含多条短线段的列表，每条短线段是一个numpy数组，形状为(step_count_per_line, 2)
    """
    if len(long_line) < step_count_per_line:
        # 如果长线上的点数少于所需的步数，则直接跳出
        return []
    
    # 计算可以生成的短线段数量
    num_lines = len(long_line) - step_count_per_line + 1
    
    # 生成短线段
    short_lines = []
    for i in range(num_lines):
        # 提取当前短线段包含的点
        short_line = long_line[i:i+step_count_per_line]
        short_lines.append(short_line)
    
    return short_lines
