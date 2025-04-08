import numpy as np
import random
import matplotlib.pyplot as plt
import src.LineGenUtils as line_utils

###################################################################################################
# Quick Note (High-Level Explanation): TODO: 这里是思路，可以喂给 gpt
# 1) We generate a random quadrilateral or use detecting_region_info to define a bounding region.
# 2) For each "line" to generate, we pick four random points A, B, C, D inside that region.
# 3) We construct a piecewise trajectory: 
#      A->B (straight line),
#      B->C (smooth cubic Bézier curve),
#      C->D (straight line).
#    We sample the piecewise path at a certain rate (e.g., 30 Hz).
# 4) Once we have a full trajectory from t=0 to t=total_time, we build two lines: line A and line B.
#    - line A is sampled at times: 0, (t1 + t2), 2*(t1 + t2), 3*(t1 + t2), ...
#    - line B is sampled at times: t1, t1 + (t1 + t2), t1 + 2*(t1 + t2), ...
#    until we go beyond the total_time of the trajectory.
# 5) Return the array of line A points and the array of line B points.
# 
# Additionally, we provide a measureLineLength function to get the total length of
# any 2D polyline (e.g., line A or line B).
#
# The aim is to have drop-in replacement function "generateLines(...)",
# similarly named and similarly returning [lines_a, lines_b] arrays.
###################################################################################################


def measureLineLength(line_points):
    """
    Measure total length of a line given by an array of [x, y] points.
    """
    if len(line_points) < 2:
        return 0.0
    total_length = 0.0
    for i in range(1, len(line_points)):
        segment = line_points[i] - line_points[i - 1]
        total_length += np.linalg.norm(segment)
    return total_length


def generate_random_point_in_region(detecting_region_info):
    """
    Generate a random point within the region by using bounding box + check approach.
    """
    xs = [detecting_region_info.v1[0], detecting_region_info.v2[0],
          detecting_region_info.v3[0], detecting_region_info.v4[0]]
    ys = [detecting_region_info.v1[1], detecting_region_info.v2[1],
          detecting_region_info.v3[1], detecting_region_info.v4[1]]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    for _ in range(1000):  # up to 1000 tries
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)
        if line_utils.pointInRegion(detecting_region_info, (rx, ry)):
            return np.array([rx, ry])
    
    # If failed, just return center
    return np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])


def generate_piecewise_trajectory(A, B, C, D, velocity, max_acceleration, sampling_rate=30):
    """
    Generate a piecewise trajectory:
      1) A->B (straight line)
      2) B->C (cubic Bézier curve, ensuring smooth transition)
      3) C->D (straight line)
    
    Returns:
      times: list of times
      positions: list of np.array([x, y]) of same length
    """

    # Helper function: sample a straight line from p0 to p1
    def sample_straight_line(p0, p1, sample_rate, start_time=0.0):
        dist = np.linalg.norm(p1 - p0)
        duration = dist / velocity if velocity > 1e-7 else 0.0
        num_samples = max(int(duration * sample_rate), 2)
        if num_samples < 2:
            num_samples = 2
        
        tvals = np.linspace(0, duration, num_samples)
        positions = []
        times = []
        for t in tvals:
            alpha = t / duration if duration > 1e-7 else 0.0
            pos = (1 - alpha) * p0 + alpha * p1
            positions.append(pos)
            times.append(start_time + t)
        return times, positions, duration

    # Helper function: sample cubic Bézier from p0 -> p1 -> p2 -> p3
    def sample_bezier_cubic(p0, p1, p2, p3, sample_rate, start_time=0.0):
        # Estimate curve length by subdividing
        # We'll just do a rough polyline approach for length estimation
        rough_subdiv = 10
        subdiv_positions = []
        length_approx = 0.0
        prev_pt = None
        
        for i in range(rough_subdiv + 1):
            t = i / rough_subdiv
            b0 = (1 - t) ** 3
            b1 = 3 * (1 - t) ** 2 * t
            b2 = 3 * (1 - t) * t ** 2
            b3 = t ** 3
            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            pt = np.array([x, y])
            subdiv_positions.append(pt)
            if prev_pt is not None:
                length_approx += np.linalg.norm(pt - prev_pt)
            prev_pt = pt
        
        duration = length_approx / velocity if velocity > 1e-7 else 0.0
        num_samples = max(int(duration * sample_rate), 2)
        if num_samples < 2:
            num_samples = 2
        
        tvals = np.linspace(0, 1, num_samples)
        positions = []
        times = []
        for i, at in enumerate(tvals):
            b0 = (1 - at) ** 3
            b1 = 3 * (1 - at) ** 2 * at
            b2 = 3 * (1 - at) * at ** 2
            b3 = at ** 3
            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            pos = np.array([x, y])
            positions.append(pos)
            times.append(start_time + at * duration)
        return times, positions, duration

    # 1) A->B
    times_AB, pos_AB, dur_AB = sample_straight_line(A, B, sampling_rate, start_time=0.0)
    
    # For the curve B->C, we attempt to ensure a smooth-ish direction.
    BC_dist = np.linalg.norm(C - B)
    if BC_dist < 1e-7:
        # degenerate
        times_BC = []
        pos_BC = []
        dur_BC = 0
    else:
        # We set some offset factor
        factor = 0.5
        # direction from B->C
        BC_dir = (C - B) / BC_dist
        # define control1 = B + factor * BC_dist * BC_dir
        # define control2 = C - factor * BC_dist * BC_dir
        control1 = B + factor * BC_dist * BC_dir
        control2 = C - factor * BC_dist * BC_dir
        
        times_BC, pos_BC, dur_BC = sample_bezier_cubic(B, control1, control2, C, 
                                                       sampling_rate,
                                                       start_time=times_AB[-1])
    
    # 3) C->D
    times_CD, pos_CD, dur_CD = sample_straight_line(C, D, sampling_rate, 
                                                    start_time=(times_BC[-1] if len(times_BC) else times_AB[-1]))
    
    # Combine them (skipping one duplicate point at the junction of each)
    all_times = times_AB[:-1] + times_BC[:-1] + times_CD  
    all_positions = pos_AB[:-1] + pos_BC[:-1] + pos_CD
    
    return all_times, all_positions


def sample_line_from_trajectory(trajectory_times, trajectory_positions, sample_times):
    """
    Given the full trajectory times & positions,
    return the positions of the line at the requested sample_times.
    
    We do linear interpolation here.
    """
    if len(trajectory_times) < 2:
        return []

    t_array = np.array(trajectory_times)
    p_array = np.array(trajectory_positions)  # shape (N, 2)

    def interp(time_val):
        # if time_val <= t_array[0], return the first
        if time_val <= t_array[0]:
            return p_array[0]
        # if time_val >= t_array[-1], return the last
        if time_val >= t_array[-1]:
            return p_array[-1]
        
        # otherwise, find idx such that t_array[idx] <= time_val < t_array[idx+1]
        idx = np.searchsorted(t_array, time_val) - 1
        idx = max(idx, 0)
        if idx >= len(t_array) - 1:
            idx = len(t_array) - 2
        
        t0 = t_array[idx]
        t1 = t_array[idx + 1]
        p0 = p_array[idx]
        p1 = p_array[idx + 1]
        alpha = (time_val - t0) / (t1 - t0) if (t1 > t0) else 0.0
        return (1 - alpha) * p0 + alpha * p1

    out_positions = []
    for st in sample_times:
        out_positions.append(interp(st))
    return out_positions


def generateLines(
    detecting_region_info,
    t1,
    t2,
    num_of_lines_to_generate,
    velocity_range=(15.0, 25.0),
    max_acceleration_range=(20.0, 30.0),
    sampling_rate=30,
    seed=None
):
    """
    New lines generator following the piecewise A->B->C->D approach.
    Return an array of lineA and lineB, each shaped (num_of_lines, N, 2) 
    (though N may differ depending on how many samples we get).
    
    Arguments:
      detecting_region_info - region boundary info
      t1, t2 - time intervals for sampling line A and line B
      num_of_lines_to_generate - how many lines we want
      velocity_range, max_acceleration_range - random range of velocity / acceleration
      sampling_rate - how many samples per second for the piecewise path
      seed - random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    lines_a = []
    lines_b = []

    for _ in range(num_of_lines_to_generate):
        velocity = random.uniform(*velocity_range)
        max_acc = random.uniform(*max_acceleration_range)
        
        A = generate_random_point_in_region(detecting_region_info)
        B = generate_random_point_in_region(detecting_region_info)
        C = generate_random_point_in_region(detecting_region_info)
        D = generate_random_point_in_region(detecting_region_info)

        traj_times, traj_positions = generate_piecewise_trajectory(
            A, B, C, D, velocity, max_acc, sampling_rate=sampling_rate
        )
        if not traj_times or not traj_positions:
            # degenerate case
            lines_a.append([])
            lines_b.append([])
            continue
        
        total_traj_time = traj_times[-1]
        
        # Times for lineA: 0, t1+t2, 2(t1+t2), ...
        lineA_sample_times = []
        current_time = 0.0
        while current_time <= total_traj_time:
            lineA_sample_times.append(current_time)
            current_time += (t1 + t2)
        
        # Times for lineB: t1, t1+(t1+t2), t1+2(t1+t2), ...
        lineB_sample_times = []
        current_time = t1
        while current_time <= total_traj_time:
            lineB_sample_times.append(current_time)
            current_time += (t1 + t2)
        
        # Sample from the trajectory
        lineA_positions = sample_line_from_trajectory(traj_times, traj_positions, lineA_sample_times)
        lineB_positions = sample_line_from_trajectory(traj_times, traj_positions, lineB_sample_times)
        
        lines_a.append(lineA_positions)
        lines_b.append(lineB_positions)

    lines_a_array = np.array([np.array(la) for la in lines_a], dtype=object)
    lines_b_array = np.array([np.array(lb) for lb in lines_b], dtype=object)
    
    return [lines_a_array, lines_b_array]


###################################################################################################
# Debug/Plot Function
###################################################################################################
def plot_trajectory_and_region(
    detecting_region_info, 
    trajectory_times, 
    trajectory_positions, 
    A, B, C, D,
    lineA_positions=None, 
    lineB_positions=None, 
    title="Debug Plot",
    show=True
):
    """
    Plot the four corners of detecting_region_info, 
    the piecewise trajectory from A->B->C->D,
    and optionally the lineA_positions, lineB_positions samples.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Region (assuming quadrilateral: v1 -> v2 -> v3 -> v4)
    v1 = detecting_region_info.v1
    v2 = detecting_region_info.v2
    v3 = detecting_region_info.v3
    v4 = detecting_region_info.v4
    
    region_x = [v1[0], v2[0], v3[0], v4[0], v1[0]]
    region_y = [v1[1], v2[1], v3[1], v4[1], v1[1]]
    ax.plot(region_x, region_y, 'k--', label="Region Boundary")

    # Plot the entire piecewise trajectory
    traj_positions_array = np.array(trajectory_positions)
    ax.plot(
        traj_positions_array[:, 0],
        traj_positions_array[:, 1],
        color="blue",
        lw=2,
        label="Trajectory (A->B->C->D)"
    )

    # Mark A, B, C, D
    ax.scatter(A[0], A[1], c='red', marker='o', zorder=5, label="A")
    ax.scatter(B[0], B[1], c='green', marker='o', zorder=5, label="B")
    ax.scatter(C[0], C[1], c='purple', marker='o', zorder=5, label="C")
    ax.scatter(D[0], D[1], c='orange', marker='o', zorder=5, label="D")

    # If line A or line B samples are provided, plot them
    if lineA_positions is not None and len(lineA_positions) > 0:
        la_arr = np.array(lineA_positions)
        ax.plot(la_arr[:, 0], la_arr[:, 1], 'ro-', label="Line A samples")
    
    if lineB_positions is not None and len(lineB_positions) > 0:
        lb_arr = np.array(lineB_positions)
        ax.plot(lb_arr[:, 0], lb_arr[:, 1], 'go-', label="Line B samples")

    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if show:
        plt.show()

# --- previous version （取消注释就是） --- 

# import numpy as np
# import matplotlib.pyplot as plt
# import random

# class TrajectoryGenerator:
#     def __init__(self, velocity=20.0, max_acceleration=25.0, name=None):
#         """初始化轨迹生成器
        
#         Args:
#             velocity: 匀速直线段的速度，单位：米/秒
#             max_acceleration: 最大加速度，单位：米/秒²
#             name: 轨迹名称（可选）
#         """
#         self.velocity = velocity
#         self.max_acceleration = max_acceleration
#         self.name = name
#         self.trajectory = []
#         self.timestamps = []
#         self.velocities = []
#         self.accelerations = []
#         self.time = 0.0
#         self.key_points = []  # 存储关键点
#         self.key_times = []   # 存储关键点对应的时间
    
#     def add_straight_line(self, start_point, end_point, duration):
#         """添加匀速直线段
        
#         Args:
#             start_point: 起点坐标 [x, y]
#             end_point: 终点坐标 [x, y]
#             duration: 直线段持续时间，单位：秒
#         """
#         start_point = np.array(start_point)
#         end_point = np.array(end_point)
        
#         # 记录关键点
#         if len(self.key_points) == 0:
#             self.key_points.append(start_point)
#             self.key_times.append(self.time)
#         self.key_points.append(end_point)
#         self.key_times.append(self.time + duration)
        
#         # 计算采样点数量
#         num_samples = max(int(duration * 10), 2)  # 10Hz采样率，至少2个点
        
#         # 生成时间戳
#         times = np.linspace(0, duration, num_samples)
        
#         # 生成位置点
#         for t in times:
#             alpha = t / duration
#             position = (1 - alpha) * start_point + alpha * end_point
#             self.trajectory.append(position)
#             self.timestamps.append(self.time + t)
        
#         # 计算速度（匀速）
#         velocity_vector = (end_point - start_point) / duration
#         speed = np.linalg.norm(velocity_vector)
#         if speed > 0:
#             velocity_unit = velocity_vector / speed
#         else:
#             velocity_unit = np.array([1.0, 0.0])  # 默认方向
            
#         # 保存实际单位速度方向（用于确保连续性）
#         self.last_velocity_direction = velocity_unit
        
#         # 添加速度和加速度数据
#         for _ in range(num_samples):
#             self.velocities.append(velocity_unit * self.velocity)
#             self.accelerations.append(np.zeros(2))  # 匀速运动加速度为0
        
#         # 更新总时间
#         self.time += duration
        
#         return velocity_unit  # 返回末端速度方向
    
#     def add_curved_transition(self, start_point, end_point, start_direction, end_direction, control_point_offset_factor=0.7):
#         """添加确保速度连续性的平滑弯曲过渡
        
#         Args:
#             start_point: 起点坐标 [x, y]
#             end_point: 终点坐标 [x, y]
#             start_direction: 起点速度方向 [dx, dy]
#             end_direction: 终点期望速度方向 [dx, dy]
#             control_point_offset_factor: 控制点偏移系数，越大弯曲越明显
#         """
#         start_point = np.array(start_point)
#         end_point = np.array(end_point)
#         start_direction = np.array(start_direction)
#         end_direction = np.array(end_direction)
        
#         # 检查起点和终点是否不同
#         if np.allclose(start_point, end_point):
#             print("警告: 起点和终点太近，跳过曲线生成")
#             return None
        
#         # 记录关键点
#         if len(self.key_points) == 0:
#             self.key_points.append(start_point)
#             self.key_times.append(self.time)
        
#         # 归一化方向向量
#         if np.linalg.norm(start_direction) > 0:
#             start_direction = start_direction / np.linalg.norm(start_direction)
#         else:
#             start_direction = np.array([1.0, 0.0])  # 默认方向
            
#         if np.linalg.norm(end_direction) > 0:
#             end_direction = end_direction / np.linalg.norm(end_direction)
#         else:
#             end_direction = np.array([1.0, 0.0])  # 默认方向
        
#         # 计算起点和终点之间的直线距离
#         direct_distance = np.linalg.norm(end_point - start_point)
        
#         # 确定两个控制点位置，使得曲线在起点和终点处的切线方向分别为start_direction和end_direction
#         # 首个控制点在起点沿起点速度方向延伸
#         control_distance_1 = direct_distance * control_point_offset_factor
#         control_point_1 = start_point + start_direction * control_distance_1
        
#         # 第二个控制点在终点沿终点速度方向反向延伸
#         control_distance_2 = direct_distance * control_point_offset_factor
#         control_point_2 = end_point - end_direction * control_distance_2
        
#         # 估计曲线长度 - 使用多段线近似
#         curve_length = (np.linalg.norm(control_point_1 - start_point) + 
#                        np.linalg.norm(control_point_2 - control_point_1) + 
#                        np.linalg.norm(end_point - control_point_2))
        
#         # 计算基于速度的时间
#         duration = curve_length / self.velocity
        
#         # 记录终点
#         self.key_points.append(end_point)
#         self.key_times.append(self.time + duration)
        
#         # 使用参数化三次贝塞尔曲线
#         num_samples = max(int(duration * 30), 30)  # 30Hz采样率，至少30个点
#         t_values = np.linspace(0, 1, num_samples)
        
#         positions = []
#         velocities = []
#         accelerations = []
        
#         # 生成三次贝塞尔曲线点
#         for t in t_values:
#             # 计算位置 - 三次贝塞尔曲线公式
#             # B(t) = (1-t)³*P₀ + 3(1-t)²t*P₁ + 3(1-t)t²*P₂ + t³*P₃
#             b0 = (1-t)**3
#             b1 = 3*(1-t)**2*t
#             b2 = 3*(1-t)*t**2
#             b3 = t**3
            
#             position = b0 * start_point + b1 * control_point_1 + b2 * control_point_2 + b3 * end_point
#             positions.append(position)
            
#             # 计算速度（贝塞尔曲线的一阶导数）
#             # B'(t) = 3(1-t)²*(P₁-P₀) + 6(1-t)t*(P₂-P₁) + 3t²*(P₃-P₂)
#             v0 = 3*(1-t)**2
#             v1 = 6*(1-t)*t
#             v2 = 3*t**2
            
#             velocity_dir = v0 * (control_point_1 - start_point) + \
#                           v1 * (control_point_2 - control_point_1) + \
#                           v2 * (end_point - control_point_2)
            
#             # 归一化速度方向
#             speed = np.linalg.norm(velocity_dir)
#             if speed > 1e-10:
#                 velocity_dir = velocity_dir / speed
#                 velocities.append(velocity_dir * self.velocity)
#             else:
#                 velocities.append(np.array([0.0, 0.0]))
            
#             # 计算加速度（贝塞尔曲线的二阶导数）
#             # B''(t) = 6(1-t)*(P₂-2P₁+P₀) + 6t*(P₃-2P₂+P₁)
#             a0 = 6*(1-t)
#             a1 = 6*t
            
#             acceleration = a0 * (control_point_2 - 2*control_point_1 + start_point) + \
#                           a1 * (end_point - 2*control_point_2 + control_point_1)
            
#             # 计算加速度大小 - 确保不超过最大值
#             accel_magnitude = self.velocity**2 * np.linalg.norm(acceleration) / (speed**2) if speed > 1e-10 else 0
            
#             # 限制加速度大小
#             if accel_magnitude > self.max_acceleration:
#                 accel_magnitude = self.max_acceleration
            
#             # 计算加速度方向 - 垂直于速度方向
#             if speed > 1e-10:
#                 # 计算法向量（垂直于速度方向）
#                 normal_vector = np.array([-velocity_dir[1], velocity_dir[0]])
                
#                 # 确保法向量与加速度方向一致
#                 if np.dot(normal_vector, acceleration) < 0:
#                     normal_vector = -normal_vector
                
#                 accelerations.append(normal_vector * accel_magnitude)
#             else:
#                 accelerations.append(np.array([0.0, 0.0]))
        
#         # 添加到轨迹
#         for i in range(num_samples):
#             self.trajectory.append(positions[i])
#             self.timestamps.append(self.time + duration * t_values[i])
#             self.velocities.append(velocities[i])
#             self.accelerations.append(accelerations[i])
        
#         # 保存最后的速度方向（用于确保连续性）
#         self.last_velocity_direction = velocities[-1] / self.velocity if velocities else end_direction
        
#         # 更新总时间
#         self.time += duration
        
#         # 返回终点实际速度方向（可能与期望方向略有不同）
#         return self.last_velocity_direction

# def is_point_inside_quadrilateral(point, quad_vertices):
#     """检查点是否在四边形内
    
#     Args:
#         point: 点坐标 [x, y]
#         quad_vertices: 四边形顶点坐标 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        
#     Returns:
#         bool: 如果点在四边形内则返回True
#     """
#     # 射线法判断点是否在多边形内
#     x, y = point
#     n = len(quad_vertices)
#     inside = False
    
#     p1x, p1y = quad_vertices[0]
#     for i in range(n + 1):
#         p2x, p2y = quad_vertices[i % n]
#         if min(p1y, p2y) < y <= max(p1y, p2y) and x <= max(p1x, p2x):
#             if p1y != p2y:
#                 xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#             if p1x == p2x or x <= xinters:
#                 inside = not inside
#         p1x, p1y = p2x, p2y
    
#     return inside

# def is_bezier_curve_inside_quadrilateral(p0, p1, p2, p3, quad_vertices, num_checks=20):
#     """检查三次贝塞尔曲线是否在四边形内
    
#     Args:
#         p0: 曲线起点 [x, y]
#         p1: 第一个控制点 [x, y]
#         p2: 第二个控制点 [x, y]
#         p3: 曲线终点 [x, y]
#         quad_vertices: 四边形顶点坐标
#         num_checks: 检查点数量
        
#     Returns:
#         bool: 如果曲线在四边形内则返回True
#     """
#     # 检查曲线上的多个点
#     for i in range(num_checks + 1):
#         t = i / num_checks
#         # 三次贝塞尔曲线公式
#         b0 = (1-t)**3
#         b1 = 3*(1-t)**2*t
#         b2 = 3*(1-t)*t**2
#         b3 = t**3
        
#         point = [
#             b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
#             b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
#         ]
        
#         if not is_point_inside_quadrilateral(point, quad_vertices):
#             return False
    
#     return True

# def generate_random_point_inside_quadrilateral(quad_vertices):
#     """在四边形内生成随机点
    
#     Args:
#         quad_vertices: 四边形顶点坐标
        
#     Returns:
#         list: 随机点坐标 [x, y]
#     """
#     # 计算四边形的边界盒
#     min_x = min(v[0] for v in quad_vertices)
#     max_x = max(v[0] for v in quad_vertices)
#     min_y = min(v[1] for v in quad_vertices)
#     max_y = max(v[1] for v in quad_vertices)
    
#     # 在边界盒内随机生成点，直到找到一个在四边形内的点
#     for _ in range(100):  # 最多尝试100次
#         x = random.uniform(min_x, max_x)
#         y = random.uniform(min_y, max_y)
#         point = [x, y]
        
#         if is_point_inside_quadrilateral(point, quad_vertices):
#             return point
    
#     # 如果随机失败，使用四边形的重心
#     centroid = np.mean(quad_vertices, axis=0)
#     return centroid.tolist()

# def generate_trajectory_with_ABCD_in_quadrilateral(quad_vertices, max_tries=100):
#     """生成一条符合要求的轨迹，包含ABCD四个点，且BC之间为明显的曲线，确保速度方向连续
    
#     Args:
#         quad_vertices: 四边形顶点坐标
#         max_tries: 最大尝试次数
        
#     Returns:
#         TrajectoryGenerator: 生成的轨迹对象，如果失败则返回None
#     """
#     for attempt in range(max_tries):
#         # 随机速度和加速度
#         velocity = random.uniform(15.0, 25.0)
#         max_acceleration = random.uniform(20.0, 30.0)
        
#         generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)
        
#         # 随机生成A、B、C、D四个点
#         points = []
#         for _ in range(4):
#             points.append(generate_random_point_inside_quadrilateral(quad_vertices))
        
#         # 确保所有点在四边形内
#         if not all(is_point_inside_quadrilateral(p, quad_vertices) for p in points):
#             continue
        
#         # 计算AB和CD的方向向量
#         ab_vector = np.array(points[1]) - np.array(points[0])
#         cd_vector = np.array(points[3]) - np.array(points[2])
        
#         # 确保AB和CD足够长
#         if np.linalg.norm(ab_vector) < 5.0 or np.linalg.norm(cd_vector) < 5.0:
#             continue
            
#         # 归一化方向向量
#         ab_direction = ab_vector / np.linalg.norm(ab_vector)
#         cd_direction = cd_vector / np.linalg.norm(cd_vector)
        
#         # 计算BC的中间方向 - 这不是BC的方向，而是用于确定曲线的形状
#         bc_vector = np.array(points[2]) - np.array(points[1])
#         if np.linalg.norm(bc_vector) < 10.0:  # 确保BC距离足够长
#             continue
            
#         bc_direction = bc_vector / np.linalg.norm(bc_vector)
        
#         # 确定BC曲线的两个控制点
#         # 第一个控制点沿着AB的末端方向延伸
#         control_distance_1 = np.linalg.norm(bc_vector) * 0.5
#         control_point_1 = np.array(points[1]) + ab_direction * control_distance_1
        
#         # 第二个控制点沿着CD的起始方向反向延伸
#         control_distance_2 = np.linalg.norm(bc_vector) * 0.5
#         control_point_2 = np.array(points[2]) - cd_direction * control_distance_2
        
#         # 检查控制点和整条贝塞尔曲线是否在四边形内
#         if not is_point_inside_quadrilateral(control_point_1.tolist(), quad_vertices) or \
#            not is_point_inside_quadrilateral(control_point_2.tolist(), quad_vertices) or \
#            not is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(), 
#                                                   control_point_2.tolist(), points[2], quad_vertices):
#             continue
        
#         try:
#             # A -> B (直线，返回B点速度方向)
#             b_velocity_direction = generator.add_straight_line(points[0], points[1], 
#                                                               np.linalg.norm(ab_vector) / velocity)
            
#             # B -> C (确保在B点速度方向连续，并在C点达到与CD直线段匹配的速度方向)
#             c_velocity_direction = generator.add_curved_transition(points[1], points[2], 
#                                                                  b_velocity_direction, cd_direction)
            
#             # C -> D (直线，起始速度方向从曲线末端继承)
#             generator.add_straight_line(points[2], points[3], np.linalg.norm(cd_vector) / velocity)
            
#             return generator
#         except Exception as e:
#             print(f"尝试 {attempt+1}: 创建轨迹失败: {e}")
    
#     print(f"在 {max_tries} 次尝试后未能生成有效轨迹")
#     return None

# def generate_multiple_trajectories(num_trajectories, quad_vertices):
#     """生成多条轨迹
    
#     Args:
#         num_trajectories: 要生成的轨迹数量
#         quad_vertices: 四边形顶点坐标
        
#     Returns:
#         list: 生成的轨迹对象列表
#     """
#     trajectories = []
    
#     for i in range(num_trajectories):
#         print(f"生成轨迹 {i+1}/{num_trajectories}...")
#         trajectory = generate_trajectory_with_ABCD_in_quadrilateral(quad_vertices)
#         if trajectory:
#             trajectory.name = f"轨迹{i+1}"
#             trajectories.append(trajectory)
    
#     return trajectories

# def plot_trajectories_in_quadrilateral(trajectories, quad_vertices, title=None, show_velocity=True):
#     """绘制四边形内的多条轨迹
    
#     Args:
#         trajectories: 轨迹对象列表
#         quad_vertices: 四边形顶点坐标
#         title: 图表标题
#         show_velocity: 是否显示速度向量
#     """
#     plt.figure(figsize=(16, 12))
    
#     # 绘制四边形边界
#     quad_vertices_closed = quad_vertices + [quad_vertices[0]]  # 闭合多边形
#     x_values = [v[0] for v in quad_vertices_closed]
#     y_values = [v[1] for v in quad_vertices_closed]
#     plt.plot(x_values, y_values, 'k-', linewidth=3)
    
#     # 标注顶点
#     for i, vertex in enumerate(quad_vertices):
#         plt.scatter(vertex[0], vertex[1], c='black', s=100, zorder=5)
#         plt.annotate(f'P{i+1} ({vertex[0]}, {vertex[1]})', 
#                    (vertex[0], vertex[1]), xytext=(5, 5), 
#                    textcoords='offset points', fontsize=12, fontweight='bold')
    
#     # 使用渐变色
#     cmap = plt.cm.tab10  # 使用离散颜色图，更好地区分轨迹
#     colors = [cmap(i % 10) for i in range(len(trajectories))]
    
#     # 绘制轨迹
#     for i, trajectory in enumerate(trajectories):
#         # 轨迹线
#         trajectory_array = np.array(trajectory.trajectory)
#         plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], color=colors[i], 
#                 linewidth=1.5, alpha=0.8)
        
#         # 标记关键点 - ABCD四个点
#         key_points_array = np.array(trajectory.key_points)
#         plt.scatter(key_points_array[:, 0], key_points_array[:, 1], color=colors[i], 
#                    s=40, alpha=0.9)
        
#         # 添加关键点标签
#         labels = ['A', 'B', 'C', 'D']
#         for j, point in enumerate(key_points_array):
#             if j < len(labels):  # 确保不超出标签数量
#                 plt.annotate(labels[j], (point[0], point[1]), xytext=(3, 3),
#                            textcoords='offset points', color=colors[i], fontsize=10, fontweight='bold')
        
#         # 显示速度向量（尤其在关键点处的速度方向）
#         if show_velocity:
#             # 找到关键点的索引
#             key_indices = []
#             for j, kp in enumerate(key_points_array):
#                 # 找到轨迹中最接近关键点的点
#                 distances = np.linalg.norm(trajectory_array - kp, axis=1)
#                 key_indices.append(np.argmin(distances))
            
#             # 绘制关键点的速度向量
#             velocities_array = np.array(trajectory.velocities)
#             for j, idx in enumerate(key_indices):
#                 # 速度向量缩放系数
#                 scale_factor = 3.0
#                 plt.quiver(trajectory_array[idx, 0], trajectory_array[idx, 1],
#                           velocities_array[idx, 0], velocities_array[idx, 1],
#                           color=colors[i], scale=20*trajectory.velocity/scale_factor,
#                           width=0.003, alpha=0.7, zorder=4)
    
#     plt.xlabel('X (m)')
#     plt.ylabel('Y (m)')
    
#     if title:
#         plt.title(title)
#     else:
#         plt.title(f'四边形区域内的{len(trajectories)}条轨迹\n每条包含ABCD四个点，BC之间为平滑弯曲，确保速度方向连续')
    
#     plt.axis('equal')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # 设置随机种子以便结果可重现
#     random.seed(42)
#     np.random.seed(42)
    
#     # 定义四边形顶点
#     quad_vertices = [
#         [0, 0],       # P1
#         [200, 0],     # P2
#         [300, 150],   # P3
#         [100, 100]    # P4
#     ]
    
#     # 生成50条轨迹
#     print("开始生成50条轨迹，每条轨迹中BC段有明显的弯曲且速度方向连续...")
#     trajectories = generate_multiple_trajectories(1, quad_vertices)
    
#     print(f"成功生成 {len(trajectories)} 条轨迹")
    
#     # 绘制轨迹
#     plot_trajectories_in_quadrilateral(trajectories, quad_vertices, show_velocity=True)