import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

    def add_straight_line(self, start_point, end_point, duration):
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

    def add_curved_transition(self, start_point, end_point, start_direction, end_direction, control_point_offset_factor=0.7):
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


def get_quad_vertices_from_detecting_region_info(detecting_region_info=None):
    """从DetectingRegionInfo对象中提取四边形顶点坐标"""
    # 定义四边形顶点，使用固定值
    quad_vertices = [
        np.array([0, 0]),       # 发射器位置 transmittor_position
        np.array([200, 0]),     # 接收器1位置 receiver_position_1
        np.array([300, 150]),   # 接收器3位置 receiver_position_3
        np.array([100, 100])    # 接收器2位置 receiver_position_2
    ]
    return quad_vertices

def get_inner_quad_vertices(quad_vertices, scale_factor):
    """
    计算按比例缩小的内部四边形顶点.
    
    Args:
        quad_vertices (list): 外部四边形的顶点列表.
        scale_factor (float): 缩放比例 (例如, 0.7 表示 70%).
        
    Returns:
        list: 内部四边形的顶点列表.
    """
    # 1. 计算四边形的几何中心（质心）
    centroid = np.mean(quad_vertices, axis=0)
    
    inner_vertices = []
    # 2. 将每个顶点向中心移动
    for vertex in quad_vertices:
        # 计算从中心到顶点的向量
        vector_to_vertex = vertex - centroid
        # 按比例缩放向量并计算新顶点位置
        new_vertex = centroid + vector_to_vertex * scale_factor
        inner_vertices.append(new_vertex)
        
    return inner_vertices


def is_point_inside_quadrilateral(point, quad_vertices):
    """检查点是否在四边形内（射线法）"""
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


def is_bezier_curve_inside_quadrilateral(p0, p1, p2, p3, quad_vertices, num_checks=20):
    """检查三次贝塞尔曲线是否完全在四边形内"""
    for i in range(num_checks + 1):
        t = i / num_checks
        # 计算贝塞尔曲线上的点
        point = (1-t)**3 * np.array(p0) + \
                3*(1-t)**2*t * np.array(p1) + \
                3*(1-t)*t**2 * np.array(p2) + \
                t**3 * np.array(p3)

        if not is_point_inside_quadrilateral(point.tolist(), quad_vertices):
            return False

    return True


def generate_random_point_inside_quadrilateral(quad_vertices):
    """在指定的四边形内生成一个随机点"""
    min_x = min(v[0] for v in quad_vertices)
    max_x = max(v[0] for v in quad_vertices)
    min_y = min(v[1] for v in quad_vertices)
    max_y = max(v[1] for v in quad_vertices)

    # 尝试在包围盒内生成点，并检查是否在四边形内
    for _ in range(100):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        point = [x, y]

        if is_point_inside_quadrilateral(point, quad_vertices):
            return point

    # 如果多次尝试失败，返回质心作为备用点
    centroid = np.mean(quad_vertices, axis=0)
    return centroid.tolist()


def generate_trajectory_with_ABCD_in_quadrilateral(boundary_vertices, max_tries=100):
    """在指定的边界内生成一条符合要求的轨迹"""
    for attempt in range(max_tries):
        velocity = random.uniform(15.0, 25.0)
        max_acceleration = random.uniform(20.0, 30.0)

        generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)

        # 在指定的边界内生成4个随机点
        points = [generate_random_point_inside_quadrilateral(boundary_vertices) for _ in range(4)]

        # 确保所有点都在边界内
        if not all(is_point_inside_quadrilateral(p, boundary_vertices) for p in points):
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

        # 计算贝塞尔曲线的控制点
        control_distance_1 = np.linalg.norm(bc_vector) * 0.5
        control_point_1 = np.array(points[1]) + ab_direction * control_distance_1

        control_distance_2 = np.linalg.norm(bc_vector) * 0.5
        control_point_2 = np.array(points[2]) - cd_direction * control_distance_2

        # 检查控制点和整条贝塞尔曲线是否都在边界内
        if not is_point_inside_quadrilateral(control_point_1.tolist(), boundary_vertices) or \
           not is_point_inside_quadrilateral(control_point_2.tolist(), boundary_vertices) or \
           not is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(),
                                                 control_point_2.tolist(), points[2], boundary_vertices):
            continue

        try:
            # 生成轨迹段
            b_velocity_direction = generator.add_straight_line(points[0], points[1], 
                                                             np.linalg.norm(ab_vector) / velocity)
            
            c_velocity_direction = generator.add_curved_transition(points[1], points[2], 
                                                                b_velocity_direction, cd_direction)
            
            generator.add_straight_line(points[2], points[3], np.linalg.norm(cd_vector) / velocity)
            
            return generator
        except Exception as e:
            print(f"生成轨迹时出错: {e}")
            continue

    return None


def generate_multiple_trajectories(num_trajectories, generation_boundary):
    """在指定的边界内生成多条轨迹"""
    trajectories = []
    
    for i in range(num_trajectories):
        # 将生成边界传递给轨迹生成函数
        trajectory = generate_trajectory_with_ABCD_in_quadrilateral(generation_boundary)
        if trajectory:
            trajectory.name = f"轨迹{i+1}"
            trajectories.append(trajectory)
    
    return trajectories


def sample_trajectory_at_time_interval(trajectory, time_interval):
    """按固定时间间隔采样轨迹点"""
    trajectory_points = np.array(trajectory.trajectory)
    timestamps = np.array(trajectory.timestamps)
    
    if len(timestamps) == 0:
        return np.array([])
        
    total_time = timestamps[-1]
    
    sample_times = np.arange(0, total_time, time_interval)
    if not sample_times.size or sample_times[-1] < total_time:
        sample_times = np.append(sample_times, total_time)
    
    sampled_points = []
    
    for t in sample_times:
        idx = np.argmin(np.abs(timestamps - t))
        sampled_points.append(trajectory_points[idx])
    
    return np.array(sampled_points)


def generate_lines_from_trajectory(trajectory_generator, time_interval):
    """从轨迹生成器按时间间隔t生成相邻点的line_a和line_b"""
    sampled_points = sample_trajectory_at_time_interval(trajectory_generator, time_interval)
    
    if len(sampled_points) < 2:
        return np.array([]), np.array([])
    
    line_a = sampled_points[:-1]
    line_b = sampled_points[1:]
    
    return line_a, line_b


def generateLines(generation_boundary, num_lines=5, time_interval=0.5, seed=42):
    """在指定边界内生成指定数量的线对"""
    random.seed(seed)
    np.random.seed(seed)
    
    trajectories = generate_multiple_trajectories(num_lines, generation_boundary)
    
    all_lines_a = []
    all_lines_b = []
    
    for trajectory in trajectories:
        line_a, line_b = generate_lines_from_trajectory(trajectory, time_interval)
        if len(line_a) > 0:
            all_lines_a.append(line_a)
            all_lines_b.append(line_b)
    
    return all_lines_a, all_lines_b


def visualize_trajectories(trajectories, outer_quad_vertices, inner_quad_vertices):
    """可视化轨迹，并显示内外边界"""
    plt.figure(figsize=(10, 8))
    
    # 绘制外部四边形边界 (黑色)
    x_outer = [v[0] for v in outer_quad_vertices] + [outer_quad_vertices[0][0]]
    y_outer = [v[1] for v in outer_quad_vertices] + [outer_quad_vertices[0][1]]
    plt.plot(x_outer, y_outer, 'k-', linewidth=2, label='原始边界')
    
    # 绘制内部四边形边界 (红色)
    x_inner = [v[0] for v in inner_quad_vertices] + [inner_quad_vertices[0][0]]
    y_inner = [v[1] for v in inner_quad_vertices] + [inner_quad_vertices[0][1]]
    plt.plot(x_inner, y_inner, 'r-', linewidth=2, label='生成区域 (70%)')

    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        if not trajectory.trajectory: continue
        trajectory_points = np.array(trajectory.trajectory)
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], '-', color=colors[i], 
                 linewidth=2.5, label=f"{trajectory.name}")
        
        key_points = np.array(trajectory.key_points)
        plt.scatter(key_points[:, 0], key_points[:, 1], color=colors[i], s=80, marker='o', edgecolors='black')
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('在缩小70%区域内生成的轨迹')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_sampled_points(lines_a, lines_b, outer_quad_vertices, inner_quad_vertices):
    """可视化采样点，并显示内外边界"""
    plt.figure(figsize=(10, 8))
    
    # 绘制外部四边形边界 (黑色)
    x_outer = [v[0] for v in outer_quad_vertices] + [outer_quad_vertices[0][0]]
    y_outer = [v[1] for v in outer_quad_vertices] + [outer_quad_vertices[0][1]]
    plt.plot(x_outer, y_outer, 'k-', linewidth=2, label='原始边界')
    
    # 绘制内部四边形边界 (红色)
    x_inner = [v[0] for v in inner_quad_vertices] + [inner_quad_vertices[0][0]]
    y_inner = [v[1] for v in inner_quad_vertices] + [inner_quad_vertices[0][1]]
    plt.plot(x_inner, y_inner, 'r-', linewidth=2, label='生成区域 (70%)')

    colors = plt.cm.viridis(np.linspace(0, 1, len(lines_a)))
    
    for i in range(len(lines_a)):
        plt.plot(lines_a[i][:, 0], lines_a[i][:, 1], 'o-', color=colors[i], markersize=6, 
                 alpha=0.8, label=f"轨迹{i+1} 点集A")
        plt.plot(lines_b[i][:, 0], lines_b[i][:, 1], 's-', color=colors[i], markersize=6, 
                 alpha=0.8, label=f"轨迹{i+1} 点集B")
        
        for j in range(len(lines_a[i])):
            plt.plot([lines_a[i][j, 0], lines_b[i][j, 0]], 
                     [lines_a[i][j, 1], lines_b[i][j, 1]], 
                     '--', color='gray', alpha=0.5)
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('轨迹采样点可视化')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def animate_trajectories(trajectories, outer_quad_vertices, inner_quad_vertices, interval=50):
    """创建轨迹动画，并显示内外边界"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制外部四边形边界 (黑色)
    x_outer = [v[0] for v in outer_quad_vertices] + [outer_quad_vertices[0][0]]
    y_outer = [v[1] for v in outer_quad_vertices] + [outer_quad_vertices[0][1]]
    ax.plot(x_outer, y_outer, 'k-', linewidth=2, label='原始边界')
    
    # 绘制内部四边形边界 (红色)
    x_inner = [v[0] for v in inner_quad_vertices] + [inner_quad_vertices[0][0]]
    y_inner = [v[1] for v in inner_quad_vertices] + [inner_quad_vertices[0][1]]
    ax.plot(x_inner, y_inner, 'r-', linewidth=2, label='生成区域 (70%)')

    min_x = min(v[0] for v in outer_quad_vertices)
    max_x = max(v[0] for v in outer_quad_vertices)
    min_y = min(v[1] for v in outer_quad_vertices)
    max_y = max(v[1] for v in outer_quad_vertices)
    
    padding = 20
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title('轨迹生成动画')
    ax.grid(True)
    ax.legend()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    points, trails, labels = [], [], []
    
    for i, trajectory in enumerate(trajectories):
        if not trajectory.key_points: continue
        key_points = np.array(trajectory.key_points)
        ax.scatter(key_points[:, 0], key_points[:, 1], color=colors[i], s=80, marker='o', edgecolors='black')
        
        point, = ax.plot([], [], 'o', color=colors[i], markersize=10)
        points.append(point)
        
        trail, = ax.plot([], [], '-', color=colors[i], linewidth=2.5, alpha=0.7)
        trails.append(trail)
        
        label = ax.text(0, 0, "", color=colors[i], fontsize=12, weight='bold')
        labels.append(label)
    
    max_time = max((traj.timestamps[-1] for traj in trajectories if traj.timestamps), default=0)
    if max_time == 0:
        plt.show()
        return None

    num_frames = int(max_time * 1000 / interval) + 1
    
    def init():
        for point, trail, label in zip(points, trails, labels):
            point.set_data([], [])
            trail.set_data([], [])
            label.set_text("")
        return points + trails + labels
    
    def animate(frame):
        current_time = frame * interval / 1000
        
        for i, trajectory in enumerate(trajectories):
            if not trajectory.timestamps: continue
            timestamps = np.array(trajectory.timestamps)
            trajectory_points = np.array(trajectory.trajectory)
            
            valid_indices = timestamps <= current_time
            if not np.any(valid_indices):
                continue
                
            valid_points = trajectory_points[valid_indices]
            trails[i].set_data(valid_points[:, 0], valid_points[:, 1])
            
            current_pos = valid_points[-1]
            points[i].set_data([current_pos[0]], [current_pos[1]])
            labels[i].set_position((current_pos[0] + 5, current_pos[1] + 5))
            labels[i].set_text(trajectory.name)
        
        return points + trails + labels
    
    anim = FuncAnimation(fig, animate, frames=num_frames, init_func=init, interval=interval, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    # 设置随机种子，保证结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 1. 获取外部四边形顶点
    outer_quad_vertices = get_quad_vertices_from_detecting_region_info()
    
    # 2. 计算按70%缩小的内部四边形顶点
    inner_quad_vertices = get_inner_quad_vertices(outer_quad_vertices, scale_factor=0.6)
    
    # 3. 在内部四边形中生成轨迹
    num_trajectories = 3
    # 将内部边界作为生成区域传入
    trajectories = generate_multiple_trajectories(num_trajectories, inner_quad_vertices)
    
    if not trajectories:
        print("未能成功生成任何轨迹。")
        return

    # 4. 可视化轨迹，同时显示内外边界
    visualize_trajectories(trajectories, outer_quad_vertices, inner_quad_vertices)
    
    # 5. 按时间间隔采样轨迹，生成线段
    time_interval = 0.3
    # 同样，使用内部边界来生成
    lines_a, lines_b = generateLines(generation_boundary=inner_quad_vertices, 
                                     num_lines=num_trajectories, 
                                     time_interval=time_interval)
    
    # 6. 可视化采样点，同时显示内外边界
    visualize_sampled_points(lines_a, lines_b, outer_quad_vertices, inner_quad_vertices)
    
    # 7. 创建动画，同时显示内外边界
    animate_trajectories(trajectories, outer_quad_vertices, inner_quad_vertices)


if __name__ == "__main__":
    main()