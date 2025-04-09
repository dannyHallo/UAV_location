import numpy as np
import matplotlib.pyplot as plt
import random

class TrajectoryGenerator:
    def __init__(self, velocity=20.0, max_acceleration=25.0, name=None):
        """初始化轨迹生成器
        
        Args:
            velocity: 匀速直线段的速度，单位：米/秒
            max_acceleration: 最大加速度，单位：米/秒²
            name: 轨迹名称（可选）
        """
        self.velocity = velocity
        self.max_acceleration = max_acceleration
        self.name = name
        self.trajectory = []
        self.timestamps = []
        self.velocities = []
        self.accelerations = []
        self.time = 0.0
        self.key_points = []  # 存储关键点
        self.key_times = []   # 存储关键点对应的时间
    
    def add_straight_line(self, start_point, end_point, duration):
        """添加匀速直线段
        
        Args:
            start_point: 起点坐标 [x, y]
            end_point: 终点坐标 [x, y]
            duration: 直线段持续时间，单位：秒
        """
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        
        # 记录关键点
        if len(self.key_points) == 0:
            self.key_points.append(start_point)
            self.key_times.append(self.time)
        self.key_points.append(end_point)
        self.key_times.append(self.time + duration)
        
        # 计算采样点数量
        num_samples = max(int(duration * 10), 2)  # 10Hz采样率，至少2个点
        
        # 生成时间戳
        times = np.linspace(0, duration, num_samples)
        
        # 生成位置点
        for t in times:
            alpha = t / duration
            position = (1 - alpha) * start_point + alpha * end_point
            self.trajectory.append(position)
            self.timestamps.append(self.time + t)
        
        # 计算速度（匀速）
        velocity_vector = (end_point - start_point) / duration
        speed = np.linalg.norm(velocity_vector)
        if speed > 0:
            velocity_unit = velocity_vector / speed
        else:
            velocity_unit = np.array([1.0, 0.0])  # 默认方向
            
        # 保存实际单位速度方向（用于确保连续性）
        self.last_velocity_direction = velocity_unit
        
        # 添加速度和加速度数据
        for _ in range(num_samples):
            self.velocities.append(velocity_unit * self.velocity)
            self.accelerations.append(np.zeros(2))  # 匀速运动加速度为0
        
        # 更新总时间
        self.time += duration
        
        return velocity_unit  # 返回末端速度方向
    
    def add_curved_transition(self, start_point, end_point, start_direction, end_direction, control_point_offset_factor=0.7):
        """添加确保速度连续性的平滑弯曲过渡
        
        Args:
            start_point: 起点坐标 [x, y]
            end_point: 终点坐标 [x, y]
            start_direction: 起点速度方向 [dx, dy]
            end_direction: 终点期望速度方向 [dx, dy]
            control_point_offset_factor: 控制点偏移系数，越大弯曲越明显
        """
        start_point = np.array(start_point)
        end_point = np.array(end_point)
        start_direction = np.array(start_direction)
        end_direction = np.array(end_direction)
        
        # 检查起点和终点是否不同
        if np.allclose(start_point, end_point):
            print("警告: 起点和终点太近，跳过曲线生成")
            return None
        
        # 记录关键点
        if len(self.key_points) == 0:
            self.key_points.append(start_point)
            self.key_times.append(self.time)
        
        # 归一化方向向量
        if np.linalg.norm(start_direction) > 0:
            start_direction = start_direction / np.linalg.norm(start_direction)
        else:
            start_direction = np.array([1.0, 0.0])  # 默认方向
            
        if np.linalg.norm(end_direction) > 0:
            end_direction = end_direction / np.linalg.norm(end_direction)
        else:
            end_direction = np.array([1.0, 0.0])  # 默认方向
        
        # 计算起点和终点之间的直线距离
        direct_distance = np.linalg.norm(end_point - start_point)
        
        # 确定两个控制点位置，使得曲线在起点和终点处的切线方向分别为start_direction和end_direction
        # 首个控制点在起点沿起点速度方向延伸
        control_distance_1 = direct_distance * control_point_offset_factor
        control_point_1 = start_point + start_direction * control_distance_1
        
        # 第二个控制点在终点沿终点速度方向反向延伸
        control_distance_2 = direct_distance * control_point_offset_factor
        control_point_2 = end_point - end_direction * control_distance_2
        
        # 估计曲线长度 - 使用多段线近似
        curve_length = (np.linalg.norm(control_point_1 - start_point) + 
                       np.linalg.norm(control_point_2 - control_point_1) + 
                       np.linalg.norm(end_point - control_point_2))
        
        # 计算基于速度的时间
        duration = curve_length / self.velocity
        
        # 记录终点
        self.key_points.append(end_point)
        self.key_times.append(self.time + duration)
        
        # 使用参数化三次贝塞尔曲线
        num_samples = max(int(duration * 30), 30)  # 30Hz采样率，至少30个点
        t_values = np.linspace(0, 1, num_samples)
        
        positions = []
        velocities = []
        accelerations = []
        
        # 生成三次贝塞尔曲线点
        for t in t_values:
            # 计算位置 - 三次贝塞尔曲线公式
            # B(t) = (1-t)³*P₀ + 3(1-t)²t*P₁ + 3(1-t)t²*P₂ + t³*P₃
            b0 = (1-t)**3
            b1 = 3*(1-t)**2*t
            b2 = 3*(1-t)*t**2
            b3 = t**3
            
            position = b0 * start_point + b1 * control_point_1 + b2 * control_point_2 + b3 * end_point
            positions.append(position)
            
            # 计算速度（贝塞尔曲线的一阶导数）
            # B'(t) = 3(1-t)²*(P₁-P₀) + 6(1-t)t*(P₂-P₁) + 3t²*(P₃-P₂)
            v0 = 3*(1-t)**2
            v1 = 6*(1-t)*t
            v2 = 3*t**2
            
            velocity_dir = v0 * (control_point_1 - start_point) + \
                          v1 * (control_point_2 - control_point_1) + \
                          v2 * (end_point - control_point_2)
            
            # 归一化速度方向
            speed = np.linalg.norm(velocity_dir)
            if speed > 1e-10:
                velocity_dir = velocity_dir / speed
                velocities.append(velocity_dir * self.velocity)
            else:
                velocities.append(np.array([0.0, 0.0]))
            
            # 计算加速度（贝塞尔曲线的二阶导数）
            # B''(t) = 6(1-t)*(P₂-2P₁+P₀) + 6t*(P₃-2P₂+P₁)
            a0 = 6*(1-t)
            a1 = 6*t
            
            acceleration = a0 * (control_point_2 - 2*control_point_1 + start_point) + \
                          a1 * (end_point - 2*control_point_2 + control_point_1)
            
            # 计算加速度大小 - 确保不超过最大值
            accel_magnitude = self.velocity**2 * np.linalg.norm(acceleration) / (speed**2) if speed > 1e-10 else 0
            
            # 限制加速度大小
            if accel_magnitude > self.max_acceleration:
                accel_magnitude = self.max_acceleration
            
            # 计算加速度方向 - 垂直于速度方向
            if speed > 1e-10:
                # 计算法向量（垂直于速度方向）
                normal_vector = np.array([-velocity_dir[1], velocity_dir[0]])
                
                # 确保法向量与加速度方向一致
                if np.dot(normal_vector, acceleration) < 0:
                    normal_vector = -normal_vector
                
                accelerations.append(normal_vector * accel_magnitude)
            else:
                accelerations.append(np.array([0.0, 0.0]))
        
        # 添加到轨迹
        for i in range(num_samples):
            self.trajectory.append(positions[i])
            self.timestamps.append(self.time + duration * t_values[i])
            self.velocities.append(velocities[i])
            self.accelerations.append(accelerations[i])
        
        # 保存最后的速度方向（用于确保连续性）
        self.last_velocity_direction = velocities[-1] / self.velocity if velocities else end_direction
        
        # 更新总时间
        self.time += duration
        
        # 返回终点实际速度方向（可能与期望方向略有不同）
        return self.last_velocity_direction

def is_point_inside_quadrilateral(point, quad_vertices):
    """检查点是否在四边形内
    
    Args:
        point: 点坐标 [x, y]
        quad_vertices: 四边形顶点坐标 [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        
    Returns:
        bool: 如果点在四边形内则返回True
    """
    # 射线法判断点是否在多边形内
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
    """检查三次贝塞尔曲线是否在四边形内
    
    Args:
        p0: 曲线起点 [x, y]
        p1: 第一个控制点 [x, y]
        p2: 第二个控制点 [x, y]
        p3: 曲线终点 [x, y]
        quad_vertices: 四边形顶点坐标
        num_checks: 检查点数量
        
    Returns:
        bool: 如果曲线在四边形内则返回True
    """
    # 检查曲线上的多个点
    for i in range(num_checks + 1):
        t = i / num_checks
        # 三次贝塞尔曲线公式
        b0 = (1-t)**3
        b1 = 3*(1-t)**2*t
        b2 = 3*(1-t)*t**2
        b3 = t**3
        
        point = [
            b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0],
            b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        ]
        
        if not is_point_inside_quadrilateral(point, quad_vertices):
            return False
    
    return True

def generate_random_point_inside_quadrilateral(quad_vertices):
    """在四边形内生成随机点
    
    Args:
        quad_vertices: 四边形顶点坐标
        
    Returns:
        list: 随机点坐标 [x, y]
    """
    # 计算四边形的边界盒
    min_x = min(v[0] for v in quad_vertices)
    max_x = max(v[0] for v in quad_vertices)
    min_y = min(v[1] for v in quad_vertices)
    max_y = max(v[1] for v in quad_vertices)
    
    # 在边界盒内随机生成点，直到找到一个在四边形内的点
    for _ in range(100):  # 最多尝试100次
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        point = [x, y]
        
        if is_point_inside_quadrilateral(point, quad_vertices):
            return point
    
    # 如果随机失败，使用四边形的重心
    centroid = np.mean(quad_vertices, axis=0)
    return centroid.tolist()

def generate_trajectory_with_ABCD_in_quadrilateral(quad_vertices, max_tries=100):
    """生成一条符合要求的轨迹，包含ABCD四个点，且BC之间为明显的曲线，确保速度方向连续
    
    Args:
        quad_vertices: 四边形顶点坐标
        max_tries: 最大尝试次数
        
    Returns:
        TrajectoryGenerator: 生成的轨迹对象，如果失败则返回None
    """
    for attempt in range(max_tries):
        # 随机速度和加速度
        velocity = random.uniform(15.0, 25.0)
        max_acceleration = random.uniform(20.0, 30.0)
        
        generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)
        
        # 随机生成A、B、C、D四个点
        points = []
        for _ in range(4):
            points.append(generate_random_point_inside_quadrilateral(quad_vertices))
        
        # 确保所有点在四边形内
        if not all(is_point_inside_quadrilateral(p, quad_vertices) for p in points):
            continue
        
        # 计算AB和CD的方向向量
        ab_vector = np.array(points[1]) - np.array(points[0])
        cd_vector = np.array(points[3]) - np.array(points[2])
        
        # 确保AB和CD足够长
        if np.linalg.norm(ab_vector) < 5.0 or np.linalg.norm(cd_vector) < 5.0:
            continue
            
        # 归一化方向向量
        ab_direction = ab_vector / np.linalg.norm(ab_vector)
        cd_direction = cd_vector / np.linalg.norm(cd_vector)
        
        # 计算BC的中间方向 - 这不是BC的方向，而是用于确定曲线的形状
        bc_vector = np.array(points[2]) - np.array(points[1])
        if np.linalg.norm(bc_vector) < 10.0:  # 确保BC距离足够长
            continue
            
        bc_direction = bc_vector / np.linalg.norm(bc_vector)
        
        # 确定BC曲线的两个控制点
        # 第一个控制点沿着AB的末端方向延伸
        control_distance_1 = np.linalg.norm(bc_vector) * 0.5
        control_point_1 = np.array(points[1]) + ab_direction * control_distance_1
        
        # 第二个控制点沿着CD的起始方向反向延伸
        control_distance_2 = np.linalg.norm(bc_vector) * 0.5
        control_point_2 = np.array(points[2]) - cd_direction * control_distance_2
        
        # 检查控制点和整条贝塞尔曲线是否在四边形内
        if not is_point_inside_quadrilateral(control_point_1.tolist(), quad_vertices) or \
           not is_point_inside_quadrilateral(control_point_2.tolist(), quad_vertices) or \
           not is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(), 
                                                  control_point_2.tolist(), points[2], quad_vertices):
            continue
        
        try:
            # A -> B (直线，返回B点速度方向)
            b_velocity_direction = generator.add_straight_line(points[0], points[1], 
                                                              np.linalg.norm(ab_vector) / velocity)
            
            # B -> C (确保在B点速度方向连续，并在C点达到与CD直线段匹配的速度方向)
            c_velocity_direction = generator.add_curved_transition(points[1], points[2], 
                                                                 b_velocity_direction, cd_direction)
            
            # C -> D (直线，起始速度方向从曲线末端继承)
            generator.add_straight_line(points[2], points[3], np.linalg.norm(cd_vector) / velocity)
            
            return generator
        except Exception as e:
            print(f"尝试 {attempt+1}: 创建轨迹失败: {e}")
    
    print(f"在 {max_tries} 次尝试后未能生成有效轨迹")
    return None

def generate_multiple_trajectories(num_trajectories, quad_vertices):
    """生成多条轨迹
    
    Args:
        num_trajectories: 要生成的轨迹数量
        quad_vertices: 四边形顶点坐标
        
    Returns:
        list: 生成的轨迹对象列表
    """
    trajectories = []
    
    for i in range(num_trajectories):
        print(f"生成轨迹 {i+1}/{num_trajectories}...")
        trajectory = generate_trajectory_with_ABCD_in_quadrilateral(quad_vertices)
        if trajectory:
            trajectory.name = f"轨迹{i+1}"
            trajectories.append(trajectory)
    
    return trajectories

def sample_trajectory_at_time(trajectory, timestamps, sample_time):
    """在指定时间点采样轨迹
    
    Args:
        trajectory: 轨迹点列表
        timestamps: 轨迹点对应的时间戳
        sample_time: 采样时间点
        
    Returns:
        np.array: 采样点坐标
    """
    if sample_time < timestamps[0]:
        return trajectory[0]  # 如果采样时间早于轨迹起始时间，返回第一个点
    
    if sample_time >= timestamps[-1]:
        return trajectory[-1]  # 如果采样时间晚于轨迹结束时间，返回最后一个点
    
    # 找到采样时间所在的时间区间
    for i in range(len(timestamps) - 1):
        if timestamps[i] <= sample_time < timestamps[i + 1]:
            # 线性插值计算采样点
            t = (sample_time - timestamps[i]) / (timestamps[i + 1] - timestamps[i])
            return trajectory[i] * (1 - t) + trajectory[i + 1] * t
    
    # 不应该到达这里
    return trajectory[-1]

def generate_lines_from_trajectory(trajectory_generator, t1, t2):
    """从轨迹生成器生成两条采样线
    
    Args:
        trajectory_generator: 轨迹生成器对象
        t1: 第一条线的起始采样时间
        t2: 采样周期
        
    Returns:
        tuple: (lines_a, lines_b) - 两条采样线的点列表
    """
    trajectory = np.array(trajectory_generator.trajectory)
    timestamps = np.array(trajectory_generator.timestamps)
    
    total_time = timestamps[-1]  # 轨迹总时长
    
    lines_a = []
    lines_b = []
    
    # 生成line A: 采样时间点 0, (t1 + t2), 2*(t1 + t2), ...
    sample_time = 0
    while sample_time <= total_time:
        point = sample_trajectory_at_time(trajectory, timestamps, sample_time)
        lines_a.append(point)
        sample_time += (t1 + t2)
    
    # 生成line B: 采样时间点 t1, t1 + (t1 + t2), t1 + 2*(t1 + t2), ...
    sample_time = t1
    while sample_time <= total_time:
        point = sample_trajectory_at_time(trajectory, timestamps, sample_time)
        lines_b.append(point)
        sample_time += (t1 + t2)
    
    return np.array(lines_a), np.array(lines_b)

def measure_line_length(line_points):
    """计算折线的总长度
    
    Args:
        line_points: 折线的点列表
        
    Returns:
        float: 折线的总长度
    """
    if len(line_points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(line_points) - 1):
        segment_length = np.linalg.norm(line_points[i+1] - line_points[i])
        total_length += segment_length
    
    return total_length

def generate_lines(quad_vertices, num_lines=10, t1=0.5, t2=0.5):
    """生成指定数量的线对
    
    Args:
        quad_vertices: 四边形顶点坐标
        num_lines: 要生成的线对数量
        t1: 第一条线的起始采样时间
        t2: 采样周期
        
    Returns:
        tuple: (all_lines_a, all_lines_b) - 所有线对
    """
    # 设置随机种子以便结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 生成轨迹
    trajectories = generate_multiple_trajectories(num_lines, quad_vertices)
    
    all_lines_a = []
    all_lines_b = []
    
    # 为每条轨迹生成对应的线对
    for trajectory in trajectories:
        lines_a, lines_b = generate_lines_from_trajectory(trajectory, t1, t2)
        all_lines_a.append(lines_a)
        all_lines_b.append(lines_b)
    
    return all_lines_a, all_lines_b

def plot_trajectories_in_quadrilateral(trajectories, quad_vertices, title=None):
    """绘制四边形内的多条轨迹
    
    Args:
        trajectories: 轨迹对象列表
        quad_vertices: 四边形顶点坐标
        title: 图表标题
    """
    plt.figure(figsize=(16, 12))
    
    # 绘制四边形边界
    quad_vertices_closed = quad_vertices + [quad_vertices[0]]  # 闭合多边形
    x_values = [v[0] for v in quad_vertices_closed]
    y_values = [v[1] for v in quad_vertices_closed]
    plt.plot(x_values, y_values, 'k-', linewidth=3)
    
    # 标注顶点
    for i, vertex in enumerate(quad_vertices):
        plt.scatter(vertex[0], vertex[1], c='black', s=100, zorder=5)
        plt.annotate(f'P{i+1} ({vertex[0]}, {vertex[1]})', 
                   (vertex[0], vertex[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 使用渐变色
    colors = plt.cm.jet(np.linspace(0, 1, len(trajectories)))
    
    # 绘制轨迹
    for i, trajectory in enumerate(trajectories):
        # 轨迹线
        trajectory_array = np.array(trajectory.trajectory)
        plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], color=colors[i], 
                linewidth=1.5, alpha=0.7)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'四边形区域内的{len(trajectories)}条平滑轨迹')
    
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lines(quad_vertices, lines_a, lines_b, title=None):
    """绘制四边形和线对
    
    Args:
        quad_vertices: 四边形顶点坐标
        lines_a: A线列表
        lines_b: B线列表
        title: 图表标题
    """
    plt.figure(figsize=(16, 12))
    
    # 绘制四边形边界
    quad_vertices_closed = quad_vertices + [quad_vertices[0]]  # 闭合多边形
    x_values = [v[0] for v in quad_vertices_closed]
    y_values = [v[1] for v in quad_vertices_closed]
    plt.plot(x_values, y_values, 'k-', linewidth=3)
    
    # 标注顶点
    for i, vertex in enumerate(quad_vertices):
        plt.scatter(vertex[0], vertex[1], c='black', s=100, zorder=5)
        plt.annotate(f'P{i+1} ({vertex[0]}, {vertex[1]})', 
                   (vertex[0], vertex[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
    
    # 使用不同颜色绘制A线和B线
    for i in range(len(lines_a)):
        color = plt.cm.tab10(i % 10)
        
        # 绘制A线
        if len(lines_a[i]) > 1:
            plt.plot(lines_a[i][:, 0], lines_a[i][:, 1], 'o-', color=color, alpha=0.7,
                    linewidth=1.5, markersize=4, label=f'Line A {i+1}' if i == 0 else "")
        
        # 绘制B线
        if len(lines_b[i]) > 1:
            plt.plot(lines_b[i][:, 0], lines_b[i][:, 1], 's--', color=color, alpha=0.7,
                    linewidth=1.5, markersize=4, label=f'Line B {i+1}' if i == 0 else "")
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    
    if title:
        plt.title(title)
    else:
        plt.title(f'四边形区域内的{len(lines_a)}对采样线')
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    random.seed(2)
    np.random.seed(42)
    
    # 定义四边形顶点
    quad_vertices = [
        [0, 0],       # P1
        [200, 0],     # P2
        [300, 150],   # P3
        [100, 100]    # P4
    ]
    
    # 生成轨迹和对应的线对
    print("开始生成轨迹和采样线...")
    all_lines_a, all_lines_b = generate_lines(quad_vertices, num_lines=1, t1=0.1, t2=0.2)
    
    # 计算线长
    for i in range(len(all_lines_a)):
        length_a = measure_line_length(all_lines_a[i])
        length_b = measure_line_length(all_lines_b[i])
        print(f"线对 {i+1}: 线A长度 = {length_a:.2f}m, 线B长度 = {length_b:.2f}m")
    
    # 绘制线对
    plot_lines(quad_vertices, all_lines_a, all_lines_b)