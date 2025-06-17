import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.trajectory_generator import TrajectoryGenerator

def get_quad_vertices_from_detecting_region_info():
    """从DetectingRegionInfo对象中提取四边形顶点坐标"""
    # 定义四边形顶点，使用固定值
    quad_vertices = [
        np.array([0, 0]),       # 发射器位置 transmittor_position
        np.array([200, 0]),     # 接收器1位置 receiver_position_1
        np.array([300, 150]),   # 接收器3位置 receiver_position_3
        np.array([100, 100])    # 接收器2位置 receiver_position_2
    ]

    return quad_vertices


def is_point_inside_quadrilateral(point, quad_vertices):
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


def is_bezier_curve_inside_quadrilateral(p0, p1, p2, p3, quad_vertices, num_checks=20):
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

        if not is_point_inside_quadrilateral(point, quad_vertices):
            return False

    return True


def generate_random_point_inside_quadrilateral(quad_vertices):
    """在四边形内生成随机点"""
    min_x = min(v[0] for v in quad_vertices)
    max_x = max(v[0] for v in quad_vertices)
    min_y = min(v[1] for v in quad_vertices)
    max_y = max(v[1] for v in quad_vertices)

    for _ in range(100):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        point = [x, y]

        if is_point_inside_quadrilateral(point, quad_vertices):
            return point

    centroid = np.mean(quad_vertices, axis=0)
    return centroid.tolist()


def generate_trajectory_with_ABC_in_quadrilateral(boundary_vertices, max_tries=100):
    """生成一条符合要求的轨迹，包含ABC三个点，其中AB为直线，BC为曲线，C点后沿切线方向继续直线运动"""
    for attempt in range(max_tries):
        velocity = random.uniform(15.0, 25.0)
        max_acceleration = random.uniform(20.0, 30.0)

        generator = TrajectoryGenerator(velocity=velocity, max_acceleration=max_acceleration)

        # 只生成ABC三个点，D点将根据C点的切线方向计算
        points = []
        for _ in range(3):
            points.append(generate_random_point_inside_quadrilateral(boundary_vertices))

        if not all(is_point_inside_quadrilateral(p, boundary_vertices) for p in points):
            continue

        ab_vector = np.array(points[1]) - np.array(points[0])
        bc_vector = np.array(points[2]) - np.array(points[1])

        if np.linalg.norm(ab_vector) < 5.0 or np.linalg.norm(bc_vector) < 10.0:
            continue

        ab_direction = ab_vector / np.linalg.norm(ab_vector)
        bc_direction = bc_vector / np.linalg.norm(bc_vector)

        # 检查控制点是否在四边形内
        control_distance_1 = np.linalg.norm(bc_vector) * 0.5
        control_point_1 = np.array(points[1]) + ab_direction * control_distance_1

        control_distance_2 = np.linalg.norm(bc_vector) * 0.5
        control_point_2 = np.array(points[2]) - bc_direction * control_distance_2

        if not is_point_inside_quadrilateral(control_point_1.tolist(), boundary_vertices) or \
           not is_point_inside_quadrilateral(control_point_2.tolist(), boundary_vertices) or \
           not is_bezier_curve_inside_quadrilateral(points[1], control_point_1.tolist(),
                                                 control_point_2.tolist(), points[2], boundary_vertices):
            continue

        try:
            # AB直线段
            b_velocity_direction = generator.add_straight_line(points[0], points[1], 
                                                             np.linalg.norm(ab_vector) / velocity)
            
            # BC曲线段，获取C点的切线方向
            c_tangent_direction = generator.add_curved_transition(points[1], points[2], 
                                                                b_velocity_direction, bc_direction)
            
            # 根据C点的切线方向计算D点
            # D点距离C点的距离可以是一个随机值或固定值
            cd_distance = random.uniform(30.0, 80.0)  # 可以调整这个距离范围
            point_d = np.array(points[2]) + c_tangent_direction * cd_distance
            
            # 检查D点是否在四边形内，如果不在，则缩短距离
            for distance_scale in [1.0, 0.8, 0.6, 0.4, 0.2]:
                temp_d = np.array(points[2]) + c_tangent_direction * (cd_distance * distance_scale)
                if is_point_inside_quadrilateral(temp_d.tolist(), boundary_vertices):
                    point_d = temp_d
                    cd_distance = cd_distance * distance_scale
                    break
            else:
                # 如果都不在四边形内，跳过这次尝试
                continue
            
            # CD直线段（沿着C点的切线方向）
            generator.add_straight_line(points[2], point_d.tolist(), cd_distance / velocity)
            
            return generator
        except Exception as e:
            print(f"生成轨迹时出错: {e}")
            continue

    return None


def generate_multiple_trajectories(num_trajectories):
    """生成多条轨迹"""
    # 提取四边形顶点
    quad_vertices = get_quad_vertices_from_detecting_region_info()
    
    trajectories = []
    
    for i in range(num_trajectories):
        trajectory = generate_trajectory_with_ABC_in_quadrilateral(quad_vertices)
        if trajectory:
            trajectory.name = f"轨迹{i+1}"
            trajectories.append(trajectory)
    
    return trajectories


def sample_trajectory_at_time_interval(trajectory, time_interval):
    """按固定时间间隔采样轨迹点"""
    trajectory_points = np.array(trajectory.trajectory)
    timestamps = np.array(trajectory.timestamps)
    
    total_time = timestamps[-1]
    
    # 计算采样时间点
    sample_times = np.arange(0, total_time, time_interval)
    if sample_times[-1] < total_time:
        sample_times = np.append(sample_times, total_time)
    
    sampled_points = []
    
    # 对每个采样时间点，找到对应的轨迹点
    for t in sample_times:
        # 找到时间最接近的轨迹点
        idx = np.argmin(np.abs(timestamps - t))
        sampled_points.append(trajectory_points[idx])
    
    return np.array(sampled_points)


def generate_lines_from_trajectory(trajectory_generator, time_interval):
    """从轨迹生成器按时间间隔t生成相邻点的line_a和line_b
    
    生成格式:
    line_a: 第1到第n-1个点
    line_b: 第2到第n个点
    
    返回:
    两个numpy数组，对应相邻时间点的位置
    """
    # 按时间间隔采样轨迹
    sampled_points = sample_trajectory_at_time_interval(trajectory_generator, time_interval)
    
    if len(sampled_points) < 2:
        return np.array([]), np.array([])
    
    # line_a是第1到第n-1个点
    line_a = sampled_points[:-1]
    
    # line_b是第2到第n个点
    line_b = sampled_points[1:]
    
    return line_a, line_b


def generateLines(detecting_region_info=None, num_lines=5, time_interval=0.5, seed=42):
    """生成指定数量的线对
    
    Args:
        detecting_region_info: 检测区域信息，默认为None
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
    trajectories = generate_multiple_trajectories(num_lines)
    
    # 初始化存储所有轨迹的列表
    all_lines_a = []
    all_lines_b = []
    
    # 为每条轨迹生成对应的线对
    for trajectory in trajectories:
        line_a, line_b = generate_lines_from_trajectory(trajectory, time_interval)
        if len(line_a) > 0:
            all_lines_a.append(line_a)
            all_lines_b.append(line_b)
    
    return all_lines_a, all_lines_b


def visualize_trajectories(trajectories, quad_vertices):
    """可视化轨迹"""
    plt.figure(figsize=(10, 8))
    
    # 绘制四边形边界
    x_quad = [v[0] for v in quad_vertices] + [quad_vertices[0][0]]
    y_quad = [v[1] for v in quad_vertices] + [quad_vertices[0][1]]
    plt.plot(x_quad, y_quad, 'k-', linewidth=2, label='边界')
    
    # 绘制每条轨迹
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    
    for i, trajectory in enumerate(trajectories):
        trajectory_points = np.array(trajectory.trajectory)
        plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], '-', color=colors[i], 
                 linewidth=2, label=f"{trajectory.name}")
        
        # 绘制关键点
        key_points = np.array(trajectory.key_points)
        plt.scatter(key_points[:, 0], key_points[:, 1], color=colors[i], s=100, marker='o')
        
        # 标注关键点
        for j, point in enumerate(key_points):
            if j == 0:
                label = 'A'
            elif j == 1:
                label = 'B'
            elif j == 2:
                label = 'C'
            else:
                label = 'D'
            plt.annotate(f'{label}', (point[0], point[1]), xytext=(5, 5), 
                        textcoords='offset points', color=colors[i], fontsize=12, fontweight='bold')
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('单次拐弯轨迹可视化 (A→B直线 + B→C曲线 + C→D沿切线直线)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_sampled_points(lines_a, lines_b, quad_vertices):
    """可视化采样点"""
    plt.figure(figsize=(10, 8))
    
    # 绘制四边形边界
    x_quad = [v[0] for v in quad_vertices] + [quad_vertices[0][0]]
    y_quad = [v[1] for v in quad_vertices] + [quad_vertices[0][1]]
    plt.plot(x_quad, y_quad, 'k-', linewidth=2, label='边界')
    
    # 绘制每条轨迹的采样点
    colors = plt.cm.viridis(np.linspace(0, 1, len(lines_a)))
    
    for i in range(len(lines_a)):
        plt.plot(lines_a[i][:, 0], lines_a[i][:, 1], 'o-', color=colors[i], markersize=6, 
                 alpha=0.7, label=f"轨迹{i+1} 点集A")
        plt.plot(lines_b[i][:, 0], lines_b[i][:, 1], 's-', color=colors[i], markersize=6, 
                 alpha=0.7, label=f"轨迹{i+1} 点集B")
        
        # 绘制点之间的连线
        for j in range(len(lines_a[i])):
            plt.plot([lines_a[i][j, 0], lines_b[i][j, 0]], 
                     [lines_a[i][j, 1], lines_b[i][j, 1]], 
                     '--', color=colors[i], alpha=0.3)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('单次拐弯轨迹采样点可视化')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def animate_trajectories(trajectories, quad_vertices, interval=50):
    """创建轨迹动画"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制四边形边界
    x_quad = [v[0] for v in quad_vertices] + [quad_vertices[0][0]]
    y_quad = [v[1] for v in quad_vertices] + [quad_vertices[0][1]]
    ax.plot(x_quad, y_quad, 'k-', linewidth=2, label='边界')
    
    # 设置坐标轴
    min_x = min(v[0] for v in quad_vertices)
    max_x = max(v[0] for v in quad_vertices)
    min_y = min(v[1] for v in quad_vertices)
    max_y = max(v[1] for v in quad_vertices)
    
    # 增加边界留白
    padding = 20
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('单次拐弯轨迹动画')
    ax.grid(True)
    
    # 为每条轨迹创建一个点对象
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))
    points = []
    trails = []
    labels = []
    
    for i, trajectory in enumerate(trajectories):
        # 标记关键点
        key_points = np.array(trajectory.key_points)
        ax.scatter(key_points[:, 0], key_points[:, 1], color=colors[i], s=100, marker='o')
        
        # 创建移动点
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8)
        points.append(point)
        
        # 创建轨迹线
        trail, = ax.plot([], [], '-', color=colors[i], linewidth=2, alpha=0.7)
        trails.append(trail)
        
        # 创建标签
        label = ax.text(0, 0, trajectory.name, color=colors[i], fontsize=10)
        labels.append(label)
    
    # 找出最长的轨迹时间
    max_time = max(trajectory.timestamps[-1] for trajectory in trajectories)
    
    # 计算动画帧数
    num_frames = int(max_time * 1000 / interval) + 1
    
    def init():
        for point, trail, label in zip(points, trails, labels):
            point.set_data([], [])
            trail.set_data([], [])
            label.set_position((0, 0))
            label.set_text("")
        return points + trails + labels
    
    def animate(frame):
        current_time = frame * interval / 1000  # 转换为秒
        
        for i, trajectory in enumerate(trajectories):
            timestamps = np.array(trajectory.timestamps)
            trajectory_points = np.array(trajectory.trajectory)
            
            # 找到当前时间点
            valid_indices = timestamps <= current_time
            if not any(valid_indices):
                continue
                
            valid_points = trajectory_points[valid_indices]
            
            # 更新轨迹线
            trails[i].set_data(valid_points[:, 0], valid_points[:, 1])
            
            # 更新当前点位置
            current_idx = np.sum(valid_indices) - 1
            if current_idx >= 0:
                points[i].set_data([trajectory_points[current_idx, 0]], 
                                   [trajectory_points[current_idx, 1]])
                
                # 更新标签位置
                labels[i].set_position((trajectory_points[current_idx, 0] + 5, 
                                        trajectory_points[current_idx, 1] + 5))
                labels[i].set_text(trajectory.name)
        
        return points + trails + labels
    
    anim = FuncAnimation(fig, animate, frames=num_frames, init_func=init, 
                         interval=interval, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


def main():
    # 设置随机种子，保证结果可重现
    random.seed(42)
    np.random.seed(42)
    
    # 获取四边形顶点
    quad_vertices = get_quad_vertices_from_detecting_region_info()
    
    # 生成轨迹
    num_trajectories = 5
    trajectories = generate_multiple_trajectories(num_trajectories)
    
    # 可视化轨迹
    visualize_trajectories(trajectories, quad_vertices)
    
    # 按时间间隔采样轨迹
    time_interval = 0.1  # 每0.1秒采样一次
    lines_a, lines_b = generateLines(num_lines=num_trajectories, time_interval=time_interval)
    
    # 可视化采样点
    visualize_sampled_points(lines_a, lines_b, quad_vertices)
    
    # 创建动画
    animate_trajectories(trajectories, quad_vertices)


if __name__ == "__main__":
    main()