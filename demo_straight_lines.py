"""
直线轨迹生成器演示 - 简化版本
展示在四边形区域内生成横竖直线轨迹的概念
"""


def demo_straight_line_generation():
    """演示直线轨迹生成的基本概念"""

    print("=== 直线轨迹生成器演示 ===")
    print()

    # 1. 定义四边形区域（发射机和三个接收机）
    print("1. 四边形区域定义:")
    print("   发射机 (T): (0, 0)")
    print("   接收机1 (R1): (100, 0)")
    print("   接收机2 (R2): (100, 100)")
    print("   接收机3 (R3): (0, 100)")
    print()

    # 2. 水平轨迹生成
    print("2. 水平轨迹生成:")
    print("   在四边形内生成水平直线，从左到右穿过区域")
    horizontal_lines = []
    for i in range(5):
        y = 20 + i * 15  # 在y=20到y=80之间生成5条水平线
        start_x, end_x = 10, 90  # 留出边距
        print(f"   水平线{i+1}: y={y}, 从({start_x},{y})到({end_x},{y})")
        horizontal_lines.append(
            {"type": "horizontal", "y": y, "start": (start_x, y), "end": (end_x, y)}
        )
    print()

    # 3. 垂直轨迹生成
    print("3. 垂直轨迹生成:")
    print("   在四边形内生成垂直直线，从上到下穿过区域")
    vertical_lines = []
    for i in range(5):
        x = 20 + i * 15  # 在x=20到x=80之间生成5条垂直线
        start_y, end_y = 10, 90  # 留出边距
        print(f"   垂直线{i+1}: x={x}, 从({x},{start_y})到({x},{end_y})")
        vertical_lines.append(
            {"type": "vertical", "x": x, "start": (x, start_y), "end": (x, end_y)}
        )
    print()

    # 4. 轨迹特性
    print("4. 轨迹特性:")
    print("   - 所有轨迹都是匀速直线运动")
    print("   - 速度: 20 m/s (可配置)")
    print("   - 时间间隔: 1秒采样一次")
    print("   - 轨迹完全在四边形区域内")
    print("   - 横竖轨迹形成网格状覆盖")
    print()

    # 5. 应用场景
    print("5. 应用场景:")
    print("   - 无人机路径规划")
    print("   - 多普勒频移数据采集")
    print("   - 机器学习训练数据生成")
    print("   - 定位算法测试")
    print()

    # 6. 可视化说明
    print("6. 可视化效果:")
    print("   - 黑色边框: 四边形检测区域")
    print("   - 红点: 发射机位置")
    print("   - 蓝/绿/橙点: 三个接收机位置")
    print("   - 蓝色线条: 水平轨迹")
    print("   - 红色线条: 垂直轨迹")
    print("   - 圆点: 轨迹起点")
    print("   - 方点: 轨迹终点")
    print()

    # 7. 代码结构说明
    print("7. 代码结构:")
    print("   - StraightLineTrajectoryGenerator: 主生成器类")
    print("   - generate_horizontal_lines(): 生成水平轨迹")
    print("   - generate_vertical_lines(): 生成垂直轨迹")
    print("   - visualize_trajectories(): 可视化功能")
    print("   - 完全兼容现有项目结构")
    print()

    print("=== 演示完成 ===")
    print()
    print("要运行完整的可视化版本，请确保环境配置正确后运行:")
    print("python straight_line_trajectory_generator.py")


def show_code_example():
    """显示代码使用示例"""
    print("\n=== 代码使用示例 ===")
    print()

    code_example = """
# 导入必要的模块
from straight_line_trajectory_generator import StraightLineTrajectoryGenerator, visualize_trajectories
from src.detecting_region_info_generator import generate_detecting_region_infos

# 1. 生成检测区域
detecting_region_infos = generate_detecting_region_infos(num_configurations=1, seed=42)
detecting_region_info = detecting_region_infos[0]

# 2. 创建轨迹生成器
generator = StraightLineTrajectoryGenerator(velocity=20.0, time_interval=1.0)

# 3. 获取四边形顶点
quad_vertices = [
    detecting_region_info.transmittor_position,
    detecting_region_info.receiver_position_1,
    detecting_region_info.receiver_position_2,
    detecting_region_info.receiver_position_3
]

# 4. 生成轨迹
horizontal_lines, vertical_lines = generator.generate_mixed_lines(
    quad_vertices=quad_vertices,
    num_horizontal=8,
    num_vertical=8,
    seed=42
)

# 5. 可视化
visualize_trajectories(
    detecting_region_info=detecting_region_info,
    horizontal_lines=horizontal_lines,
    vertical_lines=vertical_lines,
    title="四边形区域内的直线轨迹"
)
"""

    print(code_example)


if __name__ == "__main__":
    demo_straight_line_generation()
    show_code_example()

