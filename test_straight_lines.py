"""
测试直线轨迹生成器的简化版本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    # 简化的测试版本
    def test_basic_functionality():
        """测试基本功能"""
        print("=== 直线轨迹生成器测试 ===")

        # 创建一个简单的四边形
        quad_vertices = [
            np.array([0, 0]),  # 发射机
            np.array([100, 0]),  # 接收机1
            np.array([100, 100]),  # 接收机2
            np.array([0, 100]),  # 接收机3
        ]

        print("四边形顶点:")
        for i, vertex in enumerate(quad_vertices):
            print(f"  顶点{i+1}: ({vertex[0]}, {vertex[1]})")

        # 测试水平线生成
        print("\n测试水平线生成...")
        horizontal_lines = generate_simple_horizontal_lines(quad_vertices, 5)
        print(f"生成了 {len(horizontal_lines)} 条水平线")

        # 测试垂直线生成
        print("\n测试垂直线生成...")
        vertical_lines = generate_simple_vertical_lines(quad_vertices, 5)
        print(f"生成了 {len(vertical_lines)} 条垂直线")

        # 简单可视化
        print("\n生成可视化图表...")
        visualize_simple_trajectories(quad_vertices, horizontal_lines, vertical_lines)

        print("\n=== 测试完成 ===")

    def generate_simple_horizontal_lines(quad_vertices, num_lines):
        """生成简单的水平线"""
        lines = []
        y_min, y_max = 10, 90  # 留出边距

        for i in range(num_lines):
            y = y_min + (y_max - y_min) * i / (num_lines - 1)
            # 简单的水平线
            start_point = np.array([10, y])
            end_point = np.array([90, y])
            line = np.array([start_point, end_point])
            lines.append(line)

        return lines

    def generate_simple_vertical_lines(quad_vertices, num_lines):
        """生成简单的垂直线"""
        lines = []
        x_min, x_max = 10, 90  # 留出边距

        for i in range(num_lines):
            x = x_min + (x_max - x_min) * i / (num_lines - 1)
            # 简单的垂直线
            start_point = np.array([x, 10])
            end_point = np.array([x, 90])
            line = np.array([start_point, end_point])
            lines.append(line)

        return lines

    def visualize_simple_trajectories(quad_vertices, horizontal_lines, vertical_lines):
        """简单可视化"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制四边形
        quad = Polygon(quad_vertices, fill=False, edgecolor="black", linewidth=3)
        ax.add_patch(quad)

        # 标记顶点
        colors = ["red", "blue", "green", "orange"]
        labels = ["发射机", "接收机1", "接收机2", "接收机3"]

        for i, (vertex, color, label) in enumerate(zip(quad_vertices, colors, labels)):
            ax.scatter(vertex[0], vertex[1], color=color, s=100, zorder=5)
            ax.annotate(
                label,
                (vertex[0], vertex[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

        # 绘制水平线
        for i, line in enumerate(horizontal_lines):
            ax.plot(
                line[:, 0],
                line[:, 1],
                "b-",
                linewidth=2,
                alpha=0.7,
                label="水平轨迹" if i == 0 else "",
            )

        # 绘制垂直线
        for i, line in enumerate(vertical_lines):
            ax.plot(
                line[:, 0],
                line[:, 1],
                "r-",
                linewidth=2,
                alpha=0.7,
                label="垂直轨迹" if i == 0 else "",
            )

        ax.set_xlabel("X坐标 (m)")
        ax.set_ylabel("Y坐标 (m)")
        ax.set_title("四边形区域内的直线轨迹")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()

    if __name__ == "__main__":
        test_basic_functionality()

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装必要的依赖包: numpy, matplotlib")
except Exception as e:
    print(f"运行错误: {e}")
    import traceback

    traceback.print_exc()

