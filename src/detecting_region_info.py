from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


class DetectingRegionInfo:
    def __init__(
        self,
        transmittor_position: np.array,
        receiver_position_1: np.array,
        receiver_position_2: np.array,
        receiver_position_3: np.array,
    ) -> None:
        self.transmittor_position = transmittor_position
        self.receiver_position_1 = receiver_position_1
        self.receiver_position_2 = receiver_position_2
        self.receiver_position_3 = receiver_position_3

        # just an alias for the above
        self.v1 = transmittor_position
        self.v2 = receiver_position_1
        self.v3 = receiver_position_2
        self.v4 = receiver_position_3

        self.pivot = transmittor_position

    def print_info(self):
        print("Transmittor Position: ", self.transmittor_position)
        print("Receiver Position 1: ", self.receiver_position_1)
        print("Receiver Position 2: ", self.receiver_position_2)
        print("Receiver Position 3: ", self.receiver_position_3)
        print("Pivot Position: ", self.pivot)

    def get_bounding_box(self):
        x_min = min(self.v1[0], self.v2[0], self.v3[0], self.v4[0])
        x_max = max(self.v1[0], self.v2[0], self.v3[0], self.v4[0])
        y_min = min(self.v1[1], self.v2[1], self.v3[1], self.v4[1])
        y_max = max(self.v1[1], self.v2[1], self.v3[1], self.v4[1])
        return (x_min, y_min), (x_max, y_max)

    # 得到（T,RX1）的极坐标
    def get_ro_beta_t_r1(self) -> tuple[float, float]:
        """
        以 TX 为原点，计算  RX1 的极坐标：
          ρ  = |T - RX1|
          β1 = arg(T - RX1)  （弧度，范围 (-π,π]）
        返回 (rho, beta1)
        """
        delta = (
            self.receiver_position_1 - self.transmittor_position
        )  # 向量从 T 指向 RX1
        rho1 = np.hypot(delta[0], delta[1])  # 或 np.linalg.norm(delta)
        beta1 = np.arctan2(delta[1], delta[0])
        return rho1, beta1

    def get_ro_beta_t_r2(self) -> tuple[float, float]:

        delta = self.receiver_position_2 - self.transmittor_position
        rho2 = np.hypot(delta[0], delta[1])
        beta2 = np.arctan2(delta[1], delta[0])
        return rho2, beta2

    def get_ro_beta_t_r3(self) -> tuple[float, float]:

        delta = self.receiver_position_3 - self.transmittor_position
        rho3 = np.hypot(delta[0], delta[1])
        beta3 = np.arctan2(delta[1], delta[0])
        return rho3, beta3

    # === 以 T 为原点、X 轴沿 T→RX1 的局部坐标系工具 ===
    def get_tx_rx1_basis(self):
        """
        返回以 T 为原点、X 轴沿 T→RX1、Y 轴为逆时针 90° 的正交基 (u, v)。
        u 为 X 轴单位向量，v 为 Y 轴单位向量。
        """
        u = self.receiver_position_1 - self.transmittor_position
        nu = np.linalg.norm(u)
        if nu < 1e-12:
            u = np.array([1.0, 0.0], dtype=float)
        else:
            u = u / nu
        v = np.array([-u[1], u[0]], dtype=float)
        return u, v

    def transform_point_to_tx_rx1(self, p: np.ndarray) -> np.ndarray:
        """
        将全局坐标点 p 变换到 (T 为原点, T→RX1 为 +X) 的局部坐标系。
        """
        p = np.asarray(p, dtype=float)
        u, v = self.get_tx_rx1_basis()
        R = np.stack([u, v], axis=1)  # 列为全局基 u,v
        return R.T @ (p - self.transmittor_position)

    def transform_points_to_tx_rx1(self, P: np.ndarray) -> np.ndarray:
        """
        向量化：将形如 (N,2) 的点集变换到局部坐标系。
        """
        P = np.asarray(P, dtype=float)
        u, v = self.get_tx_rx1_basis()
        R = np.stack([u, v], axis=1)  # (2,2)
        return (P - self.transmittor_position) @ R

    def draw_figure(self, ax=None, color="r", title=None):
        """
        Draws the detecting region quadrilateral.

        If an `ax` (Matplotlib Axes object) is provided, it draws on that.
        Otherwise, it creates a new figure and axes to draw on and displays it.

        Args:
            ax (matplotlib.axes.Axes, optional): The axes to draw on. Defaults to None.
            color (str, optional): The color for the plot elements. Defaults to "r".
            title (str, optional): An optional title for the plot.
        """
        # If no axes are provided, create a new figure and axes.
        # This maintains the original behavior of creating a standalone plot.
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 7))
            show_plot = True

        # Compute and apply a padded bounding box to frame the plot nicely
        (x_min, y_min), (x_max, y_max) = self.get_bounding_box()
        padding = 20
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # Set plot labels and title
        ax.set_xlabel("X-coordinate")
        ax.set_ylabel("Y-coordinate")
        if title:
            ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        # Draw the quadrilateral boundary
        quad = Polygon(
            [self.v1, self.v2, self.v3, self.v4],
            fill=False,
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(quad)

        # Mark and label each vertex for clarity
        verts = {
            "T / pivot / v1": self.v1,
            "R1 / v2": self.v2,
            "R2 / v3": self.v3,
            "R3 / v4": self.v4,
        }
        for label, (x, y) in verts.items():
            ax.scatter(x, y, color=color, zorder=5)  # zorder ensures points are on top
            ax.text(
                x + padding * 0.05,
                y + padding * 0.05,
                label,
                fontsize=9,
                fontweight="bold",
                color=color,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        # If a new figure was created by this method, display it.
        if show_plot:
            plt.tight_layout()
            plt.show()
