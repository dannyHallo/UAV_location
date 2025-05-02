from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


class DetectingRegionInfo:
    def __init__(self, transmittor_position: np.array, receiver_position_1: np.array, receiver_position_2: np.array, receiver_position_3: np.array) -> None:
        self.transmittor_position = transmittor_position
        self.receiver_position_1 = receiver_position_1
        self.receiver_position_2 = receiver_position_2
        self.receiver_position_3 = receiver_position_3

        # just an alias for the above
        self.v1 = transmittor_position
        self.v2 = receiver_position_1
        self.v3 = receiver_position_2
        self.v4 = receiver_position_3
        
        self.pivot_receiver = receiver_position_1
    
    def print_info(self):
        print("Transmittor Position: ", self.transmittor_position)
        print("Receiver Position 1: ", self.receiver_position_1)
        print("Receiver Position 2: ", self.receiver_position_2)
        print("Receiver Position 3: ", self.receiver_position_3)
        print("Pivot Receiver Position: ", self.pivot_receiver)

    def get_bounding_box(self):
        x_min = min(self.v1[0], self.v2[0], self.v3[0], self.v4[0])
        x_max = max(self.v1[0], self.v2[0], self.v3[0], self.v4[0])
        y_min = min(self.v1[1], self.v2[1], self.v3[1], self.v4[1])
        y_max = max(self.v1[1], self.v2[1], self.v3[1], self.v4[1])
        return (x_min, y_min), (x_max, y_max)

    def draw_figure(self, color='r'):
        _, ax = plt.subplots()

        # compute and apply a padded bounding box
        (x_min, y_min), (x_max, y_max) = self.get_bounding_box()
        padding = 10
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)

        # draw the quadrilateral
        quad = Polygon([self.v1, self.v2, self.v4, self.v3], fill=False, edgecolor=color)
        ax.add_patch(quad)

        # mark and label each vertex
        verts = {'T / v1': self.v1, 'R1 / pivot / v2': self.v2, 'R2 / v3': self.v3, 'R3 / v4': self.v4}
        for label, (x, y) in verts.items():
            ax.scatter(x, y, color=color)                    # draw a dot
            ax.text(
                x, y, label,
                fontsize=12,
                fontweight='bold',
                color=color,
                ha='left', va='bottom',
                # offset so text doesn't sit directly on the point
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

        plt.show()