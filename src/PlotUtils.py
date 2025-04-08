import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def visualizeLines(detecting_region_info, lines, color='r'):
    # For visualization purposes, we will use matplotlib to plot the lines and the detecting region.

    _, ax = plt.subplots()
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 150)
    ax.add_patch(Polygon([detecting_region_info.v1,
                 detecting_region_info.v2, detecting_region_info.v4, detecting_region_info.v3], fill=False))

    # Plot the lines
    for i in range(len(lines)):
        line = lines[i]
        ax.plot([point[0] for point in line], [point[1]
                for point in line], color)

    plt.show()
