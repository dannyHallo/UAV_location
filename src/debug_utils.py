import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def print_lines_shape(lines):
    # Print the shape of the lines array
    print("Total lines: ", len(lines))
    for i in range(len(lines)):
        print("Line ", i, " shape: ", lines[i].shape)


def visualizeLines(detecting_region_info, lines, color="r"):
    # For visualization purposes, we will use matplotlib to plot the lines and the detecting region.

    _, ax = plt.subplots()
    ax.set_xlim(-120, 120)
    ax.set_ylim(-50, 150)
    ax.add_patch(
        Polygon(
            [
                detecting_region_info.v1,
                detecting_region_info.v2,
                detecting_region_info.v4,
                detecting_region_info.v3,
            ],
            fill=False,
        )
    )

    # Plot the lines
    for i in range(len(lines)):
        line = lines[i]
        ax.plot([point[0] for point in line], [point[1] for point in line], color)

    plt.show()
