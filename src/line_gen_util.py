import numpy as np


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def point_in_triangle(v1, v2, v3, pt):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def pointInRegion(detecting_region_info, pt) -> bool:
    v1 = detecting_region_info.v1
    v2 = detecting_region_info.v2
    v3 = detecting_region_info.v3
    v4 = detecting_region_info.v4

    # check if point is in boundary (quadrilateral -> 2 triangles)
    return point_in_triangle(v1, v2, v3, pt) or point_in_triangle(v3, v2, v4, pt)


def getTriangleSize(v1, v2, v3) -> float:
    """
    Calculate the area of a triangle using a determinant-based approach.
    """
    matrix = np.array([[v1[0], v1[1], 1], [v2[0], v2[1], 1], [v3[0], v3[1], 1]])

    det = np.linalg.det(matrix)
    area = 0.5 * np.abs(det)
    return area
