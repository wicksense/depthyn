from __future__ import annotations

import math

from depthyn.models import Point3D


def centroid(points: list[Point3D]) -> Point3D:
    if not points:
        return (0.0, 0.0, 0.0)
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def bbox(points: list[Point3D]) -> tuple[Point3D, Point3D]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def distance_xy(a: Point3D, b: Point3D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def xy_cell(point: Point3D, cell_size_m: float) -> tuple[int, int]:
    return (
        math.floor(point[0] / cell_size_m),
        math.floor(point[1] / cell_size_m),
    )

