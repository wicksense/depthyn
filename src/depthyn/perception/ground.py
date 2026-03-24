"""Ground segmentation for stationary LiDAR point clouds.

Removes ground-plane points before clustering to prevent objects from
merging with the ground surface. Uses a grid-based elevation approach
that works without external dependencies (no Open3D/scipy required).

The method:
1. Divide XY plane into cells
2. For each cell, find the minimum Z (ground estimate)
3. Points within `ground_tolerance_m` of the cell's min Z are ground
4. Return only non-ground points
"""

from __future__ import annotations

import math

from depthyn.models import Point3D


def remove_ground(
    points: list[Point3D],
    *,
    cell_size_m: float = 2.0,
    ground_tolerance_m: float = 0.15,
    min_cell_points: int = 3,
) -> list[Point3D]:
    """Remove ground-plane points using a fitted plane model.

    Fits a global ground plane (z = ax + by + c) to the lowest points,
    then removes any point within ``ground_tolerance_m`` of this plane.
    This handles tilted ground surfaces and avoids the per-cell min-Z
    problem where object points can be the lowest in their cell.

    Parameters
    ----------
    points:
        Input point cloud.
    cell_size_m:
        Not used in plane-fit mode (kept for API compatibility).
    ground_tolerance_m:
        Points within this distance above the fitted ground plane are
        removed. Default 0.15m — tight enough to keep people (~0.2m
        above ground) but removes flat ground returns.
    min_cell_points:
        Not used in plane-fit mode (kept for API compatibility).

    Returns
    -------
    Non-ground points.
    """
    if len(points) < 10:
        return list(points)

    # Collect Z values for percentile calculation
    n = len(points)
    z_values = sorted(p[2] for p in points)

    # Use the bottom 30% of points by Z as ground candidates
    cutoff_idx = max(1, int(n * 0.3))
    z_threshold = z_values[cutoff_idx]

    # Fit plane z = ax + by + c using least squares on ground candidates
    # Simplified: compute mean and covariance manually (no numpy needed)
    sum_x = sum_y = sum_z = 0.0
    sum_xx = sum_xy = sum_xz = 0.0
    sum_yy = sum_yz = 0.0
    count = 0

    for x, y, z in points:
        if z <= z_threshold:
            sum_x += x
            sum_y += y
            sum_z += z
            sum_xx += x * x
            sum_xy += x * y
            sum_xz += x * z
            sum_yy += y * y
            sum_yz += y * z
            count += 1

    if count < 3:
        return list(points)

    # Solve normal equations for z = ax + by + c
    # [sum_xx  sum_xy  sum_x] [a]   [sum_xz]
    # [sum_xy  sum_yy  sum_y] [b] = [sum_yz]
    # [sum_x   sum_y   count] [c]   [sum_z ]
    #
    # Use Cramer's rule for 3x3 system
    m = [
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x, sum_y, float(count)],
    ]
    rhs = [sum_xz, sum_yz, sum_z]

    det = _det3(m)
    if abs(det) < 1e-10:
        # Degenerate — fall back to flat plane at median Z
        median_z = z_values[n // 2]
        return [p for p in points if p[2] > median_z + ground_tolerance_m]

    a = _det3([
        [rhs[0], m[0][1], m[0][2]],
        [rhs[1], m[1][1], m[1][2]],
        [rhs[2], m[2][1], m[2][2]],
    ]) / det
    b = _det3([
        [m[0][0], rhs[0], m[0][2]],
        [m[1][0], rhs[1], m[1][2]],
        [m[2][0], rhs[2], m[2][2]],
    ]) / det
    c = _det3([
        [m[0][0], m[0][1], rhs[0]],
        [m[1][0], m[1][1], rhs[1]],
        [m[2][0], m[2][1], rhs[2]],
    ]) / det

    # Remove points near the fitted ground plane
    elevated: list[Point3D] = []
    for x, y, z in points:
        ground_z = a * x + b * y + c
        if z > ground_z + ground_tolerance_m:
            elevated.append((x, y, z))

    return elevated


def _det3(m: list[list[float]]) -> float:
    """3x3 matrix determinant."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
