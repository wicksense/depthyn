from __future__ import annotations

from collections import deque

from depthyn.models import Detection, Point3D
from depthyn.perception.preprocess import bbox, centroid, xy_cell


def cluster_points(
    points: list[Point3D],
    *,
    cell_size_m: float,
    min_cluster_points: int,
    min_cluster_cells: int,
    min_cluster_height_m: float,
    max_cluster_height_m: float,
    max_cluster_width_m: float,
) -> list[Detection]:
    if not points:
        return []

    cell_points: dict[tuple[int, int], list[Point3D]] = {}
    for point in points:
        cell = xy_cell(point, cell_size_m)
        cell_points.setdefault(cell, []).append(point)

    detections: list[Detection] = []
    visited: set[tuple[int, int]] = set()
    component_index = 0

    for cell in cell_points:
        if cell in visited:
            continue

        queue: deque[tuple[int, int]] = deque([cell])
        visited.add(cell)
        component_cells: list[tuple[int, int]] = []

        while queue:
            current = queue.popleft()
            component_cells.append(current)
            for neighbor in _neighbors(current):
                if neighbor in visited or neighbor not in cell_points:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        if len(component_cells) < min_cluster_cells:
            continue

        component_points: list[Point3D] = []
        for component_cell in component_cells:
            component_points.extend(cell_points[component_cell])

        if len(component_points) < min_cluster_points:
            continue

        bbox_min, bbox_max = bbox(component_points)
        width_x = bbox_max[0] - bbox_min[0]
        width_y = bbox_max[1] - bbox_min[1]
        height = bbox_max[2] - bbox_min[2]

        if height < min_cluster_height_m or height > max_cluster_height_m:
            continue
        if width_x > max_cluster_width_m or width_y > max_cluster_width_m:
            continue

        component_index += 1
        detections.append(
            Detection(
                detection_id=f"det-{component_index:04d}",
                centroid=centroid(component_points),
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                point_count=len(component_points),
                cell_count=len(component_cells),
            )
        )

    return detections


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = cell
    return [
        (x - 1, y - 1),
        (x - 1, y),
        (x - 1, y + 1),
        (x, y - 1),
        (x, y + 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
    ]

