from __future__ import annotations

from depthyn.models import Point3D
from depthyn.perception.preprocess import xy_cell


class BackgroundModel:
    def __init__(self, cell_size_m: float, min_hits: int) -> None:
        self.cell_size_m = cell_size_m
        self.min_hits = min_hits
        self._hits: dict[tuple[int, int], int] = {}

    def observe(self, points: list[Point3D]) -> None:
        seen = {xy_cell(point, self.cell_size_m) for point in points}
        for cell in seen:
            self._hits[cell] = self._hits.get(cell, 0) + 1

    def filter_foreground(self, points: list[Point3D]) -> list[Point3D]:
        foreground: list[Point3D] = []
        for point in points:
            cell = xy_cell(point, self.cell_size_m)
            if self._hits.get(cell, 0) < self.min_hits:
                foreground.append(point)
        return foreground

