from __future__ import annotations

import unittest

from depthyn.perception.clustering import cluster_points


def _cluster(cx: float, cy: float, *, count: int = 25) -> list[tuple[float, float, float]]:
    points = []
    for index in range(count):
        dx = (index % 5) * 0.15
        dy = (index // 5) * 0.15
        dz = 0.2 + ((index % 3) * 0.25)
        points.append((cx + dx, cy + dy, dz))
    return points


class ClusteringTests(unittest.TestCase):
    def test_finds_two_object_clusters(self) -> None:
        points = _cluster(0.0, 0.0) + _cluster(5.0, 4.5)
        detections = cluster_points(
            points,
            cell_size_m=0.5,
            min_cluster_points=10,
            min_cluster_cells=2,
            min_cluster_height_m=0.2,
            max_cluster_height_m=3.0,
            max_cluster_width_m=4.0,
        )
        self.assertEqual(len(detections), 2)

    def test_rejects_overwide_clusters(self) -> None:
        wall = [(float(x), 0.0, 1.0) for x in range(20)]
        detections = cluster_points(
            wall,
            cell_size_m=1.0,
            min_cluster_points=3,
            min_cluster_cells=2,
            min_cluster_height_m=0.2,
            max_cluster_height_m=3.0,
            max_cluster_width_m=5.0,
        )
        self.assertEqual(detections, [])


if __name__ == "__main__":
    unittest.main()

