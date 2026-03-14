from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ReplayConfig:
    input_dir: Path
    output_json: Path
    mode: str = "mobile"
    max_frames: int | None = None
    preview_point_limit: int = 1200
    voxel_size_m: float = 0.3
    min_range_m: float = 1.0
    max_range_m: float = 60.0
    z_min_m: float = -2.5
    z_max_m: float = 4.5
    cluster_cell_size_m: float = 0.75
    min_cluster_points: int = 12
    min_cluster_cells: int = 3
    min_cluster_height_m: float = 0.4
    max_cluster_height_m: float = 5.0
    max_cluster_width_m: float = 10.0
    background_warmup_frames: int = 10
    background_min_hits: int = 6
    track_max_distance_m: float = 3.0
    track_max_missed_frames: int = 6

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["input_dir"] = str(self.input_dir)
        payload["output_json"] = str(self.output_json)
        return payload
