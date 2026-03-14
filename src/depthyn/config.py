from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class DetectorConfig:
    kind: str = "baseline"
    label: str | None = None
    backend_python: str | None = None
    backend_repo: Path | None = None
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    score_threshold: float = 0.25
    device: str = "cuda:0"

    def resolved_label(self) -> str:
        return self.label or self.kind

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.backend_repo is not None:
            payload["backend_repo"] = str(self.backend_repo)
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        if self.checkpoint_path is not None:
            payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@dataclass(slots=True)
class ReplayConfig:
    input_dir: Path
    output_json: Path
    mode: str = "mobile"
    max_frames: int | None = None
    preview_point_limit: int = 1200
    detector: DetectorConfig = field(default_factory=DetectorConfig)
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
        payload["detector"] = self.detector.to_dict()
        return payload
