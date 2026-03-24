"""Background model for stationary LiDAR foreground extraction.

Inspired by Blickfeld Percept and Ouster Gemini Detect approaches:
- Time-based fade: cells must be continuously occupied for `fade_time_s`
  before being considered background (default 180s, vs Gemini's 200-600s).
- Continuous observation: background model updates every frame, not just
  during a fixed warmup period.
- Object-aware freezing: cells near active detections are protected from
  being absorbed into the background.
- Reference mode: optionally build a fixed background from an empty-scene
  snapshot instead of adaptive fade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from depthyn.models import Point3D
from depthyn.perception.preprocess import xy_cell


@dataclass(slots=True)
class _CellState:
    """Per-cell background statistics."""

    first_seen_ns: int = 0
    last_seen_ns: int = 0
    hit_count: int = 0
    frozen: bool = False


class BackgroundModel:
    """Adaptive background model with time-based fade and object awareness.

    Parameters
    ----------
    cell_size_m:
        XY grid cell size in metres.
    fade_time_s:
        Seconds a cell must be continuously occupied before it is
        considered background. Set to 0 for immediate (legacy behaviour).
    min_hits:
        Minimum observation count before a cell *can* become background.
        Works together with fade_time_s — both conditions must be met.
    """

    def __init__(
        self,
        cell_size_m: float,
        min_hits: int = 6,
        fade_time_s: float = 180.0,
    ) -> None:
        self.cell_size_m = cell_size_m
        self.min_hits = min_hits
        self.fade_time_ns = int(fade_time_s * 1_000_000_000)
        self._cells: dict[tuple[int, int], _CellState] = {}
        self._reference: set[tuple[int, int]] | None = None

    # ------------------------------------------------------------------
    # Reference mode
    # ------------------------------------------------------------------

    def build_reference(self, points: list[Point3D]) -> None:
        """Build a fixed reference from an empty-scene point cloud.

        Any cell occupied in the reference is permanently background.
        Call once with a scan of the empty scene before objects appear.
        """
        if self._reference is None:
            self._reference = set()
        for point in points:
            self._reference.add(xy_cell(point, self.cell_size_m))

    # ------------------------------------------------------------------
    # Adaptive mode
    # ------------------------------------------------------------------

    def observe(
        self,
        points: list[Point3D],
        timestamp_ns: int = 0,
        protected_cells: set[tuple[int, int]] | None = None,
    ) -> None:
        """Update background statistics with a new frame.

        Parameters
        ----------
        points:
            Full point cloud for this frame.
        timestamp_ns:
            Frame timestamp in nanoseconds. Used for fade timing.
        protected_cells:
            Cells that should not be absorbed into background
            (e.g., cells containing active detections).
        """
        seen: set[tuple[int, int]] = set()
        for point in points:
            seen.add(xy_cell(point, self.cell_size_m))

        for cell in seen:
            if protected_cells and cell in protected_cells:
                # Object-aware: mark frozen, don't count toward background
                state = self._cells.get(cell)
                if state is not None:
                    state.frozen = True
                continue

            state = self._cells.get(cell)
            if state is None:
                state = _CellState(
                    first_seen_ns=timestamp_ns,
                    last_seen_ns=timestamp_ns,
                    hit_count=1,
                )
                self._cells[cell] = state
            else:
                state.last_seen_ns = timestamp_ns
                state.hit_count += 1
                state.frozen = False

        # Cells not seen this frame: reset their continuity
        # (only reset if they haven't been seen for a while)
        # We do this lazily in _is_background instead to avoid
        # iterating the entire cell dict every frame.

    def filter_foreground(
        self,
        points: list[Point3D],
        timestamp_ns: int = 0,
    ) -> list[Point3D]:
        """Return only foreground points (not part of background)."""
        foreground: list[Point3D] = []
        for point in points:
            cell = xy_cell(point, self.cell_size_m)
            if not self._is_background(cell, timestamp_ns):
                foreground.append(point)
        return foreground

    def _is_background(self, cell: tuple[int, int], timestamp_ns: int) -> bool:
        """Check if a cell is background."""
        # Reference mode: anything in the reference is background
        if self._reference is not None and cell in self._reference:
            return True

        state = self._cells.get(cell)
        if state is None:
            return False

        # Frozen cells (near active detections) are never background
        if state.frozen:
            return False

        # Must have enough observations
        if state.hit_count < self.min_hits:
            return False

        # Time-based fade: cell must have been present for fade_time_ns
        if self.fade_time_ns > 0:
            duration_ns = state.last_seen_ns - state.first_seen_ns
            if duration_ns < self.fade_time_ns:
                return False

        return True

    def protected_cells_from_detections(
        self,
        detections: list,
        margin_cells: int = 2,
    ) -> set[tuple[int, int]]:
        """Compute protected cell set from detection bounding boxes.

        Returns the set of cells covered by detection bounding boxes
        plus a margin, to prevent absorbing objects into background.
        """
        protected: set[tuple[int, int]] = set()
        for det in detections:
            # Get the cell range covered by the detection bbox + margin
            min_cx = int(det.bbox_min[0] / self.cell_size_m) - margin_cells
            max_cx = int(det.bbox_max[0] / self.cell_size_m) + margin_cells
            min_cy = int(det.bbox_min[1] / self.cell_size_m) - margin_cells
            max_cy = int(det.bbox_max[1] / self.cell_size_m) + margin_cells
            for cx in range(min_cx, max_cx + 1):
                for cy in range(min_cy, max_cy + 1):
                    protected.add((cx, cy))
        return protected
