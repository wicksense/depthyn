from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from depthyn.config import DetectorConfig, ReplayConfig
from depthyn.detectors.base import DetectorError
from depthyn.pipeline import run_replay, write_summary


def run_detector_comparison(
    base_config: ReplayConfig,
    detectors: list[DetectorConfig],
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_runs: list[dict[str, object]] = []
    aggregate_metrics: dict[str, dict[str, object]] = {}

    for detector in detectors:
        detector_slug = detector.resolved_label().replace(" ", "-").lower()
        summary_path = output_dir / f"{detector_slug}-summary.json"
        run_config = replace(
            base_config,
            output_json=summary_path,
            detector=detector,
        )
        try:
            summary = run_replay(run_config)
            write_summary(summary, summary_path)
            detector_runs.append(
                {
                    "detector": detector.to_dict(),
                    "status": "ok",
                    "summary_path": str(summary_path),
                    "metrics": summary["metrics"],
                }
            )
            aggregate_metrics[detector.resolved_label()] = summary["metrics"]
        except DetectorError as exc:
            detector_runs.append(
                {
                    "detector": detector.to_dict(),
                    "status": "error",
                    "error": str(exc),
                    "summary_path": None,
                }
            )

    comparison = {
        "project": "Depthyn",
        "comparison_type": "detector_replay",
        "input_dir": str(base_config.input_dir),
        "mode": base_config.mode,
        "detector_runs": detector_runs,
        "aggregate_metrics": aggregate_metrics,
    }
    write_summary(comparison, output_dir / "comparison.json")
    return comparison

