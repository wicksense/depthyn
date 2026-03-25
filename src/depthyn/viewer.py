from __future__ import annotations

import http.server
import json
import socketserver
import urllib.parse
from functools import partial
from pathlib import Path


def rule_storage_path(summary_path: Path, frame: str) -> Path:
    normalized_frame = frame.strip().lower()
    if normalized_frame not in {"sensor", "world"}:
        raise ValueError(f"Unsupported rule frame: {frame}")
    return summary_path.with_name(
        f"{summary_path.stem}.rules.{normalized_frame}{summary_path.suffix}"
    )


def load_saved_rules(summary_path: Path, frame: str) -> dict[str, object] | None:
    rules_path = rule_storage_path(summary_path, frame)
    if not rules_path.exists():
        return None
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    return validate_rules_payload(payload, frame=frame)


def save_rules(summary_path: Path, frame: str, payload: dict[str, object]) -> Path:
    validated = validate_rules_payload(payload, frame=frame)
    rules_path = rule_storage_path(summary_path, frame)
    rules_path.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return rules_path


def validate_rules_payload(payload: dict[str, object], *, frame: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError("Rules payload must be a JSON object")

    raw_zones = payload.get("zones", [])
    raw_tripwires = payload.get("tripwires", [])
    if not isinstance(raw_zones, list):
        raise ValueError("zones must be a list")
    if not isinstance(raw_tripwires, list):
        raise ValueError("tripwires must be a list")

    return {
        "reference_frame": frame,
        "zones": raw_zones,
        "tripwires": raw_tripwires,
    }


class DepthynViewerRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        directory: str,
        project_root: Path,
        summary_path: Path,
        **kwargs,
    ) -> None:
        self._project_root = project_root
        self._summary_path = summary_path
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/session":
            self._send_json(
                {
                    "summary_path": str(self._summary_path),
                    "summary_name": self._summary_path.name,
                    "available_rule_frames": ["sensor", "world"],
                    "rule_paths": {
                        frame: str(rule_storage_path(self._summary_path, frame))
                        for frame in ("sensor", "world")
                    },
                }
            )
            return
        if parsed.path == "/api/rules":
            frame = urllib.parse.parse_qs(parsed.query).get("frame", ["sensor"])[0]
            try:
                payload = load_saved_rules(self._summary_path, frame)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            if payload is None:
                self._send_json({"error": "rules not found"}, status=404)
                return
            self._send_json(payload)
            return
        super().do_GET()

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/rules":
            self.send_error(404)
            return

        frame = urllib.parse.parse_qs(parsed.query).get("frame", ["sensor"])[0]
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
            saved_path = save_rules(self._summary_path, frame, payload)
        except json.JSONDecodeError:
            self._send_json({"error": "request body is not valid JSON"}, status=400)
            return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self._send_json(
            {
                "status": "saved",
                "frame": frame,
                "path": str(saved_path),
            }
        )

    def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_viewer(summary_path: Path, host: str, port: int) -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary_path = summary_path.resolve()

    try:
        summary_rel = summary_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Summary path must live under the project root: {project_root}"
        ) from exc

    handler = partial(
        DepthynViewerRequestHandler,
        directory=str(project_root),
        project_root=project_root,
        summary_path=summary_path,
    )

    with socketserver.ThreadingTCPServer((host, port), handler) as server:
        summary_param = urllib.parse.quote(str(summary_rel))
        viewer_3d_url = (
            f"http://{host}:{port}/viewer/viewer3d.html?data=/{summary_param}"
        )
        viewer_2d_url = (
            f"http://{host}:{port}/viewer/index.html?data=/{summary_param}"
        )
        print(f"Serving Depthyn viewer from {project_root}")
        print(f"3D Viewer: {viewer_3d_url}")
        print(f"2D Viewer: {viewer_2d_url}")
        print(f"API Session: http://{host}:{port}/api/session")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping viewer server.")
