from __future__ import annotations

import http.server
import socketserver
import urllib.parse
from pathlib import Path


def serve_viewer(summary_path: Path, host: str, port: int) -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary_path = summary_path.resolve()

    try:
        summary_rel = summary_path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"Summary path must live under the project root: {project_root}"
        ) from exc

    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args, directory=str(project_root), **kwargs
    )

    with socketserver.ThreadingTCPServer((host, port), handler) as server:
        summary_param = urllib.parse.quote(str(summary_rel))
        viewer_url = (
            f"http://{host}:{port}/viewer/index.html?data=/{summary_param}"
        )
        print(f"Serving Depthyn viewer from {project_root}")
        print(f"Open: {viewer_url}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping viewer server.")

