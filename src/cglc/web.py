"""Local server for the demo site: static files + the same API the Vercel
functions expose.

    python -m cglc.web              # http://127.0.0.1:8000
    python -m cglc.web --port 9000 --time-limit 180

Binds to localhost only. Keys arrive in the request body, are used for that
request, and are never logged.
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import service

ROOT = Path(__file__).resolve().parents[2] / "demo"
_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript",
          ".css": "text/css", ".png": "image/png", ".gif": "image/gif",
          ".mp4": "video/mp4", ".svg": "image/svg+xml", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    server_version = "cglc-web"

    def log_message(self, fmt, *args):  # request lines only; bodies hold keys
        pass

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status_body) -> None:
        status, body = status_body
        self._send(status, json.dumps(body).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/health":
            return self._json(service.handle_health())
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (ROOT / rel).resolve()
        if ROOT not in f.parents and f != ROOT / "index.html" or not f.is_file():
            return self._send(404, b"not found", "text/plain")
        self._send(200, f.read_bytes(), _TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        if n > service.MAX_BODY_BYTES:
            return self._json((413, {"ok": False, "kind": "input", "error": "Request too large."}))
        raw = self.rfile.read(n)
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            return self._json(service.handle_run(raw))
        if path == "/api/models":
            return self._json(service.handle_models(raw))
        self._send(404, b"not found", "text/plain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--time-limit", type=float, default=180.0,
                    help="seconds a single run may take (hosted default is 50)")
    a = ap.parse_args()
    os.environ.setdefault("CGLC_TIME_LIMIT", str(a.time_limit))
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"CGLC demo on http://127.0.0.1:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
