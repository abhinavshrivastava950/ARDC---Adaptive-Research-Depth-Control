"""Tiny helpers shared by the Vercel Python functions in api/."""
from __future__ import annotations

import json

from . import service


def respond(handler, status_body) -> None:
    status, body = status_body
    data = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def read_body(handler) -> bytes:
    n = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(min(n, service.MAX_BODY_BYTES + 1))
