"""Shared bits for the Vercel Python functions."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cglc import service  # noqa: E402


def respond(handler, status_body):
    status, body = status_body
    data = json.dumps(body).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def read_body(handler):
    n = int(handler.headers.get("Content-Length") or 0)
    return handler.rfile.read(min(n, service.MAX_BODY_BYTES + 1))
