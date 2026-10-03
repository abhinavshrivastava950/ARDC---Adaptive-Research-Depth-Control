"""Vercel entrypoint: the CGLC web API as one Flask app.

Static files (the demo site) are served by Vercel's CDN from ``public/``;
this app only handles ``/api/*``. Locally, ``python -m cglc.web`` serves
both without Flask.

Keys arrive in the request body, are used for that request only, and are
never logged (Flask/werkzeug log request lines, not bodies).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from flask import Flask, Response, request  # noqa: E402

from cglc import service  # noqa: E402

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = service.MAX_BODY_BYTES


def _json(status_body) -> Response:
    status, body = status_body
    return Response(json.dumps(body), status=status, mimetype="application/json",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/run")
def run():
    return _json(service.handle_run(request.get_data(cache=False)))


@app.post("/api/models")
def models():
    return _json(service.handle_models(request.get_data(cache=False)))


@app.get("/api/health")
def health():
    return _json(service.handle_health())


@app.errorhandler(413)
def too_large(_e):
    return _json((413, {"ok": False, "kind": "input", "error": "Request too large."}))


@app.errorhandler(404)
def not_found(_e):
    return _json((404, {"ok": False, "kind": "input", "error": "Not found."}))


@app.errorhandler(405)
def bad_method(_e):
    return _json((405, {"ok": False, "kind": "input", "error": "Method not allowed."}))
