"""Load the Vercel functions exactly as files (no package install) and call them over HTTP."""
import importlib.util
import json
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

API = Path(__file__).resolve().parents[1] / "api"


def _serve(name):
    spec = importlib.util.spec_from_file_location(f"vercel_{name}", API / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    srv = HTTPServer(("127.0.0.1", 0), mod.handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _call(srv, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_port}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_function():
    srv = _serve("health")
    try:
        status, body = _call(srv, "/api/health")
        assert status == 200 and body["ok"] and "groq" in body["providers"]
    finally:
        srv.shutdown()


def test_run_function_offline_and_validation_errors():
    srv = _serve("run")
    try:
        docs = [{"name": "a.txt", "text": "Employees may work remotely up to three days per week."}]
        status, body = _call(srv, "/api/run", {"provider": "offline", "goal": "remote days per week",
                                               "documents": docs})
        assert status == 200 and body["ok"] and body["decision"] == "ALLOW_FINALIZE"
        status, body = _call(srv, "/api/run", {"provider": "groq", "goal": "", "documents": docs})
        assert status == 400 and body["kind"] == "input"
    finally:
        srv.shutdown()


def test_models_function_requires_a_key_for_groq_without_calling_out():
    srv = _serve("models")
    try:
        status, body = _call(srv, "/api/models", {"provider": "groq", "api_key": ""})
        assert status == 400 and "key" in body["error"].lower()
    finally:
        srv.shutdown()
