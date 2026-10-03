"""The Vercel entrypoint (app.py) exercised with Flask's test client, plus site-copy sync."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("vercel_app", ROOT / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # loaded as a file, like Vercel's entrypoint
    return mod.app.test_client()


DOCS = [{"name": "a.txt", "text": "Employees may work remotely up to three days per week."}]


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json["ok"] and "groq" in r.json["providers"]


def test_run_offline_end_to_end(client):
    r = client.post("/api/run", json={"provider": "offline", "goal": "remote days per week",
                                      "documents": DOCS})
    assert r.status_code == 200 and r.json["decision"] == "ALLOW_FINALIZE"
    assert r.headers["Cache-Control"] == "no-store"


def test_validation_and_method_errors_are_json(client):
    r = client.post("/api/run", json={"provider": "groq", "goal": "", "documents": DOCS})
    assert r.status_code == 400 and r.json["kind"] == "input"
    assert client.get("/api/run").status_code == 405 and client.get("/api/run").json["ok"] is False
    assert client.get("/nope").status_code == 404


def test_models_needs_a_key_and_oversize_body_is_rejected(client):
    r = client.post("/api/models", json={"provider": "groq", "api_key": ""})
    assert r.status_code == 400 and "key" in r.json["error"].lower()
    big = client.post("/api/run", data=b"x" * 2_100_000, content_type="application/json")
    assert big.status_code == 413 and big.json["kind"] == "input"


def test_site_copies_are_in_sync():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sync_site.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
