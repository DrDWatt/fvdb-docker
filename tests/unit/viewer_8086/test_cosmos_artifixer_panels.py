"""/cosmos/* and /artifixer/* (Nov 16 2026): :8086 proxies to the Cosmos gateway (:8093) and the ArtiFixer controller (:8092).

Both upstreams are in-memory test doubles that record every call; nothing leaves the test process.
"""
import json

import httpx
import pytest


class Upstream:
    def __init__(self):
        self.calls = []
        self.jobs = [{"_id": "j1", "state": "running", "ply": "Car_model.ply", "step": "steps 1-4", "endpoint_name": "Brev"}]

    def __call__(self, method, url, **kw):
        body = kw.get("json")
        self.calls.append((method, url, body, kw.get("params")))
        path = httpx.URL(url).path
        routes = {
            "/api/activity": {"lifecycle": {"state": "stopped", "brev": {"logged_in": False, "detail": "run brev login"}},
                              "calls": [{"prompt": "How wide is the doorway?", "response": "0.9 m", "state": "ok"}]},
            "/api/lifecycle/check-brev": {"logged_in": True, "detail": "ok"},
            "/api/plys": [{"name": "Car_model.ply", "source": "Car", "derived": False},
                          {"name": "Car-orbit_artifixer3d_ppisp_sharp.ply", "source": None, "derived": True}],
            "/api/brev": {"logged_in": True, "instances": []},
            "/api/endpoints": [{"_id": "e1", "name": "Brev", "kind": "brev"}],
            "/api/jobs": self.jobs if method == "GET" else {"_id": "j2", "state": "queued", "scene": "Car-orbit", **(body or {})},
            "/api/jobs/j1/log": {"lines": ["[stage] ok"]},
            "/api/jobs/j1/cancel": {"_id": "j1", "state": "cancelled"},
        }
        if path == "/api/jobs" and method == "POST" and body.get("ply") == "nope.ply":
            return httpx.Response(422, json={"detail": "nope.ply is not in the models folder"})
        return httpx.Response(200, json=routes[path])


@pytest.fixture
def upstream(ss, monkeypatch):
    from features import artifixer_jobs, cosmos_activity
    fake = Upstream()
    monkeypatch.setattr(cosmos_activity.httpx, "request", fake)
    monkeypatch.setattr(artifixer_jobs.httpx, "request", fake)
    return fake


def test_cosmos_activity_shows_brev_login_nim_state_and_prompts(client, upstream):
    a = client.get("/cosmos/activity?limit=5").json()
    assert a["lifecycle"]["brev"]["logged_in"] is False and a["calls"][0]["prompt"] == "How wide is the doorway?"
    assert a["swagger"].endswith(":8093/docs") and upstream.calls[0][3] == {"limit": 5}
    assert client.post("/cosmos/check-brev").json()["logged_in"] is True


def test_cosmos_gateway_down_is_a_clear_502(client, ss, monkeypatch):
    from features import cosmos_activity

    def down(*a, **kw):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(cosmos_activity.httpx, "request", down)
    r = client.get("/cosmos/activity")
    assert r.status_code == 502 and "Cosmos gateway unreachable" in r.json()["detail"]


def test_artifixer_status_says_whether_the_loaded_model_can_run(client, upstream):
    st = client.get("/artifixer/status?model=Car_model.ply").json()
    assert st["model"]["source"] == "Car" and st["brev"]["logged_in"] and st["endpoints"][0]["_id"] == "e1"
    assert st["jobs"][0]["_id"] == "j1" and st["links"]["swagger"].endswith(":8092/docs")
    assert client.get("/artifixer/status?model=unknown.ply").json()["model"] is None


def test_artifixer_jobs_are_submitted_followed_and_cancelled_through_the_controller(client, upstream):
    job = client.post("/artifixer/jobs", json={"kind": "artifixer3d_plus", "ply": "Car_model.ply", "endpoint_id": "e1",
                                               "evil": "dropped"}).json()
    assert job["_id"] == "j2" and "evil" not in upstream.calls[-1][2]                     # only known fields forwarded
    bad = client.post("/artifixer/jobs", json={"ply": "nope.ply"})
    assert bad.status_code == 422 and "not in the models folder" in bad.json()["detail"]   # controller's own validation
    assert client.get("/artifixer/jobs/j1/log?tail=10").json()["lines"] == ["[stage] ok"]
    assert client.post("/artifixer/jobs/j1/cancel").json()["state"] == "cancelled"


def test_page_has_both_panels(client):
    html = client.get("/").text
    for needle in ('id="artifixer-section"', "/web/cosmos_activity.js", "/web/artifixer.js", "CosmosActivity.render", "ArtiFixer.render"):
        assert needle in html, needle
