"""Cosmos 3 activity for the :8086 Measure panel (Nov 16 2026).

Why: measurements timed out with no clue why (Nov 12: the Brev CLI login had expired, so the on-demand NIM could not
start). The panel now shows the Brev login, the NIM state (stopped / starting: loading the model / ready) and the recent
Cosmos prompts and responses. Data comes from the Cosmos gateway (~/Cosms3-mcp gateway/, :8093 on cosmos-shared),
which reads the cosmos3-mcp tools and cosmos-brev.
"""
import os

import httpx
from fastapi import APIRouter, HTTPException

COSMOS_GATEWAY_URL = os.environ.get("COSMOS_GATEWAY_URL", "http://cosmos-gateway:8093").rstrip("/")


def create_router() -> APIRouter:
    router = APIRouter(prefix="/cosmos", tags=["cosmos"])

    def _call(method: str, path: str, **kw):
        try:
            r = httpx.request(method, f"{COSMOS_GATEWAY_URL}{path}", timeout=90, **kw)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Cosmos gateway unreachable ({COSMOS_GATEWAY_URL}): {exc}") from exc
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Cosmos gateway {path}: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    @router.get("/activity")
    def activity(limit: int = 10):
        """Brev login + NIM state + recent calls (prompt, response, seconds, error)."""
        return {**_call("GET", "/api/activity", params={"limit": max(1, min(limit, 50))}),
                "swagger": os.environ.get("COSMOS_GATEWAY_PUBLIC_URL", "http://100.105.59.33:8093") + "/docs"}

    @router.post("/check-brev")
    def check_brev():
        """Re-check the Brev CLI login now (e.g. right after running `brev login`)."""
        return _call("POST", "/api/lifecycle/check-brev")

    return router
