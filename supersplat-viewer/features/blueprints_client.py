"""Connection to Splat Blueprints (~/splat-blueprints) shared by the :8086 features that use it
(measure.py: Cosmos 3 measurement jobs; orient_override.py: saved display orientations).

The bearer token is the compose secret `blueprints_token` (~/splat-blueprints/secrets/viewer.token); it is
read per request, never logged, and never sent to the browser.
"""
import os
from pathlib import Path

import httpx
from fastapi import HTTPException

BLUEPRINTS_URL = os.environ.get("BLUEPRINTS_URL", "http://blueprints-api:8000").rstrip("/")
BLUEPRINTS_TOKEN_FILE = os.environ.get("BLUEPRINTS_TOKEN_FILE", "/run/secrets/blueprints_token")
BLUEPRINTS_PROJECT = os.environ.get("BLUEPRINTS_PROJECT", "peacock")


def token() -> str:
    try:
        tok = Path(BLUEPRINTS_TOKEN_FILE).read_text().strip()
    except OSError:
        tok = ""
    if not tok:
        raise HTTPException(status_code=503, detail="Blueprints service not configured (no Blueprints token)")
    return tok


def client(timeout: float) -> httpx.Client:
    return httpx.Client(base_url=BLUEPRINTS_URL, timeout=httpx.Timeout(timeout, connect=10),
                        headers={"Authorization": f"Bearer {token()}"})
