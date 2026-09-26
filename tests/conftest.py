"""Test configuration shared by all suites.

Makes the service code importable exactly as it is laid out in the containers:
  shared/viewer_common  -> /app/viewer_common   (both viewers)
  supersplat-viewer/    -> /app                 (:8086: supersplat_service, features)
  fvdb-viewer/          -> /app                 (:8085: image_viewer_service)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for path in (REPO / "tests", REPO / "shared", REPO / "supersplat-viewer", REPO / "fvdb-viewer"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
