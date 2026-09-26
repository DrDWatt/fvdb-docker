# Agent notes – fvdb-docker

## Splat viewers
| Port | Container | Code | Compose file |
|---|---|---|---|
| 8085 | `fvdb-viewer` | `fvdb-viewer/image_viewer_service.py` (server-rendered frames, fVDB) | `docker-compose.master.yml` |
| 8086 | `supersplat-viewer` | `supersplat-viewer/supersplat_service.py` + `features/` (routers) + `web/` (JS/CSS) | `docker-compose.workflow.yml` |

- Code shared by both viewers lives in `shared/viewer_common/` (mounted at `/app/viewer_common`):
  `camera_path.py` (smoothed constant-speed flythrough path, `MIN_VIDEO_SECONDS = 30`),
  `video.py` (H.264 MP4 writer), `rag.py` (document text extraction + Ollama chat).
- Viewer code is bind-mounted: after editing, `docker restart <container>`; after changing
  mounts/env, `docker compose -f <file> up -d --no-deps <service>`.
- MP4 export needs `imageio-ffmpeg` (static ffmpeg with libx264; the OpenCV FFmpeg builds lack it).
  :8085 gets it from the host conda env (`/home/dwatkins3/miniforge3/envs/fvdb`, mounted as `/host-fvdb`);
  :8086 from its image (`supersplat-viewer/requirements.txt`).
- :8086 relies on the SuperSplat viewer API (`window.scrubTo`, `window.captureFrame`, settings.json
  `animTracks`) and on `patch_viewer.py` (`window.getCameraMatrices`) applied at image build.
  The viewer git commit is pinned in `supersplat-viewer/Dockerfile`.
- The fVDB PLYs store trained cameras as extra PLY elements (`camera_to_world_matrices`,
  `projection_matrices`, `image_sizes`, `median_depths`); `supersplat-viewer/features/flythrough.py` reads them with numpy.
  The SuperSplat viewer displays splats rotated (0, 0, 180): viewer = (-x, -y, z).

## Tests (`tests/`, `docker-compose.test.yml`)
| Suite | Where | Command |
|---|---|---|
| Unit: Python (`tests/unit`, GPU parts faked) + JS (`tests/js`, node:test) | runner container | `docker compose -f docker-compose.test.yml run --rm viewer-tests` |
| Integration (`tests/integration`, real GPU services) | isolated test stack | `docker compose -f docker-compose.test.yml --profile integration up -d` then `docker compose -f docker-compose.test.yml run --rm viewer-tests pytest -m "integration and not slow"` |
| TRELLIS.2 mesh (`-m slow`, ~5 min, needs ~40 GB free) | test stack + shared TRELLIS | `... run --rm viewer-tests pytest -m slow` |
| Browser E2E (`tests/browser`, host Chromium + GPU) | host | `node --test --test-concurrency=1 tests/browser/` |
- The test stack (`fvdb-test` project: ports 18085/18086, own `./test-models`, own rendering service) never
  touches the dev viewers; tests seed `APL-copter-ultra_model.ply` into `./test-models`.
  Tear down with `docker compose -f docker-compose.test.yml --profile integration down`.
- Each capability is covered for both viewers: navigation, segmentation, 3D extraction/reconstruction,
  typed + uploaded object/extraction metadata, RAG document upload, RAG chat, PLY upload/download,
  flythrough, 30 s+ H.264 MP4 export.

## Verification
- Health: `curl localhost:8085/health`, `curl localhost:8086/health`
- Export (≥30 s, H.264 High yuv420p): `curl -X POST -o f.mp4 "localhost:8085/flythrough/export?duration=30&fps=30"`,
  then inspect with the bundled ffmpeg: `$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())") -i f.mp4`
- RAG: `curl localhost:808{5,6}/rag/status`; Ollama model `nemotron-mini`.
