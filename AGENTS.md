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

## Verification
- Health: `curl localhost:8085/health`, `curl localhost:8086/health`
- Export (≥30 s, H.264 High yuv420p): `curl -X POST -o f.mp4 "localhost:8085/flythrough/export?duration=30&fps=30"`,
  then inspect with the bundled ffmpeg: `$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())") -i f.mp4`
- RAG: `curl localhost:808{5,6}/rag/status`; Ollama model `nemotron-mini`.
