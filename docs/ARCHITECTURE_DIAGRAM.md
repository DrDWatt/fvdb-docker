# fVDB Multi-Container Architecture

## System Overview Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    fVDB GAUSSIAN SPLATTING PLATFORM                           │
│                                      Docker Multi-Container System                            │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

                                         ┌─────────────────────┐
                                         │    USER BROWSER     │
                                         │   (Web Interface)   │
                                         └──────────┬──────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    │                               │                               │
                    ▼                               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│      📺 FVDB-VIEWER           │ │    🎓 TRAINING-SERVICE        │ │    🎨 RENDERING-SERVICE       │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Container: fvdb-viewer       │ │  Container: fvdb-training-gpu │ │  Container: fvdb-rendering    │
│  Port: 8085                   │ │  Port: 8000                   │ │  Port: 8001                   │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  UI: http://localhost:8085    │ │  UI: http://localhost:8000    │ │  UI: http://localhost:8001    │
│  API: http://localhost:8085/  │ │  API: http://localhost:8000/  │ │  API: http://localhost:8001/  │
│       docs                    │ │       api                     │ │       api                     │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Features:                    │ │  Features:                    │ │  Features:                    │
│  • Interactive 3D viewer      │ │  • Video → Gaussian Splat     │ │  • PLY file management        │
│  • SAM-2 segmentation         │ │  • Photos → Gaussian Splat    │ │  • Model downloads            │
│  • Per-object RAG summaries   │ │  • COLMAP dataset training    │ │  • Rendering controls         │
│  • GARField 3D extraction     │ │  • Job monitoring             │ │  • Model listing              │
│  GPU: Required (CUDA)         │ │  GPU: Required (CUDA)         │ │  GPU: Optional                │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘
                │                               │                               │
                └───────────────────────────────┼───────────────────────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                    │   SHARED MODELS VOL   │
                                    │     ./models          │
                                    └───────────────────────┘

┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│      📸 COLMAP-SERVICE        │ │    🔬 SAM2-SERVICE            │ │    🎬 USD-PIPELINE            │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Container: colmap-processor  │ │  Container: sam2-segmentation │ │  Container: usd-pipeline      │
│  Port: 8003                   │ │  Port: 8004                   │ │  Port: 8002                   │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  UI: http://localhost:8003    │ │  UI: http://localhost:8004    │ │  UI: http://localhost:8002    │
│  API: http://localhost:8003/  │ │  API: http://localhost:8004/  │ │  API: http://localhost:8002/  │
│       docs                    │ │       docs                    │ │       api                     │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Features:                    │ │  Features:                    │ │  Features:                    │
│  • Video frame extraction     │ │  • Object segmentation        │ │  • PLY → USD conversion       │
│  • COLMAP reconstruction      │ │  • Auto-segmentation          │ │  • High-quality rendering     │
│  • Camera pose estimation     │ │  • Click-to-segment           │ │  • Omniverse integration      │
│  • Point cloud generation     │ │  • Mask export                │ │  • Scene export               │
│  GPU: Required (CUDA)         │ │  GPU: Required (CUDA)         │ │  GPU: Optional                │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘

┌───────────────────────────────┐ ┌───────────────────────────────┐ ┌───────────────────────────────┐
│    🌐 STREAMING-SERVER        │ │    🎯 GARFIELD-SERVICE        │ │   📱 SPECTACULARAI-SERVICE    │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Container: streaming-server  │ │  Container: garfield-extract  │ │  Container: spectacularai     │
│  Port: 8080                   │ │  Port: 8006                   │ │  Port: 8007                   │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  UI: http://localhost:8080    │ │  UI: http://localhost:8006    │ │  UI: http://localhost:8007    │
│  Test: http://localhost:8080/ │ │  API: http://localhost:8006/  │ │  API: http://localhost:8007/  │
│        test                   │ │       docs                    │ │       docs                    │
│ ─────────────────────────────│ │ ─────────────────────────────│ │ ─────────────────────────────│
│  Features:                    │ │  Features:                    │ │  Features:                    │
│  • WebRTC streaming           │ │  • 3D object extraction       │ │  • SLAM-based processing      │
│  • Real-time 3D view          │ │  • Hierarchical grouping      │ │  • Mobile app integration     │
│  • Interactive controls       │ │  • Gaussian filtering         │ │  • Real-time capture          │
│  • Model metadata             │ │  • Asset export               │ │  • PLY generation             │
│  GPU: Required (Vulkan)       │ │  GPU: Required (CUDA)         │ │  GPU: Required (CUDA)         │
└───────────────────────────────┘ └───────────────────────────────┘ └───────────────────────────────┘


## Service Communication Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

   ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
   │  VIDEO   │ ───► │  COLMAP  │ ───► │ TRAINING │ ───► │ RENDERING│ ───► │  VIEWER  │
   │  UPLOAD  │      │  :8003   │      │  :8000   │      │  :8001   │      │  :8085   │
   └──────────┘      └──────────┘      └──────────┘      └──────────┘      └──────────┘
                          │                  │                  │                │
                          ▼                  ▼                  ▼                ▼
                    ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐
                    │  Camera  │      │   PLY    │      │  Model   │      │  SAM-2   │
                    │  Poses   │      │  Models  │      │  Files   │      │  Segment │
                    └──────────┘      └──────────┘      └──────────┘      └──────────┘
                                                              │
                                           ┌──────────────────┼──────────────────┐
                                           ▼                  ▼                  ▼
                                     ┌──────────┐      ┌──────────┐      ┌──────────┐
                                     │   USD    │      │ GARFIELD │      │ STREAMING│
                                     │  :8002   │      │  :8006   │      │  :8080   │
                                     └──────────┘      └──────────┘      └──────────┘
```


## Quick Reference: All Swagger API URLs

| Service | Port | Web UI | Swagger API | Health Check |
|---------|------|--------|-------------|--------------|
| **Viewer** | 8085 | http://localhost:8085 | http://localhost:8085/docs | http://localhost:8085/health |
| **Training** | 8000 | http://localhost:8000 | http://localhost:8000/api | http://localhost:8000/health |
| **Rendering** | 8001 | http://localhost:8001 | http://localhost:8001/api | http://localhost:8001/health |
| **USD Pipeline** | 8002 | http://localhost:8002 | http://localhost:8002/api | http://localhost:8002/health |
| **COLMAP** | 8003 | http://localhost:8003 | http://localhost:8003/docs | http://localhost:8003/health |
| **SAM-2** | 8004 | http://localhost:8004 | http://localhost:8004/docs | http://localhost:8004/health |
| **GARField** | 8006 | http://localhost:8006 | http://localhost:8006/docs | http://localhost:8006/health |
| **SpectacularAI** | 8007 | http://localhost:8007 | http://localhost:8007/docs | http://localhost:8007/health |
| **Streaming** | 8080 | http://localhost:8080/test | N/A | http://localhost:8080/health |


## Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCKER NETWORK: workflow-net                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   viewer    │  │  training   │  │  rendering  │  │    colmap   │        │
│  │   :8085     │  │   :8000     │  │   :8001     │  │    :8003    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                          │
│  ┌─────────────┐  ┌─────────────┐ │ ┌─────────────┐  ┌─────────────┐        │
│  │    sam2     │  │     usd     │ │ │   garfield  │  │ spectacular │        │
│  │   :8004     │  │   :8002     │ │ │    :8006    │  │    :8007    │        │
│  └─────────────┘  └─────────────┘ │ └─────────────┘  └─────────────┘        │
│                                   │                                          │
└───────────────────────────────────┼──────────────────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        │   SHARED VOLUMES      │
                        ├───────────────────────┤
                        │  ./models             │
                        │  ./colmap-data        │
                        │  ./sam2-data          │
                        │  ./garfield-data      │
                        │  ./usd-outputs        │
                        │  ./uploads            │
                        │  ./outputs            │
                        └───────────────────────┘
```


## GPU Requirements

| Service | GPU Required | VRAM Recommended | Notes |
|---------|--------------|------------------|-------|
| Training | Yes | 8GB+ | CUDA compute for Gaussian training |
| Viewer | Yes | 4GB+ | CUDA for fVDB rendering |
| SAM-2 | Yes | 8GB+ | Segment Anything Model 2 |
| GARField | Yes | 8GB+ | 3D extraction algorithms |
| COLMAP | Yes | 4GB+ | GPU-accelerated reconstruction |
| SpectacularAI | Yes | 4GB+ | SLAM processing |
| Streaming | Yes | 4GB+ | Vulkan for WebRTC |
| Rendering | Optional | 2GB+ | CPU fallback available |
| USD Pipeline | Optional | 2GB+ | CPU fallback available |


## Quick Start

```bash
# Start all services with one command
docker compose -f docker-compose.master.yml up -d

# Check all services are healthy
docker compose -f docker-compose.master.yml ps

# View logs
docker compose -f docker-compose.master.yml logs -f

# Stop all services
docker compose -f docker-compose.master.yml down
```
