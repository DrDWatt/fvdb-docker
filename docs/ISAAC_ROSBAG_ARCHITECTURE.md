# ISAAC ROSBAG Workflow Architecture

## Overview

This workflow enables conversion of ZED X stereo camera SVO files to ROSBAG format for use with NVIDIA ISAAC Sim and ISAAC Lab, with integrated visualization using SAM-2 segmentation and GARField 3D extraction.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ISAAC ROSBAG WORKFLOW                                     │
│                                                                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐            │
│  │   ZED X Camera  │     │  SVO Converter  │     │    ROSBAG       │            │
│  │   (.svo files)  │────▶│    :8009        │────▶│    Storage      │            │
│  └─────────────────┘     │  /api           │     └────────┬────────┘            │
│                          └─────────────────┘              │                      │
│                                                           │                      │
│                    ┌──────────────────────────────────────┼──────────────────┐   │
│                    │                                      │                  │   │
│                    ▼                                      ▼                  ▼   │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐   ┌───────────┐  │
│  │      ISAAC Sim          │   │       ISAAC Lab             │   │  ISAAC    │  │
│  │        :8010            │   │         :8011               │   │  Viewer   │  │
│  │        /api             │   │         /api                │   │   :8012   │  │
│  │                         │   │                             │   │   /api    │  │
│  │  • Scene Loading        │   │  • RL Training              │   │           │  │
│  │  • Robot Spawning       │   │  • Policy Evaluation        │   │           │  │
│  │  • Physics Simulation   │   │  • Checkpointing            │   │           │  │
│  │  • Sensor Simulation    │   │  • Metrics Logging          │   │           │  │
│  └─────────────────────────┘   └─────────────────────────────┘   └─────┬─────┘  │
│                                                                        │        │
│                    ┌───────────────────────────────────────────────────┘        │
│                    │                                                             │
│                    ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      AI Processing Pipeline                              │    │
│  │                                                                          │    │
│  │   ┌─────────────────────┐         ┌─────────────────────┐               │    │
│  │   │      SAM-2          │         │      GARField       │               │    │
│  │   │      :8004          │────────▶│      :8006          │               │    │
│  │   │      /docs          │         │      /docs          │               │    │
│  │   │                     │         │                     │               │    │
│  │   │  • Auto Segment     │         │  • 3D Extraction    │               │    │
│  │   │  • Point Prompts    │         │  • PLY Export       │               │    │
│  │   │  • Mask Generation  │         │  • Mesh Generation  │               │    │
│  │   └─────────────────────┘         └─────────────────────┘               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Service Details

### SVO to ROSBAG Converter
| Property | Value |
|----------|-------|
| **Port** | 8009 |
| **UI** | http://localhost:8009 |
| **Swagger** | http://localhost:8009/api |
| **ReDoc** | http://localhost:8009/api/redoc |

**Features:**
- Upload ZED X SVO/SVO2 files
- Convert to ROSBAG format
- Include depth, pointcloud, IMU data
- Background processing with progress tracking
- Download converted ROSBAG files

### ISAAC Sim
| Property | Value |
|----------|-------|
| **Port** | 8010 |
| **UI** | http://localhost:8010 |
| **Swagger** | http://localhost:8010/api |
| **ReDoc** | http://localhost:8010/api/redoc |

**Features:**
- Photorealistic robotics simulation
- Scene management (warehouse, hospital, office)
- Robot model loading (Carter, JetBot, Franka, UR10, Spot)
- ROSBAG playback support
- Physics and render configuration

### ISAAC Lab
| Property | Value |
|----------|-------|
| **Port** | 8011 |
| **UI** | http://localhost:8011 |
| **Swagger** | http://localhost:8011/api |
| **ReDoc** | http://localhost:8011/api/redoc |

**Features:**
- Reinforcement learning framework
- Pre-built RL tasks (locomotion, manipulation, navigation)
- Algorithm support (PPO, SAC, TD3)
- Live training metrics
- Model checkpointing

### ISAAC Viewer
| Property | Value |
|----------|-------|
| **Port** | 8012 |
| **UI** | http://localhost:8012 |
| **Swagger** | http://localhost:8012/api |
| **ReDoc** | http://localhost:8012/api/redoc |

**Features:**
- ROSBAG visualization
- Frame-by-frame playback
- SAM-2 segmentation integration
- GARField 3D extraction
- Export segmented objects

## Quick Start

```bash
# Start the ISAAC ROSBAG workflow
docker compose -f docker-compose.isaac-rosbag.yml up -d

# View logs
docker compose -f docker-compose.isaac-rosbag.yml logs -f

# Stop services
docker compose -f docker-compose.isaac-rosbag.yml down
```

## Typical Workflow

1. **Upload SVO File** → http://localhost:8009
   - Upload ZED X camera recording (.svo or .svo2)
   - Configure conversion options (depth, pointcloud, IMU)
   - Start conversion

2. **View in ISAAC Viewer** → http://localhost:8012
   - Load converted ROSBAG
   - Browse frames with playback controls
   - Use SAM-2 for object segmentation
   - Extract 3D objects with GARField

3. **Simulate in ISAAC Sim** → http://localhost:8010
   - Load a scene environment
   - Spawn robot models
   - Playback ROSBAG sensor data

4. **Train with ISAAC Lab** → http://localhost:8011
   - Select RL task
   - Configure training parameters
   - Monitor live metrics

## Data Flow

```
ZED X Camera
     │
     ▼ (.svo file)
┌────────────┐
│    SVO     │
│ Converter  │ ─────▶ ROSBAG files ─────┐
└────────────┘                          │
                                        ▼
                              ┌─────────────────┐
                              │  Shared Volume  │
                              │  /isaac-data/   │
                              │    rosbags/     │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
       ┌────────────┐          ┌────────────┐          ┌────────────┐
       │ ISAAC Sim  │          │ ISAAC Lab  │          │   ISAAC    │
       │            │          │            │          │   Viewer   │
       └────────────┘          └────────────┘          └─────┬──────┘
                                                             │
                                                    ┌────────┴────────┐
                                                    │                 │
                                                    ▼                 ▼
                                              ┌──────────┐     ┌──────────┐
                                              │  SAM-2   │     │ GARField │
                                              └──────────┘     └──────────┘
```

## Network Configuration

| Network | Purpose |
|---------|---------|
| `isaac-rosbag-net` | Internal communication between ISAAC services |
| `fvdb-workflow-net` | Connection to main fVDB workflow (SAM-2, GARField) |

## Volume Mounts

| Path | Purpose |
|------|---------|
| `isaac-data/svo-uploads/` | Uploaded SVO files |
| `isaac-data/rosbags/` | Converted ROSBAG files |
| `isaac-data/scenes/` | Custom simulation scenes |
| `isaac-data/checkpoints/` | RL model checkpoints |
| `isaac-data/frames/` | Extracted video frames |

## API Endpoints Summary

### SVO Converter (8009)
- `POST /convert` - Upload and convert SVO to ROSBAG
- `GET /jobs` - List conversion jobs
- `GET /jobs/{id}` - Get job status
- `GET /download/{id}` - Download ROSBAG
- `GET /rosbags` - List available ROSBAGs

### ISAAC Sim (8010)
- `POST /simulation/start` - Start simulation
- `POST /simulation/stop` - Stop simulation
- `GET /sessions` - List sessions
- `GET /scenes` - List available scenes
- `GET /robots` - List robot models

### ISAAC Lab (8011)
- `POST /training/start` - Start RL training
- `POST /training/stop` - Stop training
- `GET /jobs` - List training jobs
- `GET /tasks` - List RL tasks
- `GET /checkpoints` - List model checkpoints

### ISAAC Viewer (8012)
- `GET /rosbags` - List ROSBAGs
- `POST /rosbag/load` - Load ROSBAG
- `GET /frame/{num}` - Get frame
- `POST /segment` - Segment object
- `POST /segment/auto` - Auto segment
- `POST /extract` - Extract 3D with GARField
