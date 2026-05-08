# fVDB Gaussian Splatting Platform - Multi-Container Architecture

## Running Containers

| Container | Image | Port | Status |
|-----------|-------|------|--------|
| fvdb-viewer | fvdb-training-gpu:latest | 8085 | Healthy |
| fvdb-training-gpu | fvdb-training-gpu:latest | 8000 | Healthy |
| fvdb-rendering | fvdb-rendering-minimal:latest | 8001 | Healthy |
| colmap-processor | colmap-service:latest | 8003 | Healthy |
| sam2-segmentation | sam2-service:latest | 8004 | Healthy |
| garfield-extraction | garfield-service:latest | 8006 | Healthy |
| svo-converter | svo-rosbag-converter:latest | 8009 | Healthy |
| isaac-sim | isaac-sim-service:latest | 8010 | Healthy |
| isaac-lab | isaac-lab-service:latest | 8011 | Healthy |
| isaac-viewer | isaac-viewer:latest | 8012 | Healthy |
| trellis-reconstructor | trellis-service:latest | 8013 | Healthy |
| ollama-llm | ollama/ollama:latest | 11435 | Healthy |
| usd-pipeline | fvdb-docker-usd-pipeline:latest | 8002 | Healthy |
| streaming-server | omniverse-streaming-server:latest | 8090 | Healthy |
| registry | registry:2 | 7000 | Running |
| openshell-cluster-nemoclaw | ghcr.io/nvidia/openshell/cluster:0.0.12 | 8080 | Healthy |

## Mermaid Architecture Diagram

```mermaid
graph TB
    %% ===== CUSTOMER-FACING UIs =====
    subgraph UI["🖥️ Customer-Facing UIs"]
        VIEWER["fVDB Viewer<br/>:8085<br/>3D Gaussian Splat Viewer"]
        WORKFLOW["Training Workflow<br/>:8000/workflow<br/>Video/Photo → Splat"]
        ISAAC_UI["ISAAC Viewer<br/>:8012<br/>SVO → Gaussian Splat"]
    end

    %% ===== CORE PROCESSING PIPELINE =====
    subgraph CORE["⚙️ Core Processing Pipeline"]
        TRAINING["Training Service<br/>:8000<br/>Gaussian Splat Training"]
        COLMAP["COLMAP Service<br/>:8003<br/>Structure from Motion"]
        RENDERING["Rendering Service<br/>:8001<br/>PLY Model Rendering"]
    end

    %% ===== AI / ML SERVICES =====
    subgraph AI["🧠 AI / ML Services"]
        SAM2["SAM-2 Segmentation<br/>:8004<br/>Object Detection"]
        GARFIELD["GARField Extraction<br/>:8006<br/>3D Object Extraction"]
        OLLAMA["Ollama LLM<br/>:11435<br/>RAG Intelligence"]
        TRELLIS["TRELLIS.2<br/>:8013<br/>Image → 3D Mesh"]
    end

    %% ===== ISAAC ROBOTICS =====
    subgraph ISAAC["🤖 ISAAC Robotics"]
        SVO["SVO Converter<br/>:8009<br/>ZED X → ROSBAG"]
        ISIM["ISAAC Sim<br/>:8010<br/>Robotics Simulation"]
        ILAB["ISAAC Lab<br/>:8011<br/>RL Training"]
    end

    %% ===== INFRASTRUCTURE =====
    subgraph INFRA["🏗️ Infrastructure"]
        USD["USD Pipeline<br/>:8002<br/>PLY → USD"]
        STREAM["Streaming Server<br/>:8090<br/>WebRTC 3D Stream"]
        REG["Docker Registry<br/>:7000"]
        OSHELL["OpenShell Cluster<br/>:8080<br/>NemoClaw"]
    end

    %% ===== SHARED STORAGE =====
    subgraph STORAGE["💾 Shared Volumes"]
        MODELS[("models/<br/>PLY files")]
        COLMAP_DATA[("colmap-data/<br/>SfM outputs")]
        ISAAC_DATA[("isaac-data/<br/>SVO/ROSBAG")]
        CACHE[("cache/<br/>Model weights")]
    end

    %% ===== DATA FLOW: UI → Services =====
    VIEWER -->|"render, segment"| RENDERING
    VIEWER -->|"auto-segment, click"| SAM2
    VIEWER -->|"3D extract"| GARFIELD
    VIEWER -->|"RAG query"| OLLAMA
    VIEWER -->|"train model"| TRAINING

    WORKFLOW -->|"upload video/photos"| TRAINING
    TRAINING -->|"camera poses"| COLMAP

    ISAAC_UI -->|"convert SVO"| SVO
    ISAAC_UI -->|"COLMAP poses"| COLMAP
    ISAAC_UI -->|"train splat"| TRAINING
    ISAAC_UI -->|"view in fVDB"| VIEWER

    %% ===== SERVICE DEPENDENCIES =====
    COLMAP -->|"posed frames"| TRAINING
    TRAINING -->|"trained PLY"| RENDERING
    TRAINING -->|"PLY model"| MODELS
    SAM2 -->|"masks"| GARFIELD
    GARFIELD -->|"extracted .ply"| TRELLIS
    SVO -->|"ROSBAG"| ISIM
    ISIM -->|"scenes"| ILAB
    RENDERING -->|"PLY"| USD
    USD -->|"USD scene"| STREAM

    %% ===== VOLUME CONNECTIONS =====
    MODELS -.-|"read/write"| VIEWER
    MODELS -.-|"read/write"| TRAINING
    MODELS -.-|"read"| RENDERING
    MODELS -.-|"read"| SAM2
    MODELS -.-|"read"| GARFIELD
    COLMAP_DATA -.-|"read/write"| COLMAP
    COLMAP_DATA -.-|"read"| TRAINING
    ISAAC_DATA -.-|"read/write"| SVO
    ISAAC_DATA -.-|"read/write"| ISAAC_UI
    CACHE -.-|"read/write"| TRAINING
    CACHE -.-|"read/write"| SAM2

    %% ===== STYLING =====
    classDef ui fill:#1E54CC,stroke:#7DB5FF,color:#fff
    classDef core fill:#0B0D12,stroke:#2F6BFF,color:#fff
    classDef ai fill:#1a0033,stroke:#7DB5FF,color:#fff
    classDef isaac fill:#0a2200,stroke:#76b900,color:#fff
    classDef infra fill:#2a2a2a,stroke:#C7CBD1,color:#fff
    classDef storage fill:#484F56,stroke:#C7CBD1,color:#fff

    class VIEWER,WORKFLOW,ISAAC_UI ui
    class TRAINING,COLMAP,RENDERING core
    class SAM2,GARFIELD,OLLAMA,TRELLIS ai
    class SVO,ISIM,ILAB isaac
    class USD,STREAM,REG,OSHELL infra
    class MODELS,COLMAP_DATA,ISAAC_DATA,CACHE storage
```

## Network Topology

All services communicate over `fvdb-workflow-net` (Docker bridge network).

```mermaid
graph LR
    subgraph NET["fvdb-workflow-net (bridge)"]
        direction TB
        A["fvdb-viewer :8085"]
        B["fvdb-training-gpu :8000"]
        C["fvdb-rendering :8001"]
        D["colmap-processor :8003"]
        E["sam2-segmentation :8004"]
        F["garfield-extraction :8006"]
        G["svo-converter :8009"]
        H["isaac-sim :8010"]
        I["isaac-lab :8011"]
        J["isaac-viewer :8012"]
        K["trellis-reconstructor :8013"]
        L["ollama-llm :11434"]
        M["usd-pipeline :8002"]
    end

    subgraph HOST["Host Network"]
        N["streaming-server :8090"]
    end

    HOST ---|"host.docker.internal"| NET
```

## Service Dependency Chain (Startup Order)

```mermaid
graph TD
    MD["model-downloader"] --> TRAINING["training-service"]
    MD --> SAM2["sam2-service"]
    TRAINING --> VIEWER["fvdb-viewer"]
    RENDERING["rendering-service"] --> VIEWER
    OLLAMA["ollama"] --> VIEWER
    OLLAMA --> PULL["ollama-pull"]
    SVO["svo-converter"] --> ISIM["isaac-sim"]
    SVO --> ISAAC_V["isaac-viewer"]
    ISIM --> ILAB["isaac-lab"]
```

## GPU Allocation

| Service | GPU Requirement |
|---------|----------------|
| Training Service | All GPUs |
| fVDB Viewer | All GPUs |
| SAM-2 Segmentation | 1 GPU |
| GARField Extraction | 1 GPU |
| COLMAP | All GPUs |
| Ollama LLM | All GPUs |
| TRELLIS.2 | All GPUs |

## Port Map Summary

| Port | Service | Purpose |
|------|---------|---------|
| 7000 | Docker Registry | Local image registry |
| 8000 | Training Service | Model training + Workflow UI |
| 8001 | Rendering Service | PLY model rendering |
| 8002 | USD Pipeline | PLY → USD conversion |
| 8003 | COLMAP | Structure from Motion |
| 8004 | SAM-2 | Object segmentation |
| 8006 | GARField | 3D object extraction |
| 8009 | SVO Converter | ZED X SVO → ROSBAG |
| 8010 | ISAAC Sim | Robotics simulation |
| 8011 | ISAAC Lab | Reinforcement learning |
| 8012 | ISAAC Viewer | SVO → Gaussian Splat UI |
| 8013 | TRELLIS.2 | Image → 3D reconstruction |
| 8080 | OpenShell | NemoClaw cluster |
| 8085 | fVDB Viewer | 3D Splat Viewer UI |
| 8090 | Streaming Server | WebRTC 3D streaming |
| 11435 | Ollama | LLM for RAG queries |
