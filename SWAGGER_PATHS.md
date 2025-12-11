# 📚 API Documentation Paths - All Services

## Service URLs with Swagger Documentation

All services now have their **interactive web UI at root (`/`)** and **Swagger API docs at `/api`**

---

### 🎓 Training Service (Port 8000)
**Main Page**: http://localhost:8000  
**Swagger API**: http://localhost:8000/api  
**ReDoc**: http://localhost:8000/api/redoc

**Features**:
- Interactive training workflow UI (default page)
- Video → Gaussian Splat pipeline
- Photos → Gaussian Splat pipeline
- COLMAP dataset training
- Job monitoring and status

---

### 🎨 Rendering Service (Port 8001)
**Main Page**: http://localhost:8001  
**Swagger API**: http://localhost:8001/api  
**ReDoc**: http://localhost:8001/api/redoc

**Features**:
- PLY file manager and viewer
- Model downloads
- Interactive rendering controls
- Quick links to other services

---

### 🎬 USD Pipeline Service (Port 8002)
**Main Page**: http://localhost:8002  
**Swagger API**: http://localhost:8002/api  
**ReDoc**: http://localhost:8002/api/redoc

**Features**:
- PLY → USD conversion (interactive buttons)
- USD file downloads
- High-quality image rendering
- Interactive web interface

---

### 📡 Streaming Server (Port 8080)
**Main Page**: http://localhost:8080  
**Test Viewer**: http://localhost:8080/test  
**Health Check**: http://localhost:8080/health

**Features**:
- WebRTC Gaussian Splat streaming
- Real-time 3D visualization
- Model metadata display
- Interactive test viewer

---

## Quick Access Summary

| Service | Port | Main UI | Swagger API |
|---------|------|---------|-------------|
| Training | 8000 | http://localhost:8000 | http://localhost:8000/api |
| Rendering | 8001 | http://localhost:8001 | http://localhost:8001/api |
| USD Pipeline | 8002 | http://localhost:8002 | http://localhost:8002/api |
| Streaming | 8080 | http://localhost:8080/test | N/A (no Swagger) |

---

## Benefits of This Setup

### User-Friendly
- **Default page** (`/`) shows interactive UI for each service
- Easy to understand what each service does
- No need to remember API endpoints

### Developer-Friendly
- **Swagger at `/api`** for full API documentation
- Interactive API testing via Swagger UI
- ReDoc alternative at `/api/redoc`
- Clear separation between UI and API docs

### Consistent
- All services follow same pattern
- Predictable URLs across the stack
- Easy to navigate between services

---

## Example Usage

### Access Main UI (User-Friendly)
```bash
# Training - see workflow guide
open http://localhost:8000

# USD Conversion - click buttons to convert
open http://localhost:8002

# Rendering - manage PLY files
open http://localhost:8001
```

### Access API Docs (Developer-Friendly)
```bash
# Training API documentation
open http://localhost:8000/api

# USD Pipeline API documentation
open http://localhost:8002/api

# Rendering API documentation
open http://localhost:8001/api
```

### Programmatic API Access
```bash
# Training: Start a training job
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "my_scene", "num_training_steps": 30000}'

# USD: Convert PLY to USD
curl -X POST http://localhost:8002/convert \
  -H "Content-Type: application/json" \
  -d '{"input_file": "model.ply"}'

# Rendering: List models
curl http://localhost:8001/models
```

---

## Updated Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interfaces                       │
├─────────────────────────────────────────────────────────┤
│  http://localhost:8000  →  Training Workflow UI         │
│  http://localhost:8001  →  Rendering Manager UI         │
│  http://localhost:8002  →  USD Conversion UI            │
│  http://localhost:8080  →  WebRTC Streaming UI          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  API Documentation                       │
├─────────────────────────────────────────────────────────┤
│  http://localhost:8000/api  →  Training API (Swagger)   │
│  http://localhost:8001/api  →  Rendering API (Swagger)  │
│  http://localhost:8002/api  →  USD Pipeline API         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 What Changed

### Before
- Swagger was at root `/` (default FastAPI behavior)
- Users landed on API docs instead of friendly UI
- Confusing for non-technical users

### After
- **Custom UI at `/`** (user-friendly landing page)
- **Swagger at `/api`** (for developers)
- Clear separation of concerns
- Better user experience

---

## Implementation Details

### FastAPI Configuration
```python
# All services now use:
app = FastAPI(
    title="Service Name",
    description="Service description",
    version="1.0.0",
    docs_url="/api",      # Swagger at /api
    redoc_url="/api/redoc" # ReDoc at /api/redoc
)

# Root endpoint returns custom HTML UI
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(custom_ui_html)
```

---

## 🚀 Try It Now!

1. **Training Workflows**  
   Visit http://localhost:8000 to see the new training interface

2. **USD Conversion**  
   Visit http://localhost:8002 to convert PLY files to USD

3. **API Documentation**  
   Visit http://localhost:8000/api for full API docs

4. **Rendering**  
   Visit http://localhost:8001 to manage PLY files

All services are now optimized for both end-users and developers! 🎉
