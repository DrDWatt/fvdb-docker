# Web Viewer Access Guide

## Current Status: ✅ RUNNING

The Omniverse Web Viewer is **running and healthy** on port 5173.

## Access URLs

### Primary Access (Localhost)
```
http://localhost:5173
```

### Alternative Access (Container IP)
```
http://172.20.0.2:5173
http://172.17.0.4:5173
```

## Container Details

```bash
# Container: omniverse-web-viewer-dev
# Status: Up and running
# Port: 5173 (mapped to host)
# Vite Dev Server: Active

# Check status
docker ps | grep omniverse-web-viewer
curl -I http://localhost:5173
```

## Troubleshooting Browser Access

### If localhost:5173 Won't Load

**1. Clear Browser Cache**
- Hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R`
- Or try incognito/private mode

**2. Check Firewall**
```bash
# Check if port is listening
netstat -tuln | grep 5173

# Should show:
# tcp  0.0.0.0:5173  LISTEN
```

**3. Try Container IP Directly**
```bash
# Get container IP
docker inspect omniverse-web-viewer-dev --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'

# Access via IP
# http://172.17.0.4:5173 or http://172.20.0.2:5173
```

**4. Verify Service is Running**
```bash
# Test with curl
curl http://localhost:5173

# Should return HTML content
```

**5. Restart Container**
```bash
docker restart omniverse-web-viewer-dev
sleep 5
curl -I http://localhost:5173
```

## Port 39089 (Windsurf Proxy)

The port you're seeing (39089) is a Windsurf/IDE proxy, not the actual web viewer.

## Connecting to Rendering Service

Once you can access the web viewer at http://localhost:5173:

### Current Configuration

The web viewer is configured to connect to:
- **Server**: `172.17.0.3` (rendering service IP)
- **Port**: `49100` (WebRTC signaling)

### Update Configuration

If the rendering service IP changes:

```bash
# Get rendering service IP
RENDER_IP=$(docker inspect fvdb-rendering --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')

# Update stream.config.json
cd /home/dwatkins3/CascadeProjects/web-viewer-sample
nano stream.config.json

# Set "server": "$RENDER_IP"
```

## Services Overview

| Service | Port | Status | URL |
|---------|------|--------|-----|
| **Web Viewer** | 5173 | ✅ Running | http://localhost:5173 |
| **PLY Rendering** | 8001 | ✅ Running | http://localhost:8001 |
| **fVDB Training** | 8000 | ✅ Running | http://localhost:8000 |
| **WebRTC Signaling** | 49100 | ✅ Ready | N/A (internal) |

## Verification Commands

```bash
# Check all services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test web viewer
curl -I http://localhost:5173

# Test rendering service
curl http://localhost:8001/health

# Test training service
curl http://localhost:8000/health

# View logs
docker logs omniverse-web-viewer-dev --tail 20
docker logs fvdb-rendering --tail 20
```

## Expected Web Viewer UI

When you access http://localhost:5173, you should see:

1. **"Omniverse Embedded Web Viewer Example"** heading
2. **UI Option** selection:
   - Radio button for "UI for default streaming USD Viewer app"
   - Radio button for "UI for any streaming app"
3. **Next button** to proceed

## Complete Workflow

```
1. Access Web Viewer
   http://localhost:5173
   
2. Select UI Option
   Choose "UI for default streaming USD Viewer app"
   Click "Next"
   
3. Connection Settings
   Server: 172.17.0.3 (or configured IP)
   Port: 49100
   
4. View 3D Content
   Connect to rendering service
   Stream 3D models
```

## If You Can Access Port 39089

Port 39089 is a Windsurf proxy, but you should be able to access 5173:

```bash
# From your browser machine, test connectivity
curl http://localhost:5173
curl http://127.0.0.1:5173

# If this works but browser doesn't, try:
# - Different browser
# - Incognito mode
# - Clear all localhost cache
# - Restart browser
```

## Docker Network Details

```bash
# Web viewer networks
docker inspect omniverse-web-viewer-dev --format='{{json .NetworkSettings.Networks}}' | python3 -m json.tool

# Rendering service networks
docker inspect fvdb-rendering --format='{{json .NetworkSettings.Networks}}' | python3 -m json.tool

# Both should be on 'bridge' network
```

## Quick Reset

If nothing works:

```bash
# Stop and restart everything
cd /home/dwatkins3/CascadeProjects/web-viewer-sample
docker restart omniverse-web-viewer-dev
docker restart fvdb-rendering

# Wait for startup
sleep 5

# Test
curl -I http://localhost:5173
curl http://localhost:8001/health

# Try browser again
# http://localhost:5173
```

## Alternative: Access from Host Browser

If you're accessing from a different machine (not localhost):

```bash
# Find the DGX Spark IP address
ip addr show | grep inet

# Access from remote browser
# http://<dgx-ip>:5173
```

---

**Current Status**: All services running and healthy
**Next Step**: Access http://localhost:5173 in browser
**Issue**: Browser connectivity to localhost:5173
**Solution**: Try incognito mode, clear cache, or use container IP
