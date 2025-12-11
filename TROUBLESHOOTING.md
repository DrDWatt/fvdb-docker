# Troubleshooting - Web Viewer Blank Page

## Issue: Blank White Page on Web Viewer

### Cause
The web viewer container's `stream.config.json` was pointing to the wrong server:
- **Old**: `127.0.0.1:49100` (doesn't exist)
- **New**: `streaming-server:8080` (our custom server)

### Solution Applied
Updated the config inside the container to point to our streaming server.

## How to Use

### 1. Refresh the Web Viewer Page
```
http://localhost:5173
```
Press `Cmd+R` or `Ctrl+R` to refresh.

### 2. You Should Now See
- **UI Option** page with two radio buttons
- "UI for default streaming USD Viewer app"
- "UI for any streaming app"

### 3. Select Streaming Option
- Click the **second radio button**: "UI for any streaming app"
- Click **"Next"**

### 4. Enter Connection Details
The connection form will appear:
- **Server**: Leave as `streaming-server` (pre-configured)
- **Port**: Should show `8080` (pre-configured)
- Click **"Connect"**

### 5. Watch the Stream!
You should see:
- Live video stream (1920x1080)
- Model information overlay
- Rotating 3D visualization
- Frame counter

## If Still Blank

### Check Browser Console (F12)
Look for errors. Should see:
- No CORS errors
- No WebSocket connection errors
- React app loading successfully

### Verify Container Config
```bash
docker exec omniverse-web-viewer cat /app/stream.config.json

# Should show:
# "server": "streaming-server",
# "signalingPort": 8080
```

### Hard Refresh
```
Cmd+Shift+R (Mac)
Ctrl+Shift+R (Linux/Windows)
```

### Clear Browser Cache
1. Open DevTools (F12)
2. Right-click refresh button
3. Select "Empty Cache and Hard Reload"

### Check Streaming Server
```bash
# Verify it's running
curl http://localhost:8080/health

# Should return:
# {"status": "healthy", "model_loaded": true, ...}
```

## Connection Flow

```
1. Browser loads http://localhost:5173
   ↓
2. React app reads /app/stream.config.json
   ↓
3. Shows UI options page
   ↓
4. User selects "any streaming app"
   ↓
5. Connection form shows server: streaming-server, port: 8080
   ↓
6. User clicks "Connect"
   ↓
7. WebRTC connection established
   ↓
8. Video stream displays
```

## Common Errors

### Error: "Cannot read properties of undefined"
**Cause**: Config file malformed
**Fix**: Re-update the config (see solution above)

### Error: "WebSocket connection failed"
**Cause**: Streaming server not accessible
**Fix**: Check server is running
```bash
docker ps | grep streaming-server
curl http://localhost:8080/health
```

### Error: CORS policy blocked
**Cause**: Wrong origin/ports
**Fix**: Ensure all services on same Docker network
```bash
docker network inspect fvdb-docker_omniverse-net
```

### Blank page with no errors
**Cause**: React app not loaded
**Fix**: Hard refresh browser (Cmd+Shift+R)

## Verification Steps

### 1. Check All Containers Running
```bash
docker compose -f docker-compose.fullstack.yml ps

# All should show "Up" and "healthy"
```

### 2. Test Web Viewer HTTP
```bash
curl -I http://localhost:5173

# Should return: HTTP/1.1 200 OK
```

### 3. Test Streaming Server
```bash
curl http://localhost:8080/health

# Should return JSON with "status": "healthy"
```

### 4. View Container Logs
```bash
# Web viewer
docker logs omniverse-web-viewer --tail 20

# Should show: "VITE v5.4.21 ready in XXX ms"

# Streaming server
docker logs streaming-server --tail 20

# Should show: "Running on http://0.0.0.0:8080"
```

### 5. Test Network Connectivity
```bash
# From web viewer to streaming server
docker exec omniverse-web-viewer ping -c 2 streaming-server

# Should succeed with 0% packet loss
```

## Quick Fix Commands

```bash
# Restart everything
docker compose -f docker-compose.fullstack.yml restart

# Update config in web viewer
docker exec omniverse-web-viewer sh -c 'cat > /app/stream.config.json << EOF
{
    "source": "local",
    "stream": {"appServer": "", "streamServer": ""},
    "gfn": {"catalogClientId": "", "clientId": "", "cmsId": 0},
    "local": {
        "server": "streaming-server",
        "signalingPort": 8080,
        "mediaPort": null
    }
}
EOF'

# Restart web viewer
docker restart omniverse-web-viewer

# Wait and refresh browser
sleep 5
echo "Now refresh http://localhost:5173"
```

## Success Indicators

✅ **Web viewer page loads** - Shows "Omniverse Embedded Web Viewer Example"
✅ **No console errors** - Browser console (F12) is clean
✅ **UI options visible** - Two radio buttons displayed
✅ **Next button works** - Proceeds to connection form
✅ **Connection succeeds** - Video stream displays

## Still Having Issues?

### Get Full Logs
```bash
# Web viewer
docker logs omniverse-web-viewer > /tmp/web-viewer.log

# Streaming server
docker logs streaming-server > /tmp/streaming-server.log

# Check the logs
cat /tmp/web-viewer.log
cat /tmp/streaming-server.log
```

### Rebuild Web Viewer
```bash
cd /home/dwatkins3/CascadeProjects/web-viewer-sample
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Nuclear Option: Clean Restart
```bash
cd /home/dwatkins3/fvdb-docker

# Stop everything
docker compose -f docker-compose.fullstack.yml down

# Clean up
docker system prune -f

# Rebuild and restart
docker compose -f docker-compose.fullstack.yml up -d --build

# Wait for services
sleep 10

# Update config
docker exec omniverse-web-viewer sh -c 'cat > /app/stream.config.json << EOF
{
    "source": "local",
    "stream": {"appServer": "", "streamServer": ""},
    "gfn": {"catalogClientId": "", "clientId": "", "cmsId": 0},
    "local": {"server": "streaming-server", "signalingPort": 8080, "mediaPort": null}
}
EOF'

# Restart web viewer
docker restart omniverse-web-viewer
```

---

**After applying fix**: Refresh http://localhost:5173 and you should see the UI! 🎉
