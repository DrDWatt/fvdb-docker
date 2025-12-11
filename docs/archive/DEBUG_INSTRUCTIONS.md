# Debugging WebRTC Streaming

## Current Status

The test viewer page is loading correctly but WebSocket connections are failing with:
```
ERROR:__main__:WebSocket error: None is not in list
```

## Next Steps

### 1. Try Connecting Again

**Go to**: http://localhost:8080/test

**Click**: "Connect to Stream"

### 2. Check Detailed Logs

After clicking connect, run:
```bash
docker logs streaming-server --tail 50
```

You should now see detailed step-by-step logs showing exactly where the error occurs:
- "Creating RTCSessionDescription..."
- "Setting remote description..."
- "Creating answer..."
- etc.

The error with full traceback will show which aiortc function is failing.

### 3. Common Issues

**"None is not in list"** typically means:
- Missing codec in the WebRTC offer
- Invalid SDP format
- aiortc can't find a compatible codec/format

### 4. Alternative: Check Browser Console

In the test viewer page:
1. Open browser console (F12)
2. Click "Connect to Stream"
3. Look for errors in the console
4. Check the "Network" tab for WebSocket messages

### 5. What to Share

Please share:
1. The detailed logs from `docker logs streaming-server --tail 50`
2. Any browser console errors
3. The WebSocket frames in the browser Network tab (if available)

This will help identify if the issue is:
- Server-side (aiortc codec problem)
- Client-side (browser WebRTC API issue)
- Network/signaling issue

---

## Streaming Server is Ready

✅ Model loaded: counter_registry_test.ply (1,086,796 Gaussians)
✅ Rendering: 51,753 points per frame
✅ WebSocket endpoint: ws://localhost:8080/ws/signaling/{session_id}
✅ Test page: http://localhost:8080/test

The infrastructure is working - we just need to debug why the WebRTC handshake is failing.
