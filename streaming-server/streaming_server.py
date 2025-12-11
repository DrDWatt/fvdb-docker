"""
Custom WebRTC Streaming Server for Omniverse Web Viewer
Streams 3D Gaussian Splat models with real-time interaction
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, Set
import numpy as np
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, VideoStreamTrack
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaBlackhole
from aiortc.mediastreams import MediaStreamTrack  
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame
from av.video.codeccontext import VideoCodecContext
from av.video.frame import VideoFrame as AVVideoFrame
import time
from plyfile import PlyData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SIGNALING_PORT = 49100
HTTP_PORT = 8080
MODEL_DIR = Path("/app/models")

# Global state
pcs: Set[RTCPeerConnection] = set()
current_model = None
model_metadata = {}
gaussian_points = None  # Numpy array of XYZ coordinates
gaussian_colors = None  # Numpy array of RGB colors
gaussian_opacity = None  # Numpy array of opacity values
gaussian_scale = None  # Numpy array of scale values

class GaussianSplatVideoTrack(VideoStreamTrack):
    """
    Video track that renders Gaussian Splat models
    """
    
    def __init__(self):
        super().__init__()
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.frame_count = 0
        self.rotation = 0
        
    async def recv(self):
        """Generate video frames"""
        pts, time_base = await self.next_timestamp()
        
        # Create frame with rotating content
        frame_bgr = self.generate_frame()
        
        # Convert BGR to RGB for VideoFrame
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Create video frame with explicit format
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        self.frame_count += 1
        self.rotation = (self.rotation + 1) % 360
        
        return video_frame
    
    def generate_frame(self):
        """Generate a frame showing model info and visualization"""
        # Create black background
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(self.height):
            color = int(30 + (y / self.height) * 50)
            frame[y, :] = [color, color // 2, color // 3]
        
        # Render Gaussian Splat point cloud if loaded
        if gaussian_points is not None and len(gaussian_points) > 0:
            self._render_gaussian_splat(frame)
        else:
            # Fallback to simple visualization
            self._render_placeholder(frame)
        
        # Add overlay info
        cv2.putText(frame, "Omniverse WebRTC Streaming", (50, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, (76, 185, 0), 2)
        
        if model_metadata:
            info = f"Model: {model_metadata.get('name', 'Unknown')[:30]}"
            cv2.putText(frame, info, (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            points_info = f"Points: {model_metadata.get('num_gaussians', 'Unknown')}"
            cv2.putText(frame, points_info, (50, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add status
        cv2.putText(frame, f"Frame: {self.frame_count} | Rotation: {self.rotation}°", 
                   (50, self.height - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Peers: {len(pcs)}", (self.width - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _render_gaussian_splat(self, frame):
        """Render actual Gaussian Splat point cloud with real colors and proper splatting"""
        if gaussian_points is None or len(gaussian_points) == 0:
            return
        
        # Get rotation matrices
        angle_y = np.radians(self.rotation)
        angle_x = np.radians(self.rotation * 0.3)
        
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        
        # Rotation matrix (Y-axis then X-axis)
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        rot_matrix = rot_y @ rot_x
        
        # Apply rotation to points
        rotated_points = gaussian_points @ rot_matrix.T
        
        # Perspective projection for better 3D effect
        focal_length = 800  # Focal length for perspective
        camera_distance = 3.5  # Camera distance from center
        
        # Apply perspective
        z_persp = rotated_points[:, 2] + camera_distance
        z_persp = np.maximum(z_persp, 0.1)  # Avoid division by zero
        
        scale = focal_length / z_persp
        offset_x = self.width // 2
        offset_y = self.height // 2
        
        x_2d = (rotated_points[:, 0] * scale + offset_x).astype(int)
        y_2d = (rotated_points[:, 1] * scale + offset_y).astype(int)
        z_2d = rotated_points[:, 2]  # For depth sorting
        
        # Get colors and opacity for valid points
        valid_mask = (x_2d >= 0) & (x_2d < self.width) & (y_2d >= 0) & (y_2d < self.height)
        x_2d_valid = x_2d[valid_mask]
        y_2d_valid = y_2d[valid_mask]
        z_2d_valid = z_2d[valid_mask]
        
        if gaussian_colors is not None:
            colors_valid = gaussian_colors[valid_mask]
        else:
            # Fallback to depth-based colors
            colors_valid = None
        
        if gaussian_scale is not None:
            scale_valid = gaussian_scale[valid_mask]
        else:
            scale_valid = None
        
        if gaussian_opacity is not None:
            opacity_valid = gaussian_opacity[valid_mask]
        else:
            opacity_valid = None
        
        # Sort by depth (back to front for proper alpha blending)
        depth_order = np.argsort(z_2d_valid)
        
        # Pre-compute colors and sizes for all valid points
        if colors_valid is not None:
            point_colors = colors_valid
        else:
            # Fallback: depth-based colors
            depth_normalized = (z_2d_valid - z_2d_valid.min()) / (z_2d_valid.max() - z_2d_valid.min() + 1e-6)
            color_intensity = (100 + depth_normalized * 155).astype(int)
            point_colors = np.column_stack((
                np.zeros_like(color_intensity),
                color_intensity,
                (255 - depth_normalized * 100).astype(int)
            ))
        
        if opacity_valid is not None:
            alphas = np.clip(opacity_valid, 0.3, 1.0)
        else:
            alphas = np.full(len(z_2d_valid), 0.8)
        
        if scale_valid is not None:
            point_sizes = np.clip((scale_valid * 5).astype(int), 2, 10)
        else:
            point_sizes = np.clip((6.0 / (1.0 + np.abs(z_2d_valid) * 0.5)).astype(int), 2, 8)
        
        # Render Gaussian splats with optimized alpha blending
        # Create single overlay for better performance
        overlay = np.zeros_like(frame, dtype=np.float32)
        alpha_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        
        for idx in depth_order:
            x, y = x_2d_valid[idx], y_2d_valid[idx]
            color = tuple(map(float, point_colors[idx]))
            point_size = int(point_sizes[idx])
            alpha = float(alphas[idx])
            
            # Draw on overlay
            cv2.circle(overlay, (x, y), point_size, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(alpha_mask, (x, y), point_size, (alpha,), -1, lineType=cv2.LINE_AA)
        
        # Apply alpha blending in one operation
        alpha_mask = np.clip(alpha_mask, 0, 1)
        alpha_mask_3ch = np.stack([alpha_mask] * 3, axis=2)
        frame[:] = (overlay * alpha_mask_3ch + frame.astype(np.float32) * (1 - alpha_mask_3ch)).astype(np.uint8)
    
    def _render_placeholder(self, frame):
        """Render placeholder visualization when no model loaded"""
        center_x = self.width // 2
        center_y = self.height // 2
        radius = 200
        
        # Draw rotating circle
        cv2.circle(frame, (center_x, center_y), radius, (76, 185, 0), 3)
        
        # Rotating point
        angle_rad = np.radians(self.rotation)
        point_x = int(center_x + radius * np.cos(angle_rad))
        point_y = int(center_y + radius * np.sin(angle_rad))
        cv2.circle(frame, (point_x, point_y), 20, (0, 185, 255), -1)
        
        # 3D effect
        ellipse_scale = np.sin(angle_rad) * 0.5 + 0.5
        ellipse_radius = int(radius * ellipse_scale)
        if ellipse_radius > 0:
            cv2.ellipse(frame, (center_x, center_y), (ellipse_radius, radius // 3),
                       0, 0, 360, (100, 200, 100), 2)


async def handle_nvidia_signaling(request):
    """Handle NVIDIA Omniverse proprietary signaling protocol"""
    # Accept NVIDIA-specific WebSocket subprotocols
    protocols = request.headers.get('Sec-WebSocket-Protocol', '').split(', ')
    nvidia_protocol = None
    for proto in protocols:
        if 'x-nv-sessionid' in proto or 'PassThruSessionId' in proto:
            nvidia_protocol = proto
            break
    
    ws = web.WebSocketResponse(protocols=[nvidia_protocol] if nvidia_protocol else None)
    await ws.prepare(request)
    
    peer_id = request.rel_url.query.get('peer_id', 'unknown')
    version = request.rel_url.query.get('version', '1')
    logger.info(f"NVIDIA signaling connection: peer_id={peer_id}, version={version}")
    
    # Create peer connection with STUN and TURN servers for NAT traversal
    # TURN server provides relay when direct connection fails
    config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            RTCIceServer(
                urls=["turn:openrelay.metered.ca:80"],
                username="openrelayproject",
                credential="openrelayproject"
            ),
            RTCIceServer(
                urls=["turn:openrelay.metered.ca:443"],
                username="openrelayproject",
                credential="openrelayproject"
            ),
        ]
    )
    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
    
    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        logger.info(f"ICE connection state: {pc.iceConnectionState}")
    
    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange():
        logger.info(f"ICE gathering state: {pc.iceGatheringState}")
        if pc.iceGatheringState == "complete":
            # Log all gathered candidates
            if hasattr(pc, 'sctp') and pc.sctp:
                transport = pc.sctp.transport
                if hasattr(transport, '_ice_transport'):
                    ice = transport._ice_transport
                    logger.info(f"Gathered {len(ice.local_candidates())} local candidates")
                    for cand in ice.local_candidates():
                        logger.info(f"  Local candidate: {cand.type} {cand.protocol} {cand.ip}:{cand.port}")
    
    @pc.on("icecandidate")
    async def on_icecandidate(candidate):
        if candidate:
            logger.info(f"Sending ICE candidate to client")
            try:
                await ws.send_json({
                    "type": "ice",
                    "candidate": {
                        "candidate": candidate.candidate,
                        "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex
                    }
                })
            except Exception as e:
                logger.warning(f"Could not send ICE candidate: {e}")
    
    # Add video track with VP8 codec (software-only, works on all platforms)
    video_track = GaussianSplatVideoTrack()
    video_sender = pc.addTrack(video_track)
    
    # Log available codecs for debugging
    try:
        capabilities = RTCRtpSender.getCapabilities("video")
        logger.info(f"Available video codecs: {[c.mimeType for c in capabilities.codecs]}")
    except Exception as e:
        logger.warning(f"Could not list codecs: {e}")
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    logger.info(f"Received WebSocket message: {data.get('type', 'unknown')}")
                    
                    if data.get("type") == "offer":
                        # Handle WebRTC offer
                        logger.info(f"Processing offer with SDP length: {len(data.get('sdp', ''))}")
                        
                        try:
                            # Create RTCSessionDescription
                            logger.info("Creating RTCSessionDescription...")
                            logger.info(f"Offer SDP:\n{data['sdp'][:500]}")  # Log first 500 chars
                            offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                            logger.info(f"Offer created: type={offer.type}")
                            
                            # Set remote description
                            logger.info("Setting remote description...")
                            await pc.setRemoteDescription(offer)
                            logger.info("Remote description set successfully")
                            
                            # Create answer  
                            logger.info("Creating answer...")
                            answer = await pc.createAnswer()
                            logger.info(f"Answer created: {answer}")
                            
                            logger.info("Setting local description...")
                            await pc.setLocalDescription(answer)
                            logger.info("Local description set successfully")
                            
                            # Send answer back
                            logger.info("Sending answer to client...")
                            await ws.send_json({
                                "type": "answer",
                                "sdp": pc.localDescription.sdp
                            })
                            logger.info(f"Sent answer successfully. Active connections: {len(pcs)}")
                        except Exception as offer_error:
                            import traceback
                            logger.error(f"Error in offer handling: {offer_error}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            raise
                        
                    elif data.get("type") == "ice":
                        # Handle ICE candidate
                        if data.get("candidate"):
                            cand_data = data["candidate"]
                            cand_string = cand_data.get('candidate', '')
                            logger.info(f"Received ICE candidate: {cand_string[:50]}")
                            try:
                                # Parse the candidate string to RTCIceCandidate
                                ice_candidate = candidate_from_sdp(cand_string.split(':', 1)[1])  # Remove "candidate:" prefix
                                ice_candidate.sdpMid = cand_data.get('sdpMid')
                                ice_candidate.sdpMLineIndex = cand_data.get('sdpMLineIndex')
                                await pc.addIceCandidate(ice_candidate)
                                logger.info("✅ Added ICE candidate successfully")
                            except Exception as ice_error:
                                logger.warning(f"Could not add ICE candidate: {ice_error}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    
            elif msg.type == web.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
    finally:
        try:
            await pc.close()
        except:
            pass
        pcs.discard(pc)
        logger.info("WebSocket connection closed")
    
    return ws


async def handle_offer(request):
    """Handle WebRTC offer from client (HTTP fallback)"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    
    logger.info(f"Received offer from client")
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pcs.discard(pc)
    
    # Add video track
    video_track = GaussianSplatVideoTrack()
    pc.addTrack(video_track)
    
    # Handle offer
    await pc.setRemoteDescription(offer)
    
    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    logger.info(f"Sending answer to client. Active connections: {len(pcs)}")
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    )


async def health(request):
    """Health check endpoint"""
    return web.json_response({
        "status": "healthy",
        "service": "webrtc-streaming-server",
        "active_connections": len(pcs),
        "model_loaded": gaussian_points is not None,
        "model_info": model_metadata
    })


# NVIDIA Omniverse-specific endpoints
async def auth_login(request):
    """Authentication endpoint for Omniverse library"""
    logger.info("Authentication request received")
    return web.json_response({
        "status": "success",
        "token": "mock-auth-token-12345",
        "userId": "demo-user",
        "expiresIn": 3600
    })


async def auth_validate(request):
    """Validate authentication token"""
    logger.info("Token validation request received")
    return web.json_response({
        "valid": True,
        "userId": "demo-user"
    })


async def get_drivers(request):
    """Return driver/capability information"""
    logger.info("Drivers info request received")
    return web.json_response({
        "version": "1.0.0",
        "platform": "linux",
        "capabilities": {
            "webrtc": True,
            "h264": True,
            "vp8": True,
            "vp9": False,
            "opus": True
        },
        "drivers": {
            "video": "software",
            "audio": "software"
        }
    })


async def session_info(request):
    """Get session information"""
    session_id = request.match_info.get('session_id', 'default')
    logger.info(f"Session info request for: {session_id}")
    return web.json_response({
        "sessionId": session_id,
        "status": "ready",
        "created": "2024-01-01T00:00:00Z",
        "lastActive": "2024-01-01T00:00:00Z",
        "metadata": model_metadata
    })


async def create_session(request):
    """Create a new streaming session"""
    logger.info("Create session request received")
    data = await request.json() if request.body_exists else {}
    session_id = data.get('sessionId', 'default-session')
    
    return web.json_response({
        "sessionId": session_id,
        "status": "created",
        "signalingUrl": f"ws://localhost:8080/ws/signaling/{session_id}",
        "metadata": model_metadata
    })


async def destroy_session(request):
    """Destroy a streaming session"""
    session_id = request.match_info.get('session_id', 'default')
    logger.info(f"Destroy session request for: {session_id}")
    return web.json_response({
        "status": "destroyed",
        "sessionId": session_id
    })


async def index(request):
    """Serve index page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebRTC Streaming Server</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            h1 { color: #76b900; }
            .status { padding: 15px; background: #d4edda; border-radius: 5px; margin: 20px 0; }
            .info { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
            code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 WebRTC Streaming Server</h1>
            <div class="status">
                <h3>✅ Server Running</h3>
                <p><strong>Signaling Port:</strong> 49100</p>
                <p><strong>HTTP Port:</strong> 8080</p>
                <p><strong>Active Connections:</strong> <span id="connections">0</span></p>
            </div>
            
            <div class="info">
                <h3>📖 Quick Links</h3>
                <ul>
                    <li><a href="http://localhost:8080/workflow" target="_blank"><strong>Complete Workflow</strong></a> - Upload, COLMAP, Training</li>
                    <li><a href="http://localhost:8080/test" target="_blank">3D Viewer</a> - WebRTC Stream</li>
                    <li><a href="http://localhost:8003/api" target="_blank">COLMAP API</a></li>
                    <li><a href="http://localhost:8000/api" target="_blank">Training API</a></li>
                    <li><a href="http://localhost:8002" target="_blank">USD Pipeline</a></li>
                    <li><a href="http://localhost:8001/docs" target="_blank">Rendering API</a></li>
                </ul>
            </div>
            
            <h3>Model Information</h3>
            <pre id="model-info">Loading...</pre>
        </div>
        
        <script>
            async function updateStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    document.getElementById('connections').textContent = data.active_connections;
                    document.getElementById('model-info').textContent = JSON.stringify(data.model_info, null, 2);
                } catch (e) {
                    console.error('Failed to fetch status:', e);
                }
            }
            
            updateStatus();
            setInterval(updateStatus, 2000);
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type="text/html")


async def load_model():
    """Load model metadata and Gaussian Splat data"""
    global current_model, model_metadata, gaussian_points, gaussian_colors, gaussian_opacity, gaussian_scale
    
    # Find PLY files - prefer counter models
    ply_files = list(MODEL_DIR.glob("*.ply"))
    
    # Prioritize counter models
    counter_files = [f for f in ply_files if 'counter' in f.name.lower()]
    if counter_files:
        ply_files = counter_files
    
    if ply_files:
        # Use the registry test or first counter model
        current_model = None
        for f in ply_files:
            if 'registry_test' in f.name or 'counter' in f.name:
                current_model = f
                break
        if current_model is None:
            current_model = ply_files[0]
            
        logger.info(f"Loading model: {current_model}")
        
        # Load PLY data
        try:
            plydata = PlyData.read(str(current_model))
            vertex_data = plydata['vertex']
            
            # Extract XYZ coordinates
            x = np.array(vertex_data['x'])
            y = np.array(vertex_data['y'])
            z = np.array(vertex_data['z'])
            
            # Combine into points array
            points = np.column_stack((x, y, z))
            
            # Extract colors if available (RGB or spherical harmonics)
            colors = None
            try:
                # Try to get RGB colors
                if 'red' in vertex_data.data.dtype.names:
                    r = np.array(vertex_data['red'])
                    g = np.array(vertex_data['green'])
                    b = np.array(vertex_data['blue'])
                    colors = np.column_stack((b, g, r))  # OpenCV uses BGR
                elif 'f_dc_0' in vertex_data.data.dtype.names:
                    # Spherical harmonics - use DC component for base color
                    sh_r = np.array(vertex_data['f_dc_0'])
                    sh_g = np.array(vertex_data['f_dc_1'])
                    sh_b = np.array(vertex_data['f_dc_2'])
                    # Convert SH to RGB (simplified - DC component only)
                    C0 = 0.28209479177387814
                    colors = np.column_stack((sh_b / C0 + 0.5, sh_g / C0 + 0.5, sh_r / C0 + 0.5))
                    colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
                logger.info(f"Extracted colors from PLY data")
            except Exception as e:
                logger.warning(f"Could not extract colors: {e}")
                colors = None
            
            # Extract opacity if available
            opacity = None
            try:
                if 'opacity' in vertex_data.data.dtype.names:
                    opacity = np.array(vertex_data['opacity'])
                    # Convert from logit space to probability
                    opacity = 1.0 / (1.0 + np.exp(-opacity))
                    logger.info(f"Extracted opacity from PLY data")
            except Exception as e:
                logger.warning(f"Could not extract opacity: {e}")
            
            # Extract scale if available
            scale = None
            try:
                if 'scale_0' in vertex_data.data.dtype.names:
                    scale_0 = np.array(vertex_data['scale_0'])
                    scale_1 = np.array(vertex_data['scale_1'])
                    scale_2 = np.array(vertex_data['scale_2'])
                    # Use average scale
                    scale = np.exp((scale_0 + scale_1 + scale_2) / 3.0)
                    logger.info(f"Extracted scale from PLY data")
            except Exception as e:
                logger.warning(f"Could not extract scale: {e}")
            
            # Normalize and center the points
            points = points - points.mean(axis=0)
            max_extent = np.abs(points).max()
            if max_extent > 0:
                points = points / max_extent
            
            # Subsample for performance (use every Nth point)
            num_points = len(points)
            if num_points > 50000:
                step = num_points // 50000
                gaussian_points = points[::step]
                if colors is not None:
                    gaussian_colors = colors[::step]
                if opacity is not None:
                    gaussian_opacity = opacity[::step]
                if scale is not None:
                    gaussian_scale = scale[::step]
            else:
                gaussian_points = points
                gaussian_colors = colors
                gaussian_opacity = opacity
                gaussian_scale = scale
            
            logger.info(f"Loaded {len(gaussian_points):,} points from {num_points:,} total Gaussians")
            
            # Set metadata
            size_mb = current_model.stat().st_size / (1024 * 1024)
            model_metadata = {
                "name": current_model.name,
                "size_mb": round(size_mb, 2),
                "num_gaussians": f"{num_points:,}",
                "rendered_points": f"{len(gaussian_points):,}",
                "training_steps": "Unknown",
                "num_images": "Unknown"
            }
            
        except Exception as e:
            logger.error(f"Error loading PLY data: {e}")
            gaussian_points = None
            size_mb = current_model.stat().st_size / (1024 * 1024)
            model_metadata = {
                "name": current_model.name,
                "size_mb": round(size_mb, 2),
                "num_gaussians": "Error loading",
                "error": str(e)
            }
        
        logger.info(f"Model metadata: {model_metadata}")
    else:
        logger.warning("No PLY models found in /app/models")
        model_metadata = {"error": "No models found"}
        gaussian_points = None


async def on_shutdown(app):
    """Cleanup on shutdown"""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


def main():
    """Run the streaming server"""
    logger.info("Starting WebRTC Streaming Server...")
    
    # Load model
    asyncio.run(load_model())
    
    # Create web app
    app = web.Application()
    
    # Basic endpoints
    app.router.add_get("/", index)
    app.router.add_get("/workflow", lambda request: web.FileResponse("/app/workflow.html"))
    app.router.add_get("/health", health)
    app.router.add_get("/test", lambda request: web.FileResponse("/app/test-viewer.html"))
    
    # WebRTC signaling
    app.router.add_get("/ws/signaling/{session_id}", handle_nvidia_signaling)
    app.router.add_get("/sign_in", handle_nvidia_signaling)  # NVIDIA Omniverse signaling endpoint
    app.router.add_post("/offer", handle_offer)
    
    # NVIDIA Omniverse-specific endpoints
    app.router.add_post("/auth/login", auth_login)
    app.router.add_post("/auth/validate", auth_validate)
    app.router.add_get("/drivers.json", get_drivers)
    app.router.add_get("/api/drivers", get_drivers)
    app.router.add_get("/session/{session_id}", session_info)
    app.router.add_post("/session/create", create_session)
    app.router.add_post("/session/{session_id}/destroy", destroy_session)
    app.on_shutdown.append(on_shutdown)
    
    # Start HTTP server (for signaling)
    logger.info(f"HTTP/Signaling server starting on port {HTTP_PORT}")
    logger.info(f"WebRTC signaling on port {SIGNALING_PORT}")
    logger.info("Ready to accept WebRTC connections")
    
    web.run_app(app, host="0.0.0.0", port=HTTP_PORT, access_log=None)


if __name__ == "__main__":
    main()
