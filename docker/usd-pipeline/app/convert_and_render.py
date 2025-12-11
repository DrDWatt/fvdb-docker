"""
USD Pipeline - Convert PLY to USD and Render
High-quality USD conversion and rendering service
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn
from plyfile import PlyData
import cv2

# USD imports (will work with usd-core)
# Import our custom USD writer (works without pxr package)
try:
    from simple_usd_writer import write_usd_point_cloud, write_usd_mesh
    USD_AVAILABLE = True
    logging.info("USD writer available (programmatic USDA creation)")
except ImportError:
    USD_AVAILABLE = False
    logging.warning("USD writer not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="USD Pipeline Service",
    description="Convert PLY files to USD and render high-quality images",
    version="1.0.0",
    docs_url="/api",
    redoc_url="/api/redoc"
)

# Directories
DATA_DIR = Path("/workspace/data")
MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)


class ConversionRequest(BaseModel):
    """Request to convert PLY to USD"""
    input_file: str
    output_name: Optional[str] = None


def ply_to_usd(ply_path: Path, usd_path: Path, subsample: int = 2) -> bool:
    """Convert PLY file to USD format using programmatic writer - HIGH QUALITY"""
    if not USD_AVAILABLE:
        logger.error("USD writer not available")
        return False
    
    try:
        # Use minimal subsampling for high quality (subsample_factor=2 keeps 50% of points)
        success = write_usd_point_cloud(ply_path, usd_path, subsample_factor=subsample)
        return success
    except Exception as e:
        logger.error(f"USD conversion failed: {e}")
        return False


def render_ply_high_quality(ply_path: Path, output_path: Path, width: int = 1920, height: int = 1080):
    """Render PLY file to high-quality image"""
    try:
        # Read PLY
        plydata = PlyData.read(str(ply_path))
        vertex_data = plydata['vertex']
        
        # Extract positions
        x = np.array(vertex_data['x'])
        y = np.array(vertex_data['y'])
        z = np.array(vertex_data['z'])
        positions = np.column_stack((x, y, z))
        
        # Extract colors if available
        try:
            if 'red' in vertex_data.data.dtype.names:
                r = np.array(vertex_data['red'])
                g = np.array(vertex_data['green'])
                b = np.array(vertex_data['blue'])
                colors = np.column_stack((b, g, r))  # BGR for OpenCV
            elif 'f_dc_0' in vertex_data.data.dtype.names:
                # Spherical harmonics
                sh_r = np.array(vertex_data['f_dc_0'])
                sh_g = np.array(vertex_data['f_dc_1'])
                sh_b = np.array(vertex_data['f_dc_2'])
                C0 = 0.28209479177387814
                colors = np.column_stack((sh_b / C0 + 0.5, sh_g / C0 + 0.5, sh_r / C0 + 0.5))
                colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
            else:
                colors = None
        except Exception as e:
            logger.warning(f"Could not extract colors: {e}")
            colors = None
        
        # Normalize positions
        positions = positions - positions.mean(axis=0)
        max_extent = np.abs(positions).max()
        if max_extent > 0:
            positions = positions / max_extent
        
        # Create high-quality render
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Camera setup
        rotation = 45  # degrees
        angle_y = np.radians(rotation)
        angle_x = np.radians(rotation * 0.3)
        
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        
        rot_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        rot_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        rot_matrix = rot_y @ rot_x
        
        # Apply rotation
        rotated = positions @ rot_matrix.T
        
        # Perspective projection
        focal_length = 800
        camera_distance = 3.5
        z_persp = rotated[:, 2] + camera_distance
        z_persp = np.maximum(z_persp, 0.1)
        
        scale = focal_length / z_persp
        offset_x = width // 2
        offset_y = height // 2
        
        x_2d = (rotated[:, 0] * scale + offset_x).astype(int)
        y_2d = (rotated[:, 1] * scale + offset_y).astype(int)
        z_2d = rotated[:, 2]
        
        # Filter valid points
        valid = (x_2d >= 0) & (x_2d < width) & (y_2d >= 0) & (y_2d < height)
        x_2d, y_2d, z_2d = x_2d[valid], y_2d[valid], z_2d[valid]
        
        if colors is not None:
            colors_valid = colors[valid]
        else:
            colors_valid = None
        
        # Sort by depth
        depth_order = np.argsort(z_2d)
        
        # Render with high quality
        for idx in depth_order:
            x, y = x_2d[idx], y_2d[idx]
            
            if colors_valid is not None:
                color = tuple(map(int, colors_valid[idx]))
            else:
                # Depth-based coloring
                depth_norm = (z_2d[idx] - z_2d.min()) / (z_2d.max() - z_2d.min() + 1e-6)
                color = (int(100 + depth_norm * 155), int(100 + depth_norm * 155), 255)
            
            # Variable point size based on depth
            point_size = max(2, int(6.0 / (1.0 + abs(z_2d[idx]) * 0.5)))
            
            cv2.circle(frame, (x, y), point_size, color, -1, lineType=cv2.LINE_AA)
        
        # Save
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        logger.info(f"Rendered to {output_path}")
        
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with interactive UI"""
    # Get available models
    models = list(MODELS_DIR.glob("*.ply"))
    usd_files = list(OUTPUTS_DIR.glob("*.usda"))
    
    models_html = ""
    if models:
        models_html = "<h3>Available PLY Models</h3><div class='models-grid'>"
        for model in models:
            size_mb = model.stat().st_size / (1024 * 1024)
            model_name = model.name
            models_html += f"""
            <div class='model-card'>
                <h4>{model_name}</h4>
                <p>Size: {size_mb:.2f} MB</p>
                <button onclick="convertToUSD('{model_name}')" class="btn-primary">Convert to USD</button>
                <button onclick="renderImage('{model_name}')" class="btn-secondary">Render Image</button>
            </div>
            """
        models_html += "</div>"
    
    usd_files_html = ""
    if usd_files:
        usd_files_html = "<h3>USD Files (Ready to Download)</h3><div class='usd-grid'>"
        for usd_file in usd_files:
            size_mb = usd_file.stat().st_size / (1024 * 1024)
            usd_name = usd_file.name
            usd_files_html += f"""
            <div class='usd-card'>
                <h4>{usd_name}</h4>
                <p>Size: {size_mb:.2f} MB | Format: USDA</p>
                <a href="/download/{usd_name}" class="btn-download">Download USD</a>
            </div>
            """
        usd_files_html += "</div>"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>USD Pipeline Service</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
            h1 {{ color: #0066cc; margin-bottom: 10px; }}
            .status {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 20px 0; border: 2px solid #28a745; }}
            .models-grid, .usd-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; margin: 20px 0; }}
            .model-card, .usd-card {{ background: #f8f9fa; padding: 20px; border-radius: 10px; border: 2px solid #e9ecef; }}
            .model-card h4, .usd-card h4 {{ color: #0066cc; margin: 0 0 10px 0; }}
            .btn-primary, .btn-secondary, .btn-download {{ padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; text-decoration: none; display: inline-block; margin: 5px 5px 5px 0; font-weight: 600; }}
            .btn-primary {{ background: #0066cc; color: white; }}
            .btn-primary:hover {{ background: #0052a3; }}
            .btn-secondary {{ background: #6c757d; color: white; }}
            .btn-secondary:hover {{ background: #545b62; }}
            .btn-download {{ background: #28a745; color: white; }}
            .btn-download:hover {{ background: #218838; }}
            #result {{ margin: 20px 0; padding: 15px; border-radius: 5px; display: none; }}
            .success {{ background: #d4edda; border: 2px solid #28a745; }}
            .error {{ background: #f8d7da; border: 2px solid #dc3545; }}
            .loading {{ background: #fff3cd; border: 2px solid #ffc107; }}
            .links {{ margin: 20px 0; }}
            .links a {{ margin-right: 15px; color: #0066cc; text-decoration: none; font-weight: 600; }}
            .links a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>USD Pipeline Service</h1>
            <p>Convert PLY Gaussian Splats to USD format and render high-quality images</p>
            
            <div class="status">
                <h3>Service Running</h3>
                <p><strong>USD Available:</strong> {USD_AVAILABLE}</p>
                <p><strong>Port:</strong> 8002</p>
                <p><strong>Format:</strong> USDA (USD ASCII) - Programmatic scene creation</p>
            </div>
            
            <div class="links">
                <a href="/api" class="btn">API Documentation (Swagger)</a>
                <a href="/health" class="btn">Health Check</a>
                <a href="/models" class="btn">Models JSON</a>
            </div>
            
            <div id="result"></div>
            
            {models_html}
            {usd_files_html}
            
        </div>
        
        <script>
            async function convertToUSD(modelName) {{
                const result = document.getElementById('result');
                result.className = 'loading';
                result.style.display = 'block';
                result.innerHTML = '<strong>Converting...</strong><p>Creating USD file from ' + modelName + '</p>';
                
                try {{
                    const response = await fetch('/convert', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ 
                            input_file: modelName,
                            output_name: modelName.replace('.ply', '')
                        }})
                    }});
                    
                    const data = await response.json();
                    
                    if (response.ok) {{
                        result.className = 'success';
                        result.innerHTML = `
                            <strong>✅ Success!</strong>
                            <p><strong>Input:</strong> ${{data.input}}</p>
                            <p><strong>Output:</strong> ${{data.output}}</p>
                            <p><strong>Format:</strong> ${{data.format}}</p>
                            <p><strong>Size:</strong> ${{data.size_mb}} MB</p>
                            <p>Refresh page to see download link below!</p>
                        `;
                        setTimeout(() => location.reload(), 2000);
                    }} else {{
                        throw new Error(data.detail || 'Conversion failed');
                    }}
                }} catch (error) {{
                    result.className = 'error';
                    result.innerHTML = '<strong>Error:</strong><p>' + error.message + '</p>';
                }}
            }}
            
            async function renderImage(modelName) {{
                const result = document.getElementById('result');
                result.className = 'loading';
                result.style.display = 'block';
                result.innerHTML = '<strong>Rendering...</strong><p>Creating high-quality 1920x1080 PNG from ' + modelName + '</p>';
                
                try {{
                    const response = await fetch('/render/' + modelName, {{
                        method: 'POST'
                    }});
                    
                    if (response.ok) {{
                        const blob = await response.blob();
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = modelName.replace('.ply', '_render.png');
                        a.click();
                        
                        result.className = 'success';
                        result.innerHTML = '<strong>Rendered!</strong><p>Image downloaded: ' + a.download + '</p><p>Check your Downloads folder for the PNG file!</p>';
                    }} else {{
                        const errorText = await response.text();
                        throw new Error('Rendering failed: ' + errorText);
                    }}
                }} catch (error) {{
                    result.className = 'error';
                    result.innerHTML = '<strong>Error:</strong><p>' + error.message + '</p>';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "service": "USD Pipeline",
        "usd_available": USD_AVAILABLE
    }


@app.get("/models")
async def list_models():
    """List available PLY models"""
    models = list(MODELS_DIR.glob("*.ply"))
    return {
        "models": [{"name": m.name, "size_mb": round(m.stat().st_size / 1024 / 1024, 2)} for m in models]
    }


@app.post("/convert")
async def convert_to_usd(request: ConversionRequest):
    """Convert PLY to USD"""
    input_path = MODELS_DIR / request.input_file
    if not input_path.exists():
        raise HTTPException(404, f"Model not found: {request.input_file}")
    
    output_name = request.output_name or input_path.stem
    usd_path = OUTPUTS_DIR / f"{output_name}.usda"
    
    success = ply_to_usd(input_path, usd_path)
    
    if success:
        return {
            "status": "success",
            "input": request.input_file,
            "output": str(usd_path),
            "format": "USDA (ASCII)",
            "size_mb": round(usd_path.stat().st_size / 1024 / 1024, 2)
        }
    else:
        raise HTTPException(500, "Conversion failed")


@app.post("/render/{model_name}")
async def render_model(model_name: str, width: int = 1920, height: int = 1080):
    """Render PLY model to high-quality image"""
    input_path = MODELS_DIR / model_name
    if not input_path.exists():
        raise HTTPException(404, f"Model not found: {model_name}")
    
    output_path = OUTPUTS_DIR / f"{input_path.stem}_render.png"
    
    try:
        render_ply_high_quality(input_path, output_path, width, height)
        return FileResponse(output_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Rendering failed: {e}")


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download USD or PNG files"""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    
    # Determine media type
    if filename.endswith('.usda') or filename.endswith('.usd'):
        media_type = "text/plain"
    elif filename.endswith('.png'):
        media_type = "image/png"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(file_path, media_type=media_type, filename=filename)


if __name__ == "__main__":
    logger.info("Starting USD Pipeline Service on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002)
