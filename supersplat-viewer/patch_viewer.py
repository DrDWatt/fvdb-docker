"""Patch SuperSplat viewer index.html for HTTP/LAN deployment.

Injects:
  1. preserveDrawingBuffer polyfill — required for canvas capture (SAM3 segmentation)
  2. crypto.randomUUID polyfill — required for non-HTTPS contexts (LAN access)
  3. Default settings.json — valid V2 schema to prevent viewer crashes

Usage: python3 patch_viewer.py /path/to/viewer/static/dir
"""
import json
import sys
from pathlib import Path


def patch_index_html(viewer_dir: Path) -> None:
    """Inject polyfill scripts into index.html before the first module script."""
    index_path = viewer_dir / "index.html"
    content = index_path.read_text()

    polyfill_block = """<script>
// Force preserveDrawingBuffer so toDataURL() works for canvas capture
const origGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type, attrs) {
  if (type === 'webgl2' || type === 'webgl') {
    attrs = attrs || {};
    attrs.preserveDrawingBuffer = true;
  }
  return origGetContext.call(this, type, attrs);
};

// Polyfill crypto.randomUUID for non-secure contexts (HTTP over LAN)
if (typeof crypto !== "undefined" && !crypto.randomUUID) {
  crypto.randomUUID = function() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
      (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
  };
}
</script>
<script type="module">"""

    content = content.replace('<script type="module">', polyfill_block, 1)
    index_path.write_text(content)
    print("Patched index.html with polyfills")


def write_settings_json(viewer_dir: Path) -> None:
    """Write a valid V2 settings.json so the viewer doesn't crash on 404."""
    settings = {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": False,
        "background": {"color": [0.04, 0.04, 0.06]},
        "postEffectSettings": {
            "sharpness": {"enabled": False, "amount": 0},
            "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
            "grading": {"enabled": False, "brightness": 0, "contrast": 1,
                        "saturation": 1, "tint": [1, 1, 1]},
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3,
                         "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5}
        },
        "animTracks": [],
        "cameras": [],
        "annotations": [],
        "startMode": "default"
    }
    settings_path = viewer_dir / "settings.json"
    settings_path.write_text(json.dumps(settings))
    print("Wrote settings.json")


def patch_index_js(viewer_dir: Path) -> None:
    """Expose window.getCameraMatrices in the viewer bundle.

    Injects next to window.captureFrame where the PlayCanvas camera component
    is in scope. Used by the backend for accurate 3D gaussian extraction.
    """
    index_js = viewer_dir / "index.js"
    src = index_js.read_text()
    if "getCameraMatrices" in src:
        print("index.js already exposes getCameraMatrices")
        return

    anchor = "window.captureFrame = ({ time, width = 480, height = width, supersample } = {}) => {"
    inject = """window.getCameraMatrices = () => {
                const cc = camera.camera;
                let model = null;
                try {
                    const gs = app.root.findComponents('gsplat')[0];
                    if (gs) model = Array.from(gs.entity.getWorldTransform().data);
                } catch (e) {}
                return {
                    view: Array.from(cc.viewMatrix.data),
                    proj: Array.from(cc.projectionMatrix.data),
                    model: model
                };
            };
            """
    if anchor not in src:
        print("WARNING: captureFrame anchor not found — getCameraMatrices not injected")
        return
    src = src.replace(anchor, inject + anchor, 1)
    index_js.write_text(src)
    print("Patched index.js with getCameraMatrices")


if __name__ == "__main__":
    viewer_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/static/viewer")
    patch_index_html(viewer_dir)
    patch_index_js(viewer_dir)
    write_settings_json(viewer_dir)
    print("Viewer patching complete")
