"""Patch SuperSplat viewer index.html for HTTP/LAN deployment.

Injects:
  1. preserveDrawingBuffer polyfill — required for canvas capture (SAM3 segmentation)
  2. crypto.randomUUID polyfill — required for non-HTTPS contexts (LAN access)
  3. window.getCameraMatrices — current view/projection + splat model matrices and canvas size
     (used by the backend for 3D gaussian extraction and Cosmos 3 measurement renders)
  4. Default settings.json — valid V2 schema to prevent viewer crashes
  5. ?orient=x,y,z,w (bundle) — optional splat-entity quaternion replacing the hard-coded (0, 0, 180),
     for PLYs without cameras whose up axis is not -y (features/orientation.py). The PLY is untouched.

getCameraMatrices: viewers >= v1.32 expose the PlayCanvas app as window.app, which the index.html script
uses. Older bundles (v1.27, the build pinned in the Dockerfile) do not; for them the function is also
injected into the bundle next to window.captureFrame (where the camera is in scope) and takes precedence.

Usage: python3 patch_viewer.py /path/to/viewer/static/dir
"""
import json
import sys
from pathlib import Path

POLYFILLS = """<script>
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

// Current camera for backend projection: column-major view/proj, the splat entity's world
// transform (the viewer loads PLYs rotated (0, 0, 180)) and the canvas size in pixels.
window.getCameraMatrices = () => {
  const app = window.app;
  if (!app) return null;
  const cams = app.root.findComponents('camera')
    .filter((c) => c.enabled && c.entity.enabled && !c.renderTarget);
  const cc = cams[0];
  if (!cc) return null;
  let model = null;
  try {
    const gs = app.root.findComponents('gsplat')[0];
    if (gs) model = Array.from(gs.entity.getWorldTransform().data);
  } catch (e) {}
  return {
    view: Array.from(cc.viewMatrix.data),
    proj: Array.from(cc.projectionMatrix.data),
    model: model,
    width: app.graphicsDevice.width,
    height: app.graphicsDevice.height
  };
};
</script>
"""

MODULE_TAG = '<script type="module">'


def patch_index_html(viewer_dir: Path) -> None:
    """Inject the polyfill/camera script before the first module script."""
    index_path = viewer_dir / "index.html"
    content = index_path.read_text()
    if "window.getCameraMatrices" in content:
        print("index.html already patched")
        return
    if MODULE_TAG not in content:
        raise SystemExit("ERROR: no <script type=\"module\"> in index.html — viewer layout changed")
    index_path.write_text(content.replace(MODULE_TAG, POLYFILLS + MODULE_TAG, 1))
    print("Patched index.html with polyfills + getCameraMatrices")


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


CAMERA_ANCHOR = "window.captureFrame = ({ time, width = 480, height = width, supersample } = {}) => {"
CAMERA_INJECT = """window.getCameraMatrices = () => {
                const cc = camera.camera;
                let model = null;
                try {
                    const gs = app.root.findComponents('gsplat')[0];
                    if (gs) model = Array.from(gs.entity.getWorldTransform().data);
                } catch (e) {}
                return {
                    view: Array.from(cc.viewMatrix.data),
                    proj: Array.from(cc.projectionMatrix.data),
                    model: model,
                    width: app.graphicsDevice.width,
                    height: app.graphicsDevice.height
                };
            };
            """


def patch_camera_bundle(viewer_dir: Path) -> None:
    """Bundles without window.app (v1.27): inject getCameraMatrices where the camera is in scope."""
    index_js = viewer_dir / "index.js"
    src = index_js.read_text()
    if "window.getCameraMatrices" in src:
        print("index.js already defines getCameraMatrices")
        return
    if CAMERA_ANCHOR not in src:
        print("index.js has no v1.27 captureFrame anchor (viewer >= v1.32 uses window.app from index.html)")
        return
    index_js.write_text(src.replace(CAMERA_ANCHOR, CAMERA_INJECT + CAMERA_ANCHOR, 1))
    print("Patched index.js with getCameraMatrices (v1.27 bundle)")


ORIENT_ANCHOR = "entity.setLocalEulerAngles(0, 0, 180);"
ORIENT_PATCH = ("window.__splatEntity = entity; "  # web/orient.js flips/rotates it live (no 2.5 GB reload)
                "{ const __o = new URL(location.href).searchParams.get('orient'); "
                "const __q = __o ? __o.split(',').map(Number) : null; "
                "if (__q && __q.length === 4 && __q.every(Number.isFinite)) entity.setLocalRotation(__q[0], __q[1], __q[2], __q[3]); "
                "else entity.setLocalEulerAngles(0, 0, 180); }")


def patch_orientation(viewer_dir: Path) -> None:
    """Let ?orient=x,y,z,w set the splat entity rotation (default stays Euler (0, 0, 180))."""
    index_js = viewer_dir / "index.js"
    src = index_js.read_text()
    if "searchParams.get('orient')" in src:
        print("index.js already supports ?orient=")
        return
    if src.count(ORIENT_ANCHOR) != 1:
        print(f"WARNING: orientation anchor found {src.count(ORIENT_ANCHOR)}x — ?orient= not available")
        return
    index_js.write_text(src.replace(ORIENT_ANCHOR, ORIENT_PATCH, 1))
    print("Patched index.js with ?orient=")


if __name__ == "__main__":
    viewer_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/app/static/viewer")
    patch_index_html(viewer_dir)
    patch_camera_bundle(viewer_dir)
    patch_orientation(viewer_dir)
    write_settings_json(viewer_dir)
    print("Viewer patching complete")
