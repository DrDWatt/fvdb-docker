// Orientation controls for PLYs without trained cameras: flip upside down / rotate 90° about the current view
// direction, applied live to the loaded splat (no reload of a multi-GB file) and saved per file
// (POST /orientation/<model>, stored in Splat Blueprints). Reset returns to the automatic estimate.
//
// Why manual: new_peacock.ply loaded upside down (2026-10-23); neither geometry nor Cosmos 3 could tell reliably
// which way is up for it (see features/orient_override.py).
(() => {
    const $ = (id) => document.getElementById(id);
    const viewer = () => $('viewer-iframe').contentWindow;
    const model = () => $('model-select').value;
    const state = { info: null, busy: false };

    const setStatus = (text, color) => { const el = $('orient-status'); el.textContent = text; el.style.color = color || ''; };

    // ----- small 3x3 / quaternion helpers (row-major 3x3, quaternions [x, y, z, w]) -----
    const mul = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
    const apply = (A, v) => A.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
    function rotation(axis, deg) {  // Rodrigues: right-handed rotation about a unit axis
        const n = Math.hypot(...axis), [x, y, z] = axis.map((a) => a / n), t = (deg * Math.PI) / 180;
        const c = Math.cos(t), s = Math.sin(t), C = 1 - c;
        return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C]];
    }
    function quatOf(m) {
        const tr = m[0][0] + m[1][1] + m[2][2];
        let q;
        if (tr > 0) {
            const s = 2 * Math.sqrt(tr + 1);
            q = [(m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, s / 4];
        } else {
            const i = [0, 1, 2].reduce((a, b) => (m[b][b] > m[a][a] ? b : a), 0), j = (i + 1) % 3, k = (i + 2) % 3;
            const s = 2 * Math.sqrt(1 + m[i][i] - m[j][j] - m[k][k]);
            q = [0, 0, 0, 0];
            q[i] = s / 4; q[j] = (m[j][i] + m[i][j]) / s; q[k] = (m[k][i] + m[i][k]) / s; q[3] = (m[k][j] - m[j][k]) / s;
        }
        const n = Math.hypot(...q);
        return q.map((v) => v / n);
    }
    // column-major 4x4 (PlayCanvas Mat4.data) -> [R (row-major 3x3), t]
    const split = (d) => [[[d[0], d[4], d[8]], [d[1], d[5], d[9]], [d[2], d[6], d[10]]], [d[12], d[13], d[14]]];

    /** New splat transform after turning the scene `deg` about the camera's viewing axis (through the camera). */
    function turned(cam, deg) {
        const [Rm, tm] = split(cam.model);
        const [Rv, tv] = split(cam.view);                            // world -> camera
        const p = [0, 1, 2].map((i) => -(Rv[0][i] * tv[0] + Rv[1][i] * tv[1] + Rv[2][i] * tv[2]));  // camera position
        const f = [-Rv[2][0], -Rv[2][1], -Rv[2][2]];                 // viewing direction (camera looks down -Z)
        const Q = rotation(f, deg);
        const R = mul(Q, Rm);
        const t = apply(Q, tm.map((v, i) => v - p[i])).map((v, i) => v + p[i]);
        return { R, t, quat: quatOf(R) };
    }

    async function turn(deg) {
        if (state.busy) return;
        const win = viewer(), entity = win && win.__splatEntity;
        const cam = win && typeof win.getCameraMatrices === 'function' ? win.getCameraMatrices() : null;
        if (!entity || !cam || !cam.model) { setStatus('Viewer still loading…', '#ffc107'); return; }
        state.busy = true;
        try {
            const { R, t, quat } = turned(cam, deg);
            entity.setLocalPosition(t[0], t[1], t[2]);
            entity.setLocalRotation(quat[0], quat[1], quat[2], quat[3]);
            if (entity._app) entity._app.renderNextFrame = true;
            const resp = await fetch(`/orientation/${encodeURIComponent(model())}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ quat, note: deg === 180 ? 'flip' : `rotate ${deg}°` }),
            });
            const out = await resp.json();
            if (!resp.ok) throw new Error(out.detail || `HTTP ${resp.status}`);
            state.info = out;
            setStatus('✅ Saved — loads this way next time', '#28a745');
            return R;
        } catch (e) {
            setStatus(`❌ ${e.message}`, '#dc3545');
        } finally {
            state.busy = false;
        }
    }

    async function reset() {
        try {
            const resp = await fetch(`/orientation/${encodeURIComponent(model())}`, { method: 'DELETE' });
            const out = await resp.json();
            if (!resp.ok) throw new Error(out.detail || `HTTP ${resp.status}`);
            $('viewer-iframe').src = Flythrough.viewerSrc(model(), undefined, out.quat);
            setStatus('Reset to automatic — reloading…');
            if (window.ModelLoading) await ModelLoading.track(model());
            await refresh();
        } catch (e) {
            setStatus(`❌ ${e.message}`, '#dc3545');
        }
    }

    async function refresh() {
        try {
            const r = await fetch(`/orientation/${encodeURIComponent(model())}`);
            state.info = await r.json();
        } catch (e) { state.info = null; }
        const editable = !!(state.info && state.info.can_edit);
        for (const id of ['orient-flip', 'orient-left', 'orient-right', 'orient-reset']) $(id).disabled = !editable;
        setStatus(!state.info ? 'Orientation unavailable'
            : !editable ? 'Orientation from trained cameras'
            : state.info.source === 'saved' ? 'Orientation: saved for this file' : 'Orientation: automatic — upside down? Flip it');
    }

    function init() {
        $('orient-flip').addEventListener('click', () => turn(180));
        $('orient-left').addEventListener('click', () => turn(90));
        $('orient-right').addEventListener('click', () => turn(-90));
        $('orient-reset').addEventListener('click', reset);
    }

    window.Orient = { init, refresh, turn, reset, _turned: turned, _quatOf: quatOf };
})();
