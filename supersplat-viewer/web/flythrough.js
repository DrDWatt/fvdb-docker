// Flythrough + MP4 export for the SuperSplat viewer page.
//
// Playback uses the viewer's native camera animation track (served by
// /flythrough/settings/<model>): the parent page advances time on
// requestAnimationFrame and calls the viewer's window.scrubTo(t), so the camera
// is spline-interpolated on the GPU at display refresh rate - no server frames.
// Export calls window.captureFrame({time}) for every output frame (deterministic,
// supersampled, offscreen), JPEG-encodes it and streams it to the H.264 encoder.
(() => {
    const MIN_SECONDS = window.FLY_MIN_SECONDS || 30;
    const MAX_SECONDS = 120;
    const UPLOAD_BATCH = 24;          // frames per upload request
    const SETTLE_MS = 1200;           // viewer blends into the anim camera over 1s
    const $ = (id) => document.getElementById(id);

    const state = { playing: false, t: 0, lastTs: 0, raf: 0, exporting: false, cancel: false, info: null };

    const iframe = () => $('viewer-iframe');
    const viewer = () => iframe().contentWindow;
    const model = () => $('model-select').value;

    function duration() {
        // Videos are never shorter than the minimum length
        const el = $('fly-duration');
        const d = Math.min(MAX_SECONDS, Math.max(MIN_SECONDS, parseFloat(el.value) || MIN_SECONDS));
        el.value = d;
        return d;
    }

    function viewerSrc(modelName, seconds) {
        const settings = `/flythrough/settings/${encodeURIComponent(modelName)}?duration=${seconds || duration()}`;
        return `/viewer/index.html?content=/models/${encodeURIComponent(modelName)}` +
            `&settings=${encodeURIComponent(settings)}&noanim&noui&webgl`;
    }

    const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    const loopDuration = () => {
        try { return viewer().animationDuration || 0; } catch (e) { return 0; }
    };
    const setStatus = (text, color) => {
        const el = $('fly-status');
        el.textContent = text;
        el.style.color = color || '#888';
    };

    function updateTimeline() {
        const loop = loopDuration();
        if (!loop) return;
        $('fly-progress').value = Math.round((state.t / loop) * 1000);
        const oneWay = state.info && state.info.mode === 'camera_path' ? duration() : loop;
        const shown = state.t <= oneWay ? state.t : loop - state.t;
        $('fly-time').textContent = `${fmt(shown)} / ${fmt(oneWay)}${state.t > oneWay ? ' (returning)' : ''}`;
    }

    // ----- Playback -----
    function tick(ts) {
        if (!state.playing) return;
        const loop = loopDuration();
        const win = viewer();
        if (loop && typeof win.scrubTo === 'function') {
            state.t = (state.t + Math.min(0.1, (ts - state.lastTs) / 1000)) % loop;
            win.scrubTo(state.t).catch(() => {});
            updateTimeline();
        }
        state.lastTs = ts;
        state.raf = requestAnimationFrame(tick);
    }

    function play() {
        if (state.exporting) return;
        if (typeof viewer().scrubTo !== 'function') {
            setStatus('Viewer still loading...', '#ffc107');
            return;
        }
        state.playing = true;
        state.lastTs = performance.now();
        $('fly-play').textContent = '⏸ Pause';
        $('fly-play').classList.replace('btn-primary', 'btn-danger');
        setStatus('Playing', '#17a2b8');
        state.raf = requestAnimationFrame(tick);
    }

    function pause(reason) {
        state.playing = false;
        cancelAnimationFrame(state.raf);
        $('fly-play').textContent = '▶ Play';
        $('fly-play').classList.replace('btn-danger', 'btn-primary');
        if (!state.exporting) setStatus(reason || 'Paused');
    }

    function seek(permille) {
        const loop = loopDuration();
        if (!loop || typeof viewer().scrubTo !== 'function') return;
        state.t = (permille / 1000) * loop;
        viewer().scrubTo(state.t).catch(() => {});
        updateTimeline();
    }

    // Pause when the user grabs the camera, so playback doesn't fight their input
    function watchViewerInput() {
        try {
            const doc = iframe().contentDocument;
            ['pointerdown', 'wheel', 'keydown'].forEach((ev) =>
                doc.addEventListener(ev, () => { if (state.playing) pause('Paused (camera moved)'); }, { passive: true }));
        } catch (e) { /* cross-origin never happens here */ }
    }

    async function loadInfo() {
        const name = model();
        if (!name) return;
        try {
            const resp = await fetch(`/flythrough/info/${encodeURIComponent(name)}?duration=${duration()}`);
            state.info = await resp.json();
            const i = state.info;
            $('fly-info').textContent = i.mode === 'camera_path'
                ? `Smoothed path through ${i.num_cameras} trained cameras`
                : i.mode === 'orbit' ? 'Orbit (model has no trained cameras)' : 'Viewer default camera motion';
        } catch (e) {
            $('fly-info').textContent = 'Flythrough info unavailable';
        }
    }

    // Called by the page after the viewer iframe (re)loads a model
    function onViewerLoaded() {
        pause('Ready');
        state.t = 0;
        watchViewerInput();
        loadInfo().then(updateTimeline);
    }

    function reloadViewer() {
        pause();
        iframe().src = viewerSrc(model());
    }

    // ----- MP4 export -----
    async function toJpeg(frame, canvas, ctx) {
        // captureFrame returns raw RGBA as base64; decode natively, force opaque alpha
        const buf = await (await fetch(`data:application/octet-stream;base64,${frame.data}`)).arrayBuffer();
        const px = new Uint8ClampedArray(buf);
        for (let i = 3; i < px.length; i += 4) px[i] = 255;
        ctx.putImageData(new ImageData(px, frame.width, frame.height), 0, 0);
        return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
    }

    async function uploadBatch(sid, blobs, firstIndex) {
        const form = new FormData();
        blobs.forEach((b, k) => form.append('frames', b, `f${String(firstIndex + k).padStart(6, '0')}.jpg`));
        const resp = await fetch(`/flythrough/export/${sid}/frames`, { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`upload failed (HTTP ${resp.status})`);
    }

    async function exportMp4() {
        if (state.exporting) { state.cancel = true; return; }
        const win = viewer();
        if (typeof win.captureFrame !== 'function') {
            setStatus('Viewer still loading...', '#ffc107');
            return;
        }
        pause();
        const [width, height] = $('fly-size').value.split('x').map(Number);
        const fps = parseInt($('fly-fps').value, 10);
        const seconds = duration();
        const btn = $('fly-export');
        state.exporting = true;
        state.cancel = false;
        btn.textContent = '✖ Cancel';
        $('fly-play').disabled = true;
        $('fly-export-progress').classList.remove('hidden');
        let sid = null;
        try {
            const start = await fetch('/flythrough/export/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: model(), duration: seconds, fps, width, height })
            });
            if (!start.ok) throw new Error((await start.json()).detail || `HTTP ${start.status}`);
            const session = await start.json();
            sid = session.session_id;
            const total = session.num_frames;

            setStatus('Positioning camera...', '#ffc107');
            await win.scrubTo(0);
            await new Promise((r) => setTimeout(r, SETTLE_MS));

            const canvas = document.createElement('canvas');
            canvas.width = session.width;
            canvas.height = session.height;
            const ctx = canvas.getContext('2d');
            let batch = [];
            let batchStart = 0;
            let pending = Promise.resolve();
            const began = performance.now();
            for (let i = 0; i < total; i++) {
                if (state.cancel) throw new Error('cancelled');
                const frame = await win.captureFrame({ time: i / fps, width: session.width, height: session.height, supersample: 2 });
                batch.push(await toJpeg(frame, canvas, ctx));
                if (batch.length === UPLOAD_BATCH || i === total - 1) {
                    const blobs = batch, first = batchStart;
                    batch = [];
                    batchStart = i + 1;
                    await pending;                       // keep uploads in order
                    pending = uploadBatch(sid, blobs, first);
                }
                const pct = ((i + 1) / total) * 100;
                $('fly-export-bar').style.width = `${pct.toFixed(1)}%`;
                const eta = ((performance.now() - began) / (i + 1)) * (total - i - 1) / 1000;
                setStatus(`Rendering ${i + 1}/${total} (~${Math.ceil(eta)}s left)`, '#ffc107');
            }
            await pending;

            setStatus('Encoding H.264...', '#ffc107');
            const done = await fetch(`/flythrough/export/${sid}/finish`, { method: 'POST' });
            sid = null;
            if (!done.ok) throw new Error((await done.json()).message || `HTTP ${done.status}`);
            const url = URL.createObjectURL(await done.blob());
            const a = document.createElement('a');
            a.href = url;
            a.download = `${model().replace(/\.ply$/i, '')}_flythrough.mp4`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(url), 10000);
            setStatus(`✅ Exported ${seconds}s @ ${fps}fps (${session.width}x${session.height})`, '#28a745');
        } catch (e) {
            if (sid) fetch(`/flythrough/export/${sid}`, { method: 'DELETE' }).catch(() => {});
            setStatus(e.message === 'cancelled' ? 'Export cancelled' : `❌ Export failed: ${e.message}`,
                e.message === 'cancelled' ? '#888' : '#dc3545');
        } finally {
            state.exporting = false;
            btn.textContent = '📥 Export MP4';
            $('fly-play').disabled = false;
            setTimeout(() => $('fly-export-progress').classList.add('hidden'), 1500);
        }
    }

    function render(container) {
        container.innerHTML = `
            <h3>Flythrough</h3>
            <p class="status-text" id="fly-info">Loading camera path...</p>
            <div class="btn-row">
                <button class="btn-primary" id="fly-play">▶ Play</button>
                <button class="btn-purple" id="fly-export">📥 Export MP4</button>
            </div>
            <div class="fly-options">
                <label>Seconds <input type="number" id="fly-duration" value="${MIN_SECONDS}" min="${MIN_SECONDS}" max="${MAX_SECONDS}"
                       title="Video length (minimum ${MIN_SECONDS}s)"></label>
                <label>FPS <select id="fly-fps"><option>24</option><option selected>30</option><option>60</option></select></label>
                <label>Size <select id="fly-size">
                    <option value="1280x720" selected>720p</option>
                    <option value="1920x1080">1080p</option>
                    <option value="1024x768">1024x768</option>
                </select></label>
            </div>
            <input type="range" id="fly-progress" min="0" max="1000" value="0">
            <div class="fly-meta"><span id="fly-time">0:00 / ${fmt(MIN_SECONDS)}</span><span id="fly-status"></span></div>
            <div id="fly-export-progress" class="progress-bar-bg hidden"><div class="progress-bar-fill" id="fly-export-bar"></div></div>`;
        $('fly-play').onclick = () => (state.playing ? pause() : play());
        $('fly-export').onclick = exportMp4;
        $('fly-progress').oninput = (e) => seek(parseInt(e.target.value, 10));
        $('fly-duration').onchange = () => { duration(); reloadViewer(); };
    }

    window.Flythrough = { render, viewerSrc, onViewerLoaded, pause };
})();
