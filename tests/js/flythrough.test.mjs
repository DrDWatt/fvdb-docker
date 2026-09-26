// Unit tests for supersplat-viewer/web/flythrough.js (flythrough playback + MP4 export).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript, flush } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/flythrough.js', import.meta.url));

function setup({ loop = 60, finishStatus = 200, captureDelayMs = 0 } = {}) {
    const dom = new FakeDom();
    const viewerDom = new FakeDom();
    const viewerDoc = viewerDom.document;
    const viewer = {
        animationDuration: loop, scrubs: [], captures: [], location: { href: '' },
        scrubTo(t) { this.scrubs.push(t); return Promise.resolve(); },
        async captureFrame({ time, width, height }) {
            this.captures.push(time);
            if (captureDelayMs) await flush(captureDelayMs);
            const px = new Uint8Array(width * height * 4).fill(this.captures.length % 256);
            return { width, height, data: Buffer.from(px).toString('base64') };
        },
    };
    dom.add('iframe', 'viewer-iframe', { contentWindow: viewer, contentDocument: viewerDoc, src: '' });
    dom.add('select', 'model-select', { value: 'my scan (v2).ply' });
    const container = dom.add('div', 'flythrough-section');

    const requests = [];
    const uploads = [];
    const frames = { queue: [] };
    const fetch = async (url, opts = {}) => {
        if (url.startsWith('data:')) {
            const buf = Buffer.from(url.split(',')[1], 'base64');
            return { arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length) };
        }
        requests.push({ url, method: opts.method || 'GET', body: opts.body });
        if (url.startsWith('/flythrough/info/')) {
            return { ok: true, json: async () => ({ mode: 'camera_path', num_cameras: 189 }) };
        }
        if (url === '/flythrough/export/start') {
            const req = JSON.parse(opts.body);
            const numFrames = Math.round(Math.max(30, req.duration) * req.fps);
            return { ok: true, json: async () => ({ session_id: 's1', num_frames: numFrames, width: 8, height: 4 }) };
        }
        if (url.endsWith('/frames')) {
            uploads.push(opts.body.getAll('frames').map((f) => f.name));
            return { ok: true, json: async () => ({}) };
        }
        if (url.endsWith('/finish')) {
            return finishStatus === 200
                ? { ok: true, blob: async () => new Blob(['mp4'], { type: 'video/mp4' }) }
                : { ok: false, status: finishStatus, json: async () => ({ message: 'Incomplete export' }) };
        }
        return { ok: true, json: async () => ({}) };
    };
    const URLx = class extends URL {
        static createObjectURL() { return 'blob:mp4'; }
        static revokeObjectURL() {}
    };
    const sandbox = loadScript(SCRIPT, dom, {
        fetch, URL: URLx, FLY_MIN_SECONDS: 30,
        requestAnimationFrame: (fn) => { frames.queue.push(fn); return frames.queue.length; },
        cancelAnimationFrame: () => { frames.queue.length = 0; },
    });
    sandbox.Flythrough.render(container);
    // rAF timestamps share performance.now()'s timebase, as in browsers
    const step = (ms) => {
        const fn = frames.queue.shift();
        if (fn) fn(frames.now = (frames.now ?? performance.now()) + ms);
    };
    return { dom, viewerDom, viewer, sandbox, requests, uploads, step, $: (id) => dom.document.getElementById(id) };
}

test('viewer URL loads the model with the flythrough settings, paused', () => {
    const { sandbox } = setup();
    const url = new URL(sandbox.Flythrough.viewerSrc('my scan (v2).ply'), 'http://x');
    assert.equal(url.pathname, '/viewer/index.html');
    assert.equal(url.searchParams.get('content'), '/models/my scan (v2).ply');
    assert.equal(decodeURIComponent(url.searchParams.get('settings')),
        '/flythrough/settings/my scan (v2).ply?duration=30');
    for (const flag of ['noanim', 'noui', 'webgl']) assert.ok(url.searchParams.has(flag), flag);
});

test('duration is clamped to 30..120 seconds and changing it reloads the viewer', () => {
    const { $ } = setup();
    for (const [input, expected] of [['10', '30'], ['500', '120'], ['45', '45'], ['abc', '30']]) {
        $('fly-duration').value = input;
        $('fly-duration').onchange();
        assert.equal(String($('fly-duration').value), expected);
        assert.match($('viewer-iframe').src, new RegExp(`duration%3D${expected}`));
    }
});

test('play advances the native animation track on every animation frame and loops', async () => {
    const { viewer, step, $ } = setup({ loop: 60 });
    $('fly-play').click();
    assert.equal($('fly-play').textContent, '⏸ Pause');
    for (let i = 0; i < 10; i++) step(100);
    assert.equal(viewer.scrubs.length, 10);
    assert.ok(viewer.scrubs.every((t, i) => i === 0 || t > viewer.scrubs[i - 1]), 'time moves forward');
    assert.ok(Math.abs(viewer.scrubs.at(-1) - 1.0) < 0.05);          // 10 frames x 100 ms
    for (let i = 0; i < 700; i++) step(100);                          // past the 60 s loop
    assert.ok(viewer.scrubs.at(-1) < 60);
    $('fly-play').click();
    assert.equal($('fly-play').textContent, '▶ Play');
    const count = viewer.scrubs.length;
    step(100);
    assert.equal(viewer.scrubs.length, count, 'paused: no more scrubbing');
});

test('seek jumps within the out-and-back loop; the label shows the one-way flythrough time', async () => {
    const { viewer, sandbox, $ } = setup({ loop: 60 });
    sandbox.Flythrough.onViewerLoaded();                           // loads camera_path info
    await flush();
    $('fly-progress').oninput({ target: { value: '250' } });
    assert.equal(viewer.scrubs.at(-1), 15);
    assert.equal($('fly-time').textContent, '0:15 / 0:30');
    $('fly-progress').oninput({ target: { value: '750' } });      // 45 s = returning leg
    assert.equal($('fly-time').textContent, '0:15 / 0:30 (returning)');
});

test('viewer load shows the path info; grabbing the camera pauses playback', async () => {
    const { viewerDom, sandbox, step, $ } = setup({ loop: 60 });
    sandbox.Flythrough.onViewerLoaded();
    await flush();
    assert.match($('fly-info').textContent, /Smoothed path through 189 trained cameras/);
    $('fly-play').click();
    step(16);
    assert.equal($('fly-status').textContent, 'Playing');
    viewerDom.fire('pointerdown');
    assert.equal($('fly-status').textContent, 'Paused (camera moved)');
    assert.equal($('fly-play').textContent, '▶ Play');
});

test('export captures every frame deterministically and streams them in order', async () => {
    const { viewer, requests, uploads, dom, $ } = setup();
    $('fly-duration').value = '10';                                   // below minimum -> 30 s
    await $('fly-export').onclick();
    const start = requests.find((r) => r.url === '/flythrough/export/start');
    assert.deepEqual(JSON.parse(start.body), { model: 'my scan (v2).ply', duration: 30, fps: 30, width: 1280, height: 720 });
    assert.equal(viewer.scrubs[0], 0, 'camera parked at t=0 before capturing');
    assert.equal(viewer.captures.length, 900);
    assert.ok(viewer.captures.every((t, i) => Math.abs(t - i / 30) < 1e-9), 'frame i captured at t = i / fps');
    const names = uploads.flat();
    assert.equal(names.length, 900);
    assert.deepEqual(names.slice(0, 2), ['f000000.jpg', 'f000001.jpg']);
    assert.equal(names.at(-1), 'f000899.jpg');
    assert.ok(uploads.every((batch) => batch.length <= 24));
    assert.ok(requests.some((r) => r.url === '/flythrough/export/s1/finish' && r.method === 'POST'));
    const link = dom.created.find((el) => el.tagName === 'A');
    assert.equal(link.download, 'my scan (v2)_flythrough.mp4');
    assert.ok(link.clicked);
    assert.match($('fly-status').textContent, /Exported 30s @ 30fps/);
    assert.equal($('fly-export').textContent, '📥 Export MP4');
});

test('export honours the selected fps', async () => {
    const { viewer, $ } = setup();
    $('fly-fps').value = '24';
    await $('fly-export').onclick();
    assert.equal(viewer.captures.length, 720);
});

test('a failed finish is reported and the session is not left dangling', async () => {
    const { requests, $ } = setup({ finishStatus: 400 });
    await $('fly-export').onclick();
    assert.match($('fly-status').textContent, /Export failed: Incomplete export/);
    assert.equal($('fly-play').disabled, false);
});

test('export can be cancelled; the server session is aborted', async () => {
    const { requests, viewer, $ } = setup({ captureDelayMs: 5 });
    const done = $('fly-export').onclick();
    await flush(1500);                                               // past the settle delay, mid-capture
    $('fly-export').onclick();                                       // second click = cancel
    await done;
    assert.ok(viewer.captures.length < 900);
    assert.ok(requests.some((r) => r.url === '/flythrough/export/s1' && r.method === 'DELETE'));
    assert.equal($('fly-status').textContent, 'Export cancelled');
});

test('export waits for the viewer to be ready', async () => {
    const { viewer, requests, $ } = setup();
    delete viewer.captureFrame;
    await $('fly-export').onclick();
    assert.equal($('fly-status').textContent, 'Viewer still loading...');
    assert.equal(requests.length, 0);
});
