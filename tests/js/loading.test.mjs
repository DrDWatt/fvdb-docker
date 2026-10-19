// Unit tests for supersplat-viewer/web/loading.js (real model-loading progress overlay).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/loading.js', import.meta.url));

function setup(progressSeq) {
    const dom = new FakeDom();
    for (const id of ['loading-overlay', 'loading-text', 'loading-bar']) dom.add('div', id);
    const viewer = { ready: false };
    const win = { location: { href: 'http://x/viewer/index.html?content=/models/old.ply' },
        getCameraMatrices: () => (viewer.ready ? { model: [1] } : { model: null }) };
    dom.add('iframe', 'viewer-iframe', { contentWindow: win });
    const texts = [];
    let i = 0;
    const fetch = async () => {
        const p = progressSeq[Math.min(i++, progressSeq.length - 1)];
        texts.push(dom.document.getElementById('loading-text').textContent);
        if (i === progressSeq.length) { viewer.ready = true; win.location.href = `http://x/viewer/index.html?content=${encodeURIComponent('/models/' + model())}`; }
        return { json: async () => p };
    };
    const fastTimeout = (fn) => setTimeout(fn, 0);
    let current = '';
    const model = () => current;
    const sandbox = loadScript(SCRIPT, dom, { fetch, setTimeout: fastTimeout });
    return { dom, sandbox, texts, setModel: (m) => { current = m; }, win, viewer, $: (id) => dom.document.getElementById(id) };
}

test('shows real download progress, then build, then ready with throughput', async () => {
    const now = 1000;
    const p = (sent, done = false) => ({ started: true, started_at: now, server_now: now + 1, bytes_sent: sent, total: 710e6, done,
        seconds: done ? 14.2 : 5, mb_per_s: done ? 50 : 40 });
    const { sandbox, texts, $, setModel } = setup([p(200e6), p(500e6), p(710e6, true), p(710e6, true)]);
    setModel('frankfurt.ply');
    const res = await sandbox.ModelLoading.track('frankfurt.ply');
    assert.ok(texts.some((t) => /Downloading frankfurt\.ply: 200 \/ 710 MB · 40 MB\/s/.test(t)), texts.join(' | '));
    assert.ok(texts.some((t) => /Building 710 MB of splats/.test(t)));
    assert.match($('loading-text').textContent, /Ready in .*download 14\.2s at 50 MB\/s/);
    assert.equal(res.download.done, true);
});

test('a stale download record (older than this load) is ignored', async () => {
    const stale = { started: true, started_at: 0, server_now: 10_000, bytes_sent: 5, total: 5, done: true, seconds: 1, mb_per_s: 5 };
    const { sandbox, $, setModel } = setup([stale, stale]);
    setModel('x.ply');
    await sandbox.ModelLoading.track('x.ply');
    assert.match($('loading-text').textContent, /Ready in .* from browser cache/);
});


test('the previous model still in the iframe is not reported as ready', () => {
    const { sandbox, viewer } = setup([{ started: false }]);
    viewer.ready = true;  // old viewer document has a splat
    assert.equal(sandbox.ModelLoading.viewerReady('graz.ply'), false);
    assert.equal(sandbox.ModelLoading.viewerReady('old.ply'), true);
});
