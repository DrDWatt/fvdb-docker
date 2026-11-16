// Unit tests for web/cosmos_activity.js and web/artifixer.js helpers (Nov 16 2026).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const COSMOS = fileURLToPath(new URL('../../supersplat-viewer/web/cosmos_activity.js', import.meta.url));
const AF = fileURLToPath(new URL('../../supersplat-viewer/web/artifixer.js', import.meta.url));
const load = (path) => loadScript(path, new FakeDom(), { fetch: async () => ({ ok: true, json: async () => ({}) }), setTimeout: () => 0, clearTimeout: () => {} });

test('Brev login badge tells the user what to do when logged out', () => {
    const { CosmosActivity: c } = load(COSMOS);
    assert.equal(c.brevBadge({ logged_in: true, detail: 'ok' }).text, 'Brev: logged in');
    const out = c.brevBadge({ logged_in: false, detail: 'expired' });
    assert.equal(out.text, 'Brev: logged out');
    assert.match(out.hint, /brev login/);
    assert.equal(c.brevBadge(null).text, 'Brev: unknown');
});

test('NIM line explains each state, and starting/in-flight polls fast', () => {
    const { CosmosActivity: c } = load(COSMOS);
    assert.match(c.nimLine({ state: 'stopped', instance: 'okay-blue-platypus' }), /starts on the next question/);
    assert.match(c.nimLine({ state: 'starting', detail: 'loading the model' }), /starting: loading the model/);
    assert.equal(c.busy({ state: 'starting' }), true);
    assert.equal(c.busy({ state: 'ready', inflight: 1 }), true);
    assert.equal(c.busy({ state: 'ready', inflight: 0 }), false);
});

test('calls render prompt, response and errors (escaped)', () => {
    const { CosmosActivity: c } = load(COSMOS);
    const html = c.renderCalls([
        { state: 'ok', started: 100, endpoint: 'brev', seconds: 12, prompt: 'How wide is <the> doorway?', media: ['view.png'], response: '0.9 m' },
        { state: 'error', started: 50, endpoint: 'brev', seconds: 1, prompt: 'desk height', media: [], error: 'brev CLI is logged out' },
    ], 160);
    assert.match(html, /How wide is &lt;the&gt; doorway\?/);
    assert.match(html, /0\.9 m/);
    assert.match(html, /brev CLI is logged out/);
    assert.match(html, /view\.png/);
    assert.equal(c.ago(100, 160), '1 min ago');
});

test('ArtiFixer3D+ needs the capture; cleanup works on any listed PLY', () => {
    const { ArtiFixer: a } = load(AF);
    assert.equal(a.canRun({ name: 'Car_model.ply', source: 'Car', derived: false }, 'artifixer3d_plus').ok, true);
    assert.equal(a.canRun({ name: 'x.ply', source: null, derived: false }, 'artifixer3d_plus').ok, false);
    assert.equal(a.canRun({ name: 'y_artifixer3d.ply', source: null, derived: true }, 'artifixer3d_plus').ok, false);
    assert.equal(a.canRun({ name: 'y_artifixer3d.ply', source: null, derived: true }, 'cleanup').ok, true);
    assert.equal(a.canRun(null, 'cleanup').ok, false);
    assert.equal(a.duration({ started: 0, finished: 5400 }, 9999), '1 h 30 min');
    assert.match(a.renderJobs([{ _id: 'j1', state: 'running', ply: 'Car_model.ply', endpoint_name: 'Brev', step: 'steps 1-4', started: 0 }], 120),
        /data-cancel="j1"/);
});
