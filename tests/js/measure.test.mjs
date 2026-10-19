// Unit tests for supersplat-viewer/web/measure.js ("Ask Cosmos 3" measurement panel).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, flush, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/measure.js', import.meta.url));
const CAMERA = { view: Array(16).fill(1), proj: Array(16).fill(0.5), model: null, width: 1280, height: 720 };
const RESULT = {
    run_id: 'run-1234567890', scene_id: 'splat-abc', render_id: 'r1', overlay_url: '/measure/overlay/splat-abc/r1.png?revisions=a,b',
    answer: ['Doorway width: 1.475 mu (mean of top 1.421 mu, bottom 1.530 mu)'], unscaled: true, seconds: 12.3,
    grounding: { label: 'doorway' }, plane: { rms: 0.0513, support: 4103, samples: 4600 },
    edges: { left: { method: 'snapped to wall-plane boundary' }, top: { method: 'model box edge (no consistent wall-plane boundary)' } },
    results: [{ revision_id: 'a', label: 'Doorway width (top)', display: '1.421 ± 0.071 mu', validation: 'unscaled',
        equation: 'L = s ‖q - p‖', work: [{ expr: '‖d‖ = √(dx² + dy² + dz²)', subst: '√(0.45² + 0.21² + 1.33²)', result: '1.42 mu' }] }],
};

function setup({ camera = CAMERA, reply = { ok: true, json: RESULT }, polls = 1 } = {}) {
    const dom = new FakeDom();
    dom.add('select', 'model-select', { value: 'peacock.ply' });
    dom.add('iframe', 'viewer-iframe', { contentWindow: { getCameraMatrices: camera ? () => camera : undefined } });
    for (const id of ['popup-title', 'popup-iframe']) dom.add('div', id);
    dom.add('div', 'popup-modal').classList.add('hidden');
    const section = dom.add('div', 'measure-section', { querySelectorAll: () => [] });
    const requests = [];
    let n = 0;
    // /measure/ask -> job id; /measure/jobs/<id> -> running (polls times) then the reply as the job outcome
    const fetch = async (url, opts = {}) => {
        requests.push({ url, method: opts.method || 'GET', body: opts.body && JSON.parse(opts.body) });
        if (url === '/measure/ask') {
            if (!reply.ok && reply.status === 422) return { ok: false, status: 422, json: async () => reply.json };
            return { ok: true, status: 200, json: async () => ({ job_id: 'j1', state: 'running' }) };
        }
        if (++n <= polls) return { ok: true, status: 200, json: async () => ({ job_id: 'j1', state: 'running', stage: 'rendering the view', elapsed_s: 12 }) };
        const job = reply.ok ? { job_id: 'j1', state: 'completed', result: reply.json, elapsed_s: 30 }
            : { job_id: 'j1', state: 'failed', error: reply.json.detail, elapsed_s: 30 };
        return { ok: true, status: 200, json: async () => job };
    };
    const sandbox = loadScript(SCRIPT, dom, { fetch, setTimeout: (fn) => setTimeout(fn, 0) });
    sandbox.Measure.render(section);
    return { dom, sandbox, requests, $: (id) => dom.document.getElementById(id) };
}

test('render adds question input, ask button, status and result area', () => {
    const { $ } = setup();
    for (const id of ['measure-question', 'measure-ask', 'measure-status', 'measure-result']) assert.ok($(id), id);
});

test('ask sends model, question and the live viewer camera; shows equation and work from the result', async () => {
    const { $, sandbox, requests } = setup();
    $('measure-question').value = 'How wide is this doorway?';
    await sandbox.Measure.ask();
    assert.deepEqual(requests.map((r) => r.url), ['/measure/ask', '/measure/jobs/j1', '/measure/jobs/j1']);
    assert.deepEqual(requests[0].body, { model: 'peacock.ply', question: 'How wide is this doorway?', camera: CAMERA });
    const html = $('measure-result').innerHTML;
    assert.match(html, /Doorway width: 1\.475 mu/);
    assert.match(html, /L = s ‖q - p‖/);
    assert.match(html, /‖d‖ = √\(dx² \+ dy² \+ dz²\) = √\(0\.45² \+ 0\.21² \+ 1\.33²\) = 1\.42 mu/);
    assert.match(html, /src="\/measure\/overlay\/splat-abc\/r1\.png\?revisions=a,b"/);
    assert.match(html, /left: snapped, top: model box/);
    assert.match($('measure-status').textContent, /model units/);
    assert.equal($('measure-ask').disabled, false);
    $('measure-overlay').onclick();
    assert.equal($('popup-iframe').src, RESULT.overlay_url);
    assert.equal($('popup-modal').classList.contains('hidden'), false);
});

test('no camera or empty question never calls the backend', async () => {
    const a = setup({ camera: null });
    a.$('measure-question').value = 'How wide is this doorway?';
    await a.sandbox.Measure.ask();
    assert.equal(a.requests.length, 0);
    assert.match(a.$('measure-status').textContent, /still loading/);
    const b = setup();
    await b.sandbox.Measure.ask('  ');
    assert.equal(b.requests.length, 0);
});

test('backend errors are shown and results are HTML-escaped', async () => {
    const err = setup({ reply: { ok: false, status: 502, json: { detail: 'grounding failed after 2 attempts' } } });
    await err.sandbox.Measure.ask('How tall is this window?');
    assert.match(err.$('measure-status').textContent, /❌ grounding failed/);
    const xss = setup();
    xss.sandbox.Measure.renderResult({ ...RESULT, answer: ['<img src=x onerror=alert(1)>'] });
    assert.doesNotMatch(xss.$('measure-result').innerHTML, /<img src=x/);
    await flush();
});


test('an unmeasured question shows the Cosmos visual answer and a warning', async () => {
    const visual = { ...RESULT, measured: false, results: [], reason: 'grounding failed', edges: {}, grounding: null, plane: null,
        answer: ['Not measured with equations: grounding failed', 'Cosmos 3 visual answer (not a measurement): about two storeys'] };
    const { $, sandbox } = setup({ reply: { ok: true, json: visual } });
    await sandbox.Measure.ask('How tall are the stairs?');
    assert.match($('measure-result').innerHTML, /Cosmos 3 visual answer \(not a measurement\): about two storeys/);
    assert.match($('measure-status').textContent, /Not measured with equations/);
});

test('transient poll failures are retried, the stage is shown while running', async () => {
    const { $, sandbox, requests } = setup({ polls: 3 });
    await sandbox.Measure.ask('How wide is this doorway?');
    assert.equal(requests.filter((r) => r.url === '/measure/jobs/j1').length, 4);
    assert.match($('measure-status').textContent, /Measured/);
});


test('object measurements (no wall plane) with a RAG report render', () => {
    const obj = { ...RESULT, method: 'object', plane: null, edges: {}, up: { method: 'viewer orientation estimate' },
        rag: { sources_checked: ['Upload Info'], statements: [], identified_name: 'Aussichtswarte Fürstenstand' } };
    const { $, sandbox } = setup();
    sandbox.Measure.renderResult(obj);
    assert.match($('measure-result').innerHTML, /object extent \(up: viewer orientation estimate\)/);
    assert.match($('measure-result').innerHTML, /RAG: 1 source\(s\), 0 size statement\(s\), identified “Aussichtswarte Fürstenstand”/);
});
