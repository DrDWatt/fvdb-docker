// Unit tests for supersplat-viewer/web/rag.js (RAG chat, model info upload, metadata links).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { File } from 'node:buffer';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/rag.js', import.meta.url));

// SSE body delivered in awkward chunks (a JSON line split across reads)
const sseChunks = (tokens) => {
    const body = tokens.map((t) => `data: ${JSON.stringify({ token: t })}\n\n`).join('') +
        `data: ${JSON.stringify({ done: true })}\n\n`;
    return [body.slice(0, 17), body.slice(17, 50), body.slice(50)];
};

function setup(routes = {}) {
    const dom = new FakeDom();
    dom.add('select', 'model-select', { value: 'scene.ply' });
    const requests = [];
    const fetch = async (url, opts = {}) => {
        requests.push({ url, method: opts.method || 'GET', body: opts.body });
        const route = Object.entries(routes).find(([prefix]) => url.startsWith(prefix));
        if (!route) return { ok: true, json: async () => ({}) };
        const r = typeof route[1] === 'function' ? route[1](url, opts) : route[1];
        if (r.chunks) {
            const enc = new TextEncoder();
            const stream = new ReadableStream({ start(c) { r.chunks.forEach((x) => c.enqueue(enc.encode(x))); c.close(); } });
            return { ok: true, body: stream };
        }
        return { ok: r.ok ?? true, status: r.status ?? 200, json: async () => r.json };
    };
    const sandbox = loadScript(SCRIPT, dom, { fetch });
    dom.fire('DOMContentLoaded');
    return { dom, sandbox, requests, $: (id) => dom.document.getElementById(id) };
}

test('mount adds the RAG pane, chat and upload modals and the metadata panel', () => {
    const { $ } = setup();
    for (const id of ['rag-pane', 'rag-ask-btn', 'rag-upload-btn', 'rag-modal', 'rag-query', 'upload-info-modal',
        'meta-panel', 'meta-label', 'meta-text', 'meta-files', 'meta-save', 'meta-upload']) {
        assert.ok($(id), id);
    }
    assert.ok($('rag-modal').classList.contains('hidden') && $('meta-panel').classList.contains('hidden'));
});

test('Ask AI shows the linked scene context and LLM status', async () => {
    const { $ } = setup({
        '/rag/context': { json: { model_name: 'scene.ply', model_summary: 'Supercharger site', segments_count: 2,
            segment_labels: ['Tesla'], extractions_count: 1, documents_count: 3 } },
        '/rag/status': { json: { available: true, model: 'nemotron-mini:latest' } },
    });
    await $('rag-ask-btn').onclick();
    assert.ok(!$('rag-modal').classList.contains('hidden'));
    assert.equal($('rag-summary').textContent, 'Model: scene.ply\nSupercharger site\nSegments: 2 SAM3 object(s) (Tesla)\n' +
        'Extractions: 1 3D extraction(s)\nDocuments: 3 linked');
    assert.match($('rag-llm-status').innerHTML, /LLM ready.*nemotron-mini:latest/);
});

test('chat streams the answer token by token and keeps history', async () => {
    const { $, requests } = setup({ '/rag/query': { chunks: sseChunks(['A red ', 'Tesla ', 'at stall 4.']) } });
    $('rag-query').value = 'Which car?';
    await $('rag-send').onclick();
    const chat = $('rag-chat').children;
    assert.equal(chat.length, 2);
    assert.equal(chat[0].children[0].textContent, 'Which car?');
    assert.equal(chat[1].children[0].textContent, 'A red Tesla at stall 4.');
    const body = JSON.parse(requests.find((r) => r.url === '/rag/query').body);
    assert.deepEqual(body, { query: 'Which car?', model: 'scene.ply', history: [] });

    $('rag-query').value = 'And the charger?';
    await $('rag-send').onclick();
    const second = JSON.parse(requests.filter((r) => r.url === '/rag/query')[1].body);
    assert.deepEqual(second.history, [{ role: 'user', content: 'Which car?' },
        { role: 'assistant', content: 'A red Tesla at stall 4.' }]);
    assert.equal($('rag-send').disabled, false);
});

test('chat shows streamed errors instead of an answer', async () => {
    const { $ } = setup({ '/rag/query': { chunks: [`data: ${JSON.stringify({ error: 'Ollama error 500' })}\n\n`] } });
    $('rag-query').value = 'hi';
    await $('rag-send').onclick();
    const bubble = $('rag-chat').children[1].children[0];
    assert.equal(bubble.textContent, 'Error: Ollama error 500');
    assert.equal(bubble.style.color, '#dc3545');
});

test('Upload Info sends the document for the current model', async () => {
    const { $, requests } = setup({ '/upload_summary': { json: { message: 'Summary uploaded successfully' } } });
    $('rag-upload-btn').onclick();
    await $('upload-info-send').onclick();
    assert.equal($('upload-info-status').textContent, 'Please select a file');
    $('upload-info-file').files = [new File(['8 stalls'], 'site.md', { type: 'text/markdown' })];
    await $('upload-info-send').onclick();
    const req = requests.find((r) => r.url.startsWith('/upload_summary'));
    assert.equal(req.url, '/upload_summary?model=scene.ply');
    assert.equal(req.body.get('file').name, 'site.md');
    assert.equal($('upload-info-status').textContent, '✅ Summary uploaded successfully');
});

test('metadata panel loads, saves typed fields and uploads documents for an object', async () => {
    const summary = { label: 'Tesla', text: 'Red Model S', training_data: { files_count: 1, last_upload: 'now' },
        files: [{ idx: 0, name: 'spec.txt', size: '0.1 KB' }] };
    const { $, requests, sandbox } = setup({
        '/metadata/object/scene.ply%7Ccar%7C0/upload': { json: { summary: { ...summary, files: [...summary.files,
            { idx: 1, name: 'photo.jpg', size: '2 KB' }], training_data: { files_count: 2, last_upload: 'now' } } } },
        '/metadata/object/scene.ply%7Ccar%7C0': (url, opts) => (opts.method === 'POST'
            ? { json: { status: 'ok' } } : { json: { summary } }),
    });
    await sandbox.MetadataLinks.open('object', 'scene.ply|car|0', 'car #1');
    assert.ok(!$('meta-panel').classList.contains('hidden'));
    assert.equal($('meta-title').textContent, '📋 Object Metadata');
    assert.equal($('meta-label').value, 'Tesla');
    assert.equal($('meta-text').value, 'Red Model S');
    assert.equal($('meta-file-list').children[0].children[0].attributes.href, '/metadata/object/scene.ply%7Ccar%7C0/file/0');

    $('meta-label').value = 'Tesla Model S';
    await $('meta-save').onclick();
    const save = requests.find((r) => r.method === 'POST' && r.url === '/metadata/object/scene.ply%7Ccar%7C0');
    assert.deepEqual([save.body.get('label'), save.body.get('text'), save.body.get('model')],
        ['Tesla Model S', 'Red Model S', 'scene.ply']);

    $('meta-files').files = [new File(['jpg'], 'photo.jpg', { type: 'image/jpeg' })];
    await $('meta-upload').onclick();
    const upload = requests.find((r) => r.url.endsWith('/upload'));
    assert.equal(upload.body.getAll('files')[0].name, 'photo.jpg');
    assert.equal(upload.body.get('model'), 'scene.ply');
    assert.equal($('meta-file-list').children.length, 2);
    assert.match($('meta-training').innerHTML, /2 document\(s\) linked/);
});

test('extraction metadata starts from the default label when nothing is linked yet', async () => {
    const { $, sandbox } = setup({ '/metadata/extraction/ab12': { json: { summary: null } } });
    await sandbox.MetadataLinks.open('extraction', 'ab12', 'Extraction ab12');
    assert.equal($('meta-title').textContent, '📋 Extraction Metadata');
    assert.equal($('meta-label').value, 'Extraction ab12');
    assert.equal($('meta-training').innerHTML, 'No documents linked');
    sandbox.MetadataLinks.close();
    assert.ok($('meta-panel').classList.contains('hidden'));
});
