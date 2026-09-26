// End-to-end browser tests for the fVDB image viewer (:8085): keyboard/slider navigation,
// flythrough playback, MP4 export download, auto segmentation + object metadata, RAG chat.
// Run on the host: node --test tests/browser/
import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { URL_8085, inspectMp4, launch, sleep } from './cdp.mjs';

let page;
const imgSrc = 'document.getElementById("render").src';

before(async () => {
    page = await launch();
    await page.open(`${URL_8085}/`);
    await page.waitFor(`${imgSrc}.startsWith("blob:")`, 120000);        // first server render shown
});
after(() => page?.close());

test('arrow keys orbit and +/- zoom request new server renders', async () => {
    const first = await page.eval(imgSrc);
    await page.key('ArrowRight', 'ArrowRight', 39);
    await page.waitFor(`document.getElementById("az-val").textContent === "10"`, 10000);
    await page.waitFor(`${imgSrc} !== ${JSON.stringify(first)}`, 20000);
    const orbited = await page.eval(imgSrc);
    await page.key('ArrowUp', 'ArrowUp', 38);
    await page.waitFor(`document.getElementById("el-val").textContent === "10"`, 10000);
    await page.key('+', 'Equal', 187);
    await page.waitFor(`${imgSrc} !== ${JSON.stringify(orbited)}`, 20000);
    assert.notEqual(await page.eval('document.getElementById("zoom-val").textContent'), '1.0');
});

test('flythrough plays frames along the camera path and pauses', async () => {
    assert.equal(await page.eval('document.getElementById("flyDuration").value'), '30');
    await page.eval('toggleFlythrough()');
    await sleep(4000);
    const label = await page.eval('document.getElementById("flyFrameLabel").textContent');
    const frame = Number(label.match(/Frame (\d+) \/ 900/)[1]);
    assert.ok(frame >= 20, label);                                      // ~30 fps target, network permitting
    await page.eval('toggleFlythrough()');
    const pausedAt = await page.eval('document.getElementById("flyFrameLabel").textContent');
    await sleep(800);
    assert.equal(await page.eval('document.getElementById("flyFrameLabel").textContent'), pausedAt);
});

test('Export MP4 downloads a 30 s H.264 video', async () => {
    await page.interceptMp4();
    await page.eval('document.getElementById("flyDuration").value = "10"');    // clamped to 30
    await page.eval('exportFlythrough()');
    await page.waitFor('window.__mp4 !== null', 300000, 1000);
    const mp4 = inspectMp4(await page.takeMp4());
    assert.ok(mp4.h264 && mp4.faststart, JSON.stringify(mp4));
    assert.ok(mp4.seconds >= 29.95, JSON.stringify(mp4));
    assert.equal(await page.eval('document.getElementById("flyDuration").value'), '30');
});

test('Auto Segment finds objects; typed metadata is linked to one of them', async () => {
    await page.eval('runAutoSegmentation()');
    await page.waitFor('fetch("/segment/labels").then((r) => r.json()).then((d) => d.num_segments > 0)', 180000, 2000);
    await page.eval(`openObjectSummary(0);
        document.getElementById("obj-summary-name").value = "e2e-object";
        document.getElementById("obj-summary-text").value = "Linked from the browser test";
        saveObjectSummary()`);
    const summary = await page.waitFor(`fetch("/object_summary/0").then((r) => r.json())
        .then((d) => d.summary && d.summary.label === "e2e-object" && d.summary)`, 10000);
    assert.equal(summary.text, 'Linked from the browser test');
});

test('Ask AI streams an answer grounded on the scene', async () => {
    await page.eval('showSummary()');
    await page.waitFor('/LLM ready/.test(document.getElementById("rag-llm-status").textContent)', 30000);
    await page.eval(`document.getElementById("rag-query-input").value = "What objects are in the scene?"; sendRagQuery()`);
    const answer = await page.waitFor(`(() => { const s = document.querySelectorAll("#rag-chat-history span");
        const t = s.length && s[s.length - 1].textContent; return !document.getElementById("rag-send-btn").disabled
        && t && t !== "Thinking..." && t; })()`, 180000, 1000);
    assert.ok(!answer.startsWith('Error'), answer);
    assert.ok(answer.length > 10, answer);
});

test('no uncaught page errors', () => {
    assert.deepEqual(page.errors, []);
});
