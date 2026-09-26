// End-to-end browser tests for the SuperSplat viewer (:8086): WebGL navigation, native
// flythrough playback, in-browser MP4 export, segmentation + metadata linking, RAG chat.
// Run on the host (needs Chromium + GPU): node --test tests/browser/
import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { URL_8086, inspectMp4, launch, sleep } from './cdp.mjs';

let page;
const viewer = 'document.getElementById("viewer-iframe").contentWindow';
const camera = `JSON.stringify(${viewer}.getCameraMatrices().view.map((v) => +v.toFixed(4)))`;

before(async () => {
    page = await launch();
    await page.open(`${URL_8086}/`);
    await page.waitFor(`typeof ${viewer}.captureFrame === "function"`, 180000);   // model on the GPU
    await sleep(1500);                                                              // settle into the anim camera
});
after(() => page?.close());

test('mouse drag orbits and the wheel zooms the camera (client-side navigation)', async () => {
    const start = await page.eval(camera);
    await page.drag(1000, 500, 1200, 560);
    await sleep(1200);
    const orbited = await page.eval(camera);
    assert.notEqual(orbited, start, 'drag changes the view');
    await page.wheel(1000, 500, -400);
    await sleep(1200);
    assert.notEqual(await page.eval(camera), orbited, 'wheel changes the view');
});

test('flythrough plays the trained-camera track natively and can be paused', async () => {
    assert.match(await page.eval('document.getElementById("fly-info").textContent'), /trained cameras/);
    const before = await page.eval(camera);
    await page.eval('document.getElementById("fly-play").click()');
    await sleep(3000);
    const time = await page.eval('document.getElementById("fly-time").textContent');
    assert.match(time, /^0:0[2-4] \/ 0:30$/, time);
    assert.notEqual(await page.eval(camera), before, 'camera moves along the path');
    await page.eval('document.getElementById("fly-play").click()');
    const paused = await page.eval(camera);
    await sleep(800);
    assert.equal(await page.eval(camera), paused, 'paused camera stays put');
});

test('Export MP4 renders every frame in the browser and downloads a 30 s H.264 video', async () => {
    await page.interceptMp4();
    await page.eval('document.getElementById("fly-size").value = "1024x768"');
    await page.eval('document.getElementById("fly-export").click()');
    const status = await page.waitFor(`(() => { const s = document.getElementById("fly-status").textContent;
        return /Exported|failed|cancelled/.test(s) && s; })()`, 900000, 2000);
    assert.match(status, /Exported 30s @ 30fps \(1024x768\)/, status);
    const mp4 = inspectMp4(await page.takeMp4());
    assert.ok(mp4.h264 && mp4.faststart, JSON.stringify(mp4));
    assert.ok(mp4.seconds >= 29.95, JSON.stringify(mp4));
});

test('SAM3 segmentation from the viewer, then metadata linked to the detected object', async () => {
    await page.eval(`${viewer}.scrubTo(0)`);                         // first trained view shows the table
    await sleep(1500);
    await page.eval('document.getElementById("seg-prompt").value = "table"');
    await page.eval('segmentWithText()');
    const status = await page.waitFor(`(() => { const s = document.getElementById("seg-status").textContent;
        return /Found|❌/.test(s) && s; })()`, 300000, 1000);
    assert.match(status, /Found [1-9]\d* object/, status);
    await page.eval('document.querySelector("#seg-object-list .meta-link").click()');
    await page.waitFor('!document.getElementById("meta-panel").classList.contains("hidden")');
    await page.eval(`document.getElementById("meta-label").value = "e2e-table";
        document.getElementById("meta-text").value = "Linked from the browser test";
        document.getElementById("meta-save").click()`);
    const model = await page.eval('document.getElementById("model-select").value');
    const items = await page.waitFor(`fetch("/metadata?model=${model}").then((r) => r.json())
        .then((d) => d.items.find((i) => i.label === "e2e-table"))`, 10000);
    assert.equal(items.text, 'Linked from the browser test');
});

test('Ask AI streams an answer grounded on the scene', async () => {
    await page.eval('document.getElementById("rag-ask-btn").click()');
    await page.waitFor('/LLM ready/.test(document.getElementById("rag-llm-status").textContent)', 30000);
    assert.match(await page.eval('document.getElementById("rag-summary").textContent'), /Segments: \d+ SAM3 object/);
    await page.eval(`document.getElementById("rag-query").value = "What objects are in this scene?";
        document.getElementById("rag-send").click()`);
    const answer = await page.waitFor(`(() => { const b = document.querySelectorAll("#rag-chat .rag-assistant span");
        const t = b.length && b[b.length - 1].textContent; return !document.getElementById("rag-send").disabled
        && t && t !== "Thinking..." && t; })()`, 180000, 1000);
    assert.ok(!answer.startsWith('Error'), answer);
    assert.ok(answer.length > 10, answer);
});

test('no uncaught page errors', () => {
    assert.deepEqual(page.errors, []);
});
