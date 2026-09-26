// Tiny Chrome DevTools Protocol driver for the viewer browser tests (no npm deps).
// Launches the host Chromium with GPU WebGL (ANGLE/Vulkan) so the SuperSplat viewer
// renders exactly as it does for users.
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const URL_8085 = process.env.VIEWER_8085_URL || 'http://localhost:18085';
export const URL_8086 = process.env.VIEWER_8086_URL || 'http://localhost:18086';
const CHROME = process.env.CHROME_BIN || '/usr/bin/chromium-browser';
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function launch({ port = 9300 + Math.floor(Math.random() * 500), width = 1600, height = 1000 } = {}) {
    const proc = spawn(CHROME, ['--headless=new', `--remote-debugging-port=${port}`, '--no-first-run', '--no-sandbox',
        `--window-size=${width},${height}`, '--use-angle=vulkan', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
        `--user-data-dir=${mkdtempSync(join(tmpdir(), 'viewer-e2e-'))}`, 'about:blank'], { stdio: 'ignore' });
    let ws;
    for (let i = 0; i < 100 && !ws; i++) {
        try {
            const page = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page');
            ws = new WebSocket(page.webSocketDebuggerUrl);
            await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
        } catch { ws = null; await sleep(200); }
    }
    if (!ws) { proc.kill(); throw new Error(`Chromium did not start (${CHROME})`); }
    return new Page(proc, ws);
}

class Page {
    constructor(proc, ws) {
        this.proc = proc; this.ws = ws; this.id = 0; this.pending = new Map(); this.errors = [];
        ws.onmessage = (m) => {
            const msg = JSON.parse(m.data);
            if (msg.id && this.pending.has(msg.id)) { this.pending.get(msg.id)(msg); this.pending.delete(msg.id); }
            if (msg.method === 'Runtime.exceptionThrown') this.errors.push(msg.params.exceptionDetails.exception?.description);
        };
    }
    send(method, params = {}) {
        return new Promise((resolve) => { const id = ++this.id; this.pending.set(id, resolve); this.ws.send(JSON.stringify({ id, method, params })); });
    }
    async open(url) {
        await this.send('Runtime.enable');
        await this.send('Network.enable');
        await this.send('Network.setCacheDisabled', { cacheDisabled: true });
        await this.send('Page.navigate', { url });
        await this.waitFor('document.readyState === "complete"', 30000);
    }
    async eval(expression) {
        const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
        if (r.result.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || expression);
        return r.result.result.value;
    }
    async waitFor(expression, timeout = 60000, interval = 250) {
        const end = Date.now() + timeout;
        while (Date.now() < end) {
            try { const v = await this.eval(expression); if (v) return v; } catch { /* page still loading */ }
            await sleep(interval);
        }
        throw new Error(`timed out waiting for: ${expression}`);
    }
    async drag(x0, y0, x1, y1, steps = 12) {
        await this.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0, y: y0 });
        await this.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: x0, y: y0, button: 'left', clickCount: 1 });
        for (let i = 1; i <= steps; i++) {
            await this.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: x0 + ((x1 - x0) * i) / steps,
                y: y0 + ((y1 - y0) * i) / steps, button: 'left', buttons: 1 });
            await sleep(16);
        }
        await this.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: x1, y: y1, button: 'left', clickCount: 1 });
    }
    async wheel(x, y, deltaY) {
        await this.send('Input.dispatchMouseEvent', { type: 'mouseWheel', x, y, deltaX: 0, deltaY });
    }
    async key(key, code = key, keyCode = 0) {
        for (const type of ['keyDown', 'keyUp']) {
            await this.send('Input.dispatchKeyEvent', { type, key, code, windowsVirtualKeyCode: keyCode });
        }
    }
    // Capture the next MP4 the page hands to URL.createObjectURL (the download path)
    async interceptMp4() {
        await this.eval(`(() => { const o = URL.createObjectURL; window.__mp4 = null;
            URL.createObjectURL = (b) => { if (b && b.type === 'video/mp4') window.__mp4 = b; return o.call(URL, b); }; })()`);
    }
    async takeMp4() {
        const b64 = await this.eval(`(async () => { const b = new Uint8Array(await window.__mp4.arrayBuffer()); let s = '';
            for (let i = 0; i < b.length; i += 0x8000) s += String.fromCharCode.apply(null, b.subarray(i, i + 0x8000));
            return btoa(s); })()`);
        return Buffer.from(b64, 'base64');
    }
    close() { try { this.ws.close(); } catch { /* already closed */ } this.proc.kill(); }
}

// Minimal MP4 inspection: H.264 track, moov before mdat (faststart), duration from mvhd
export function inspectMp4(buf) {
    const find = (tag) => buf.indexOf(Buffer.from(tag, 'ascii'));
    const mvhd = find('mvhd');
    const version = buf[mvhd + 4];
    const timescale = version === 1 ? buf.readUInt32BE(mvhd + 24) : buf.readUInt32BE(mvhd + 16);
    const duration = version === 1 ? Number(buf.readBigUInt64BE(mvhd + 28)) : buf.readUInt32BE(mvhd + 20);
    return { h264: find('avc1') > 0 && find('avcC') > 0, faststart: find('moov') >= 0 && find('moov') < find('mdat'),
        seconds: duration / timescale, bytes: buf.length };
}
