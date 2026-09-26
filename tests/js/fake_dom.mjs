// Minimal DOM + browser globals for unit-testing the viewer's plain-JS modules in Node.
// Elements are registered from the HTML strings the modules inject (by id), which is
// all web/flythrough.js and web/rag.js rely on.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

class ClassList {
    constructor() { this.set = new Set(); }
    add(...c) { c.forEach((x) => this.set.add(x)); }
    remove(...c) { c.forEach((x) => this.set.delete(x)); }
    contains(c) { return this.set.has(c); }
    replace(a, b) { if (this.set.delete(a)) this.set.add(b); }
    toggle(c) { this.set.has(c) ? this.set.delete(c) : this.set.add(c); }
}

export class FakeElement {
    constructor(dom, tag, id = '') {
        Object.assign(this, { dom, tagName: tag.toUpperCase(), id, children: [], style: {}, attributes: {},
            value: '', textContent: '', disabled: false, files: [], listeners: {}, classList: new ClassList() });
        this._html = '';
    }
    set innerHTML(html) { this._html = html; this.children = []; this.dom.register(html); }
    get innerHTML() { return this._html; }
    insertAdjacentHTML(_pos, html) { this.dom.register(html); }
    appendChild(child) { this.children.push(child); return child; }
    append(...kids) { kids.forEach((k) => this.children.push(k)); }
    setAttribute(k, v) { this.attributes[k] = v; if (k === 'class') v.split(/\s+/).forEach((c) => this.classList.add(c)); }
    addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
    dispatch(ev, event = {}) { (this.listeners[ev] || []).forEach((fn) => fn(event)); }
    focus() {}
    click() { this.clicked = true; if (this.onclick) return this.onclick(); }
}

export class FakeDom {
    constructor() {
        this.byId = new Map();
        this.created = [];
        this.document = {
            getElementById: (id) => this.byId.get(id) || null,
            createElement: (tag) => this.createElement(tag),
            addEventListener: (ev, fn) => (this.docListeners[ev] ||= []).push(fn),
            documentElement: {},
        };
        this.docListeners = {};
        this.document.body = new FakeElement(this, 'body');
    }
    createElement(tag) {
        const el = new FakeElement(this, tag);
        if (tag === 'canvas') {
            el.getContext = () => ({ putImageData: (img) => { el.lastImage = img; } });
            el.toBlob = (cb, type, quality) => cb(new Blob([`jpeg:${el.lastImage?.data?.[0]}`], { type }));
        }
        this.created.push(el);
        return el;
    }
    add(tag, id, props = {}) {
        const el = Object.assign(new FakeElement(this, tag, id), props);
        this.byId.set(id, el);
        return el;
    }
    // create elements for every id="..." in injected HTML, with inputs' default values
    register(html) {
        for (const m of html.matchAll(/<(\w+)([^>]*?)\bid="([^"]+)"([^>]*)>/g)) {
            const attrs = m[2] + m[4];
            const el = this.add(m[1], m[3]);
            const cls = attrs.match(/class="([^"]+)"/);
            if (cls) el.setAttribute('class', cls[1]);
            const value = attrs.match(/value="([^"]*)"/);
            if (value) el.value = value[1];
            if (m[1] === 'select') {
                const body = html.slice(m.index).split('</select>')[0];
                const sel = body.match(/<option(?: value="([^"]*)")?\s+selected>([^<]*)</);
                if (sel) el.value = sel[1] ?? sel[2];
            }
        }
    }
    fire(ev, event = {}) { (this.docListeners[ev] || []).forEach((fn) => fn(event)); }
}

// Load a browser script into a sandbox whose window/document are the fake DOM.
export function loadScript(path, dom, globals = {}) {
    const sandbox = {
        document: dom.document, console, setTimeout, clearTimeout, Promise, URL, URLSearchParams,
        Blob, FormData, TextDecoder, TextEncoder, Uint8Array, Uint8ClampedArray, ArrayBuffer, JSON, Math,
        String, Number, Object, Array, Error, parseInt, parseFloat, encodeURIComponent, decodeURIComponent,
        atob, btoa, performance, ReadableStream, Response,
        ImageData: class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } },
        alert: (msg) => sandbox.alerts.push(msg), alerts: [],
        ...globals,
    };
    sandbox.window = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(readFileSync(path, 'utf8'), sandbox, { filename: path });
    return sandbox;
}

export const flush = (ms = 0) => new Promise((r) => setTimeout(r, ms));
