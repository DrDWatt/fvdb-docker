// Unit tests for supersplat-viewer/web/orient.js (flip / rotate about the view direction).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/orient.js', import.meta.url));
const near = (a, b, eps = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) < eps);

function load() {
    const dom = new FakeDom();
    for (const id of ['orient-status', 'orient-flip', 'orient-left', 'orient-right', 'orient-reset', 'model-select']) dom.add('div', id);
    dom.add('iframe', 'viewer-iframe', { contentWindow: {} });
    return loadScript(SCRIPT, dom, {});
}
// column-major 4x4 from row-major 3x3 + t
const mat4 = (R, t) => [R[0][0], R[1][0], R[2][0], 0, R[0][1], R[1][1], R[2][1], 0, R[0][2], R[1][2], R[2][2], 0, t[0], t[1], t[2], 1];

test('flipping the viewer default (0, 0, 180) seen from +z gives the identity (new_peacock upright)', () => {
    const { Orient } = load();
    const def = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]];
    // camera at (0, 0, 5) looking down -z: view = translate(0, 0, -5)
    const out = Orient._turned({ model: mat4(def, [0, 0, 0]), view: mat4([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, -5]) }, 180);
    assert.ok(near(out.quat.map(Math.abs), [0, 0, 0, 1], 1e-9), JSON.stringify(out.quat));
    assert.ok(near(out.t, [0, 0, 0]));  // the axis passes through the camera and the origin: no shift
});

test('turning about an off-origin camera keeps the scene in front of it', () => {
    const { Orient } = load();
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    // camera at (2, 1, 5) looking down -z
    const out = Orient._turned({ model: mat4(I, [0, 0, 0]), view: mat4(I, [-2, -1, -5]) }, 180);
    assert.ok(near(out.t, [4, 2, 0]));   // origin mirrored through the camera axis
    const q90 = Orient._turned({ model: mat4(I, [0, 0, 0]), view: mat4(I, [0, 0, -5]) }, 90).quat;
    assert.ok(near(q90, [0, 0, Math.SQRT1_2, -Math.SQRT1_2]) || near(q90, [0, 0, -Math.SQRT1_2, Math.SQRT1_2]), JSON.stringify(q90));
});
