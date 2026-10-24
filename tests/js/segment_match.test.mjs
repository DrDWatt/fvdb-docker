// Unit tests for supersplat-viewer/web/segment_match.js (Object Metadata suggestions from the RAG database).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/segment_match.js', import.meta.url));
const CAND = { name: 'Baluster vase, from a five-piece garniture', url: 'https://m/items/show/3397', score: 6.6,
    height_m: 0.403, dims_text: 'Dimensions: HxWxD: 40.3 x 13.0 x 13.0 cm', location: { Wall: 'East', 'Shelf Number': '82' },
    details: { 'Object Name': 'Vase', Dimensions: 'HxWxD: 40.3 x 13.0 x 13.0 cm', Wall: 'East', 'Shelf Number': '82' },
    why: { shape: 0.9 } };
const LAMP = { name: 'Arco floor lamp', url: 'https://shop.example/arco', score: 5, height_m: 2.4, dims_text: 'Height: 240 cm; Width: 200 cm',
    details: { Product: 'Arco floor lamp', Material: 'Carrara marble, steel' }, location: {}, why: { shape: null } };

test('summary and the description written into Object Metadata', () => {
    const dom = new FakeDom();
    const { SegmentMatch } = loadScript(SCRIPT, dom, {});
    assert.equal(SegmentMatch._summary({ candidates: [CAND] }), '→ Baluster vase, from a five-piece garniture · Dimensions: HxWxD: 40.3 x 13.0 x 13.0 cm (score 6.6)');
    assert.equal(SegmentMatch._summary({ candidates: [], note: 'no crawled pages' }), 'no crawled pages');
    assert.equal(SegmentMatch._describe(CAND),
        'Object Name: Vase\nDimensions: HxWxD: 40.3 x 13.0 x 13.0 cm\nWall: East\nShelf Number: 82\nSource: https://m/items/show/3397');
    // a non-museum site: its own labels, size line added from the canonical size
    assert.equal(SegmentMatch._describe(LAMP),
        'Product: Arco floor lamp\nMaterial: Carrara marble, steel\nSize: Height: 240 cm; Width: 200 cm\nSource: https://shop.example/arco');
});


test('scale summary: single, verified and disagreeing links', () => {
    const { SegmentMatch } = loadScript(SCRIPT, new FakeDom(), {});
    assert.match(SegmentMatch._scaleSummary({ n: 1, scale: '0.33 m/mu' }), /single reference, unverified/);
    assert.match(SegmentMatch._scaleSummary({ n: 2, scale: '0.5 m/mu', verified: true, leave_one_out: [{ name: 'A', error_pct: 1.2 }] }), /verified \(leave-one-out: A: \+1.2%\)/);
    assert.match(SegmentMatch._scaleSummary({ n: 3, scale: '0.8 m/mu', verified: false, applied: false, leave_one_out: [{ name: 'A', error_pct: 394.3 }] }),
        /wrong catalogue object; no scale set/);
});
