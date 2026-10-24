// Unit tests for supersplat-viewer/web/segment_match.js (Object Metadata suggestions from the RAG database).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { FakeDom, loadScript } from './fake_dom.mjs';

const SCRIPT = fileURLToPath(new URL('../../supersplat-viewer/web/segment_match.js', import.meta.url));
const CAND = { name: 'Baluster vase, from a five-piece garniture', url: 'https://m/items/show/3397', score: 6.6,
    fields: { 'Object Name': 'Vase', Dimensions: 'HxWxD: 40.3 x 13.0 x 13.0 cm', Wall: 'East', 'Shelf Number': '82' },
    why: { shape: 0.9 } };

test('summary and the description written into Object Metadata', () => {
    const dom = new FakeDom();
    const { SegmentMatch } = loadScript(SCRIPT, dom, {});
    assert.equal(SegmentMatch._summary({ candidates: [CAND] }), '→ Baluster vase, from a five-piece garniture · HxWxD: 40.3 x 13.0 x 13.0 cm (score 6.6)');
    assert.equal(SegmentMatch._summary({ candidates: [], note: 'no crawled pages' }), 'no crawled pages');
    assert.equal(SegmentMatch._describe(CAND),
        'Object Name: Vase\nDimensions: HxWxD: 40.3 x 13.0 x 13.0 cm\nWall: East\nShelf Number: 82\nSource: https://m/items/show/3397');
});
