// Find each SAM3 segment in the RAG database (crawled catalogue pages) and offer proper name + description in
// Object Metadata. Cosmos 3 describes each crop; the ranking (object type, words, published H/W vs mask H/W) is
// deterministic. Nothing is linked automatically — "Use" fills the label/notes and Save stores them.
(() => {
    const $ = (id) => document.getElementById(id);
    const model = () => $('model-select').value;
    const state = { prompt: '', masks: [], results: {}, running: false };
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    // Called by the page after SAM3 returns masks
    function offer(masks, prompt) {
        state.prompt = prompt;
        state.masks = masks;
        state.results = {};
        let btn = $('seg-match-all');
        if (!btn) {
            btn = document.createElement('button');
            btn.id = 'seg-match-all';
            btn.className = 'btn-primary';
            btn.style.cssText = 'width:100%;margin-top:6px;';
            btn.addEventListener('click', matchAll);
            $('seg-object-list').insertAdjacentElement('afterend', btn);
        }
        btn.textContent = `🔎 Find ${masks.length} object(s) in RAG database`;
        btn.disabled = !masks.length;
    }

    async function fetchMatches(index) {
        if (state.results[index]) return state.results[index];
        const resp = await fetch(`/segment/${index}/rag-matches`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: model() }),
        });
        const out = await resp.json();
        if (!resp.ok) throw new Error(out.detail || `HTTP ${resp.status}`);
        state.results[index] = out;
        return out;
    }

    function summary(out) {
        const top = out.candidates && out.candidates[0];
        if (!top) return out.note || 'no match in RAG database';
        const dims = top.fields && top.fields.Dimensions ? ` · ${top.fields.Dimensions}` : '';
        return `→ ${top.name}${dims} (score ${top.score})`;
    }

    async function matchAll() {
        if (state.running) return;
        state.running = true;
        const btn = $('seg-match-all');
        btn.disabled = true;
        try {
            for (let k = 0; k < state.masks.length; k++) {   // one at a time: Cosmos 3 serves one request at a time
                const i = state.masks[k].index;
                btn.textContent = `🔎 Matching ${k + 1}/${state.masks.length}…`;
                const slot = $(`seg-match-${i}`);
                try {
                    const out = await fetchMatches(i);
                    if (slot) { slot.textContent = summary(out); slot.title = out.description || ''; }
                } catch (e) {
                    if (slot) slot.textContent = `⚠ ${e.message}`;
                }
            }
            btn.textContent = '✅ Matched — open 📄 to review and use a suggestion';
        } finally {
            state.running = false;
            btn.disabled = false;
        }
    }

    function describeCandidate(c) {
        const f = c.fields || {};
        const keys = ['Object Name', 'Title', 'Dimensions', 'Medium', 'Period', 'Dated', 'Wall', 'Shelf Number', 'Object Number'];
        return keys.filter((k) => f[k]).map((k) => `${k}: ${f[k]}`).concat([`Source: ${c.url}`]).join('\n');
    }

    // Object Metadata panel: suggestions for segment `index`
    async function showIn(containerId, index) {
        const box = $(containerId);
        if (!box) return;
        box.innerHTML = '<p class="status-text">🔎 Looking for this object in the RAG database…</p>';
        try {
            const out = await fetchMatches(index);
            const slot = $(`seg-match-${index}`);
            if (slot) slot.textContent = summary(out);
            if (!out.candidates || !out.candidates.length) {
                box.innerHTML = `<p class="status-text">${esc(out.note || 'No matching page in the RAG database.')}</p>`;
                return;
            }
            box.innerHTML = `<p class="status-text">Possible matches (Cosmos 3 sees: “${esc(out.description || out.label)}”) — confirm one:</p>` +
                out.candidates.map((c, k) => `
                    <div class="meta-suggest-item">
                        <div><b>${esc(c.name)}</b> <small>score ${c.score}${c.why.shape !== null ? ` · shape ${c.why.shape}` : ''}</small><br>
                        <small>${esc((c.fields || {}).Dimensions || '')} ${esc((c.fields || {}).Wall ? '· ' + c.fields.Wall + ' wall' : '')}
                        ${esc((c.fields || {})['Shelf Number'] ? '· shelf ' + c.fields['Shelf Number'] : '')}</small>
                        <a href="${esc(c.url)}" target="_blank" rel="noopener">page ↗</a></div>
                        <div class="meta-suggest-actions"><button class="btn-success" data-k="${k}">Use</button>
                        ${(c.fields || {}).Dimensions ? `<button class="btn-purple" data-scale="${k}" title="Measure this object in 3D and use its catalogue height to scale the scene">📏 Use for scale</button>` : ''}</div>
                    </div>`).join('') + '<p class="status-text" id="meta-scale-status"></p>';
            box.querySelectorAll('button[data-k]').forEach((b) => b.addEventListener('click', () => {
                const c = out.candidates[Number(b.dataset.k)];
                $('meta-label').value = c.name;
                $('meta-text').value = describeCandidate(c);
                box.querySelectorAll('.meta-suggest-item').forEach((el) => el.classList.remove('chosen'));
                b.closest('.meta-suggest-item').classList.add('chosen');
            }));
            box.querySelectorAll('button[data-scale]').forEach((b) => b.addEventListener('click', () => linkScale(index, out.candidates[Number(b.dataset.scale)], b)));
        } catch (e) {
            box.innerHTML = `<p class="status-text" style="color:#dc3545;">❌ ${esc(e.message)}</p>`;
        }
    }

    function scaleSummary(r) {
        const loo = (r.leave_one_out || []).map((x) => `${x.name}: ${x.error_pct > 0 ? '+' : ''}${x.error_pct}%`).join(', ');
        return `📏 Scene scale ${r.scale} from ${r.n} linked object(s) — ` +
            (r.n < 2 ? 'single reference, unverified: link another object to check it'
                : r.verified ? `verified (leave-one-out: ${loo})`
                : `objects disagree (leave-one-out: ${loo}) — at least one link is the wrong catalogue object; no scale set`);
    }

    async function linkScale(index, cand, btn) {
        const status = $('meta-scale-status');
        btn.disabled = true;
        status.textContent = '📏 Measuring this object in 3D…';
        status.style.color = '#17a2b8';
        try {
            $('meta-label').value = cand.name;
            $('meta-text').value = describeCandidate(cand);
            const resp = await fetch(`/segment/${index}/link-scale`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: model(), candidate: cand }),
            });
            const r = await resp.json();
            if (!resp.ok) throw new Error(r.detail || `HTTP ${resp.status}`);
            status.textContent = scaleSummary(r) + (r.applied === false ? '' : ' — measurements now come out in m / cm.');
            status.style.color = r.n >= 2 && !r.verified ? '#ffc107' : '#28a745';
        } catch (e) {
            status.textContent = `❌ ${e.message}`;
            status.style.color = '#dc3545';
        } finally {
            btn.disabled = false;
        }
    }

    window.SegmentMatch = { offer, matchAll, showIn, linkScale, _summary: summary, _describe: describeCandidate, _scaleSummary: scaleSummary };
})();
