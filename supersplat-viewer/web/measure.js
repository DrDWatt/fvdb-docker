// "Ask Cosmos 3" measurement panel for the SuperSplat viewer page.
//
// Sends the user's question + the viewer's current camera (window.getCameraMatrices in the viewer
// iframe) to /measure/ask. The backend renders the same view, Cosmos 3 locates the feature, and the
// Splat Blueprints engine computes the dimension. Every number shown here comes from a persisted
// calculation: value, equation and shown work (substitutions). The model never supplies numbers.
(() => {
    const $ = (id) => document.getElementById(id);
    const EXAMPLES = ['How wide is this doorway?', 'How tall is this window?', 'How tall is this doorway?'];
    const state = { busy: false, last: null };
    const POLL_MS = 2000;
    const MAX_POLL_MISSES = 10;  // ~20 s of consecutive network errors before giving up
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

    function camera() {
        try {
            const win = $('viewer-iframe').contentWindow;
            const m = typeof win.getCameraMatrices === 'function' ? win.getCameraMatrices() : null;
            if (m && !m.width) {  // older viewer builds do not report the canvas size; read the drawing buffer
                const c = win.document.querySelector('canvas');
                if (c) Object.assign(m, { width: c.width, height: c.height });
            }
            return m;
        } catch (e) { return null; }
    }

    function setStatus(text, color) {
        const el = $('measure-status');
        el.textContent = text;
        el.style.color = color || '#888';
    }

    // Results: one card per persisted calculation (value, equation, work lines)
    function renderResult(out) {
        const cards = out.results.map((r) => `
            <div class="measure-card" data-revision="${esc(r.revision_id)}">
                <div class="measure-title">${esc(r.label)}</div>
                <div class="measure-value">${esc(r.display)} <span class="measure-tag">${esc(r.validation)}</span></div>
                <div class="measure-eq">${esc(r.equation)}</div>
                ${r.work.map((w) => `<div class="measure-work">${esc(`${w.expr} = ${w.subst} = ${w.result}`)}</div>`).join('')}
            </div>`).join('');
        const edges = Object.entries(out.edges || {}).map(([k, e]) => `${k}: ${e.method.startsWith('snapped') ? 'snapped' : 'model box'}`).join(', ');
        const geometry = out.plane
            ? `edges ${esc(edges)}; wall plane rms ${out.plane.rms.toFixed(4)} (${out.plane.support}/${out.plane.samples} samples)`
            : `object extent${out.up ? ` (up: ${esc(out.up.method)})` : ''}`;
        const rag = out.rag ? ` RAG: ${out.rag.sources_checked.length} source(s), ${out.rag.statements.length} size statement(s)` +
            `${out.rag.identified_name ? `, identified “${esc(out.rag.identified_name)}”` : ''}.` : '';
        const meta = out.measured === false
            ? `No equation-based measurement: ${esc(out.reason || '')}; ${out.seconds}s.`
            : `Located by Cosmos 3 as “${esc(out.grounding.label)}”; ${geometry}; ${out.seconds}s.${rag}`;
        $('measure-result').innerHTML = `
            ${out.answer.map((a) => `<p class="measure-answer">${esc(a)}</p>`).join('')}
            <img id="measure-overlay" class="measure-overlay" alt="Measurement overlay" title="Click to enlarge" src="${esc(out.overlay_url)}">
            ${cards}
            <p class="status-text measure-meta">${meta} Run ${esc(out.run_id.slice(0, 8))} · scene ${esc(out.scene_id)}</p>`;
        $('measure-overlay').onclick = () => openOverlay(out.overlay_url);
    }

    function openOverlay(url) {
        $('popup-title').textContent = 'Cosmos 3 measurement (equation + work)';
        $('popup-iframe').src = url;
        $('popup-modal').classList.remove('hidden');
    }

    async function ask(question) {
        if (state.busy) return;
        const q = (question ?? $('measure-question').value).trim();
        if (!q) { setStatus('Type a question, e.g. "How wide is this doorway?"', '#ffc107'); return; }
        const cam = camera();
        if (!cam) { setStatus('Viewer still loading — try again in a moment', '#ffc107'); return; }
        state.busy = true;
        $('measure-ask').disabled = true;
        $('measure-result').innerHTML = '';
        setStatus('📐 Rendering this view, asking Cosmos 3 to locate the feature, measuring…', '#17a2b8');
        try {
            const resp = await fetch('/measure/ask', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: $('model-select').value, question: q, camera: cam }),
            });
            let job = await resp.json();
            if (!resp.ok) throw new Error(job.detail || `HTTP ${resp.status}`);
            // Large splats take minutes on the CPU renderer: poll the job (one short request every 2 s)
            // instead of one long request that a Wi-Fi hiccup kills ("Failed to fetch").
            let misses = 0;
            while (job.state === 'running') {
                await sleep(POLL_MS);
                try {
                    const r = await fetch(`/measure/jobs/${job.job_id}`);
                    const next = await r.json();
                    if (!r.ok) throw new Error(next.detail || `HTTP ${r.status}`);
                    job = next;
                    misses = 0;
                } catch (e) {
                    if (++misses > MAX_POLL_MISSES) throw new Error(`lost contact with the viewer server (${e.message})`);
                    continue;  // transient network error: keep the job, retry
                }
                setStatus(`📐 ${job.stage} · ${Math.round(job.elapsed_s || 0)} s`, '#17a2b8');
            }
            if (job.state !== 'completed') throw new Error(job.error || 'measurement failed');
            const out = job.result;
            state.last = out;
            renderResult(out);
            setStatus(out.measured === false ? '⚠️ Not measured with equations — Cosmos 3 visual answer shown'
                : out.unscaled ? '✅ Measured (model units — no metric scale yet)' : '✅ Measured', out.measured === false ? '#ffc107' : '#28a745');
        } catch (e) {
            setStatus(`❌ ${e.message}`, '#dc3545');
        } finally {
            state.busy = false;
            $('measure-ask').disabled = false;
        }
    }

    function render(container) {
        container.innerHTML = `
            <h3>Measure (Cosmos 3)</h3>
            <p class="status-text">Point the view at a feature, then ask. Dimensions come from equations, not the model.</p>
            <input type="text" id="measure-question" class="measure-input" placeholder="How wide is this doorway?">
            <div class="btn-row">
                <button class="btn-primary" id="measure-ask">📐 Ask Cosmos 3</button>
            </div>
            <div class="measure-examples">${EXAMPLES.map((e) => `<button class="measure-chip" data-q="${esc(e)}">${esc(e)}</button>`).join('')}</div>
            <p class="status-text" id="measure-status"></p>
            <div id="measure-result"></div>`;
        $('measure-ask').onclick = () => ask();
        $('measure-question').onkeydown = (e) => { if (e.key === 'Enter') ask(); };
        container.querySelectorAll('.measure-chip').forEach((b) => {
            b.onclick = () => { $('measure-question').value = b.dataset.q; ask(b.dataset.q); };
        });
    }

    window.Measure = { render, ask, renderResult, camera, state };
})();
