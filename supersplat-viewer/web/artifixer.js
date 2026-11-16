// ArtiFixer3D+ panel (Nov 16 2026): run ArtiFixer3D+ or the quick floater cleanup on the loaded PLY and follow the job,
// from the :8086 sidebar. Talks to /artifixer/* (server-side proxy to the ArtiFixer controller :8092).
(() => {
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const POLL_MS = 10000;
    const CLEAR_KEY = 'artifixer-cleared-at';
    const store = () => { try { return window.localStorage; } catch (e) { return null; } };
    const state = { timer: null, status: null, openLog: null, clearedAt: Number((store() && store().getItem(CLEAR_KEY)) || 0) };

    // Pure helpers (unit-tested in tests/js/artifixer.test.mjs)
    function canRun(model, kind) {
        if (!model) return { ok: false, why: 'This PLY is not in the PLY service.' };
        if (kind === 'cleanup') return { ok: true, why: 'Writes <name>_cleaned.ply next to it (~1 min, Spark CPU).' };
        if (model.derived) return { ok: false, why: 'This is already an ArtiFixer/cleanup output; run it on the original fVDB model.' };
        if (!model.source) return { ok: false, why: 'No COLMAP capture found for this PLY (needed for ArtiFixer3D+). Quick cleanup still works.' };
        return { ok: true, why: `Capture ${model.source}; ~1.5 h on the GPU endpoint, then shown in the :8089 viewer.` };
    }
    function duration(job, now) {
        if (job.started == null) return '';
        const s = Math.round((job.finished || now) - job.started);
        return s >= 3600 ? `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min` : s >= 60 ? `${Math.round(s / 60)} min` : `${s} s`;
    }
    // Nov 16 2026: the list got cluttered. "Clear" hides finished jobs in this browser (queued/running always stay);
    // nothing is deleted on the server, "Show all" brings them back.
    function visibleJobs(jobs, clearedAt) {
        return jobs.filter((j) => ['queued', 'running'].includes(j.state) || (j.created || 0) > clearedAt);
    }
    const ICON = { queued: '⏳', running: '🔄', done: '✅', failed: '❌', cancelled: '⏹', interrupted: '⚠️' };

    function renderJobs(all, now) {
        const jobs = visibleJobs(all, state.clearedAt), hidden = all.length - jobs.length;
        const bar = `<div class="btn-row af-bar"><span class="status-text">${jobs.length} job(s)${hidden ? `, ${hidden} cleared` : ''}</span>
            ${jobs.some((j) => !['queued', 'running'].includes(j.state)) ? '<button class="measure-chip" data-clear="1">Clear</button>' : ''}
            ${hidden ? '<button class="measure-chip" data-showall="1">Show all</button>' : ''}</div>`;
        if (!jobs.length) return bar + (all.length ? '' : '<p class="status-text">No jobs yet.</p>');
        return bar + jobs.map((j) => {
            const res = j.results || {};
            const links = [res.viewer ? `<a href="${esc(res.viewer)}" target="_blank">View in :8089</a>` : '',
                ...(res.plys || []).map((p) => `<span class="af-ply">${esc(p)}</span>`)].join(' ');
            return `<div class="af-job" data-id="${esc(j._id)}">
                <div>${ICON[j.state] || ''} <b>${esc(j.state)}</b> · ${esc(j.ply)} · ${esc(j.endpoint_name)} · ${esc(duration(j, now))}</div>
                <div class="status-text">${esc(j.step)}${j.error ? ` — <span style="color:#dc3545">${esc(j.error)}</span>` : ''}</div>
                <div class="btn-row">${['queued', 'running'].includes(j.state) ? `<button class="measure-chip" data-cancel="${esc(j._id)}">Cancel</button>` : ''}
                    <button class="measure-chip" data-log="${esc(j._id)}">Log</button> ${links}</div>
                ${state.openLog === j._id ? '<pre class="af-log" id="af-log"></pre>' : ''}
            </div>`;
        }).join('');
    }

    async function api(path, opts) {
        const r = await fetch(path, opts);
        const body = await r.json();
        if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
        return body;
    }

    function updateRunHint() {
        const st = state.status;
        if (!st) return;
        const kind = $('af-kind').value, verdict = canRun(st.model, kind);
        $('af-endpoint').style.display = kind === 'cleanup' ? 'none' : '';
        $('af-run').disabled = !verdict.ok;
        $('af-hint').textContent = verdict.why;
    }

    async function refresh() {
        const model = $('model-select') ? $('model-select').value : '';
        try {
            const st = await api(`/artifixer/status?model=${encodeURIComponent(model)}`);
            state.status = st;
            const b = st.brev || {};
            $('af-brev').innerHTML = b.logged_in ? '<span style="color:#28a745">● Brev: logged in</span>'
                : `<span style="color:#dc3545" title="${esc(b.detail)}">● Brev: logged out — run ~/.local/bin/brev login</span>`;
            const keep = $('af-endpoint').value;
            $('af-endpoint').innerHTML = st.endpoints.map((e) => `<option value="${esc(e._id)}">${esc(e.name)} (${esc(e.kind)})</option>`).join('');
            if (st.endpoints.some((e) => e._id === keep)) $('af-endpoint').value = keep;
            $('af-jobs').innerHTML = renderJobs(st.jobs, Date.now() / 1000);
            $('af-links').innerHTML = `<a href="${esc(st.links.controller)}" target="_blank">Controller</a> · <a href="${esc(st.links.swagger)}" target="_blank">Swagger</a> · MCP ${esc(st.links.mcp)}`;
            updateRunHint();
            if (state.openLog) showLog(state.openLog);
        } catch (e) {
            $('af-hint').textContent = `ArtiFixer controller unavailable: ${e.message}`;
        }
        clearTimeout(state.timer);
        state.timer = setTimeout(refresh, POLL_MS);
    }

    async function showLog(id) {
        const el = $('af-log');
        if (!el) return;
        try { el.textContent = (await api(`/artifixer/jobs/${id}/log?tail=40`)).lines.join('\n') || '(no output yet)'; } catch (e) { el.textContent = e.message; }
    }

    async function run() {
        const kind = $('af-kind').value, body = { kind, ply: $('model-select').value };
        if (kind !== 'cleanup') body.endpoint_id = $('af-endpoint').value;
        $('af-run').disabled = true;
        try {
            const job = await api('/artifixer/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
            $('af-hint').textContent = `Queued job ${job._id}${job.scene ? ` (scene ${job.scene})` : ''}.`;
        } catch (e) { $('af-hint').textContent = `❌ ${e.message}`; }
        refresh();
    }

    function setCleared(ts) {
        state.clearedAt = ts;
        if (store()) store().setItem(CLEAR_KEY, String(ts));
        if (state.status) $('af-jobs').innerHTML = renderJobs(state.status.jobs, Date.now() / 1000);
    }

    async function onClick(ev) {
        const t = ev.target.dataset || {};
        if (t.clear) { state.openLog = null; return setCleared(Date.now() / 1000); }
        if (t.showall) return setCleared(0);
        if (t.cancel) { try { await api(`/artifixer/jobs/${t.cancel}/cancel`, { method: 'POST' }); } catch (e) { $('af-hint').textContent = e.message; } refresh(); }
        if (t.log) { state.openLog = state.openLog === t.log ? null : t.log; $('af-jobs').innerHTML = renderJobs(state.status.jobs, Date.now() / 1000); showLog(t.log); }
    }

    function render(container) {
        container.innerHTML = `
            <h3>ArtiFixer3D+</h3>
            <p class="status-text">Clean the loaded splat with NVIDIA ArtiFixer (orbit path, PPISP, ArtiFixer3D+) or the quick floater cleanup.</p>
            <p class="status-text" id="af-brev">Brev: …</p>
            <select id="af-kind" class="measure-input"><option value="artifixer3d_plus">ArtiFixer3D+ (GPU endpoint, ~1.5 h)</option>
                <option value="cleanup">Quick floater cleanup (Spark CPU, ~1 min)</option></select>
            <select id="af-endpoint" class="measure-input"></select>
            <div class="btn-row"><button class="btn-primary" id="af-run">🧹 Run on this model</button></div>
            <p class="status-text" id="af-hint"></p>
            <div id="af-jobs"></div>
            <p class="status-text" id="af-links"></p>`;
        $('af-kind').onchange = updateRunHint;
        $('af-run').onclick = run;
        $('af-jobs').onclick = onClick;
        if ($('model-select')) $('model-select').addEventListener('change', refresh);
        refresh();
    }

    window.ArtiFixer = { render, refresh, canRun, duration, renderJobs, visibleJobs, state };
})();
