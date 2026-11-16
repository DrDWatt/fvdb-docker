// Cosmos 3 activity panel (Nov 16 2026), under "Measure (Cosmos 3)" on the :8086 page.
//
// Shows what used to be invisible when a measurement timed out: the Brev CLI login, the on-demand NIM state
// (stopped / starting: loading the model / ready) and the recent Cosmos prompts and responses (/cosmos/activity,
// from the Cosmos gateway :8093). Polls every 3 s while a measurement runs or the NIM is starting, else every 30 s.
(() => {
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    const FAST_MS = 3000, SLOW_MS = 30000;
    const CLEAR_KEY = 'cosmos-cleared-at';
    const store = () => { try { return window.localStorage; } catch (e) { return null; } };
    const state = { timer: null, last: null, open: false, clearedAt: Number((store() && store().getItem(CLEAR_KEY)) || 0) };

    // Pure helpers (unit-tested in tests/js/cosmos_activity.test.mjs)
    function brevBadge(brev) {
        if (!brev || brev.logged_in == null) return { text: 'Brev: unknown', color: '#888', hint: brev ? brev.detail : '' };
        return brev.logged_in ? { text: 'Brev: logged in', color: '#28a745', hint: brev.detail }
            : { text: 'Brev: logged out', color: '#dc3545', hint: 'Run ~/.local/bin/brev login on the Spark, then Re-check.' };
    }
    function nimLine(lc) {
        if (!lc) return 'NIM: unknown';
        const words = { stopped: 'stopped (starts on the next question, ~3.5 min)', starting: `starting: ${lc.detail}`,
            ready: `ready (${lc.detail})`, stopping: 'stopping', error: `error: ${lc.detail}` };
        return `NIM on ${lc.instance || 'Brev'}: ${words[lc.state] || lc.state}`;
    }
    function busy(lc) { return lc && (lc.state === 'starting' || (lc.inflight || 0) > 0); }
    function ago(ts, now) {
        const s = Math.max(0, Math.round(now - ts));
        return s < 60 ? `${s}s ago` : s < 3600 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
    }

    // Nov 16 2026: "Clear" hides past calls in this browser (a running call always stays); "Show all" undoes it.
    function visibleCalls(calls, clearedAt) { return (calls || []).filter((c) => c.state === 'running' || c.started > clearedAt); }

    function renderCalls(all, now) {
        const calls = visibleCalls(all, state.clearedAt), hidden = (all || []).length - calls.length;
        const bar = `<div class="btn-row af-bar"><span class="status-text">${calls.length} call(s)${hidden ? `, ${hidden} cleared` : ''}</span>
            ${calls.length ? '<button class="measure-chip" data-cclear="1">Clear</button>' : ''}
            ${hidden ? '<button class="measure-chip" data-cshowall="1">Show all</button>' : ''}</div>`;
        if (!calls.length) return bar + (hidden ? '' : '<p class="status-text">No Cosmos calls since the server started.</p>');
        return bar + calls.map((c) => `
            <details class="cosmos-call ${c.state}">
                <summary>${c.state === 'ok' ? '✅' : c.state === 'running' ? '⏳' : '❌'} ${esc(ago(c.started, now))} · ${esc(c.endpoint)}
                    · ${c.seconds != null ? esc(c.seconds) + ' s' : 'running'} · ${esc(c.prompt.slice(0, 60))}${c.prompt.length > 60 ? '…' : ''}</summary>
                <div class="cosmos-io"><b>Prompt</b>${c.media && c.media.length ? ` (media: ${esc(c.media.join(', '))})` : ''}<pre>${esc(c.prompt)}</pre>
                ${c.error ? `<b>Error</b><pre class="err">${esc(c.error)}</pre>` : `<b>Response</b><pre>${esc(c.response ?? '')}</pre>`}</div>
            </details>`).join('');
    }

    async function refresh() {
        let next = SLOW_MS;
        try {
            const r = await fetch('/cosmos/activity?limit=10');
            const a = await r.json();
            if (!r.ok) throw new Error(a.detail || `HTTP ${r.status}`);
            state.last = a;
            const lc = a.lifecycle || {}, badge = brevBadge(lc.brev);
            $('cosmos-brev').innerHTML = `<span style="color:${badge.color}" title="${esc(badge.hint)}">● ${esc(badge.text)}</span>`;
            $('cosmos-nim').textContent = nimLine(lc);
            $('cosmos-hint').textContent = lc.brev && lc.brev.logged_in === false ? badge.hint : '';
            $('cosmos-calls').innerHTML = (a.calls_error ? `<p class="status-text" style="color:#dc3545">${esc(a.calls_error)}</p>` : '')
                + renderCalls(a.calls, a.at || Date.now() / 1000);
            $('cosmos-swagger').href = a.swagger || '#';
            if (busy(lc) || (window.Measure && Measure.state.busy)) next = FAST_MS;
        } catch (e) {
            $('cosmos-nim').textContent = `Cosmos activity unavailable: ${e.message}`;
        }
        clearTimeout(state.timer);
        state.timer = setTimeout(refresh, next);
    }

    async function recheck() {
        $('cosmos-recheck').disabled = true;
        try { await fetch('/cosmos/check-brev', { method: 'POST' }); } finally { $('cosmos-recheck').disabled = false; refresh(); }
    }

    function setCleared(ts) {
        state.clearedAt = ts;
        if (store()) store().setItem(CLEAR_KEY, String(ts));
        if (state.last) $('cosmos-calls').innerHTML = renderCalls(state.last.calls, state.last.at || Date.now() / 1000);
    }

    function render(container) {
        container.insertAdjacentHTML('beforeend', `
            <div class="cosmos-activity">
                <div class="btn-row" style="justify-content:space-between;align-items:center">
                    <span id="cosmos-brev" class="status-text">Brev: …</span>
                    <button class="measure-chip" id="cosmos-recheck" title="Re-check the Brev login">↻ Re-check</button>
                </div>
                <p class="status-text" id="cosmos-nim">NIM: …</p>
                <p class="status-text" id="cosmos-hint" style="color:#ffc107"></p>
                <details id="cosmos-log"><summary class="status-text">Cosmos activity (prompts &amp; responses)</summary>
                    <div id="cosmos-calls"></div>
                    <p class="status-text"><a id="cosmos-swagger" target="_blank">Cosmos API (Swagger)</a></p></details>
            </div>`);
        $('cosmos-recheck').onclick = recheck;
        $('cosmos-calls').onclick = (ev) => {
            const t = ev.target.dataset || {};
            if (t.cclear) setCleared(Date.now() / 1000);
            if (t.cshowall) setCleared(0);
        };
        refresh();
    }

    // A measurement just started: switch to fast polling right away
    function kick() { clearTimeout(state.timer); state.timer = setTimeout(refresh, 500); }

    window.CosmosActivity = { render, refresh, kick, brevBadge, nimLine, busy, ago, renderCalls, visibleCalls, state };
})();
