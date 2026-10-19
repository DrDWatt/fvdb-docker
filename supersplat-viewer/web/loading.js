// Real model-loading progress for the SuperSplat viewer page.
//
// The viewer iframe runs with ?noui, so it shows nothing while it downloads and parses a PLY. The old
// overlay closed on the iframe's `load` event (HTML loaded), long before a 700 MB-1.3 GB splat arrived,
// leaving a black canvas. This polls the server's /models-progress/<model> (bytes sent, MB/s) and the
// viewer's getCameraMatrices().model (the splat entity exists), and only then reports "Ready".
(() => {
    const $ = (id) => document.getElementById(id);
    const POLL_MS = 500;
    const MAX_MS = 15 * 60 * 1000;  // give up after 15 min (huge file on a slow link)
    let token = 0;

    const mb = (b) => (b / 1e6).toFixed(0);

    // Ready = the iframe document is the viewer for THIS model (right after a model switch the iframe still
    // holds the previous viewer for a moment) and its splat entity exists.
    function viewerReady(model) {
        try {
            const win = $('viewer-iframe').contentWindow;
            const content = new URL(win.location.href).searchParams.get('content') || '';
            if (model && decodeURIComponent(content.split('/').pop()) !== model) return false;
            const m = win.getCameraMatrices();
            return !!(m && m.model);
        } catch (e) { return false; }
    }

    function show(text, pct) {
        const overlay = $('loading-overlay');
        overlay.classList.remove('fade-out');
        overlay.style.display = 'flex';
        $('loading-text').textContent = text;
        if (pct !== undefined) $('loading-bar').style.width = `${Math.max(2, Math.min(100, pct)).toFixed(0)}%`;
    }

    function hide(text) {
        $('loading-text').textContent = text;
        $('loading-bar').style.width = '100%';
        setTimeout(() => {
            $('loading-overlay').classList.add('fade-out');
            setTimeout(() => { $('loading-overlay').style.display = 'none'; }, 500);
        }, 1200);
    }

    // Track one model load; a newer call cancels older ones. Resolves with timing details.
    async function track(model) {
        const mine = ++token;
        const t0 = Date.now();
        let download = null;
        show(`Loading ${model}…`, 2);
        // Latest download record of THIS load (compared on the server clock: no browser/server skew)
        async function poll() {
            try {
                const p = await (await fetch(`/models-progress/${encodeURIComponent(model)}`)).json();
                if (p.started && (p.server_now - p.started_at) * 1000 <= Date.now() - t0 + 2000) download = p;
            } catch (e) { /* keep polling */ }
        }
        const finished = (d) => d && (d.done || d.bytes_sent >= d.total);
        while (mine === token && Date.now() - t0 < MAX_MS) {
            if (viewerReady(model)) {
                // The same-origin viewer iframe blocks this page's JS while it builds the splats, so the last
                // poll may predate the end of the download: refresh it once before reporting.
                await poll();
                const total = (Date.now() - t0) / 1000;
                const dl = finished(download)
                    ? ` · download ${download.seconds.toFixed(1)}s at ${download.mb_per_s.toFixed(0)} MB/s` : ' · from browser cache';
                hide(`✅ Ready in ${total.toFixed(1)}s${dl}`);
                return { model, seconds: total, download };
            }
            await poll();
            if (download && !finished(download)) {
                show(`Downloading ${model}: ${mb(download.bytes_sent)} / ${mb(download.total)} MB · ${download.mb_per_s.toFixed(0)} MB/s`,
                     85 * download.bytes_sent / download.total);
            } else if (finished(download)) {
                show(`Building ${mb(download.total)} MB of splats in the browser (${download.seconds.toFixed(0)}s download)…`, 92);
            } else if (Date.now() - t0 > 3000) {
                show(`Loading ${model} (browser cache or starting download)…`, 10);
            }
            await new Promise((r) => setTimeout(r, POLL_MS));
        }
        if (mine === token) hide('⚠️ Model did not finish loading (see browser console)');
        return null;
    }

    window.ModelLoading = { track, viewerReady };
})();
