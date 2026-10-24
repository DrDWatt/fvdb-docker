// "Crawl a website" in the Upload Info dialog: the agent follows links from a URL about this model (same site,
// bounded depth/pages, robots.txt honoured), pulls titles, descriptions, labelled fields (Object Name,
// Dimensions, …), size statements and images, and adds them to the RAG store used by Ask AI and Cosmos 3.
(() => {
    const $ = (id) => document.getElementById(id);
    const model = () => $('model-select').value;
    const POLL_MS = 2000;
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

    function setStatus(text, color) { const el = $('crawl-status'); el.textContent = text; el.style.color = color || ''; }

    function mount() {
        const dialog = $('upload-info-status');
        if (!dialog) return;
        dialog.insertAdjacentHTML('afterend', `
            <div class="crawl-box">
                <h3>🌐 Crawl a website about this model</h3>
                <p class="status-text">The agent follows links on the same site (under the same path), reads titles,
                   descriptions, fields such as Dimensions, size statements and images, and adds them to the RAG database.</p>
                <input type="url" id="crawl-url" placeholder="https://peacockroom.wayne.edu/items/browse?collection=3">
                <div class="crawl-row">
                    <label>Depth <input type="number" id="crawl-depth" min="0" max="6" value="3"></label>
                    <label>Max pages <input type="number" id="crawl-pages" min="1" max="2000" value="400"></label>
                    <label><input type="checkbox" id="crawl-site"> whole site</label>
                </div>
                <div class="btn-row"><button class="btn-primary" id="crawl-start">🌐 Crawl</button>
                    <button class="btn-danger" id="crawl-forget">Forget crawled pages</button></div>
                <div id="crawl-status" class="status-text"></div>
            </div>`);
        $('crawl-start').addEventListener('click', start);
        $('crawl-forget').addEventListener('click', forget);
        refreshCount();
    }

    async function refreshCount() {
        try {
            const d = await (await fetch(`/rag/web-documents?model=${encodeURIComponent(model())}&limit=1`)).json();
            if (d.total) setStatus(`${d.total} crawled page(s) in the RAG database for this model`);
        } catch (e) { /* Blueprints unavailable */ }
    }

    async function start() {
        const url = $('crawl-url').value.trim();
        if (!/^https?:\/\//.test(url)) { setStatus('Enter an http(s):// URL', '#dc3545'); return; }
        $('crawl-start').disabled = true;
        try {
            const resp = await fetch('/rag/crawl', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url, model: model(), max_depth: Number($('crawl-depth').value),
                    max_pages: Number($('crawl-pages').value), scope: $('crawl-site').checked ? 'site' : 'path' }),
            });
            let job = await resp.json();
            if (!resp.ok) throw new Error(job.detail || `HTTP ${resp.status}`);
            setStatus('🌐 Crawling…', '#17a2b8');
            let misses = 0;
            while (['queued', 'running'].includes(job.state)) {
                await sleep(POLL_MS);
                try {
                    const r = await fetch(`/rag/crawl/${job.job_id}`);
                    const next = await r.json();
                    if (!r.ok) throw new Error(next.detail || `HTTP ${r.status}`);
                    job = next;
                    misses = 0;
                } catch (e) {
                    if (++misses > 10) throw new Error(`lost contact (${e.message})`);
                    continue;
                }
                setStatus(`🌐 ${job.stage || 'crawling'}…`, '#17a2b8');
            }
            if (job.state !== 'completed') throw new Error(job.error || job.state);
            const r = job.result;
            const blocked = r.images_robots_blocked ? `, ${r.images_robots_blocked} not downloaded (robots.txt)` : '';
            setStatus(`✅ ${r.pages} page(s): ${r.fields} fields, ${r.statements} size statement(s), ` +
                      `${r.images_found} image(s) (${r.images_saved} saved${blocked}) added to the RAG database`, '#28a745');
        } catch (e) {
            setStatus(`❌ ${e.message}`, '#dc3545');
        } finally {
            $('crawl-start').disabled = false;
        }
    }

    async function forget() {
        if (!confirm(`Remove all crawled pages for ${model()} from the RAG database?`)) return;
        try {
            const d = await (await fetch(`/rag/web-documents?model=${encodeURIComponent(model())}`, { method: 'DELETE' })).json();
            setStatus(`Removed ${d.deleted_documents || 0} crawled page(s)`);
        } catch (e) { setStatus(`❌ ${e.message}`, '#dc3545'); }
    }

    window.Crawl = { start, forget, refreshCount };
    document.addEventListener('DOMContentLoaded', mount);
})();
