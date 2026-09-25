// Metadata linking + RAG query UI for the SuperSplat viewer page (mirrors :8085).
//   * RAG pane: "Ask AI" (scene-grounded chat, streamed from Ollama) and
//     "Upload Info" (model-level document, shared with :8085 via the rendering service)
//   * Metadata panel: label / notes / documents linked to a SAM3 object or a
//     GARField extraction; everything linked becomes RAG context.
(() => {
    const $ = (id) => document.getElementById(id);
    const model = () => $('model-select').value;
    let chatHistory = [];
    let active = null;   // {kind, id}

    const el = (tag, attrs = {}, text = '') => {
        const node = document.createElement(tag);
        Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
        if (text) node.textContent = text;
        return node;
    };

    function mount() {
        document.body.insertAdjacentHTML('beforeend', `
            <div id="rag-pane">
                <span class="rag-title">📄 RAG Query</span>
                <button class="btn-warning" id="rag-ask-btn">Ask AI</button>
                <button class="btn-primary" id="rag-upload-btn">⬆️ Upload Info</button>
            </div>

            <div id="rag-modal" class="rag-modal hidden">
                <div class="rag-dialog">
                    <div class="rag-dialog-head">
                        <h2>📄 Model Summary &amp; RAG Query</h2>
                        <button class="popup-close" id="rag-close">&times;</button>
                    </div>
                    <div id="rag-summary" class="rag-summary"></div>
                    <div id="rag-llm-status" class="status-text"></div>
                    <div id="rag-chat" class="rag-chat"></div>
                    <div class="rag-input-row">
                        <input type="text" id="rag-query" placeholder="Ask about the scene (e.g. 'What objects are in the splat?')">
                        <button class="btn-primary" id="rag-send">Ask</button>
                    </div>
                </div>
            </div>

            <div id="upload-info-modal" class="rag-modal hidden">
                <div class="rag-dialog rag-dialog-small">
                    <div class="rag-dialog-head">
                        <h2>⬆️ Upload Model Info</h2>
                        <button class="popup-close" id="upload-info-close">&times;</button>
                    </div>
                    <p class="status-text">Upload a summary document (PDF, TXT, JSON, MD) for this model.
                       It is shared with the fVDB viewer (:8085).</p>
                    <input type="file" id="upload-info-file" accept=".pdf,.txt,.json,.md,.text">
                    <div class="btn-row"><button class="btn-primary" id="upload-info-send">Upload</button></div>
                    <div id="upload-info-status" class="status-text"></div>
                </div>
            </div>

            <div id="meta-panel" class="hidden">
                <div class="rag-dialog-head">
                    <h2 id="meta-title">📋 Metadata</h2>
                    <button class="popup-close" id="meta-close">&times;</button>
                </div>
                <input type="text" id="meta-label" placeholder="Label (e.g. Chair, Charger)">
                <textarea id="meta-text" placeholder="Description / notes..."></textarea>
                <input type="file" id="meta-files" multiple accept=".pdf,.txt,.json,.md,.csv,.doc,.docx,.png,.jpg">
                <ul id="meta-file-list"></ul>
                <div id="meta-training" class="status-text">No documents linked</div>
                <div class="btn-row">
                    <button class="btn-success" id="meta-save">💾 Save</button>
                    <button class="btn-primary" id="meta-upload">⬆️ Upload Files</button>
                </div>
            </div>`);

        $('rag-ask-btn').onclick = openChat;
        $('rag-close').onclick = () => $('rag-modal').classList.add('hidden');
        $('rag-send').onclick = sendQuery;
        $('rag-query').onkeydown = (e) => { if (e.key === 'Enter') sendQuery(); };
        $('rag-upload-btn').onclick = () => { $('upload-info-status').textContent = ''; $('upload-info-modal').classList.remove('hidden'); };
        $('upload-info-close').onclick = () => $('upload-info-modal').classList.add('hidden');
        $('upload-info-send').onclick = uploadModelInfo;
        $('meta-close').onclick = closeMeta;
        $('meta-save').onclick = saveMeta;
        $('meta-upload').onclick = uploadMetaFiles;
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') { $('rag-modal').classList.add('hidden'); $('upload-info-modal').classList.add('hidden'); }
        });
    }

    // ----- Ask AI -----
    async function openChat() {
        chatHistory = [];
        $('rag-chat').innerHTML = '';
        $('rag-summary').textContent = 'Loading context...';
        $('rag-llm-status').textContent = 'Checking LLM...';
        $('rag-modal').classList.remove('hidden');
        $('rag-query').focus();
        try {
            const d = await (await fetch(`/rag/context?model=${encodeURIComponent(model())}`)).json();
            const lines = [];
            if (d.model_summary) lines.push(`Model: ${d.model_name}\n${d.model_summary}`);
            if (d.segments_count > 0) lines.push(`Segments: ${d.segments_count} SAM3 object(s)` +
                (d.segment_labels.length ? ` (${d.segment_labels.join(', ')})` : ''));
            if (d.extractions_count > 0) lines.push(`Extractions: ${d.extractions_count} 3D extraction(s)`);
            if (d.documents_count > 0) lines.push(`Documents: ${d.documents_count} linked`);
            $('rag-summary').textContent = lines.length ? lines.join('\n')
                : 'No context data yet. Segment objects, extract 3D objects or upload documents to enrich the knowledge base.';
        } catch (e) {
            $('rag-summary').textContent = 'No summary available.';
        }
        try {
            const s = await (await fetch('/rag/status')).json();
            $('rag-llm-status').innerHTML = s.available
                ? `<span style="color:#28a745;">● LLM ready</span> <span style="color:#666;">(${s.model})</span>`
                : `<span style="color:#dc3545;">● LLM unavailable</span>`;
            if (!s.available) $('rag-llm-status').append(el('span', { style: 'color:#888;' }, ` ${s.error || ''}`));
        } catch (e) {
            $('rag-llm-status').innerHTML = '<span style="color:#dc3545;">● LLM unavailable</span>';
        }
    }

    function chatBubble(role, text) {
        const row = el('div', { class: `rag-msg rag-${role}` });
        const bubble = el('span', {}, text);
        row.appendChild(bubble);
        $('rag-chat').appendChild(row);
        $('rag-chat').scrollTop = $('rag-chat').scrollHeight;
        return bubble;
    }

    async function sendQuery() {
        const input = $('rag-query');
        const query = input.value.trim();
        if (!query) return;
        input.value = '';
        $('rag-send').disabled = true;
        chatBubble('user', query);
        const bubble = chatBubble('assistant', 'Thinking...');
        try {
            const resp = await fetch('/rag/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, model: model(), history: chatHistory })
            });
            if (!resp.ok) throw new Error((await resp.json()).error || `HTTP ${resp.status}`);
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let full = '', buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();                 // keep a partial SSE line for the next chunk
                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const msg = JSON.parse(line.slice(6));
                    if (msg.token) { full += msg.token; bubble.textContent = full; }
                    if (msg.error) throw new Error(msg.error);
                    if (msg.done) chatHistory.push({ role: 'user', content: query }, { role: 'assistant', content: full });
                }
                $('rag-chat').scrollTop = $('rag-chat').scrollHeight;
            }
        } catch (e) {
            bubble.textContent = `Error: ${e.message}`;
            bubble.style.color = '#dc3545';
        } finally {
            $('rag-send').disabled = false;
        }
    }

    async function uploadModelInfo() {
        const file = $('upload-info-file').files[0];
        const status = $('upload-info-status');
        if (!file) { status.textContent = 'Please select a file'; status.style.color = '#dc3545'; return; }
        const form = new FormData();
        form.append('file', file);
        status.textContent = 'Uploading...';
        status.style.color = '#ffc107';
        try {
            const resp = await fetch(`/upload_summary?model=${encodeURIComponent(model())}`, { method: 'POST', body: form });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
            status.textContent = `✅ ${data.message || 'Uploaded'}`;
            status.style.color = '#28a745';
            $('upload-info-file').value = '';
            setTimeout(() => $('upload-info-modal').classList.add('hidden'), 2000);
        } catch (e) {
            status.textContent = `Error: ${e.message}`;
            status.style.color = '#dc3545';
        }
    }

    // ----- Metadata links (SAM3 objects / GARField extractions) -----
    const metaUrl = () => `/metadata/${active.kind}/${encodeURIComponent(active.id)}`;

    function renderMeta(summary) {
        const list = $('meta-file-list');
        list.innerHTML = '';
        (summary?.files || []).forEach((f) => {
            const li = el('li');
            const a = el('a', { href: `${metaUrl()}/file/${f.idx}`, target: '_blank' }, `📄 ${f.name}`);
            li.append(a, el('small', {}, ` (${f.size})`));
            list.appendChild(li);
        });
        const td = summary?.training_data || {};
        $('meta-training').innerHTML = td.files_count
            ? `<span style="color:#28a745;">✓ ${td.files_count} document(s) linked</span> <small>Last: ${td.last_upload}</small>`
            : 'No documents linked';
    }

    async function open(kind, id, defaultLabel) {
        active = { kind, id };
        $('meta-title').textContent = kind === 'object' ? '📋 Object Metadata' : '📋 Extraction Metadata';
        $('meta-label').value = defaultLabel || '';
        $('meta-text').value = '';
        renderMeta(null);
        $('meta-panel').classList.remove('hidden');
        try {
            const data = await (await fetch(metaUrl())).json();
            if (data.summary) {
                $('meta-label').value = data.summary.label || defaultLabel || '';
                $('meta-text').value = data.summary.text || '';
                renderMeta(data.summary);
            }
        } catch (e) { /* new item */ }
    }

    function closeMeta() {
        $('meta-panel').classList.add('hidden');
        active = null;
    }

    function flash(btn, text) {
        const original = btn.textContent;
        btn.textContent = text;
        setTimeout(() => { btn.textContent = original; btn.disabled = false; }, 1500);
    }

    async function saveMeta() {
        if (!active) return;
        const form = new FormData();
        form.append('label', $('meta-label').value);
        form.append('text', $('meta-text').value);
        form.append('model', model());
        const resp = await fetch(metaUrl(), { method: 'POST', body: form });
        flash($('meta-save'), resp.ok ? '✓ Saved!' : '❌ Failed');
    }

    async function uploadMetaFiles() {
        const input = $('meta-files');
        if (!active || !input.files.length) { alert('Please select files to upload'); return; }
        const form = new FormData();
        for (const f of input.files) form.append('files', f);
        form.append('model', model());
        const btn = $('meta-upload');
        btn.disabled = true;
        btn.textContent = '⏳ Uploading...';
        try {
            const resp = await fetch(`${metaUrl()}/upload`, { method: 'POST', body: form });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
            renderMeta(data.summary);
            input.value = '';
            flash(btn, '✓ Uploaded!');
        } catch (e) {
            flash(btn, '❌ Failed');
        }
    }

    window.MetadataLinks = { open, close: closeMeta };
    window.Rag = { openChat };
    document.addEventListener('DOMContentLoaded', mount);
})();
