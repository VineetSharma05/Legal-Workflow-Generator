
const app = {
    messages: [],        // { id, role, content, pipeline?, trace?, timestamp }
    activeTab: 'chat',
    isLoading: false,
};

const USER_ICON = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="12" cy="8" r="3.2" stroke="currentColor" stroke-width="1.4"/>
  <path d="M6 19c0-3.3 2.7-6 6-6s6 2.7 6 6" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
</svg>`;

const BOT_ICON = `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M12 4l1.6 6.4L20 12l-6.4 1.6L12 20l-1.6-6.4L4 12l6.4-1.6L12 4z" fill="currentColor"/>
</svg>`;

// ═══════════════════════════════════════════════════════════════════════════════
// DOM References
// ═══════════════════════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    chatView:       $('#chat-view'),
    labView:        $('#lab-view'),
    chatMessages:   $('#chat-messages'),
    welcomeScreen:  $('#welcome-screen'),
    chatInput:      $('#chat-input'),
    sendBtn:        $('#send-btn'),
    providerSelect: $('#provider-select'),
    topkSlider:     $('#topk-slider'),
    topkValue:      $('#topk-value'),
    labUnit:        $('#lab-unit'),
    labProvider:    $('#lab-provider'),
    labTopk:        $('#lab-topk'),
    labTopkValue:   $('#lab-topk-value'),
    labQuery:       $('#lab-query'),
    labRunBtn:      $('#lab-run-btn'),
    labResults:     $('#lab-results'),
    labResultsTitle: $('#lab-results-title'),
    labResultsTime:  $('#lab-results-time'),
    labResultsBody:  $('#lab-results-body'),
};

// ═══════════════════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════════════════

function init() {
    // Tab switching
    $$('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Chat input
    dom.chatInput.addEventListener('input', onChatInputChange);
    dom.chatInput.addEventListener('keydown', onChatKeydown);
    dom.sendBtn.addEventListener('click', onSendClick);

    // Settings
    dom.topkSlider.addEventListener('input', () => {
        dom.topkValue.textContent = dom.topkSlider.value;
    });
    dom.labTopk.addEventListener('input', () => {
        dom.labTopkValue.textContent = dom.labTopk.value;
    });

    // Suggestion buttons
    $$('.suggestion-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            dom.chatInput.value = btn.dataset.query;
            onChatInputChange();
            sendChatMessage();
        });
    });

    // Test Lab
    dom.labRunBtn.addEventListener('click', runLabTest);
    dom.labQuery.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runLabTest();
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tab Switching
// ═══════════════════════════════════════════════════════════════════════════════

function switchTab(tab) {
    app.activeTab = tab;
    $$('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    dom.chatView.classList.toggle('active', tab === 'chat');
    dom.labView.classList.toggle('active', tab === 'lab');
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chat Input Handling
// ═══════════════════════════════════════════════════════════════════════════════

function onChatInputChange() {
    // Auto-resize
    dom.chatInput.style.height = 'auto';
    dom.chatInput.style.height = Math.min(dom.chatInput.scrollHeight, 120) + 'px';
    // Enable/disable send
    dom.sendBtn.disabled = !dom.chatInput.value.trim() || app.isLoading;
}

function onChatKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (dom.chatInput.value.trim() && !app.isLoading) {
            sendChatMessage();
        }
    }
}

function onSendClick() {
    if (dom.chatInput.value.trim() && !app.isLoading) {
        sendChatMessage();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chat Messaging
// ═══════════════════════════════════════════════════════════════════════════════

async function sendChatMessage() {
    const query = dom.chatInput.value.trim();
    if (!query) return;

    // Hide welcome screen
    if (dom.welcomeScreen) {
        dom.welcomeScreen.style.display = 'none';
    }

    // Add user message
    const userMsg = {
        id: genId(),
        role: 'user',
        content: query,
        timestamp: new Date(),
    };
    app.messages.push(userMsg);
    renderMessage(userMsg);

    // Clear input
    dom.chatInput.value = '';
    dom.chatInput.style.height = 'auto';
    dom.sendBtn.disabled = true;

    // Show typing indicator
    app.isLoading = true;
    const typingEl = showTypingIndicator();
    scrollToBottom();

    // Call API
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                provider: dom.providerSelect.value,
                top_k: parseInt(dom.topkSlider.value),
            }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();

        // Remove typing indicator
        typingEl.remove();

        // Add assistant message
        const assistantMsg = {
            id: genId(),
            role: 'assistant',
            content: data.answer,
            pipeline: data.pipeline,
            trace: data.trace,
            abstained: data.abstained,
            timestamp: new Date(),
        };
        app.messages.push(assistantMsg);
        renderMessage(assistantMsg);

    } catch (err) {
        typingEl.remove();
        renderErrorMessage(err.message);
    } finally {
        app.isLoading = false;
        dom.sendBtn.disabled = !dom.chatInput.value.trim();
        scrollToBottom();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Message Rendering
// ═══════════════════════════════════════════════════════════════════════════════

function renderMessage(msg) {
    const el = document.createElement('div');
    el.className = `message ${msg.role}`;
    el.dataset.msgId = msg.id;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = msg.role === 'user' ? USER_ICON : BOT_ICON;

    const body = document.createElement('div');
    body.className = 'msg-body';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    if (msg.role === 'user') {
        bubble.textContent = msg.content;
    } else {
        bubble.innerHTML = formatAnswer(msg.content);
    }

    body.appendChild(bubble);

    // Pipeline inspector for assistant messages
    if (msg.role === 'assistant' && msg.pipeline) {
        body.appendChild(renderPipelineInspector(msg.pipeline, msg.trace));
    }

    el.appendChild(avatar);
    el.appendChild(body);
    dom.chatMessages.appendChild(el);
    scrollToBottom();
}

function renderErrorMessage(errorText) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.textContent = '⚠️';

    const body = document.createElement('div');
    body.className = 'msg-body';

    const errorEl = document.createElement('div');
    errorEl.className = 'msg-error';
    errorEl.textContent = `Error: ${errorText}`;
    body.appendChild(errorEl);

    el.appendChild(avatar);
    el.appendChild(body);
    dom.chatMessages.appendChild(el);
}

function showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'message assistant';
    el.id = 'typing-msg';

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = BOT_ICON;

    const body = document.createElement('div');
    body.className = 'msg-body';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';

    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    bubble.appendChild(typing);
    body.appendChild(bubble);
    el.appendChild(avatar);
    el.appendChild(body);
    dom.chatMessages.appendChild(el);
    return el;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Answer Formatting
// ═══════════════════════════════════════════════════════════════════════════════

function formatAnswer(text) {
    if (!text) return '';

    // Escape HTML
    let html = escapeHtml(text);

    // Lines of special characters (━━━, ===, ---) → <hr>
    html = html.replace(/^[━═─\-=]{4,}$/gm, '<hr class="answer-hr">');

    // Section titles (all-caps lines like SUMMARY, ACTION CHECKLIST, SOURCES, etc.)
    html = html.replace(/^([A-Z][A-Z\s&\/]{3,})$/gm, '<span class="answer-section-title">$1</span>');

    // Warning lines (⚠)
    html = html.replace(/^(⚠.*)$/gm, '<span class="answer-warning">$1</span>');

    // Newlines → <br>
    html = html.replace(/\n/g, '<br>');

    return html;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pipeline Inspector
// ═══════════════════════════════════════════════════════════════════════════════

function renderPipelineInspector(pipeline, trace) {
    const container = document.createElement('div');
    container.className = 'pipeline-inspector';

    // Toggle button
    const toggle = document.createElement('button');
    toggle.className = 'pipeline-toggle';
    toggle.innerHTML = '<span>Inspect Pipeline</span> <span class="chevron">▸</span>';

    const content = document.createElement('div');
    content.className = 'pipeline-content';

    const timeline = document.createElement('div');
    timeline.className = 'pipeline-timeline';

    // ── Build stages ──
    const stages = buildPipelineStages(pipeline, trace);
    stages.forEach(s => timeline.appendChild(renderStage(s)));

    content.appendChild(timeline);

    // Toggle logic
    toggle.addEventListener('click', () => {
        const isOpen = content.classList.contains('open');
        content.classList.toggle('open');
        toggle.classList.toggle('open');
    });

    container.appendChild(toggle);
    container.appendChild(content);
    return container;
}

function buildPipelineStages(p, trace) {
    const stages = [];

    // 1. Classification
    if (p.classification) {
        const c = p.classification;
        const confidence = typeof c.confidence === 'number' ? c.confidence.toFixed(2) : '?';
        stages.push({
            name: 'Classification',
            summary: `${c.domain || '?'} • ${confidence}`,
            status: c.domain && c.domain !== 'unknown' ? 'success' : 'error',
            details: [
                { label: 'Intent', value: c.intent },
                { label: 'Domain', value: c.domain },
                { label: 'All Domains', value: (c.all_domains || []).join(', ') || '—' },
                { label: 'Confidence', value: confidence },
                { label: 'Normalized', value: c.normalized_query },
                { label: 'Keywords', value: (c.keywords || []).join(', ') || '—' },
            ],
        });
    }

    // 2. Retrieval
    if (p.retrieval) {
        const r = p.retrieval;
        const docCount = (r.docs || []).length;
        stages.push({
            name: 'Retrieval',
            summary: `${docCount} document${docCount !== 1 ? 's' : ''} retrieved`,
            status: docCount > 0 ? 'success' : 'warning',
            details: [
                ...(r.rewritten_query ? [{ label: 'Rewritten', value: r.rewritten_query }] : []),
                { label: 'Retries', value: String(r.retry_count || 0) },
            ],
            docsTable: r.docs,
        });
    }

    // 3. Context Grading
    if (p.context_grading && p.context_grading.grade) {
        const g = p.context_grading;
        stages.push({
            name: 'Context Grading',
            summary: g.grade,
            status: g.grade === 'sufficient' ? 'success' : 'error',
            details: [
                { label: 'Grade', value: g.grade, badge: g.grade === 'sufficient' ? 'success' : 'error' },
                { label: 'Reason', value: g.reason },
            ],
        });
    }

    // 4. Generation
    if (p.generation) {
        const g = p.generation;
        const citCount = (g.citations || []).length;
        stages.push({
            name: 'Generation',
            summary: `${citCount} citation${citCount !== 1 ? 's' : ''}`,
            status: 'info',
            details: [
                { label: 'Citations', value: (g.citations || []).join(', ') || '—' },
                { label: 'Gen Retries', value: String(g.retry_count || 0) },
            ],
        });
    }

    // 5. Citation Verification
    if (p.citation_verification) {
        const cv = p.citation_verification;
        const vCount = (cv.verified || []).length;
        const fCount = (cv.failed || []).length;
        stages.push({
            name: 'Citation Verification',
            summary: `${vCount} verified, ${fCount} failed`,
            status: fCount === 0 ? 'success' : 'warning',
            details: [
                { label: 'Verified', value: (cv.verified || []).join(', ') || '—' },
                ...(fCount > 0 ? [{ label: 'Failed', value: cv.failed.join(', ') }] : []),
            ],
        });
    }

    // 6. Groundedness
    if (p.groundedness && p.groundedness.grade) {
        const g = p.groundedness;
        stages.push({
            name: 'Groundedness',
            summary: g.grade,
            status: g.grade === 'grounded' ? 'success' : 'error',
            details: [
                { label: 'Grade', value: g.grade, badge: g.grade === 'grounded' ? 'success' : 'error' },
                { label: 'Reason', value: g.reason },
            ],
        });
    }

    // 7. Answerability
    if (p.answerability && p.answerability.grade) {
        const a = p.answerability;
        stages.push({
            name: 'Answerability',
            summary: a.grade,
            status: a.grade === 'answers' ? 'success' : 'error',
            details: [
                { label: 'Grade', value: a.grade, badge: a.grade === 'answers' ? 'success' : 'error' },
                { label: 'Reason', value: a.reason },
            ],
        });
    }

    // 8. Abstain (if triggered)
    if (p.abstain_info && p.abstain_info.triggered) {
        stages.push({
            name: 'Abstained',
            summary: 'Pipeline abstained',
            status: 'error',
            details: [
                { label: 'Reason', value: p.abstain_info.reason },
            ],
        });
    }

    // 9. Trace log (always)
    if (trace && trace.length > 0) {
        stages.push({
            name: 'Full Trace',
            summary: `${trace.length} entries`,
            status: 'info',
            traceLog: trace,
        });
    }

    return stages;
}

function renderStage(stage) {
    const el = document.createElement('div');
    el.className = `stage ${stage.status}`;

    // Dot
    const dot = document.createElement('div');
    dot.className = 'stage-dot';
    el.appendChild(dot);

    // Header
    const header = document.createElement('div');
    header.className = 'stage-header';
    header.innerHTML = `
        <span class="stage-name">${escapeHtml(stage.name)}</span>
        <span class="stage-summary">${escapeHtml(stage.summary || '')}</span>
        <span class="stage-chevron">▸</span>
    `;

    // Body
    const body = document.createElement('div');
    body.className = 'stage-body';

    // Details
    if (stage.details) {
        stage.details.forEach(d => {
            const row = document.createElement('div');
            row.className = 'stage-detail';

            const label = document.createElement('span');
            label.className = 'detail-label';
            label.textContent = d.label;

            const value = document.createElement('span');
            value.className = 'detail-value';

            if (d.badge) {
                const badge = document.createElement('span');
                badge.className = `status-badge ${d.badge}`;
                badge.textContent = d.value || '—';
                value.appendChild(badge);
            } else {
                value.textContent = d.value || '—';
            }

            row.appendChild(label);
            row.appendChild(value);
            body.appendChild(row);
        });
    }

    // Docs table
    if (stage.docsTable && stage.docsTable.length > 0) {
        body.appendChild(renderDocsTable(stage.docsTable));
    }

    // Trace log
    if (stage.traceLog) {
        body.appendChild(renderTraceLog(stage.traceLog));
    }

    // Toggle
    header.addEventListener('click', () => {
        el.classList.toggle('open');
    });

    el.appendChild(header);
    el.appendChild(body);
    return el;
}

function renderDocsTable(docs) {
    const table = document.createElement('table');
    table.className = 'docs-table';

    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>#</th>
            <th>Provision ID</th>
            <th>Title</th>
            <th>Score</th>
        </tr>
    `;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    docs.forEach((d, i) => {
        const tr = document.createElement('tr');
        const score = typeof d.combined_score === 'number' ? d.combined_score.toFixed(3) : '—';
        tr.innerHTML = `
            <td>${i + 1}</td>
            <td>${escapeHtml(d.provision_id || '—')}</td>
            <td>${escapeHtml(d.title || '—')}</td>
            <td class="score">${score}</td>
        `;

        // Add tooltip/expand for text preview if available
        if (d.text) {
            tr.title = d.text.substring(0, 200) + '…';
            tr.style.cursor = 'help';
        }

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    return table;
}

function renderTraceLog(trace) {
    const log = document.createElement('div');
    log.className = 'trace-log';
    trace.forEach(entry => {
        const line = document.createElement('div');
        line.className = 'trace-entry';
        line.textContent = entry;
        log.appendChild(line);
    });
    return log;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Lab
// ═══════════════════════════════════════════════════════════════════════════════

async function runLabTest() {
    const query = dom.labQuery.value.trim();
    if (!query) {
        dom.labQuery.focus();
        return;
    }

    const unit = dom.labUnit.value;
    const provider = dom.labProvider.value;
    const topK = parseInt(dom.labTopk.value);

    // Loading state
    dom.labRunBtn.classList.add('loading');
    dom.labRunBtn.disabled = true;
    dom.labResults.classList.remove('visible');

    const startTime = Date.now();

    // Map unit to endpoint
    const endpoints = {
        query:    '/api/test/query',
        rag:      '/api/test/rag',
        classify: '/api/test/classify',
    };

    try {
        const res = await fetch(endpoints[unit], {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, provider, top_k: topK }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

        // Render result
        dom.labResultsTitle.textContent = `${unitLabel(unit)} — Output`;
        dom.labResultsTime.textContent = `${elapsed}s`;
        dom.labResultsBody.innerHTML = '';
        dom.labResultsBody.appendChild(renderJsonHighlighted(data));
        dom.labResults.classList.add('visible');

    } catch (err) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        dom.labResultsTitle.textContent = 'Error';
        dom.labResultsTime.textContent = `${elapsed}s`;
        dom.labResultsBody.innerHTML = '';
        const errEl = document.createElement('div');
        errEl.className = 'lab-error';
        errEl.textContent = err.message;
        dom.labResultsBody.appendChild(errEl);
        dom.labResults.classList.add('visible');
    } finally {
        dom.labRunBtn.classList.remove('loading');
        dom.labRunBtn.disabled = false;
    }
}

function unitLabel(unit) {
    return {
        query: 'Query Processing',
        rag: 'RAG Pipeline',
        classify: 'Domain Classification',
    }[unit] || unit;
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON Syntax Highlighting
// ═══════════════════════════════════════════════════════════════════════════════

function renderJsonHighlighted(data) {
    const pre = document.createElement('pre');
    const json = JSON.stringify(data, null, 2);

    // Simple syntax highlighting
    const highlighted = json.replace(
        /("(?:\\.|[^"\\])*")\s*:/g,
        '<span class="json-key">$1</span>:'
    ).replace(
        /:\s*("(?:\\.|[^"\\])*")/g,
        ': <span class="json-string">$1</span>'
    ).replace(
        /:\s*(\d+\.?\d*)/g,
        ': <span class="json-number">$1</span>'
    ).replace(
        /:\s*(true|false)/g,
        ': <span class="json-bool">$1</span>'
    ).replace(
        /:\s*(null)/g,
        ': <span class="json-null">$1</span>'
    );

    pre.innerHTML = highlighted;
    return pre;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════════

function genId() {
    return 'msg_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 7);
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', init);
