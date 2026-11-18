
let documentRevision = null;
function escapeHTML(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function metric(value, digits=3) { return value == null ? 'N/A' : value.toFixed(digits); }
function proxyPercent(value) { return value == null ? 'N/A' : (value * 100).toFixed(0) + '%'; }
async function apiFetch(url, options = {}) {
  const token = document.getElementById('owner-token').value;
  options.headers = {...options.headers, ...(token ? {'Authorization':'Bearer ' + token} : {})};
  try {
    const response = await window.fetch(url, options);
    if (!response.ok) {
      const detail = await response.text();
      throw new Error('HTTP ' + response.status + ': ' + detail);
    }
    document.getElementById('app-status').textContent = 'Connected. Lexical overlap does not establish answer correctness.';
    return response;
  } catch (error) {
    document.getElementById('app-status').textContent = error.message;
    throw error;
  }
}
async function readDocumentFile(input) {
  const file = input.files[0];
  if (!file) return;
  if (file.size > 500000) { document.getElementById('app-status').textContent = 'File exceeds 500 KB'; return; }
  document.getElementById('doc-text').value = await file.text();
  if (!document.getElementById('doc-title').value) document.getElementById('doc-title').value = file.name;
}
async function refreshDocuments() {
  const data = await (await apiFetch('/api/v1/documents')).json();
  documentRevision = data.corpus_revision;
  document.getElementById('doc-revision').textContent = 'Corpus revision: ' + documentRevision;
  const list = document.getElementById('doc-list'); list.replaceChildren();
  for (const doc of data.documents) {
    const row = document.createElement('div'); row.className = 'chunk-item';
    const label = document.createElement('span');
    label.textContent = doc.doc_id + ' · ' + doc.title + ' · v' + (doc.version || '?') + ' · ' + doc.characters + ' chars ';
    row.appendChild(label);
    const edit = document.createElement('button'); edit.className = 'btn-secondary'; edit.textContent = 'Edit';
    edit.onclick = async () => {
      const result = await (await apiFetch('/api/v1/documents/' + encodeURIComponent(doc.doc_id))).json();
      documentRevision = result.corpus_revision;
      for (const [field, key] of Object.entries({'id':'doc_id','title':'title','domain':'domain','version':'version','source':'source','license':'license','text':'text'}))
        document.getElementById('doc-' + field).value = result.document[key] || '';
    };
    const remove = document.createElement('button'); remove.className = 'btn-secondary'; remove.textContent = 'Delete';
    remove.onclick = async () => {
      if (!window.confirm('Delete ' + doc.doc_id + ' and rebuild its index?')) return;
      await apiFetch('/api/v1/documents/' + encodeURIComponent(doc.doc_id), {method:'DELETE',headers:{'If-Match':documentRevision}});
      await refreshDocuments(); await refreshStats();
    };
    row.append(edit, remove); list.appendChild(row);
  }
}
async function saveDocument() {
  const button = document.getElementById('doc-save'); button.disabled = true;
  try {
    const payload = {};
    for (const [field, key] of Object.entries({'id':'doc_id','title':'title','domain':'domain','version':'version','source':'source','license':'license','text':'text'}))
      payload[key] = document.getElementById('doc-' + field).value;
    await apiFetch('/api/v1/documents', {method:'POST',headers:{'Content-Type':'application/json',...(documentRevision ? {'If-Match':documentRevision} : {})},body:JSON.stringify(payload)});
    await refreshDocuments(); await refreshStats();
  } finally { button.disabled = false; }
}

function switchTab(id, btn) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
}

async function refreshStats() {
  const res = await apiFetch('/api/v1/stats');
  const data = await res.json();
  const t = data.telemetry;
  document.getElementById('pill-docs').textContent = t.corpus_documents;
  document.getElementById('pill-chunks').textContent = t.indexed_chunks;
  document.getElementById('pill-hit').textContent = (t.effective_cache_hit_rate * 100).toFixed(1) + '%';
  document.getElementById('pill-breaker').textContent = t.circuit_breaker_state;

  const domSel = document.getElementById('q-domain');
  if (domSel.options.length <= 1) {
    data.domains.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d; opt.textContent = d;
      domSel.appendChild(opt);
    });
  }

  const chips = document.getElementById('sample-chips');
  if (chips.children.length === 0) {
    data.sample_queries.slice(0, 10).forEach(s => {
      const btn = document.createElement('button');
      btn.className = 'chip';
      btn.textContent = `[${s.category}] ${s.question.slice(0, 48)}...`;
      btn.onclick = () => { document.getElementById('q-input').value = s.question; runQuery(); };
      chips.appendChild(btn);
    });
  }
}

function renderQueryResponse(data) {
  document.getElementById('answer-box').textContent = data.answer;
  const tierClass = data.blocked_by_guardrail ? 'b-bad' : (data.cache_tier === 'miss' ? 'b-warn' : 'b-ok');
  document.getElementById('resp-badges').innerHTML = `
    <span class="badge ${tierClass}">Tier: ${data.cache_tier}</span>
    <span class="badge b-info">Lexical Grounding Proxy: ${proxyPercent(data.lexical_grounding_proxy)}</span>
    <span class="badge b-info">Heuristic score: ${data.confidence.toFixed(3)}</span>
  `;
  document.getElementById('metrics-strip').innerHTML = `
    <span>Generation: <strong>${escapeHTML(data.generation_mode)}</strong></span>
    <span>Latency: <strong>${data.latency_ms} ms</strong></span>
    <span>Citations: <strong>${escapeHTML(data.citations.join(', ') || 'None')}</strong></span>
    <span>PII Redacted: <strong>${data.pii_redacted}</strong></span>
    <span>L2 Sim: <strong>${data.l2_similarity}</strong></span>
  `;

  const evDiv = document.getElementById('evidence-list');
  if (!data.evidence || data.evidence.length === 0) {
    evDiv.innerHTML = '<div class="chunk-item">No chunks passed security/confidence gates.</div>';
  } else {
    evDiv.innerHTML = data.evidence.map((ev, i) => `
      <div class="chunk-item">
        <div class="chunk-meta">
          <strong>#${i + 1} [${escapeHTML(ev.doc_id)}] ${escapeHTML(ev.title)}</strong>
          <span class="badge b-info">${escapeHTML(ev.domain)}</span>
        </div>
        <div style="color:var(--muted); font-size:0.78rem; margin-bottom:6px;">
          Dense: <code>${ev.dense_score}</code> · BM25: <code>${ev.bm25_score}</code> · RRF: <code>${ev.rrf_score}</code> · Rerank: <code>${ev.rerank_score}</code>
        </div>
        <div>${escapeHTML(ev.parent_text.slice(0, 320))}...</div><small>Source: ${escapeHTML(ev.source || "synthetic / unspecified")} · Version: ${escapeHTML(ev.version || "unspecified")}</small>
      </div>
    `).join('');
  }
  refreshStats();
}

async function runQuery() {
  const btn = document.getElementById('btn-run-query');
  btn.disabled = true;
  try {
    const payload = {
      query: document.getElementById('q-input').value,
      retrieval_mode: document.getElementById('q-mode').value,
      use_reranker: document.getElementById('q-rerank').value === 'true',
      use_cache: document.getElementById('q-cache').value === 'true',
      domain_filter: document.getElementById('q-domain').value || null,
    };
    const res = await apiFetch('/api/v1/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
    });
    const data = await res.json();
    renderQueryResponse(data);
  } finally { btn.disabled = false; }
}

async function runQueryStream() {
  const ansBox = document.getElementById('answer-box');
  ansBox.textContent = '';
  const payload = {
    query: document.getElementById('q-input').value,
    retrieval_mode: document.getElementById('q-mode').value,
    use_reranker: document.getElementById('q-rerank').value === 'true',
    use_cache: document.getElementById('q-cache').value === 'true',
    domain_filter: document.getElementById('q-domain').value || null,
  };
  const res = await apiFetch('/api/v1/query/stream', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const parts = buf.split('\n\n');
    buf = parts.pop() || '';
    for (const part of parts) {
      const lines = part.split('\n');
      const evLine = lines.find(l => l.startsWith('event: '));
      const dataLine = lines.find(l => l.startsWith('data: '));
      if (!evLine || !dataLine) continue;
      const ev = evLine.slice(7).trim();
      const obj = JSON.parse(dataLine.slice(6));
      if (ev === 'delta') ansBox.textContent += obj.token;
      if (ev === 'done') {
        document.getElementById('resp-badges').innerHTML = `
          <span class="badge b-info">Buffered SSE Replay</span>
          <span class="badge b-info">Lexical Grounding Proxy: ${proxyPercent(obj.lexical_grounding_proxy)}</span>
          <span class="badge b-info">Latency: ${obj.latency_ms} ms</span>
        `;
        refreshStats();
      }
    }
  }
}

async function loadAblation(refresh = false) {
  const res = await apiFetch('/api/v1/ablation?refresh=' + refresh);
  const data = await res.json();
  const sig = data.significance_flagship_vs_baseline;
  document.getElementById('ablation-sig').textContent =
    (data.report_matches_current_data === false ? 'Saved report uses different data. ' : '') + `Full pipeline vs Baseline: Recall@5 +${sig.recall_at_5.delta.toFixed(4)} (bootstrap fraction=${sig.recall_at_5.bootstrap_nonpositive_fraction}) | NDCG@5 +${sig.ndcg_at_5.delta.toFixed(4)} (bootstrap fraction=${sig.ndcg_at_5.bootstrap_nonpositive_fraction})`;

  const tbody = document.getElementById('ablation-tbody');
  tbody.innerHTML = data.experiments.map((exp, idx) => {
    const m = exp.metrics;
    const isBest = idx === data.experiments.length - 1;
    return `<tr style="${isBest ? 'background:rgba(99,102,241,0.12); font-weight:600;' : ''}">
      <td>${exp.spec.label}</td>
      <td>${exp.num_chunks}</td>
      <td>${metric(m.recall_at_3.mean, 3)}</td>
      <td>${metric(m.recall_at_5.mean, 3)} <span style="color:var(--muted);font-size:0.78em;">[${metric(m.recall_at_5.ci_low, 2)}, ${metric(m.recall_at_5.ci_high, 2)}]</span></td>
      <td>${metric(m.mrr.mean, 3)}</td>
      <td>${metric(m.ndcg_at_5.mean, 3)} <span style="color:var(--muted);font-size:0.78em;">[${metric(m.ndcg_at_5.ci_low, 2)}, ${metric(m.ndcg_at_5.ci_high, 2)}]</span></td>
      <td>${metric(m.lexical_grounding_proxy.mean, 3)}</td>
      <td>${metric(m.refusal_accuracy.mean, 3)}</td>
      <td>${metric(m.latency_ms.mean, 2)} ms</td>
    </tr>`;
  }).join('');

  const cats = data.experiments[data.experiments.length - 1].by_category;
  document.getElementById('category-cards').innerHTML = Object.entries(cats).map(([k, v]) => `
    <div class="card" style="margin:0;">
      <div class="kpi-label">${k} (${v.count})</div>
      <div class="kpi-val">${((v.recall_at_5 ?? v.refusal_accuracy) * 100).toFixed(1)}%</div>
      <div class="kpi-sub">${v.mrr !== undefined ? `MRR: ${v.mrr.toFixed(3)} · NDCG@5: ${v.ndcg_at_5.toFixed(3)}` : `Lexical Grounding Proxy: ${(v.lexical_grounding_proxy == null ? 'N/A' : v.lexical_grounding_proxy.toFixed(3))}`}</div>
    </div>
  `).join('');
}

function renderBenchmark(rep) {
  const s = rep.summary;
  const ol = rep.overall_latency_ms;
  document.getElementById('bench-kpis').innerHTML = `
    <div class="card"><div class="kpi-label">Completion Rate</div><div class="kpi-val">${(s.success_rate * 100).toFixed(2)}%</div><div class="kpi-sub">${s.succeeded.toLocaleString()} / ${rep.config.total_requests.toLocaleString()} OK</div></div>
    <div class="card"><div class="kpi-label">Achieved Throughput</div><div class="kpi-val">${s.achieved_rps.toLocaleString()} RPS</div><div class="kpi-sub">Wall time: ${s.wall_time_seconds} s</div></div>
    <div class="card"><div class="kpi-label">P50 / P99 Latency</div><div class="kpi-val">${ol.p50} ms</div><div class="kpi-sub">P99: ${ol.p99} ms · Peak Gate: ${s.peak_backend_concurrency}</div></div>
    <div class="card"><div class="kpi-label">Cache + Singleflight Offload</div><div class="kpi-val">${(s.effective_cache_and_coalesce_rate * 100).toFixed(1)}%</div><div class="kpi-sub">Avoided backend executions in this experiment</div></div>
  `;
  document.getElementById('bench-tbody').innerHTML = Object.entries(rep.tier_breakdown).map(([tier, info]) => `
    <tr>
      <td><code>${tier}</code></td>
      <td>${info.count.toLocaleString()}</td>
      <td>${info.share_pct}%</td>
      <td>${info.latency_ms.p50}</td>
      <td>${info.latency_ms.p95}</td>
      <td>${info.latency_ms.p99}</td>
      <td style="width:180px;"><div class="bar-track"><div class="bar-fill" style="width:${info.share_pct}%"></div></div></td>
    </tr>
  `).join('');
}

async function loadBenchmark() {
  const res = await apiFetch('/api/v1/benchmark');
  renderBenchmark(await res.json());
}

async function runLiveBenchmark() {
  const btn = document.getElementById('btn-bench');
  btn.disabled = true;
  btn.textContent = 'Running Experiment...';
  try {
    const n = parseInt(document.getElementById('bench-size').value, 10);
    const res = await apiFetch('/api/v1/benchmark/run', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({total_requests: n, simulated_llm_latency_ms: 2.0})
    });
    renderBenchmark(await res.json());
    refreshStats();
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Synthetic Experiment';
  }
}

function calcLittles() {
  const L = parseFloat(document.getElementById('ll-users').value) || 10000;
  const hit = Math.min(99, Math.max(0, parseFloat(document.getElementById('ll-hit').value) || 75)) / 100;
  const coldMs = parseFloat(document.getElementById('ll-cold').value) || 320;
  const cacheMs = 2.0;
  const blendedW = (hit * cacheMs + (1 - hit) * coldMs) / 1000.0;
  const coldOnlyW = coldMs / 1000.0;
  const lambdaBlended = L / coldOnlyW;
  const coldRequestsPerSec = lambdaBlended * (1 - hit);
  const coldConcurrentSlots = coldRequestsPerSec * coldOnlyW;
  document.getElementById('ll-output').textContent =
    `Blended Latency W = ${(blendedW * 1000).toFixed(1)} ms (vs ${coldMs} ms uncached)\n` +
    `Fixed offered throughput: ${Math.round(lambdaBlended).toLocaleString()} RPS; mean in-flight after caching: ${Math.round(lambdaBlended * blendedW)}\n` +
    `Active Cold LLM Slots Needed: ${Math.round(coldConcurrentSlots).toLocaleString()} slots (vs ${L.toLocaleString()} without cache -> ${((1 - coldConcurrentSlots / L) * 100).toFixed(1)}% fewer cold slots; model assumptions apply)`;
}

refreshStats();
loadAblation(false);
loadBenchmark();
refreshDocuments();
calcLittles();
