const chatEl = document.getElementById('chat');
const questionEl = document.getElementById('question');
const baseUrlEl = document.getElementById('baseUrl');
const serviceStatus = document.getElementById('serviceStatus');
const metaLine = document.getElementById('metaLine');
const hint = document.getElementById('hint');
const retrievedPanel = document.getElementById('retrievedPanel');
const retrievedCount = document.getElementById('retrievedCount');
const promptPreview = document.getElementById('promptPreview');
const metricRetrievalMs = document.getElementById('metricRetrievalMs');
const metricDenseMs = document.getElementById('metricDenseMs');
const metricBm25Ms = document.getElementById('metricBm25Ms');
const metricMemoryMs = document.getElementById('metricMemoryMs');
const metricRerankMs = document.getElementById('metricRerankMs');
const metricPromptChars = document.getElementById('metricPromptChars');
const chipsEl = document.getElementById('chips');
const refreshQuickBtn = document.getElementById('btnRefreshQuick');
const quickFilters = document.querySelectorAll('.quick-filter');
const btnShowRetrieved = document.getElementById('btnShowRetrieved');
const retrievedModal = document.getElementById('retrievedModal');
const btnCloseRetrieved = document.getElementById('btnCloseRetrieved');
const retrievedModalBody = document.getElementById('retrievedModalBody');
const retrievedRelevantBody = document.getElementById('retrievedRelevantBody');
const retrievedIrrelevantBody = document.getElementById('retrievedIrrelevantBody');
const retrievedMemoryBody = document.getElementById('retrievedMemoryBody');

const sessionId = crypto.randomUUID();
const userId = 'user_' + Math.random().toString(36).slice(2, 10);

let lastRetrieved = [];
let lastRetrievedContext = [];

function apiBase() { return baseUrlEl.value.replace(/\/$/, ''); }

function addMessage(text, cls = 'assistant') {
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setStatus(text, ok = true) {
  serviceStatus.textContent = text;
  serviceStatus.style.color = ok ? '#d1fae5' : '#fecaca';
}

function getPayload() {
  return {
    session_id: sessionId,
    user_id: userId,
    question: questionEl.value.trim(),
    role_name: document.getElementById('roleName').value,
    top_k: Number(document.getElementById('topK').value || 8),
    bm25_top_k: Number(document.getElementById('bm25TopK').value || 30),
    rerank_top_k: Number(document.getElementById('rerankTopK').value || 5),
    max_tokens: Number(document.getElementById('maxTokens').value || 1024),
  };
}

function makeItemCard(item) {
  const div = document.createElement('div');
  div.className = 'modal-item';
  const content = item.content || item.vector_text || '';
  div.innerHTML = `
    <div class="hdr"><span>${item.retrieval_source || ''}/${item.memory_type || ''}</span><span>${item.id || ''}</span></div>
    <div class="ttl">${item.title || '未命名'}</div>
    <div class="txt">${content || '无内容'}</div>
  `;
  return div;
}

function renderRetrievedList(data) {
  const items = Array.isArray(data) ? data : [];
  lastRetrievedContext = items;
  retrievedCount.textContent = String(items.length);
  retrievedPanel.innerHTML = '';
  if (!items.length) {
    retrievedPanel.innerHTML = '<div class="footer-note">暂无检索结果。</div>';
    btnShowRetrieved.disabled = true;
    return;
  }
  btnShowRetrieved.disabled = false;
  for (const item of items) {
    const div = document.createElement('div');
    div.className = 'retrieved-item';
    const content = item.content || item.vector_text || '';
    div.innerHTML = `
      <div class="hdr"><span>${item.retrieval_source || ''}/${item.memory_type || ''}</span><span>${item.id || ''}</span></div>
      <div class="ttl">${item.title || '未命名'}</div>
      <div class="txt">${content.slice(0, 260)}${content.length > 260 ? '...' : ''}</div>
    `;
    retrievedPanel.appendChild(div);
  }
}

function updateMetrics(data = {}) {
  const timings = data.timings_ms || {};
  metricRetrievalMs.textContent = timings.total_retrieval_ms != null ? `${timings.total_retrieval_ms} ms` : '-';
  metricDenseMs.textContent = timings.dense_ms != null ? `${timings.dense_ms} ms` : '-';
  metricBm25Ms.textContent = timings.bm25_ms != null ? `${timings.bm25_ms} ms` : '-';
  metricMemoryMs.textContent = timings.memory_ms != null ? `${timings.memory_ms} ms` : '-';
  metricRerankMs.textContent = timings.rerank_ms != null ? `${timings.rerank_ms} ms` : '-';
  metricPromptChars.textContent = data.prompt_chars != null ? String(data.prompt_chars) : '-';
}

function openRetrievedModal() {
  const relevant = Array.isArray(window.__retrievedRelevant) ? window.__retrievedRelevant : [];
  const irrelevant = Array.isArray(window.__retrievedIrrelevant) ? window.__retrievedIrrelevant : [];
  const memory = Array.isArray(window.__retrievedMemory) ? window.__retrievedMemory : [];
  retrievedRelevantBody.innerHTML = '';
  retrievedIrrelevantBody.innerHTML = '';
  retrievedMemoryBody.innerHTML = '';

  if (relevant.length) relevant.forEach((item) => retrievedRelevantBody.appendChild(makeItemCard(item)));
  else retrievedRelevantBody.innerHTML = '<div class="footer-note">暂无相关检索。</div>';

  if (irrelevant.length) irrelevant.forEach((item) => retrievedIrrelevantBody.appendChild(makeItemCard(item)));
  else retrievedIrrelevantBody.innerHTML = '<div class="footer-note">暂无无关检索。</div>';

  if (memory.length) memory.forEach((item) => retrievedMemoryBody.appendChild(makeItemCard(item)));
  else retrievedMemoryBody.innerHTML = '<div class="footer-note">暂无长期记忆。</div>';

  retrievedModal.hidden = false;
}

function closeRetrievedModal() {
  retrievedModal.hidden = true;
}

async function checkHealth() {
  try {
    const res = await fetch(`${apiBase()}/health`);
    const data = await res.json();
    setStatus(data.status === 'ok' ? '服务正常' : '异常', data.status === 'ok');
  } catch (e) {
    setStatus('后端未连接', false);
  }
}

function domainToCategory(domain) {
  const value = String(domain || '').toLowerCase();
  if (value.includes('法')) return 'law';
  if (value.includes('医')) return 'medical';
  return 'assistant';
}

function extractUserQuestion(text) {
  const raw = String(text || '').trim();
  if (!raw) return '';
  const match = raw.match(/用户问题[:：]\s*([\s\S]+)/);
  return (match ? match[1] : raw)
    .replace(/^【[^】]*】\s*/g, '')
    .replace(/^[\s\-—:：]+/, '')
    .replace(/[。.!！?？]+$/g, '')
    .trim();
}

function formatQuestionLikeText(text) {
  const cleaned = extractUserQuestion(text);
  if (!cleaned) return '这个知识点怎么理解？';
  if (cleaned.includes('如何') || cleaned.includes('怎么') || cleaned.includes('什么') || cleaned.includes('是否')) {
    return cleaned.endsWith('？') ? cleaned : `${cleaned}？`;
  }
  const short = cleaned.slice(0, 12);
  const templates = [
    `${short}一般怎么理解？`,
    `${short}通常怎么判断？`,
    `${short}怎么处理？`,
    `${short}需要注意什么？`,
    `${short}由谁负责？`,
  ];
  return templates[Math.floor(Math.random() * templates.length)];
}

async function loadRandomQuickQuestions(category = 'all') {
  try {
    const res = await fetch(`${apiBase()}/knowledge/random?limit=4&category=${encodeURIComponent(category)}&t=${Date.now()}`);
    const data = await res.json();
    const items = Array.isArray(data.items) ? data.items : [];
    if (!items.length) return;
    chipsEl.innerHTML = '';
    for (const item of items) {
      const chip = document.createElement('div');
      chip.className = 'chip';
      const sourceText = item.question || item.content || item.title || '随机知识';
      chip.textContent = formatQuestionLikeText(sourceText);
      chip.dataset.question = extractUserQuestion(item.question || item.content || item.title || '');
      chipsEl.appendChild(chip);
    }
  } catch (e) {
    // 保持默认快捷提问，不影响页面使用
  }
}

async function sendChat(stream = false) {
  const payload = getPayload();
  if (!payload.question) {
    hint.textContent = '请输入问题后再发送。';
    return;
  }
  const questionText = payload.question;
  questionEl.value = '';
  const endpoint = stream ? '/chat/stream' : '/chat';
  addMessage(questionText, 'user');
  const placeholder = addMessage(stream ? '正在思考...' : '等待返回...', 'assistant');
  metaLine.textContent = stream ? '流式模式已启动...' : '请求处理中...';
  promptPreview.textContent = questionText;
  hint.textContent = '';
  btnShowRetrieved.disabled = true;
  lastRetrievedContext = [];

  try {
    if (stream) {
      const res = await fetch(apiBase() + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let answer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith('data: ')) continue;
          const payloadText = line.slice(6);
          if (payloadText === '[DONE]') continue;
          const event = JSON.parse(payloadText);
          if (event.type === 'meta') {
            metaLine.textContent = `检索：${event.retrieval_stats?.merged ?? 0} 条，重排：${event.retrieval_stats?.reranked ?? 0} 条`;
            if (event.prompt_preview) promptPreview.textContent = event.prompt_preview.split('\n').pop() || event.prompt_preview;
            if (Array.isArray(event.retrieved_all)) {
              lastRetrieved = event.retrieved_all;
              renderRetrievedList(lastRetrieved);
            }
            updateMetrics(event);
          } else if (event.type === 'delta') {
            answer += event.content || '';
            placeholder.textContent = answer || '正在思考...';
            chatEl.scrollTop = chatEl.scrollHeight;
          } else if (event.type === 'done') {
            if (event.answer) {
              answer = event.answer;
              placeholder.textContent = answer;
            } else if (answer) {
              placeholder.textContent = answer;
            }
            if (Array.isArray(event.retrieved_all)) {
              lastRetrieved = event.retrieved_all;
              renderRetrievedList(lastRetrieved);
            }
            window.__retrievedRelevant = Array.isArray(event.retrieved_relevant) ? event.retrieved_relevant : window.__retrievedRelevant || [];
            window.__retrievedIrrelevant = Array.isArray(event.retrieved_irrelevant) ? event.retrieved_irrelevant : window.__retrievedIrrelevant || [];
            window.__retrievedMemory = [];
            if (event.memory_saved !== undefined) {
              metaLine.textContent += ` | 长期记忆：${event.memory_saved ? '已保存' : '未保存'}`;
            }
          }
        }
      }
      if (!answer) placeholder.textContent = '无返回内容';
    } else {
      const res = await fetch(apiBase() + endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      placeholder.textContent = data.answer || '无返回内容';
      metaLine.textContent = `检索数：${data.retrieval_count || 0}，耗时：${data.latency_ms || 0} ms`;
      if (Array.isArray(data.retrieved)) {
        lastRetrieved = data.retrieved;
        renderRetrievedList(lastRetrieved);
      }
      updateMetrics(data);
      if (data.prompt_preview) promptPreview.textContent = data.prompt_preview.split('\n').pop() || data.prompt_preview;
      btnShowRetrieved.disabled = !(Array.isArray(data.retrieved) && data.retrieved.length);
      lastRetrievedContext = Array.isArray(data.retrieved) ? data.retrieved : [];
      window.__retrievedRelevant = Array.isArray(data.retrieved_relevant) ? data.retrieved_relevant : [];
      window.__retrievedIrrelevant = Array.isArray(data.retrieved_irrelevant) ? data.retrieved_irrelevant : [];
      window.__retrievedMemory = [];
      if (answer) placeholder.textContent = answer;
    }
  } catch (e) {
    placeholder.textContent = `请求失败：${e.message}`;
    setStatus('请求失败', false);
  }
}

async function resetSession() {
  try {
    const url = new URL(apiBase() + '/session/reset');
    url.searchParams.set('session_id', sessionId);
    const res = await fetch(url.toString(), { method: 'POST' });
    const data = await res.json();
    chatEl.innerHTML = '';
    questionEl.value = '';
    retrievedPanel.innerHTML = '<div class="footer-note">回答后会显示最近一次检索结果。</div>';
    retrievedCount.textContent = '0';
    promptPreview.textContent = '暂无预览内容。';
    metaLine.textContent = '';
    updateMetrics({});
    hint.textContent = '会话已重置，输入框和回答区已清空。';
    lastRetrieved = [];
    lastRetrievedContext = [];
    btnShowRetrieved.disabled = true;
    closeRetrievedModal();
    addMessage(`会话已重置：${data.session_id}`, 'meta');
    const active = document.querySelector('.quick-filter.active');
    loadRandomQuickQuestions(active?.dataset.category || 'all');
  } catch (e) {
    addMessage(`重置失败：${e.message}`, 'meta');
  }
}

document.getElementById('btnSend').onclick = () => sendChat(true);
document.getElementById('btnStream').onclick = () => sendChat(true);
document.getElementById('btnChat').onclick = () => sendChat(false);
document.getElementById('btnReset').onclick = resetSession;
document.getElementById('btnHealth').onclick = checkHealth;

document.getElementById('question').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat(true);
  }
});

chipsEl.addEventListener('click', (e) => {
  if (e.target.classList.contains('chip')) {
    questionEl.value = e.target.dataset.question || e.target.textContent.trim();
    questionEl.focus();
  }
});

btnShowRetrieved.addEventListener('click', openRetrievedModal);
btnCloseRetrieved.addEventListener('click', closeRetrievedModal);
retrievedModal.addEventListener('click', (e) => {
  if (e.target === retrievedModal) closeRetrievedModal();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !retrievedModal.hidden) closeRetrievedModal();
});

refreshQuickBtn.addEventListener('click', () => {
  const active = document.querySelector('.quick-filter.active');
  loadRandomQuickQuestions(active?.dataset.category || 'all');
});

quickFilters.forEach((btn) => {
  btn.addEventListener('click', () => {
    quickFilters.forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    loadRandomQuickQuestions(btn.dataset.category || 'all');
  });
});

addMessage('我是多角色问答系统，支持法律、医疗与通用问答，请开始提问。', 'assistant');
checkHealth();
loadRandomQuickQuestions();
