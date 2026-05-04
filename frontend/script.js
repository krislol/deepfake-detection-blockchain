'use strict';

const API_BASE = 'http://localhost:8000';
const MAX_FILE_BYTES = 500 * 1024 * 1024; // 500 MB
const ALLOWED_EXTS = new Set(['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']);

// ── Element refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const apiBadge        = $('apiBadge');
const badgeDot        = $('badgeDot');
const apiStatus       = $('apiStatus');

const uploadSection   = $('uploadSection');
const uploadArea      = $('uploadArea');
const browseBtn       = $('browseBtn');
const fileInput       = $('fileInput');
const selectedFile    = $('selectedFile');
const fileName        = $('fileName');
const fileSize        = $('fileSize');
const clearFileBtn    = $('clearFileBtn');
const aggregationSel  = $('aggregationSelect');
const analyseBtn      = $('analyseBtn');

const loadingSection  = $('loadingSection');
const elapsedTime     = $('elapsedTime');

const resultsSection  = $('resultsSection');
const resultVerdict   = $('resultVerdict');
const verdictLabel    = $('verdictLabel');
const confidenceFill  = $('confidenceFill');
const confidenceThumb = $('confidenceThumb');
const confidencePct   = $('confidencePct');
const confTrack       = $('confTrack');
const riFilename      = $('riFilename');
const riFrames        = $('riFrames');
const riFaces         = $('riFaces');
const riTime          = $('riTime');
const riHash          = $('riHash');
const copyHashBtn     = $('copyHashBtn');
const registerBtn     = $('registerBtn');
const verifyBtn       = $('verifyBtn');
const bcResult        = $('bcResult');
const newUploadBtn    = $('newUploadBtn');

const errorSection    = $('errorSection');
const errorMessage    = $('errorMessage');
const retryBtn        = $('retryBtn');

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile   = null;
let lastFileHash  = null;
let elapsedTimer  = null;
let elapsedSecs   = 0;

// ── API health check ──────────────────────────────────────────────────────────
async function checkAPIHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      setApiStatus('online', 'API Online');
    } else {
      setApiStatus('offline', 'API Error');
    }
  } catch {
    setApiStatus('offline', 'API Offline');
  }
}

function setApiStatus(state, label) {
  badgeDot.className = 'badge-dot ' + state;
  apiStatus.textContent = label;
}

// Recheck every 30 s
checkAPIHealth();
setInterval(checkAPIHealth, 30_000);

// ── File selection ────────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('click', e => {
  if (e.target !== browseBtn) fileInput.click();
});

uploadArea.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});

clearFileBtn.addEventListener('click', () => resetFileSelection());

// ── Drag & drop ───────────────────────────────────────────────────────────────
uploadArea.addEventListener('dragover', e => {
  e.preventDefault();
  uploadArea.classList.add('dragover');
});

['dragleave', 'dragend'].forEach(ev =>
  uploadArea.addEventListener(ev, () => uploadArea.classList.remove('dragover'))
);

uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

// Global drop target so the whole page accepts drops
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});

// ── Handle file selection ─────────────────────────────────────────────────────
function handleFileSelect(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();

  if (!ALLOWED_EXTS.has(ext)) {
    showError(`Invalid file type "${ext}". Please use MP4, AVI, MOV, MKV, FLV, or WMV.`);
    return;
  }

  if (file.size > MAX_FILE_BYTES) {
    showError(`File is ${formatBytes(file.size)} — the limit is 500 MB.`);
    return;
  }

  currentFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  selectedFile.hidden  = false;
  analyseBtn.disabled  = false;
  hideError();
}

function resetFileSelection() {
  currentFile = null;
  lastFileHash = null;
  fileInput.value = '';
  selectedFile.hidden = true;
  analyseBtn.disabled = true;
}

// ── Analyse button ────────────────────────────────────────────────────────────
analyseBtn.addEventListener('click', () => {
  if (currentFile) uploadVideo(currentFile);
});

async function uploadVideo(file) {
  showLoading();

  const formData = new FormData();
  formData.append('file', file);
  const aggregation = aggregationSel.value;

  try {
    const res = await fetch(`${API_BASE}/detect?aggregation=${aggregation}`, {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || `Server error (${res.status})`);
    }

    if (!data.success) {
      throw new Error(data.detail || 'Detection failed.');
    }

    displayResults(data);
  } catch (err) {
    if (err.name === 'TypeError' && err.message.includes('fetch')) {
      showError('Cannot reach the API. Is the server running on localhost:8000?');
    } else {
      showError(err.message);
    }
  }
}

// ── Loading state ─────────────────────────────────────────────────────────────
function showLoading() {
  uploadSection.hidden  = true;
  errorSection.hidden   = true;
  resultsSection.hidden = true;
  loadingSection.hidden = false;

  elapsedSecs = 0;
  elapsedTime.textContent = '0s';
  elapsedTimer = setInterval(() => {
    elapsedSecs++;
    elapsedTime.textContent = elapsedSecs + 's';
  }, 1000);
}

function stopLoading() {
  clearInterval(elapsedTimer);
  loadingSection.hidden = true;
}

// ── Display results ───────────────────────────────────────────────────────────
function displayResults(data) {
  stopLoading();

  const isFake      = data.classification === 'FAKE';
  const fakeProb    = data.fake_probability;
  const pct         = Math.round(fakeProb * 100);

  // Verdict banner
  resultVerdict.className = `result-verdict ${isFake ? 'fake' : 'real'}`;
  verdictLabel.textContent = isFake ? 'FAKE' : 'REAL';

  // Confidence bar (fill = fake probability)
  requestAnimationFrame(() => {
    confidenceFill.style.width       = `${pct}%`;
    confidenceThumb.style.left       = `${pct}%`;
  });

  const confLabel = isFake
    ? `${pct}% fake probability — ${Math.round(data.confidence * 100)}% confidence`
    : `${100 - pct}% real probability — ${Math.round(data.confidence * 100)}% confidence`;
  confidencePct.textContent = confLabel;
  confTrack.setAttribute('aria-valuenow', pct);

  // Detail grid
  riFilename.textContent = data.filename;
  riFrames.textContent   = data.frames_analysed.toLocaleString();
  riFaces.textContent    = data.faces_detected.toLocaleString();
  riTime.textContent     = `${data.processing_time_seconds}s`;
  riHash.textContent     = data.file_hash;
  lastFileHash           = data.file_hash;

  // Reset blockchain panel
  bcResult.hidden = true;
  bcResult.textContent = '';
  registerBtn.disabled = false;

  resultsSection.hidden = false;
}

// ── Copy hash ─────────────────────────────────────────────────────────────────
copyHashBtn.addEventListener('click', async () => {
  const hash = riHash.textContent;
  if (!hash) return;
  try {
    await navigator.clipboard.writeText(hash);
    copyHashBtn.textContent = 'Copied!';
    copyHashBtn.classList.add('copied');
    setTimeout(() => {
      copyHashBtn.textContent = 'Copy';
      copyHashBtn.classList.remove('copied');
    }, 2000);
  } catch {
    copyHashBtn.textContent = 'Failed';
    setTimeout(() => { copyHashBtn.textContent = 'Copy'; }, 2000);
  }
});

// ── Blockchain: register ──────────────────────────────────────────────────────
registerBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  registerBtn.disabled = true;
  registerBtn.textContent = 'Registering…';
  bcResult.hidden = true;

  const formData = new FormData();
  formData.append('file', currentFile);

  try {
    const res  = await fetch(`${API_BASE}/register`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Registration failed');

    const blockRow = data.block_number != null
      ? `<div class="bc-detail-row">
           <span class="bc-key">Block</span>
           <span>#${data.block_number}</span>
         </div>
         <div class="bc-detail-row">
           <span class="bc-key">Block Hash</span>
           <span class="bc-mono">${data.block_hash}</span>
         </div>`
      : `<div class="bc-detail-row">
           <span class="bc-key">Status</span>
           <span>Pending — ${data.transactions_until_block} more transaction(s) until next block</span>
         </div>`;

    showBcResultHtml('bc-ok', `
      <div class="bc-status">Registered on Polygon (Simulation)</div>
      <div class="bc-detail-row">
        <span class="bc-key">TX Hash</span>
        <span class="bc-mono">${data.transaction_hash}</span>
      </div>
      ${blockRow}
      <a class="bc-link" href="blockchain.html" target="_blank" rel="noopener">
        View Full Blockchain Explorer
      </a>
    `);
  } catch (err) {
    showBcResult('bc-fail', err.message);
    registerBtn.disabled = false;
  }

  registerBtn.textContent = 'Register on Blockchain';
});

// ── Blockchain: verify ────────────────────────────────────────────────────────
verifyBtn.addEventListener('click', async () => {
  if (!lastFileHash) return;
  verifyBtn.disabled = true;
  verifyBtn.textContent = 'Verifying…';
  bcResult.hidden = true;

  try {
    const res  = await fetch(`${API_BASE}/verify/${lastFileHash}`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Verification failed');

    if (data.is_registered) {
      const blockRow = data.block_number != null
        ? `<div class="bc-detail-row">
             <span class="bc-key">Block</span>
             <span>#${data.block_number}</span>
           </div>
           <div class="bc-detail-row">
             <span class="bc-key">Block Hash</span>
             <span class="bc-mono">${data.block_hash}</span>
           </div>`
        : `<div class="bc-detail-row">
             <span class="bc-key">Status</span>
             <span>Pending — not yet in a block</span>
           </div>`;

      showBcResultHtml('bc-ok', `
        <div class="bc-status">File is registered and unmodified</div>
        <div class="bc-detail-row">
          <span class="bc-key">Filename</span>
          <span>${escapeHtml(data.original_filename || '—')}</span>
        </div>
        <div class="bc-detail-row">
          <span class="bc-key">Registered</span>
          <span>${data.registered_at ? new Date(data.registered_at).toLocaleString() : '—'}</span>
        </div>
        ${blockRow}
        <a class="bc-link" href="blockchain.html" target="_blank" rel="noopener">
          View Full Blockchain Explorer
        </a>
      `);
    } else {
      showBcResult('bc-neutral', data.message);
    }
  } catch (err) {
    showBcResult('bc-fail', err.message);
  }

  verifyBtn.disabled = false;
  verifyBtn.textContent = 'Verify Existing Hash';
});

function showBcResult(cls, text) {
  bcResult.className = `bc-result ${cls}`;
  bcResult.textContent = text;
  bcResult.hidden = false;
}

function showBcResultHtml(cls, html) {
  bcResult.className = `bc-result ${cls}`;
  bcResult.innerHTML = html;
  bcResult.hidden = false;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Error state ───────────────────────────────────────────────────────────────
function showError(msg) {
  stopLoading();
  uploadSection.hidden  = false;
  resultsSection.hidden = true;
  errorMessage.textContent = msg;
  errorSection.hidden = false;
}

function hideError() {
  errorSection.hidden = true;
}

retryBtn.addEventListener('click', () => {
  hideError();
});

// ── New upload ────────────────────────────────────────────────────────────────
newUploadBtn.addEventListener('click', () => {
  resultsSection.hidden = true;
  resetFileSelection();
  uploadSection.hidden = false;
});

// ── Utilities ─────────────────────────────────────────────────────────────────
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 ** 2)  return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 ** 3)  return (bytes / 1024 ** 2).toFixed(1) + ' MB';
  return (bytes / 1024 ** 3).toFixed(1) + ' GB';
}
