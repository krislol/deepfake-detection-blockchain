'use strict';

const API_BASE = 'http://localhost:8000';

async function loadChain() {
  try {
    const res = await fetch(`${API_BASE}/blockchain`, {
      signal: AbortSignal.timeout(6000)
    });
    if (!res.ok) throw new Error(`API returned ${res.status}`);
    const data = await res.json();
    render(data);
    show('chainContainer');
    hide('errorState');
  } catch (err) {
    const msg = err.message.includes('fetch') || err.name === 'TimeoutError'
      ? 'Cannot reach the API. Is the server running on localhost:8000?'
      : err.message;
    document.getElementById('errorMsg').textContent = msg;
    show('errorState');
  }
}

function render(data) {
  const { chain, pending_transactions, stats } = data;

  // Stats
  document.getElementById('statBlocks').textContent  = stats.total_blocks;
  document.getElementById('statTx').textContent       = stats.total_confirmed_transactions;
  document.getElementById('statPending').textContent  = stats.pending_transactions;
  document.getElementById('statUntilNext').textContent = stats.transactions_until_next_block;

  // Pending
  const pendingSection = document.getElementById('pendingSection');
  if (pending_transactions.length > 0) {
    const remaining = stats.transactions_until_next_block;
    document.getElementById('pendingNote').textContent =
      `${remaining} more transaction(s) needed to mine the next block.`;
    document.getElementById('pendingList').innerHTML =
      pending_transactions.map(renderTxRow).join('');
    pendingSection.hidden = false;
  } else {
    pendingSection.hidden = true;
  }

  // Chain — show newest-first, genesis always last
  const container = document.getElementById('chainContainer');
  const emptyState = document.getElementById('emptyState');

  const confirmed = chain.slice(1).reverse(); // skip genesis, reverse

  if (confirmed.length === 0 && pending_transactions.length === 0) {
    emptyState.hidden = false;
    container.innerHTML = '';
    return;
  }
  emptyState.hidden = true;

  const html = confirmed.map(b => renderBlock(b, false)).join('');
  const genesis = chain[0] ? renderBlock(chain[0], true) : '';
  container.innerHTML = html + genesis;
}

function renderBlock(block, isGenesis) {
  const date     = new Date(block.timestamp + 'Z').toLocaleString();
  const txRows   = isGenesis ? '' : block.transactions.map(renderTxRow).join('');
  const label    = isGenesis ? '<span class="genesis-badge">GENESIS</span>' : '';
  const txLine   = isGenesis
    ? '<div class="block-tx-count genesis-note">Chain origin — no transactions</div>'
    : `<div class="block-tx-count">${block.transaction_count} transaction${block.transaction_count !== 1 ? 's' : ''}</div>`;

  return `
    <div class="block-card${isGenesis ? ' block-genesis' : ''}">
      <div class="block-header">
        <div class="block-number">Block #${block.block_number}${label}</div>
        <div class="block-time">${date}</div>
      </div>
      <div class="block-hashes">
        <div class="hash-row">
          <span class="hash-key">Block Hash</span>
          <code class="hash-val">${block.block_hash}</code>
        </div>
        <div class="hash-row">
          <span class="hash-key">Previous</span>
          <code class="hash-val hash-muted">${block.previous_block_hash}</code>
        </div>
      </div>
      ${txLine}
      ${txRows ? `<div class="tx-list">${txRows}</div>` : ''}
    </div>
  `;
}

function renderTxRow(tx) {
  const date = tx.registered_at
    ? new Date(tx.registered_at + 'Z').toLocaleString()
    : '—';
  return `
    <div class="tx-row">
      <div class="tx-hash">
        <span class="tx-label">TX</span>
        <code>${tx.tx_hash}</code>
      </div>
      <div class="tx-meta">
        <span class="tx-filename">${escapeHtml(tx.filename)}</span>
        <span class="tx-time">${date}</span>
      </div>
      <div class="tx-filehash">
        <span class="tx-label">File SHA-256</span>
        <code>${tx.file_hash}</code>
      </div>
    </div>
  `;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function show(id) { document.getElementById(id).hidden = false; }
function hide(id) { document.getElementById(id).hidden = true; }

loadChain();
setInterval(loadChain, 15000);
