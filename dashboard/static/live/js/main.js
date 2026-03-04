'use strict';

// ═══════════════════════════════════════════════════════
// RELOJ
// ═══════════════════════════════════════════════════════
setInterval(() => {
    const clock = document.getElementById('clock');
    if (clock) {
        clock.textContent = new Date().toLocaleTimeString('es-CO', { timeZone: 'America/Bogota', hour12: false });
    }
}, 1000);

// ═══════════════════════════════════════════════════════
// SISTEMA DE PESTAÑAS
// ═══════════════════════════════════════════════════════
const TAB_IDS = ['candles', 'pnl', 'indicators', 'trades', 'results', 'log'];

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        // Ocultar todo
        TAB_IDS.forEach(id => {
            document.getElementById(`tab-${id}`).classList.remove('visible');
            document.getElementById(`tbtn-${id}`).classList.remove('active');
        });
        // Mostrar el seleccionado
        document.getElementById(`tab-${tab}`).classList.add('visible');
        btn.classList.add('active');
        // Redimensionar charts si aplica
        if (tab === 'candles') {
            candleChart.applyOptions({
                width: document.getElementById('candle-chart').clientWidth,
                height: document.getElementById('candle-chart').clientHeight,
            });
        }
        if (tab === 'pnl') pnlChart.resize();
    });
});

// ═══════════════════════════════════════════════════════
// ─── SYMBOL SELECTOR LOGIC ───
let selectedSymbols = [];
let suggestedSymbols = [];
let focusSymbol = null; // Símbolo que el usuario está viendo actualmente
let manualFocus = false; // Si el usuario cambió de pestaña manualmente

function toggleSymbolModal() {
    const modal = document.getElementById('symbol-selection-modal');
    const overlay = document.getElementById('symbol-modal-overlay');
    modal.classList.toggle('visible');
    overlay.classList.toggle('visible');
    if (modal.classList.contains('visible')) {
        renderSuggested();
        renderTags();
    }
}

async function loadSuggestions() {
    try {
        const res = await fetch('/api/active_symbols?mode=live');
        const data = await res.json();
        if (data.status === 'success') {
            suggestedSymbols = data.symbols;
            renderSuggested();
        }
    } catch (e) { console.error("Error loading suggestions", e); }
}

function renderSuggested() {
    const container = document.getElementById('suggested-symbols');
    if (!container) return;
    container.innerHTML = suggestedSymbols.map(sym => {
        const isSelected = selectedSymbols.includes(sym);
        return `<div class="suggestion-pill ${isSelected ? 'selected' : ''}" onclick="toggleSymbol('${sym}')">${sym}</div>`;
    }).join('');
}

function renderTags() {
    const container = document.getElementById('selected-tags');
    if (!container) return;
    if (selectedSymbols.length === 0) {
        container.innerHTML = '<span style="color:#444; font-size:11px; padding:5px;">Ninguno seleccionado</span>';
    } else {
        container.innerHTML = selectedSymbols.map(sym => `
    <div class="symbol-tag">
        ${sym}
        <span class="remove" onclick="toggleSymbol('${sym}')">×</span>
    </div>
    `).join('');
    }
    document.getElementById('selected-count-badge').textContent = selectedSymbols.length;
}

function toggleSymbol(sym) {
    sym = sym.toUpperCase().trim();
    if (selectedSymbols.includes(sym)) {
        selectedSymbols = selectedSymbols.filter(s => s !== sym);
    } else {
        selectedSymbols.push(sym);
    }
    renderTags();
    renderSuggested();
}

function addCustomSymbol() {
    const input = document.getElementById('custom-symbol-input');
    const sym = input.value.toUpperCase().trim();
    if (sym && !selectedSymbols.includes(sym)) {
        selectedSymbols.push(sym);
        input.value = '';
        renderTags();
        renderSuggested();
    }
}

function selectAllSymbols() {
    selectedSymbols = [...suggestedSymbols];
    renderTags();
    renderSuggested();
}

loadSuggestions();

// GAUGE WIN RATE
// ═══════════════════════════════════════════════════════
function updateGauge(pct) {
    const offset = 116 - (116 * Math.min(100, Math.max(0, pct)) / 100);
    document.getElementById('gauge-arc').style.strokeDashoffset = offset;
}

// ═══════════════════════════════════════════════════════
// MINI PnL CHART (Chart.js)
// ═══════════════════════════════════════════════════════
const pnlCtx = document.getElementById('pnl-chart').getContext('2d');
const pnlDataset = {
    labels: [], datasets: [{
        data: [], borderColor: '#00d4aa',
        backgroundColor: (c) => {
            try {
                const g = c.chart.ctx.createLinearGradient(0, 0, 0, 300);
                g.addColorStop(0, 'rgba(0,212,170,0.25)');
                g.addColorStop(1, 'rgba(0,212,170,0)');
                return g;
            } catch (e) { return 'rgba(0,212,170,0.1)'; }
        },
        fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2
    }]
};
const pnlChart = new Chart(pnlCtx, {
    type: 'line', data: pnlDataset,
    options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#0d1626', borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1, titleColor: '#546180', bodyColor: '#e2e8f0',
                callbacks: { label: i => ` PnL: $${i.raw.toFixed(2)}` }
            }
        },
        scales: {
            x: { display: false },
            y: {
                position: 'right',
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: '#546180', font: { family: 'JetBrains Mono', size: 11 }, callback: v => `$${v.toFixed(0)}` }
            }
        }
    }
});
let pnlHistory = [];

// ═══════════════════════════════════════════════════════
// CANDLESTICK CHART (TradingView Lightweight Charts)
// ═══════════════════════════════════════════════════════
const candleEl = document.getElementById('candle-chart');
const candleChart = LightweightCharts.createChart(candleEl, {
    width: candleEl.clientWidth || 800,
    height: candleEl.clientHeight || 400,
    layout: { background: { color: '#07101f' }, textColor: '#546180' },
    grid: { vertLines: { color: 'rgba(255,255,255,0.03)' }, horzLines: { color: 'rgba(255,255,255,0.03)' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: 'rgba(255,255,255,0.07)' },
    timeScale: { borderColor: 'rgba(255,255,255,0.07)', timeVisible: true, secondsVisible: false },
});

const candleSeries = candleChart.addCandlestickSeries({
    upColor: '#00e676', downColor: '#ff3d5a',
    borderUpColor: '#00e676', borderDownColor: '#ff3d5a',
    wickUpColor: 'rgba(0,230,118,0.6)', wickDownColor: 'rgba(255,61,90,0.6)',
});
const ema200Series = candleChart.addLineSeries({
    color: '#ffd740', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, title: 'EMA200',
});
const vwapSeries = candleChart.addLineSeries({
    color: '#a78bfa', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted, title: 'VWAP',
});

// Redimensionar al cambiar tamaño de ventana
new ResizeObserver(() => {
    if (document.getElementById('tab-candles').classList.contains('visible')) {
        candleChart.applyOptions({ width: candleEl.clientWidth, height: candleEl.clientHeight });
    }
}).observe(candleEl);

// Estado del gráfico
let prevSymbol = '';
let prevCandleTs = 0;
let tradeMarkersList = [];

function updateCandleChart(state) {
    const candles = state.candles;
    if (!candles || candles.length < 2) return;
    const sym = state.symbol || '';
    const lastTs = candles[candles.length - 1]?.time || 0;
    const symbolChanged = sym !== prevSymbol;
    if (symbolChanged) {
        // Símbolo nuevo: cargar todo desde cero 
        prevSymbol = sym; prevCandleTs = 0; tradeMarkersList = [];
        try {
            candleSeries.setData(candles);
            ema200Series.setData(candles.map(c => ({ time: c.time, value: state.ema_200 || 0 })));
            vwapSeries.setData(candles.map(c => ({ time: c.time, value: state.vwap || 0 })));
            candleChart.timeScale().scrollToRealTime();
        } catch (e) { }
    } else if (lastTs > prevCandleTs) {
        // Solo actualizar la última vela
        try {
            const last = candles[candles.length - 1];
            candleSeries.update(last);
            if (state.ema_200) ema200Series.update({ time: last.time, value: state.ema_200 });
            if (state.vwap) vwapSeries.update({ time: last.time, value: state.vwap });
        } catch (e) { }
    }
    prevCandleTs = lastTs;
}

// ═══════════════════════════════════════════════════════
// LOG
// ═══════════════════════════════════════════════════════
const MAX_LOG = 200;
let logLines = [];
function classifyLog(l) {
    if (l.includes('🟢 BUY') || l.includes('ORDER_FILLED | BUY')) return 'order-buy';
    if (l.includes('🔴 SELL') || l.includes('ORDER_FILLED | SELL')) return 'order-sell';
    if (l.includes('| ERROR')) return 'error';
    if (l.includes('| WARNING')) return 'warning';
    if (l.includes('| INFO')) return 'info';
    return 'default';
}
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function appendLog(lines) {
    if (!lines || !lines.length) return;
    // Deduplicar: Solo agregar líneas que no estén ya en el final del buffer
    const lastLines = logLines.slice(-30);
    const newFiltered = lines.filter(l => !lastLines.includes(l) && l.trim());

    if (newFiltered.length === 0) return;

    logLines.push(...newFiltered);
    if (logLines.length > MAX_LOG) logLines = logLines.slice(-MAX_LOG);

    const term = document.getElementById('log-terminal');
    if (!term) return;

    const wasBottom = term.scrollTop + term.clientHeight >= term.scrollHeight - 20;
    term.innerHTML = logLines.map(l =>
        `<span class="log-line ${classifyLog(l)}">${esc(l)}</span>`
    ).join('\n');
    if (wasBottom) term.scrollTop = term.scrollHeight;
}

// ═══════════════════════════════════════════════════════
// OPERACIONES Y NOTIFICACIONES
// ═══════════════════════════════════════════════════════
let lastTradeCount = 0;

function showToast(trade) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    const isBuy = trade.side === 'BUY';
    toast.className = `toast ${trade.side}`;

    toast.innerHTML = `
        <div class="toast-side">${isBuy ? '🟢 ORDEN EJECUTADA: COMPRA' : '🔴 ORDEN EJECUTADA: VENTA'}</div>
        <div class="toast-details">
            Activo: <b>${trade.symbol || ''}</b><br>
            Precio: $${(trade.price || 0).toFixed(2)}<br>
            ${!isBuy ? `PnL: <span style="color:${trade.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${trade.pnl >= 0 ?
            '+' : ''}${(trade.pnl || 0).toFixed(2)}</span>` : ''}
        </div>
        `;
    container.appendChild(toast);

    // Redirigir automáticamente a la pestaña de Operaciones (o Velas si se prefiere)
    // if (document.querySelector('.tab-btn[data-tab="trades"]')) {
    // document.querySelector('.tab-btn[data-tab="trades"]').click();
    // }

    // Sonido opcional si hubiera (bip)
    // const audio = new Audio(isBuy ? 'buy.mp3' : 'sell.mp3'); audio.play();

    // Eliminar a los 5 segundos
    setTimeout(() => {
        toast.classList.add('toast-fadeout');
        toast.addEventListener('animationend', () => toast.remove());
    }, 5000);
}

// ═══════════════════════════════════════════════════════
// RESULTADOS Y TABLA DE BITÁCORA ML
// ═══════════════════════════════════════════════════════
let lastResultCount = 0;
let totalNetValue = 0;
let TOTAL_SYMBOLS = 46; // Fallback default

// Mapa vivo de régimen por símbolo (se actualiza con cada tick del WebSocket)
let liveRegimes = {};

// Mapeo de códigos de régimen a etiqueta y color
const REGIME_META = {
    TREND_UP: { label: '🚀 Tendencia Alcista', color: '#00c781', bg: 'rgba(0,199,129,0.12)' },
    TREND_DOWN: { label: '📉 Tendencia Bajista', color: '#ff4f4f', bg: 'rgba(255,79,79,0.12)' },
    RANGE: { label: '🎯 Mercado Lateral', color: '#40c4ff', bg: 'rgba(64,196,255,0.12)' },
    MOMENTUM: { label: '⚡ Momentum Explosivo', color: '#ffab40', bg: 'rgba(255,171,64,0.12)' },
    CHAOS: { label: '🌩️ Alta Volatilidad', color: '#ff9800', bg: 'rgba(255,152,0,0.12)' },
    NEUTRAL: { label: '☁️ Neutro', color: '#888', bg: 'rgba(128,128,128,0.10)' },
};

function buildRegimeBadge(regime) {
    const m = REGIME_META[regime] || REGIME_META['NEUTRAL'];
    return `<span style="
                display:inline-block; padding:4px 10px; border-radius:20px;
                background:${m.bg}; color:${m.color};
                border:1px solid ${m.color}55; font-size:11px; font-weight:600;
                white-space:nowrap; letter-spacing:0.3px;
            ">${m.label}</span>`;
}

function renderBlockingSummary(summary, mlCount) {
    if ((!summary || Object.keys(summary).length === 0) && !mlCount) return '<span style="color:var(--muted)">Sin bloqueos</span>';
    let html = '';
    if (summary) {
        for (const [key, val] of Object.entries(summary)) {
            html += `<span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:1px 4px; border-radius:3px; color:#aaa; margin-right:4px; margin-bottom:4px; display:inline-block;">${key}:${val}</span>`;
        }
    }
    if (mlCount) {
        html += `<span style="background:rgba(191,149,255,0.1); border:1px solid rgba(191,149,255,0.3); padding:1px 4px; border-radius:3px; color:var(--purple); font-weight:bold; display:inline-block;">🧠 AI:${mlCount}</span>`;
    }
    return html;
}

async function fetchConfig() {
    try {
        const res = await fetch('/api/config?mode=live');
        const data = await res.json();
        if (data && data.total_symbols) {
            TOTAL_SYMBOLS = data.total_symbols;
        }
    } catch (e) { console.error("Error loading config:", e); }
}

// Fetch config once on load
fetchConfig();

async function loadNeuralStats() {
    try {
        const res = await fetch('/api/neural_stats');
        const d = await res.json();
        const modeColors = { 'MLP': '#03a9f4', 'cold-start': '#ff9800', 'unavailable': '#888' };
        const modeIcons = { 'MLP': '🟢 MLP Activo', 'cold-start': '🟡 Cold-Start (heurístico)', 'unavailable': '⚠️ No disponible' };
        const mode = d.mode || 'unavailable';
        const el = (id) => document.getElementById(id);
        if (el('neural-mode-label')) el('neural-mode-label').textContent = modeIcons[mode] || mode;
        if (el('neural-samples')) el('neural-samples').textContent = d.total_samples ?? '─';
        if (el('neural-accuracy')) {
            el('neural-accuracy').textContent = d.model_accuracy > 0 ? `${d.model_accuracy}%` : '─';
        }
        if (el('neural-winrate')) {
            el('neural-winrate').textContent = d.win_rate_hist > 0 ? `${d.win_rate_hist}%` : '─';
            el('neural-winrate').style.color = (d.win_rate_hist >= 50) ? '#4caf50' : '#f44336';
        }
        if (el('neural-stats-card')) {
            el('neural-stats-card').style.borderColor = modeColors[mode] + '55';
        }
    } catch (e) {
        const el = document.getElementById('neural-mode-label');
        if (el) el.textContent = '⚠️ Sin conexión al servidor';
    }
}
// ── Auto-refresh: corre loadResults cada 8s solo si el tab está visible ──
let _resultsRefreshInterval = null;
function _startResultsAutoRefresh() {
    if (_resultsRefreshInterval) return; // ya corriendo
    loadResults();  // carga inmediata
    _resultsRefreshInterval = setInterval(async () => {
        // Solo actualizar si el tab de resultados está visible (no congela el resto)
        const tab = document.getElementById('tab-results');
        if (tab && tab.classList.contains('visible')) {
            await loadResults();
            // Parpadeo sutil del indicador EN VIVO
            const badge = document.getElementById('results-live-badge');
            if (badge) {
                badge.style.opacity = '0.4';
                setTimeout(() => { badge.style.opacity = '1'; }, 300);
            }
        }
    }, 8000);
}
// Iniciar auto-refresh al cargar la página
_startResultsAutoRefresh();

loadNeuralStats();
setInterval(loadNeuralStats, 15000);

async function loadResults() {
    try {
        // EN EL DASHBOARD LIVE YA NO MOSTRAMOS LA BD DEL PASADO (SIMULACIÓN)
        // Aquí solo se mostrarán las operaciones en vivo construidas dinámicamente
        // por updateUI() bajo el prefijo 'current-session-row'.
        const tbody = document.getElementById('results-table-body');
        if (tbody) {
            Array.from(tbody.querySelectorAll('tr')).forEach(tr => {
                if (!tr.id || !tr.id.startsWith('current-session-row')) {
                    tr.remove();
                }
            });
        }
        const summaryDiv = document.getElementById('ml-progress');
        if (summaryDiv) {
            summaryDiv.style.display = 'none';
        }
    } catch (e) {
        console.error("Error en loadResults limpiezas:", e);
    }
}

async function runAITraining() {
    const btn = document.querySelector('button[onclick="runAITraining()"]');
    const originalText = btn ? btn.innerHTML : '🧠 Analizar mis Errores con IA (Live)';
    try {
        if (btn) {
            btn.innerHTML = '⏳ Analizando...';
            btn.disabled = true;
        }
        const res = await fetch('/api/train_ai', { method: 'POST' });
        const data = await res.json();

        if (btn) {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }

        if (data.status === 'success') {
            const modalHtml = `
                        <div id="ai-modal-overlay" style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); z-index:9999; display:flex; align-items:center; justify-content:center; backdrop-filter: blur(4px);">
                            <div style="background:var(--bg-card); padding:30px; border-radius:12px; border:1px solid var(--border); width:80%; max-width:800px; max-height:85vh; overflow-y:auto; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); padding-bottom:15px; margin-bottom:20px;">
                                    <h2 style="margin:0; color:var(--accent); font-size:1.4em;">🧠 Análisis de Patrones mediante IA</h2>
                                    <button onclick="document.getElementById('ai-modal-overlay').remove()" style="background:var(--red); color:white; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">CERRAR</button>
                                </div>
                                <pre style="white-space:pre-wrap; font-family:'JetBrains Mono', monospace; font-size:13px; color:var(--text); background:rgba(0,0,0,0.3); padding:20px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);">${esc(data.output)}</pre>
                            </div>
                        </div>`;
            document.body.insertAdjacentHTML('beforeend', modalHtml);
        } else {
            alert("Error ejecutando IA: " + data.message);
        }
    } catch (e) {
        console.error("Error AI endpoint:", e);
        alert("Error de conexión al ejecutar IA.");
        if (btn) {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    }
}

// ═══════════════════════════════════════════════════════
// PAPER TRADING & BANK MODAL
// ═══════════════════════════════════════════════════════
// loadActiveSymbols has been removed - using loadSuggestions now.

function _showLoadingOverlay(msg) {
    let overlay = document.getElementById('global-loader');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'global-loader';
        overlay.style.cssText = 'position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.85); z-index:9999; display:flex; flex-direction:column; justify-content:center; align-items:center; color:white; font-family:var(--font); backdrop-filter:blur(5px);';
        // Add CSS spinner
        overlay.innerHTML = `
                    <style>@keyframes spin { 100% { transform:rotate(360deg); } }</style>
                    <div style="width:40px; height:40px; border:4px solid rgba(255,255,255,0.2); border-top-color:#ffab40; border-radius:50%; animation:spin 1s linear infinite; margin-bottom:20px;"></div>
                    <div id="loader-msg" style="font-size:18px; font-weight:bold; letter-spacing:1px;"></div>
                `;
        document.body.appendChild(overlay);
    }
    document.getElementById('loader-msg').textContent = msg;
    overlay.style.display = 'flex';
}

function _hideLoadingOverlay() {
    const overlay = document.getElementById('global-loader');
    if (overlay) overlay.style.display = 'none';
}

// Modal bonito tipo Wipe Memory
function _showConfirmModal(title, text, confirmBtnText, onConfirm, extraHtml = '') {
    const modalHtml = `
            <div id="custom-confirm-modal" style="position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:9999; display:flex; justify-content:center; align-items:center;">
                <div style="background:#161b22; border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding:30px; width:400px; text-align:center; box-shadow:0 10px 40px rgba(0,0,0,0.5);">
                    <div style="font-size:40px; margin-bottom:15px;">🚀</div>
                    <h2 style="margin:0 0 10px 0; font-size:20px;">${title}</h2>
                    <p style="color:var(--muted); font-size:14px; margin-bottom:25px; line-height:1.5;">${text}</p>
                    <div style="margin-bottom:25px;">${extraHtml}</div>
                    <div style="display:flex; justify-content:center; gap:10px;">
                        <button onclick="document.getElementById('custom-confirm-modal').remove()" style="background:transparent; border:1px solid rgba(255,255,255,0.2); color:white; padding:10px 20px; border-radius:6px; cursor:pointer; font-weight:bold;">Cancelar</button>
                        <button id="btn-confirm-action" style="background:var(--accent); border:none; color:white; padding:10px 20px; border-radius:6px; cursor:pointer; font-weight:bold;">${confirmBtnText}</button>
                    </div>
                </div>
            </div>`;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    document.getElementById('btn-confirm-action').onclick = () => {
        onConfirm();
        document.getElementById('custom-confirm-modal').remove();
    };
}

async function startPaperTrade() {
    _showLoadingOverlay('Cargando Símbolos Activos...');
    try {
        const resAssets = await fetch('/api/all_symbols?mode=live');
        const dataAssets = await resAssets.json();
        if (dataAssets && dataAssets.status === 'success') {
            window.configAssets = dataAssets;
        }
    } catch (e) {
        console.error("Error validando assets antes de arrancar", e);
    }
    _hideLoadingOverlay();

    // Obtener los símbolos activos del Gestor de Símbolos (assets_live.json)
    if (!window.configAssets || !window.configAssets.assets) return alert("Cargando activos... intenta de nuevo.");

    const activeAssets = window.configAssets.assets.filter(a => a.enabled).map(a => a.symbol);

    if (activeAssets.length === 0) {
        return _showConfirmModal(
            "Agrega Símbolos",
            "Debes habilitar al menos un símbolo en el GESTOR DE SÍMBOLOS antes de iniciar Live con Alpaca Paper.",
            "Entendido",
            () => { }
        );
    }

    const symbolsList = activeAssets.join(',');

    const formattedBadges = activeAssets.map(s =>
        `<span style="display:inline-block; background:rgba(255,171,64,0.15); border:1px solid rgba(255,171,64,0.4); color:var(--accent); font-family:'JetBrains Mono', monospace; font-size:12px; padding:4px 8px; border-radius:4px; margin:3px;">${s}</span>`
    ).join('');

    _showConfirmModal(
        "Iniciar Live con Alpaca Paper",
        `Se lanzará un hilo independiente analizando datos de mercado en tiempo real para <strong>${activeAssets.length} empresas</strong>.<br><br><div style="display:flex; flex-wrap:wrap; justify-content:center; gap:5px; margin: 15px 0;">${formattedBadges}</div><br><span style="font-size:12px; opacity:0.7;">(La simulación general NO será afectada).</span>`,
        "Iniciar Alpaca Paper",
        async () => {
            _showLoadingOverlay('INICIANDO ALPACA PAPER...');
            try {
                const res = await fetch('/api/live_alpaca_start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symbols: symbolsList })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    setTimeout(() => location.reload(), 800);
                } else {
                    document.getElementById('global-loader').style.display = 'none';
                    alert("Error: " + data.message);
                }
            } catch (e) {
                document.getElementById('global-loader').style.display = 'none';
                alert("Error conectando. ¿Está corriendo el contenedor trading-bot-live-alpaca?");
            }
        }
    );
}

async function stopPaperTrade() {
    _showConfirmModal(
        "Detener Alpaca Paper",
        "Se detendrán los hilos de Live. La simulación sigue corriendo independientemente.",
        "Detener",
        async () => {
            _showLoadingOverlay('DETENIENDO ALPACA PAPER...');
            try {
                const res = await fetch('/api/live_alpaca_stop', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    setTimeout(() => location.reload(), 800);
                } else {
                    document.getElementById('global-loader').style.display = 'none';
                    alert("Error: " + data.message);
                }
            } catch (e) {
                document.getElementById('global-loader').style.display = 'none';
                alert("Error deteniendo Alpaca Paper");
            }
        }
    );
}


function openBankModal() {
    const modalHtml = `
        <div id="bank-modal-overlay"
            style="position:fixed; top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.8); z-index:9999; display:flex; align-items:center; justify-content:center;">
            <div style="background:#1e212b; padding:25px; border-radius:10px; border:1px solid #3c404b; width:350px;">
                <h3 style="margin-top:0; color:#0057ff; display:flex; align-items:center; gap:8px;">🏦 Hapi Bank
                    Simulator</h3>
                <p style="font-size:12px; color:#aaa; margin-bottom:20px;">Fondea o retira ganancias. (Retiros tienen
                    comisión Hapi de $4.99)</p>

                <label style="font-size:11px; font-weight:bold;">Monto a mover ($USD)</label>
                <input type="number" id="bank-amount" value="100.00" step="10.0"
                    style="width:100%; padding:10px; margin-top:5px; margin-bottom:15px; background:#111; color:white; border:1px solid #333; border-radius:5px; box-sizing:border-box;">

                <div style="display:flex; gap:10px; margin-bottom: 20px;">
                    <button onclick="handleBank('deposit')"
                        style="flex:1; padding:10px; background:var(--green); color:#111; font-weight:bold; border:none; border-radius:5px; cursor:pointer;">↓
                        DEPOSITAR</button>
                    <button onclick="handleBank('withdraw')"
                        style="flex:1; padding:10px; background:var(--accent); color:white; font-weight:bold; border:none; border-radius:5px; cursor:pointer;">↑
                        RETIRAR</button>
                </div>
                <button onclick="document.getElementById('bank-modal-overlay').remove()"
                    style="width:100%; padding:8px; background:transparent; color:#888; border:1px solid #444; border-radius:5px; cursor:pointer;">CERRAR</button>
            </div>
        </div>`;
    document.body.insertAdjacentHTML('beforeend', modalHtml);
}

async function handleBank(action) {
    const amt = parseFloat(document.getElementById('bank-amount').value);
    if (!amt || amt <= 0) return alert("Monto inválido"); try {
        const res = await fetch(`/api/bank/${action}`, {
            method: 'POST', body: JSON.stringify({ amount: amt }), headers: { 'Content-Type': 'application/json' }
        });
        const data = await res.json(); if (data.status === 'success') {
            alert(data.message);
            document.getElementById('bank-modal-overlay').remove();
        } else {
            alert("Error: " + data.message);
        }
    } catch (e) { alert(" Error connecting to Bank API"); }
}

// Llamada inicial 
loadResults();

// Cargar historial de trades inicial
fetch('/api/trades_history')
    .then(res => res.json())
    .then(data => {
        if (data && data.length > 0) {
            renderTrades(data);
        }
    })
    .catch(err => console.error("Error loading trades history:", err));

let sessionTrades = []; // Cache para persistencia en vivo

async function renderTrades(trades) {
    if (!trades) return;

    const now = new Date();
    const today = now.getFullYear() + '-' +
        String(now.getMonth() + 1).padStart(2, '0') + '-' +
        String(now.getDate()).padStart(2, '0');

    // 1. Integrar nuevos trades al cache global de la sesión
    trades.forEach(t => {
        if (!t.date || (!t.date.startsWith(today) && t.date !== today)) return;

        // Evitar duplicados (mismo símbolo, hora y precio)
        const exists = sessionTrades.some(st =>
            st.symbol === t.symbol && st.time === t.time && st.price === t.price
        );
        if (!exists) {
            sessionTrades.push(t);
        }
    });

    // 2. Gestionar notificaciones (toasts) basado en el total acumulado
    if (lastTradeCount === 0 && sessionTrades.length > 0) {
        lastTradeCount = sessionTrades.length;
    } else if (sessionTrades.length > lastTradeCount && lastTradeCount !== 0) {
        const newOnes = sessionTrades.slice(lastTradeCount);
        newOnes.forEach(nt => showToast(nt));
        lastTradeCount = sessionTrades.length;
    }

    const list = document.getElementById('trades-list');

    // Si no hay nada hoy, mostrar mensaje de espera
    if (sessionTrades.length === 0) {
        list.innerHTML = `
            <div class="no-trades" style="text-align:center; padding:30px; opacity:0.6;">
                <div style="font-size:24px; margin-bottom:10px;">📅</div>
                Sin operaciones hoy (${today})<br>
                <span style="font-size:10px;">Las operaciones de días anteriores están ocultas.</span>
            </div>`;
        return;
    }

    // 3. Renderizar siempre el cache completo (ordenado del más nuevo al más viejo)
    list.innerHTML = [...sessionTrades].sort((a, b) => (b.time || '').localeCompare(a.time || '')).reverse().map(t => {
        const confs = t.confirmations || [];
        const hasML = t.ml_prob > 0;
        const hasMult = t.conf_mult > 1.0;
        const isSell = t.side === 'SELL';

        // Formatear precios para mostrar Compra -> Venta si es SELL
        const priceDisplay = isSell
            ? `<div style="display:flex; flex-direction:column;">
                         <span style="font-size:10px; color:var(--muted); text-decoration:line-through;">Entrada: $${(t.entry_price || 0).toFixed(2)}</span>
                         <span>Salida: $${(t.price || 0).toFixed(2)}</span>
                       </div>`
            : `$${(t.price || 0).toFixed(2)}`;

        let tFees = t.fees || 0;
        let tPnl = t.pnl || 0;

        // Simulación retroactiva para órdenes en RAM (previas a actualización)
        if (isSell && tFees === 0 && Math.abs(tPnl) > 0 && t.qty > 0) {
            const estGross = t.price * t.qty;
            const comm = Math.max(0.35, Math.min(0.01 * estGross, 0.0035 * t.qty));
            const sec = Math.max(0.01, estGross * 0.0000278);
            const taf = Math.max(0.01, Math.min(5.95, t.qty * 0.000166));
            tFees = comm + sec + taf;
            tPnl = tPnl - tFees; // Adjust historical Pnl
        }

        return `
                <div class="trade-card">
                    <div class="trade-header">
                        <div class="trade-main-info">
                            <span class="trade-symbol">${esc(t.symbol || '---')}</span>
                            <span class="trade-pill ${t.side.toLowerCase()}">${t.side}</span>
                            ${hasML ? `<span class="ml-badge">🧠 IA ${(t.ml_prob * 100).toFixed(0)}%</span>` : ''}
                            ${hasMult ? `<span class="mult-badge">🔥 x${t.conf_mult.toFixed(1)}</span>` : ''}
                        </div>
                        <div style="text-align:right">
                            <div class="clock" style="font-size:10px; color:var(--text);">${t.date || ''}</div>
                            <div class="clock" style="font-size:9px;">${t.time || ''}</div>
                        </div>
                    </div>

                    <div class="trade-stats-grid">
                        <div class="trade-field">
                            <div class="trade-field-label">${isSell ? 'Precios' : 'Precio'}</div>
                            <div class="trade-field-val">${priceDisplay}</div>
                        </div>
                        <div class="trade-field">
                            <div class="trade-field-label">Cantidad</div>
                            <div class="trade-field-val">${(t.qty || 0).toFixed(4)}</div>
                        </div>
                        <div class="trade-field">
                            <div class="trade-field-label">Inversión</div>
                            <div class="trade-field-val">$${(isSell ? (t.entry_price * t.qty) : (t.price * t.qty)).toFixed(2)}</div>
                        </div>
                        <div class="trade-field">
                            <div class="trade-field-label">PnL Neto</div>
                            <div class="trade-field-val" style="color:${isSell ? (tPnl >= 0 ? 'var(--green)' : 'var(--red)') : 'var(--muted)'}; font-size: 15px;">
                                ${isSell ? (tPnl >= 0 ? '+' : '') + (tPnl || 0).toFixed(2) : '─'}
                            </div>
                        </div>
                    </div>

                    <div class="trade-details">
                        <div class="trade-reason" style="margin-bottom: 4px;">
                            <span style="opacity:0.6; font-weight: bold;">📝 Motivo:</span>
                            <span style="color:var(--accent); font-weight: 600;">${esc(t.reason || 'S/E')}</span>
                            ${isSell && t.entry_reason ? `
                                <div style="font-size:11px; margin-top:2px;">
                                    <span style="opacity:0.5;">↳ Origen:</span> ${esc(t.entry_reason)}
                                </div>
                            ` : ''}
                            ${isSell && tFees ? `
                                <div style="font-size:11px; margin-top:4px; color:var(--red);">
                                    <span style="opacity:0.6; font-weight:bold;">🏛 Broker Fees:</span> -$${tFees.toFixed(2)}
                                </div>
                            ` : ''}
                        </div>
                        
                        ${confs.length > 0 ? `
                        <div style="margin-top:5px;">
                            <div style="font-size:10px; color:var(--muted); margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">🛡️ Filtros de Confluencia:</div>
                            <div class="trade-confs">
                                ${confs.map(c => `<span class="conf-tag" style="background:rgba(217,70,239,0.1); border-color:rgba(217,70,239,0.2); color:#f5d0fe;">${esc(c)}</span>`).join('')}
                            </div>
                        </div>
                        ` : ''}
                    </div>
                </div>
                `;
    }).join('');
}

// ═══════════════════════════════════════════════════════
// POSICIÓN ABIERTA
// ═══════════════════════════════════════════════════════
function renderPosition(state) {
    const area = document.getElementById('pos-area');
    const pos = state.position;
    if (!pos) {
        area.innerHTML = '<div style="padding:12px 16px;font-size:11px;color:var(--muted);">🏖 Sin posición abierta</div > ';
        return;
    }
    const entry = pos.entry_price, sl = pos.stop_loss, tp = pos.take_profit;
    const bid = state.bid ?? entry;
    const pct = Math.max(0, Math.min(100, ((bid - sl) / (tp - sl)) * 100));
    const pnl = (bid - entry) * pos.qty;
    area.innerHTML = `
            <div class="pos-area">
                <div class="pos-sym">${pos.symbol}</div>
                <div class="pos-qty">${pos.qty.toFixed(4)} acciones · entrada $${entry.toFixed(2)}</div>
                <div class="pos-pnl-val" style="color:${pnl >= 0 ? 'var(--green)' : 'var(--red)'};">${pnl >= 0 ? '+' :
            ''}$${pnl.toFixed(2)}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:${pct}%;"></div>
                </div>
                <div class="progress-labels"><span>SL $${sl.toFixed(2)}</span><span>TP $${tp.toFixed(2)}</span></div>
                <div class="levels-grid">
                    <div class="level-box sl">
                        <div class="level-label">STOP LOSS</div>
                        <div class="level-val">$${sl.toFixed(2)}</div>
                    </div>
                    <div class="level-box tp">
                        <div class="level-label">TAKE PROFIT</div>
                        <div class="level-val">$${tp.toFixed(2)}</div>
                    </div>
                </div>
            </div>
            `;
}

// ═══════════════════════════════════════════════════════
// CONFLUENCIAS
// ═══════════════════════════════════════════════════════
function renderConfirmations(confs) {
    const list = document.getElementById('confirm-list');
    if (!confs || confs.length === 0) {
        list.innerHTML = '<div class="confirm-item muted">Sin señal activa</div>';
        return;
    }
    list.innerHTML = confs.map(c => {
        const cls = c.startsWith('✅') ? 'ok' : c.startsWith('❌') ? 'block' : 'muted';
        return `<div class="confirm-item ${cls}">${esc(c)}</div>`;
    }).join('');
}

async function setFocusSymbol(sym) {
    if (focusSymbol === sym) return;
    focusSymbol = sym;
    manualFocus = true;
    console.log("Focus changed to:", sym);

    // Si el símbolo no es el que el bot está procesando activamente,
    // traemos su historia vía API para mostrar el gráfico.
    try {
        const res = await fetch(`/api/history?symbol=${sym}`);
        const data = await res.json();
        if (data.status === 'success' && data.candles) {
            const cdata = data.candles;
            candleSeries.setData(cdata);
            // Actualizar indicadores visuales
            if (cdata.length > 0) {
                const last = cdata[cdata.length - 1];
                document.getElementById('ind-price').textContent = `$${last.close.toFixed(2)}`;
            }
        }
    } catch (e) { console.error("Error fetching history for focus", e); }

    updateSymbolTabs();
}

function updateSymbolTabs(forceSymbols = []) {
    const container = document.getElementById('symbol-tabs-container');
    if (forceSymbols.length === 0 && !focusSymbol) {
        container.style.display = 'none';
        return;
    }
    container.style.display = 'flex';

    const symbols = forceSymbols.length > 0 ? forceSymbols : (selectedSymbols.length > 0 ? selectedSymbols :
        [focusSymbol]);

    container.innerHTML = symbols.map(s => {
        const isActive = (s === focusSymbol);
        return `
            <div class="symbol-tab ${isActive ? 'active' : ''}" onclick="setFocusSymbol('${s}')">
                ${s}
            </div>
            `;
    }).join('');
}

// ═══════════════════════════════════════════════════════
// ACTUALIZACIÓN PRINCIPAL
// ═══════════════════════════════════════════════════════
let prevBid = 0;
let currentSymbol = null;

// Seleccionador manual de symbol eliminado: la estrategia corre en estricto modo secuencial.

function updateUI(fullState) {
    if (!fullState) return;

    // Si hay símbolos forzados, actualizar pestañas
    if (fullState.force_symbols) {
        updateSymbolTabs(fullState.force_symbols);
    }

    // Seleccionar el estado del símbolo enfocado
    // focusSymbol puede venir de hacer clic en una pestaña
    let state = fullState[focusSymbol] || fullState._main || fullState;

    // Si el estado seleccionado no es válido (ej. focusSymbol viejo), buscar el primero disponible
    if (!state.symbol || state.symbol === '─') {
        const availableSyms = Object.keys(fullState).filter(k => k !== 'force_symbols' && k !== '_main');
        if (availableSyms.length > 0) {
            state = fullState[availableSyms[0]];
            if (!manualFocus) focusSymbol = state.symbol;
        }
    }

    const sym = state.symbol || '─';
    const bid = state.bid ?? 0;
    const ask = state.ask ?? 0;

    // Si el bot cambió de símbolo y el usuario NO está mirando una pestaña específica,
    // (En monitoreo paralelo real, esto es menos común pero lo dejamos como fallback)
    if (!manualFocus && sym !== '─' && sym !== focusSymbol) {
        focusSymbol = sym;
    }

    // Actualizar la fecha simulada
    const simDateDisplay = document.getElementById('sim-date');

    // Inyectar indicador de MOCK TIME o SIMULACIÓN si está activo
    const liveBadge = document.getElementById('results-live-badge');
    const modeText = document.getElementById('results-mode-text');
    if (liveBadge && modeText) {
        liveBadge.style.color = '#ff9800'; // Naranja
        liveBadge.querySelector('span').style.background = '#ff9800';
        modeText.textContent = 'ALPACA PAPER TRADING';
    }

    if (simDateDisplay && state.timestamp) {
        try {
            const d = new Date(state.timestamp);
            simDateDisplay.textContent = d.toISOString().split('T')[0];
        } catch (e) { }
    }

    // Auto-actualizar ML Dataset si hay cambio de símbolo
    if (currentSymbol !== null && sym !== '─' && sym !== currentSymbol) {
        loadResults();
    }
    if (sym !== '─') {
        currentSymbol = sym;
    }

    // Actualizar las filas "En progreso..." en la tabla de resultados para TODOS los símbolos activos
    // Ignoramos claves genéricas y extraemos solo los estados de símbolos individuales
    const activeStates = Object.entries(fullState)
        .filter(([k, v]) => {
            return k !== 'force_symbols' && k !== '_main' &&
                v && v.symbol && v.symbol !== '─' &&
                (v.mode === 'LIVE_ALPACA' || v.mode === 'LIVE_REAL' || (v.mode && v.mode.startsWith('LIVE')));
        })
        .map(([k, v]) => v);

    if (activeStates.length > 0) {
        activeStates.forEach(sState => {
            const sSym = sState.symbol;
            let currentRow = document.getElementById(`current-session-row-${sSym}`);
            const tbody = document.getElementById('results-table-body');

            if (!currentRow && tbody) {
                currentRow = document.createElement('tr');
                currentRow.id = `current-session-row-${sSym}`;
                tbody.insertBefore(currentRow, tbody.firstChild);
            }

            if (currentRow) {
                // Si previamente estaba en "Esperando resultados..."
                const noData = document.querySelector('.no-trades');
                if (noData && noData.closest('tbody') === tbody) noData.parentElement.remove();

                const grossPnlRaw = (sState.gross_profit || 0) - Math.abs(sState.gross_loss || 0);

                let dynamicFees = sState.total_fees || 0;
                let dynamicSlip = sState.total_slippage || 0;
                if (Math.abs(grossPnlRaw) > 0 && dynamicFees === 0) {
                    dynamicFees = 0.75; // Mock IBKR average para trades antiguos en memoria
                    dynamicSlip = 0.15;
                }

                const pnl = grossPnlRaw - dynamicFees - dynamicSlip;
                const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
                const pnlSign = pnl > 0 ? '+' : '';

                currentRow.style.background = 'rgba(255, 255, 255, 0.05)'; // Resaltado sutil

                // ─── DEDUPLICACIÓN AGRESIVA ───
                // Buscamos si ya existe una fila TR estática para este símbolo
                const staticRows = Array.from(tbody.querySelectorAll('tr')).filter(r =>
                    !r.id.startsWith('current-session-row') &&
                    r.cells[0] &&
                    r.cells[0].textContent.replace('🔄', '').trim() === sSym
                );

                const highestHistoricalSession = (window.completedSessions || {})[sSym] || 0;
                const currentSession = sState.session || 0;

                let shouldShow = true;

                // 1. Si la sesión actual es menor o igual a la más alta grabada, ocultar
                if (currentSession > 0 && currentSession <= highestHistoricalSession) {
                    shouldShow = false;
                }

                // 2. Si hay una fila estática ya renderizada con el mismo número de sesión, ocultar
                if (shouldShow && staticRows.length > 0) {
                    const matchSameSession = staticRows.some(r => {
                        const sText = r.cells[2].textContent.replace('#', '').trim();
                        return parseInt(sText) === currentSession;
                    });
                    if (matchSameSession) shouldShow = false;
                }

                // 3. Si acabamos de resetear (sesión 0 o 1) y ya hay datos históricos, probablemente sea ruido
                if (shouldShow && currentSession <= 1 && highestHistoricalSession >= 1) {
                    shouldShow = false;
                }

                currentRow.style.display = shouldShow ? 'table-row' : 'none';

                const grossPnl = grossPnlRaw;
                const grossPnlColor = grossPnl >= 0 ? 'var(--green)' : 'var(--red)';
                const grossPnlSign = grossPnl > 0 ? '+' : '';

                currentRow.innerHTML = `
                <td style="padding:10px; font-weight:bold; color:var(--accent2);">${sSym} 🔄</td>
                <td style="padding:10px; font-weight:bold; color:var(--green);">$${(sState.bid || 0).toFixed(2)}</td>
                <td style="padding:10px; color:var(--muted);">#${sState.session || 0}</td>
                <td style="padding:10px;">${sState.total_trades || 0}</td>
                <td style="padding:10px;">${(sState.win_rate || 0).toFixed(1)}%</td>
                <td style="padding:10px; color:${grossPnlColor}; font-weight:bold;">${grossPnlSign}$${grossPnl.toFixed(2)}</td>
                <td style="padding:10px; color:var(--red);">-$${dynamicFees.toFixed(2)}</td>
                <td style="padding:10px; color:var(--red);">-$${dynamicSlip.toFixed(2)}</td>
                <td style="padding:10px; color:${pnlColor}; font-weight:bold;">${pnlSign}$${pnl.toFixed(2)}</td>
                <td style="padding:10px; font-size:11px; color:var(--accent); font-style:italic;">
                    ${(() => {
                        const blockMsg = (sState.blocks && sState.blocks.length > 0) ? sState.blocks[0] : null;
                        if (blockMsg) {
                            return `<span style="color:var(--yellow);">${blockMsg}</span>` +
                                (sState.is_waiting ? ` <span style="color:var(--muted); opacity:0.6;">(Escaneo en ${sState.next_scan_in}s)</span>` : '');
                        }
                        if (sState.is_waiting) return `⏳ Esperando próximo escaneo (${sState.next_scan_in}s)...`;

                        const sig = (sState.signal || 'HOLD').toUpperCase();
                        if (sig === 'HOLD') {
                            return `<span style="opacity:0.6;">📉 ESPERANDO SEÑAL</span>`;
                        }
                        return `<span style="color:var(--green); font-weight:bold;">🚀 SEÑAL DE ${sig}</span>`;
                    })()}
                </td>
                        `;
            }
        });
    }

    // Actualizar indicadores solo si es el símbolo enfocado
    const isFocusActive = (sym === focusSymbol);
    if (isFocusActive) {
        // Header
        document.getElementById('hdr-price').textContent = bid ? `$${bid.toFixed(2)} ` : '$─.──';
        const chg = prevBid > 0 ? bid - prevBid : 0;
        const chgEl = document.getElementById('hdr-change');
        if (chg !== 0) {
            chgEl.textContent = `${chg > 0 ? '+' : ''}${chg.toFixed(3)} `;
            chgEl.style.color = chg > 0 ? 'var(--green)' : 'var(--red)';
        }
        prevBid = bid;

        const sig = state.signal || 'HOLD';
        const sigEl = document.getElementById('hdr-sig');
        sigEl.textContent = sig;
        sigEl.className = `sig - badge ${sig} `;

        // Gráfico de velas
        if (state.candles && state.candles.length > 0) {
            candleSeries.setData(state.candles);
        }

        // Símbolo en header (Combobox)
        const symDisplay = document.getElementById('hdr-sym');
        if (symDisplay && symDisplay.tagName === 'SELECT') {
            const activeSyms = Object.keys(fullState).filter(k => k !== 'force_symbols' && k !== '_main' && k !== 'logs');

            const currentOptions = Array.from(symDisplay.options).map(o => o.value);
            if (activeSyms.length > 0 && JSON.stringify(activeSyms.sort()) !== JSON.stringify(currentOptions.sort())) {
                symDisplay.innerHTML = '';
                activeSyms.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s;
                    opt.textContent = `👁 ${s}`;
                    symDisplay.appendChild(opt);
                });
            }
            if (symDisplay.value !== sym) {
                symDisplay.value = sym;
            }
        }

        // Indicadores pestaña
        document.getElementById('ind-price').textContent = bid ? `$${bid.toFixed(2)} ` : '─';
        document.getElementById('ind-ema12').textContent = (state.ema_fast || 0).toFixed(4);
        document.getElementById('ind-ema26').textContent = (state.ema_slow || 0).toFixed(4);
        document.getElementById('ind-ema200').textContent = (state.ema_200 || 0).toFixed(4);
        document.getElementById('ind-rsi').textContent = (state.rsi || 0).toFixed(1);
        document.getElementById('ind-macd').textContent = (state.macd_hist || 0).toFixed(4);
        document.getElementById('ind-vwap').textContent = (state.vwap || 0).toFixed(2);
        document.getElementById('ind-atr').textContent = (state.atr || 0).toFixed(4);
    }

    // Sidebar estadísticas (esto es global de la cuenta)
    let globalGross = 0;
    let globalFees = 0;
    let globalSlippage = 0;

    // Iteramos por TODOS los símbolos cargados para construir el balance real
    for (const [k, v] of Object.entries(fullState)) {
        if (k === 'force_symbols' || k === '_main' || k === 'logs') continue;
        if (v && v.symbol && v.symbol !== '─') {
            const sg = (v.gross_profit || 0) - Math.abs(v.gross_loss || 0);
            let sf = (v.total_fees || 0);
            let sl = (v.total_slippage || 0);
            if (Math.abs(sg) > 0 && sf === 0) {
                sf = 0.75; // retroactive fee placeholder
                sl = 0.15;
            }
            globalGross += sg;
            globalFees += sf;
            globalSlippage += sl;
        }
    }

    const globalNet = globalGross - globalFees - globalSlippage;

    const init = 25000;
    const avail = init + globalNet; // Pnl Total sumado al balance principal
    const pnl = globalNet; // Esto es el PNL GLOBAL

    const pnlEl = document.getElementById('pnl-val');
    pnlEl.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} `;
    pnlEl.className = `pnl-big-val ${pnl > 0 ? 'g' : pnl < 0 ? 'r' : 'n'} `;

    // Widgets Superiores Live / Simulación (Compatibilidad de Diseño UI)
    const summaryGross = document.getElementById('summary-gross-pnl');
    const summaryFees = document.getElementById('summary-total-fees');
    const summaryNet = document.getElementById('summary-net-pnl');
    const summaryBalance = document.getElementById('summary-final-balance');

    if (summaryGross) {
        summaryGross.textContent = `${globalGross >= 0 ? '+' : ''}$${globalGross.toFixed(2)}`;
        summaryGross.style.color = globalGross >= 0 ? 'var(--green)' : 'var(--red)';
    }
    if (summaryFees) summaryFees.textContent = `-$${globalFees.toFixed(2)}`;
    if (summaryNet) {
        summaryNet.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}`;
        summaryNet.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
    }
    if (summaryBalance) summaryBalance.textContent = `$${avail.toFixed(2)}`;

    const isLivePaperModeVal = document.body.classList.contains('mode-live-paper');
    if (isLivePaperModeVal) {
        if (document.getElementById('session-label')) document.getElementById('session-label').style.display = 'none';
        if (document.getElementById('iter-label')) document.getElementById('iter-label').style.display = 'none';
    } else {
        if (document.getElementById('session-label')) document.getElementById('session-label').style.display = 'block';
        if (document.getElementById('iter-label')) document.getElementById('iter-label').style.display = 'block';
    }

    document.getElementById('session-num').textContent = state.session ?? 1;
    document.getElementById('iter-num').textContent = state.iteration ?? 0;
    document.getElementById('cash-avail').textContent = `$${avail.toFixed(2)} `;
    document.getElementById('cash-settle').textContent = `$${(state.settlement ?? 0).toFixed(2)} `;

    const wr = state.win_rate ?? 0;
    const total = state.total_trades ?? 0;
    const wins = state.winning_trades ?? 0;

    document.getElementById('wr-val').textContent = `${wr.toFixed(1)}% `;
    document.getElementById('trades-val').textContent = `${total} trade${total !== 1 ? 's' : ''} `;
    document.getElementById('wins-val').textContent = wins;
    document.getElementById('losses-val').textContent = total - wins;

    // Countdown visibility 
    const cdContainer = document.getElementById('scan-countdown');
    if (cdContainer) {
        if (state.is_waiting && state.next_scan_in > 0) {
            cdContainer.style.display = 'flex';
            document.getElementById('countdown-val').textContent = `${state.next_scan_in} s`;
        } else {
            cdContainer.style.display = 'none';
        }
    }

    updateGauge(wr);

    const gp = state.gross_profit ?? 0;
    const gl = Math.abs(state.gross_loss ?? 0);
    const pf = gl > 0 ? gp / gl : null;
    const pfEl = document.getElementById('pf-val');
    pfEl.textContent = pf != null ? pf.toFixed(2) : '─';
    pfEl.className = `kv - val ${pf == null ? '' : pf >= 1.5 ? 'g' : pf >= 1 ? 'y' : 'r'} `;

    // Header modo y distinción visual
    const modeInd = document.getElementById('mode-indicator');
    let modeTitle = 'LIVE PAPER T+1 🟠';
    let bgConfig = '#1a0000'; // Dark red/live tint

    if (modeInd) modeInd.innerHTML = modeTitle;

    // Cambiar fondo global para distinguir modos
    document.documentElement.style.setProperty('--bg', bgConfig);
    const modeChip = document.getElementById('mode-chip');
    if (modeChip) modeChip.textContent = modeTitle;

    // Botones visibilidad
    const anyRunning = activeStates && activeStates.length > 0;
    const btnStart = document.getElementById('btn-start-paper');
    const btnStop = document.getElementById('btn-stop-paper');
    if (btnStart) btnStart.style.display = anyRunning ? 'none' : 'flex';
    if (btnStop) btnStop.style.display = anyRunning ? 'flex' : 'none';

    // Body class para CSS
    document.body.classList.remove('mode-live-paper');
    document.body.classList.add('mode-live-paper');

    // Mostrar/ocultar botón de actualización en modal
    const updateC = document.getElementById('update-symbols-container');
    if (updateC) updateC.style.display = anyRunning ? 'block' : 'none';

    // PnL chart — solo agregar punto si cambió
    const ts = new Date().toLocaleTimeString('es-CO', { timeZone: 'America/Bogota', hour12: false });
    pnlHistory.push({ x: ts, y: pnl });
    if (pnlHistory.length > 300) pnlHistory.shift();
    pnlDataset.datasets[0].borderColor = pnl >= 0 ? '#00d4aa' : '#ff3d5a';
    pnlDataset.labels = pnlHistory.map(p => p.x);
    pnlDataset.datasets[0].data = pnlHistory.map(p => p.y);
    // Solo actualizar el PnL chart si la pestaña está visible (para no gastar CPU)
    if (document.getElementById('tab-pnl').classList.contains('visible')) {
        pnlChart.update('none');
    }

    // Candlestick
    updateCandleChart(state);

    // Confluencias
    renderConfirmations(state.confirmations);

    // Posición
    renderPosition(state);

    // Trades
    renderTrades(state.trades);

    // Log - Leer de la raíz del mensaje (fullState), no del sub-estado
    if (fullState && fullState.logs) appendLog(fullState.logs);

    // ─── Tab Indicadores ───
    const price = state.bid ?? 0;
    const ema200 = state.ema_200 ?? 0;
    const vwap = state.vwap ?? 0;
    const atr = state.atr ?? 0;
    const rsi = state.rsi ?? 50;
    const macd = state.macd_hist ?? 0;

    document.getElementById('ind-price').textContent = price ? `$${price.toFixed(2)} ` : '─';
    document.getElementById('ind-ema12').textContent = state.ema_fast ? `$${state.ema_fast.toFixed(2)} ` :
        '─';
    document.getElementById('ind-ema26').textContent = state.ema_slow ? `$${state.ema_slow.toFixed(2)} ` :
        '─';
    document.getElementById('ind-ema200').textContent = ema200 ? `$${ema200.toFixed(2)} ` : '─';

    const vsEma = ema200 > 0 ? ((price - ema200) / ema200 * 100) : null;
    const vsEmaEl = document.getElementById('ind-vs-ema200');
    if (vsEma != null) {
        vsEmaEl.textContent = `${vsEma >= 0 ? '▲' : ''} ${Math.abs(vsEma).toFixed(2)}% ${vsEma >= 0 ?
            '(alcista)' : '(bajista)'
            } `;
        vsEmaEl.style.color = vsEma >= 0 ? 'var(--green)' : 'var(--red)';
    }

    document.getElementById('ind-rsi').textContent = rsi.toFixed(1);
    document.getElementById('rsi-needle').style.left = `${rsi}% `;
    const rsiState = rsi > 65 ? '🔴 Sobrecomprado' : rsi < 35 ? '🟢 Sobreventa (oportunidad)'
        : '✅ Zona neutral'; document.getElementById('ind-rsi-state').textContent = rsiState;
    document.getElementById('ind-rsi-state').style.color = rsi > 65 ? 'var(--red)' : rsi < 35
        ? 'var(--green)' : 'var(--accent)';
    document.getElementById('ind-macd').textContent = macd.toFixed(4);
    document.getElementById('ind-macd').style.color = macd > 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('ind-macd-dir').textContent = macd > 0 ? '📈 Positivo (alcista)' : '📉 Negativo (bajista)';

    document.getElementById('ind-vwap').textContent = vwap ? `$${vwap.toFixed(2)} ` : '─';
    const vsVwap = vwap > 0 ? ((price - vwap) / vwap * 100) : null;
    const vsVwapEl = document.getElementById('ind-vs-vwap');
    if (vsVwap != null) {
        vsVwapEl.textContent = vsVwap >= 0 ? `▲ ${vsVwap.toFixed(2)}% (por encima)` : `▼ ${Math.abs(vsVwap).toFixed(2)}% (por debajo)`;
        vsVwapEl.style.color = vsVwap >= 0 ? 'var(--green)' : 'var(--red)';
    }
    document.getElementById('ind-vwap-spread').textContent = vwap > 0 ? `$${Math.abs(price -
        vwap).toFixed(2)
        } de distancia` : '─';

    document.getElementById('ind-atr').textContent = atr ? `$${atr.toFixed(2)} ` : '─';
    const atrPct = price > 0 && atr > 0 ? (atr / price * 100) : 0;
    const atrSlDist = atr * 1.5;
    document.getElementById('ind-atr-pct').textContent = atrPct ? `${atrPct.toFixed(2)}% ` : '─';
    document.getElementById('ind-atr-sl').textContent = atrSlDist ? `$${atrSlDist.toFixed(2)} (precio SL ≈ $${(price - atrSlDist).toFixed(2)})` : '─';
    const atrStateEl = document.getElementById('ind-atr-state');
    if (atrPct > 3) {
        atrStateEl.textContent = '🔴 Alta — Bot bloqueado'; atrStateEl.style.color = 'var(--red)';
    } else if (atrPct > 1.5) {
        atrStateEl.textContent = '🟡 Media — Operable con cautela'; atrStateEl.style.color =
            'var(--yellow)';
    } else {
        atrStateEl.textContent = '🟢 Baja — Condiciones óptimas'; atrStateEl.style.color =
            'var(--green)';
    }

    document.getElementById('ind-cash-init').textContent = `$${init.toFixed(2)} `;
    document.getElementById('ind-cash-avail').textContent = `$${avail.toFixed(2)} `;
    document.getElementById('ind-gp').textContent = `$${gp.toFixed(2)} `;
    document.getElementById('ind-gl').textContent = `$${gl.toFixed(2)} `;
}

// ═══════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`);

    ws.onopen = () => {
        document.getElementById('status-dot').classList.remove('off');
        document.getElementById('status-text').textContent = 'Bot activo';
        document.getElementById('status-text').style.color = 'var(--green)';
        setInterval(() => { if (ws.readyState === 1) ws.send('ping'); }, 20000);
    };

    ws.onmessage = (e) => {
        try {
            const d = JSON.parse(e.data);
            // Guardar régimen en vivo por símbolo (viene en el estado del WS)
            if (d && typeof d === 'object') {
                Object.keys(d).forEach(sym => {
                    if (d[sym] && d[sym].regime) {
                        liveRegimes[d[sym].symbol || sym] = d[sym].regime;
                    }
                });
            }
            updateUI(d);
        }
        catch (err) { /* silencioso */ }
    };

    ws.onclose = () => {
        document.getElementById('status-dot').classList.add('off');
        document.getElementById('status-text').textContent = 'Reconectando…';
        document.getElementById('status-text').style.color = 'var(--muted)';
        setTimeout(connectWS, 3000);
    };

    ws.onerror = () => ws.close();
}

// Funciones de simulación eliminadas en modo Live


// ── Pre-cargar config de activos en background para evitar fallos del botón Live Paper ──
async function _preloadConfigAssets() {
    try {
        const res = await fetch('/api/all_symbols?mode=live');
        const data = await res.json();
        if (data.status === 'success') {
            window.configAssets = data;
        }
    } catch (e) {
        console.error("Error cargando configuración de activos en background:", e);
    }
}

_preloadConfigAssets();
connectWS();
