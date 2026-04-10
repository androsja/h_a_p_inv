'use strict';

// ═══════════════════════════════════════════════════════
// ESTADO DEL BOT — Fuente de verdad única para los botones
// Solo updateBotStatus() puede modificar _botRunning.
// updateUI() NO toca los botones para evitar conflictos con el WebSocket.
// ═══════════════════════════════════════════════════════
let _botRunning = null; // null = desconocido, true = corriendo, false = detenido

// ═══════════════════════════════════════════════════════
// RELOJ
// ═══════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════
// El reloj secundario (local) fue eliminado a petición del usuario.
// ═══════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
// SISTEMA DE PESTAÑAS
// ═══════════════════════════════════════════════════════
const TAB_IDS = ['candles', 'pnl', 'indicators', 'trades', 'results', 'log'];

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        // Ocultar todo
        TAB_IDS.forEach(id => {
            const tabEl = document.getElementById(`tab-${id}`);
            const btnEl = document.getElementById(`tbtn-${id}`);
            if (tabEl) tabEl.classList.remove('visible');
            if (btnEl) btnEl.classList.remove('active');
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
        if (tab === 'log') {
            fetch('/api/logs')
                .then(res => res.json())
                .then(lines => {
                    if (Array.isArray(lines)) appendLog(lines);
                })
                .catch(err => console.error("Error fetching logs:", err));
        }
    });
});

// ═══════════════════════════════════════════════════════
// ─── SYMBOL SELECTOR LOGIC ───
let selectedSymbols = [];
let suggestedSymbols = [];
let focusSymbol = null; // Símbolo que el usuario está viendo actualmente
let currentSymbol = null; // Símbolo para estadísticas MLP
let fullState = null;     // Guardar el último estado completo del bot
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
    // Null-check: el badge no existe en la página live, solo en la sim
    const badge = document.getElementById('selected-count-badge');
    if (badge) badge.textContent = selectedSymbols.length;
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
    } else if (lastTs >= prevCandleTs) {
        // Actualizar la vela actual o agregar una nueva
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
    if (l.includes('[ORDER_SENT]')) return 'order-sent';
    if (l.includes('[TRADE_AVOIDED]')) return 'avoided';
    if (l.includes('[GHOST_OPEN]')) return 'ghost-open';
    if (l.includes('[GHOST_CLOSE]')) return 'ghost-close';
    if (l.includes('| ERROR')) return 'error';
    if (l.includes('| WARNING')) return 'warning';
    if (l.includes('| INFO')) return 'info';
    return 'default';
}
function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
function appendLog(lines) {
    if (!lines || !lines.length) return;
    
    // El backend envía un snapshot integro de los últimos 200 logs
    // Verificamos si hay algún cambio real para evitar re-renderizar todo
    if (logLines.length > 0 && logLines[logLines.length - 1] === lines[lines.length - 1] && logLines.length === lines.length) {
        return; 
    }

    logLines = lines; // Reemplazar directamente en lugar de duplicar/mezclar array slices

    const term = document.getElementById('log-terminal');
    if (!term) return;

    const wasBottom = term.scrollTop + term.clientHeight >= term.scrollHeight - 20;
    term.innerHTML = logLines.map(l =>
        `<span class="log-line ${classifyLog(l)}">${esc(l)}</span>`
    ).join('\n');
    if (wasBottom) term.scrollTop = term.scrollHeight;
}

function clearLogs() {
    logLines = [];
    const term = document.getElementById('log-terminal');
    if (term) {
        term.innerHTML = '<div class="log-line info" style="text-align:center; padding: 20px; opacity: 0.5;">— Bitácora limpiada. Esperando nuevos eventos... —</div>';
    }
    console.log("Log terminal cleared by user.");
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
        <div class="toast-side" style="margin-bottom: 2px;">
            ${isBuy ? '🟢 COMPRA' : '🔴 VENTA'} | <b>${trade.symbol || ''}</b>
        </div>
        <div class="toast-details" style="font-size: 11px; line-height: 1.4;">
            <div style="display: flex; justify-content: space-between;">
               <span>Precio: $${(trade.price || 0).toFixed(2)}</span>
               ${!isBuy ? `<span>PnL: <b style="color:${trade.pnl >= 0 ? 'var(--green)' : 'var(--red)'}">${trade.pnl >= 0 ? '+' : ''}${(trade.pnl || 0).toFixed(2)}</b></span>` : ''}
            </div>
            <div style="margin-top: 4px; color: var(--muted); border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px;">
               ${trade.reason ? `Motivo: ${trade.reason}<br>` : ''}
               ${trade.ml_prob ? `Certeza IA: ${(trade.ml_prob * 100).toFixed(1)}%` : ''}
            </div>
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

// Global AI stats function removed as it is now symbol-specific in the table
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

// Removed redundant global AI metrics refresh calls

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

function _showToastMsg(msg, color = '#00c781') {
    const t = document.createElement('div');
    t.style.cssText = `position:fixed;bottom:28px;left:50%;transform:translateX(-50%);z-index:99999;
        background:#161b22;border:1px solid ${color}55;border-radius:10px;padding:14px 22px;
        color:white;box-shadow:0 8px 32px rgba(0,0,0,0.7);font-size:13px;font-weight:600;
        pointer-events:none;animation:wipeSlideIn 0.2s ease;max-width:460px;text-align:center;`;
    t.innerHTML = `<span style="color:${color};">${msg}</span>`;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

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

    // Obtener los símbolos activos del Gestor de Símbolos
    if (!window.configAssets || !window.configAssets.assets) {
        _showToastMsg('⚠️ Error cargando activos. Recarga la página e intenta de nuevo.', '#ff9800');
        return;
    }

    const activeAssets = window.configAssets.assets.filter(a => a.enabled).map(a => a.symbol);

    if (activeAssets.length === 0) {
        return _showConfirmModal(
            "Agrega Símbolos",
            "Debes habilitar al menos un símbolo en el <strong>GESTOR DE SÍMBOLOS</strong> antes de iniciar Live con Alpaca Paper.",
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
        `Se lanzará un hilo independiente analizando datos de mercado en tiempo real para <strong>${activeAssets.length} empresas</strong>.<br><br>
         <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:5px; margin: 10px 0;">${formattedBadges}</div>
         <div style="margin-top:15px; text-align:left; background:rgba(255,171,64,0.05); border:1px solid rgba(255,171,64,0.2); padding:12px; border-radius:8px;">
            <label style="font-size:11px; color:#ffab40; font-weight:bold; display:block; margin-bottom:5px;">⏪ REPLAY HISTÓRICO (OPCIONAL)</label>
            <div style="display:flex; gap:10px; align-items:center;">
                <input type="text" id="replay-datetime-input" placeholder="Opcional: Seleccionar fecha y hora..." style="background:#111; border:1px solid #333; color:white; padding:8px; border-radius:4px; font-size:13px; width:100%;">
            </div>
            <div style="font-size:10px; color:var(--muted); margin-top:5px; line-height:1.3;">
                Elige fecha y hora para correr en <b>tiempo simulado (acelerado)</b>. Si dejas la fecha vacía, operará en <b>tiempo real (Live)</b>.
            </div>
         </div>
         <br><span style="font-size:11px; opacity:0.6;">(La simulación general NO será afectada).</span>`,
        "Iniciar Alpaca Paper",
        async () => {
            const dtVal = document.getElementById('replay-datetime-input').value;
            const fullStartStr = dtVal ? dtVal : null;
            
            _showLoadingOverlay('INICIANDO ALPACA PAPER...');
            try {
                const res = await fetch('/api/paper_trade_start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        symbols: symbolsList,
                        mockTime: !!dtVal,
                        sim_start_date: fullStartStr
                    })
                });
                const data = await res.json();
                if (data.status === 'success') {
                    setTimeout(() => location.reload(), 800);
                } else {
                    _hideLoadingOverlay();
                    _showToastMsg('❌ Error: ' + (data.message || 'desconocido'), '#ff3d5a');
                }
            } catch (e) {
                _hideLoadingOverlay();
                _showToastMsg('❌ Error de red.', '#ff3d5a');
            }
        }
    );

    // Initialize Flatpickr for unified datetime selection
    flatpickr("#replay-datetime-input", {
        enableTime: true,
        dateFormat: "Y-m-d H:i",
        time_24hr: true,
        allowInput: true,
        defaultHour: 9,
        defaultMinute: 30
    });
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

// Removed historical /api/trades_history fetch to keep session clean

let sessionTrades = []; // Cache para persistencia en vivo

async function renderTrades(trades) {
    if (!trades) return;

    // 1. Integrar nuevos trades al cache de la sesión
    trades.forEach(t => {
        if (!t.date) return;

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

    // Si no hay nada, mostrar mensaje de espera
    if (sessionTrades.length === 0) {
        list.innerHTML = `
            <div class="no-trades" style="text-align:center; padding:30px; opacity:0.6;">
                <div style="font-size:24px; margin-bottom:10px;">📅</div>
                Sin operaciones en esta sesión aún<br>
            </div>`;
        return;
    }

    // 3. Renderizar siempre el cache completo (ordenado del más nuevo al más viejo)
    list.innerHTML = [...sessionTrades].sort((a, b) => (b.time || '').localeCompare(a.time || '')).reverse().map(t => {
        const confs = t.confirmations || [];
        const hasML = t.ml_prob > 0;
        const hasMult = t.conf_mult > 1.0;
        const isSell = t.side === 'SELL' || t.side === 'BUY/SELL';

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
// TRADES SUB-ROW (expandible bajo cada símbolo)
// ═══════════════════════════════════════════════════════
function toggleTradesSubRow(sym) {
    let subRow = document.getElementById(`trades-subrow-${sym}`);
    const mainRow = document.getElementById(`current-session-row-${sym}`);
    if (!mainRow) return;

    if (subRow) {
        // Toggle visibilidad
        const isVisible = subRow.style.display !== 'none';
        subRow.style.display = isVisible ? 'none' : 'table-row';
        // Actualizar icono en el símbolo
        const arrow = mainRow.querySelector('td:first-child span');
        if (arrow) arrow.textContent = isVisible ? '▶' : '▼';
        if (!isVisible) _renderTradesSubRow(sym);
        return;
    }

    // Crear sub-fila
    subRow = document.createElement('tr');
    subRow.id = `trades-subrow-${sym}`;
    subRow.style.background = 'rgba(0,0,0,0.35)';
    mainRow.insertAdjacentElement('afterend', subRow);
    _renderTradesSubRow(sym);

    // Actualizar icono
    const arrow = mainRow.querySelector('td:first-child span');
    if (arrow) arrow.textContent = '▼';
}

function _renderTradesSubRow(sym) {
    const subRow = document.getElementById(`trades-subrow-${sym}`);
    if (!subRow) return;

    // FUENTE DE VERDAD: leer directamente del estado WebSocket más reciente
    // (en lugar del cache sessionTrades que puede estar vacío al inicio)
    const symState = fullState && fullState[sym];
    const directTrades = (symState && symState.trades) ? [...symState.trades] : [];
    
    // También integrar cualquier trade en sessionTrades para no perder datos históricos de la sesión
    const cachedTrades = sessionTrades.filter(t => t.symbol === sym);
    
    // Mezclar ambas fuentes evitando duplicados
    const allKeys = new Set(directTrades.map(t => `${t.time}|${t.price}|${t.side}`));
    cachedTrades.forEach(t => {
        const key = `${t.time}|${t.price}|${t.side}`;
        if (!allKeys.has(key)) {
            directTrades.push(t);
            allKeys.add(key);
        }
    });
    
    const trades = directTrades.sort((a, b) => (b.time || '').localeCompare(a.time || ''));

    if (trades.length === 0) {
        subRow.innerHTML = `<td colspan="11" style="padding:10px 20px; font-size:11px; color:var(--muted); border-bottom:1px solid rgba(255,255,255,0.05);">
            📭 Sin operaciones registradas para <strong>${sym}</strong> en esta sesión
        </td>`;
        return;
    }

    const rows = trades.map(t => {
        const isSell = t.side === 'SELL' || t.side === 'BUY/SELL';
        const pnl = t.pnl || 0;
        const pnlColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const sideColor = t.side === 'BUY' ? '#00e676' : '#ff3d5a';
        const price = isSell
            ? `<span style="font-size:10px; color:var(--muted); text-decoration:line-through;">$${(t.entry_price||0).toFixed(2)}</span> → $${(t.price||0).toFixed(2)}`
            : `$${(t.price||0).toFixed(2)}`;
        return `
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04); font-size:11px;">
            <td style="padding:6px 10px 6px 30px; color:var(--muted);">${t.time || '--:--'}</td>
            <td><span style="background:${sideColor}22; color:${sideColor}; border:1px solid ${sideColor}55; padding:2px 7px; border-radius:4px; font-weight:bold; font-size:10px;">${t.side}</span></td>
            <td style="padding:6px 8px;">${price}</td>
            <td style="padding:6px 8px; color:var(--muted);">${(t.qty||0).toFixed(4)} acc.</td>
            <td style="padding:6px 8px; color:${pnlColor}; font-weight:bold;">${isSell ? (pnl>=0?'+':'')+(pnl).toFixed(2) : '─'}</td>
            <td colspan="6" style="padding:6px 8px; color:var(--muted); font-style:italic;">${esc(t.reason || '')}</td>
        </tr>`;
    }).join('');

    subRow.innerHTML = `<td colspan="11" style="padding:0; border-bottom:2px solid rgba(0,212,170,0.2);">
        <table style="width:100%; border-collapse:collapse; background:rgba(0,10,20,0.4);">
            <thead>
                <tr style="font-size:10px; color:var(--muted); border-bottom:1px solid rgba(255,255,255,0.07);">
                    <th style="padding:5px 10px 5px 30px; text-align:left; font-weight:500;">Hora</th>
                    <th style="padding:5px 8px; text-align:left; font-weight:500;">Lado</th>
                    <th style="padding:5px 8px; text-align:left; font-weight:500;">Precio</th>
                    <th style="padding:5px 8px; text-align:left; font-weight:500;">Cantidad</th>
                    <th style="padding:5px 8px; text-align:left; font-weight:500;">PnL Neto</th>
                    <th style="padding:5px 8px; text-align:left; font-weight:500;">Motivo</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    </td>`;
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

// Seleccionador manual de symbol eliminado: la estrategia corre en estricto modo secuencial.

function updateUI(data) {
    if (!data) return;
    fullState = data; // Guardar globalmente para otros módulos (como Stats)

    // Si hay símbolos forzados, actualizar pestañas
    if (fullState.force_symbols) {
        updateSymbolTabs(fullState.force_symbols);
    }

    // Seleccionar el estado del símbolo enfocado
    // focusSymbol puede venir de hacer clic en una pestaña
    let state = fullState[focusSymbol] || fullState._main || fullState;

    // Si el estado seleccionado no es válido (ej. focusSymbol viejo), buscar el primero disponible
    const RESERVED_KEYS = new Set(['force_symbols', '_main', 'logs']);
    if (!state.symbol || state.symbol === '─') {
        const availableSyms = Object.keys(fullState).filter(k => !RESERVED_KEYS.has(k) && fullState[k] && fullState[k].symbol);
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
        if (state.mock_time) {
            liveBadge.style.color = '#03a9f4'; // Azul (Simulación/Replay)
            liveBadge.querySelector('span').style.background = '#03a9f4';
            modeText.textContent = 'REPLAY HISTÓRICO AI';
        } else {
            liveBadge.style.color = '#ff9800'; // Naranja (Live Paper)
            liveBadge.querySelector('span').style.background = '#ff9800';
            modeText.textContent = 'ALPACA PAPER TRADING';
        }
    }

    if (state.mock_time) {
        const [datePart, timePart] = state.mock_time.split(' ');
        const isDone = (timePart === '16:00:00');

        if (simDateDisplay) {
            simDateDisplay.textContent = state.mock_time + (isDone ? ' (FIN)' : '');
            simDateDisplay.style.color = isDone ? '#ff3d5a' : '#ffab40';
            simDateDisplay.style.fontSize = '24px';
        }
    } else {
        if (simDateDisplay) {
            if (state.timestamp) {
                try {
                    const d = new Date(state.timestamp);
                    simDateDisplay.textContent = d.toISOString().replace('T', ' ').split('.')[0];
                } catch (e) { }
            }
            simDateDisplay.style.fontSize = ''; // Reset size
            simDateDisplay.style.color = '';
        }
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
            return !RESERVED_KEYS.has(k) &&
                v && typeof v === 'object' && v.symbol && v.symbol !== '─' &&
                (v.mode === 'LIVE_ALPACA' || v.mode === 'LIVE_REAL' || v.mode === 'LIVE_PAPER' || (v.mode && v.mode.startsWith('LIVE')));
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
                
                // Determinar el régimen de mercado para este símbolo
                const regimeCode = sState.regime || liveRegimes[sSym] || 'NEUTRAL';
                const regimeBadge = buildRegimeBadge(regimeCode);

                // Número de trades de esta sesión para este símbolo
                const tradesCount = sState.total_trades || 0;
                const tradesBadge = tradesCount > 0
                    ? `<span onclick="event.stopPropagation(); toggleTradesSubRow('${sSym}')" style="cursor:pointer; background:rgba(0,212,170,0.15); border:1px solid rgba(0,212,170,0.35); color:var(--accent); font-size:11px; padding:2px 7px; border-radius:10px; font-weight:bold;" title="Ver trades">${tradesCount} ▼</span>`
                    : `<span style="color:var(--muted);">0</span>`;

                // Capturar sSym en closure para el onclick de la fila
                const _symForClick = sSym;
                currentRow.style.cursor = 'pointer';
                currentRow.onclick = () => toggleTradesSubRow(_symForClick);

                const spread = Math.abs((sState.ask || 0) - (sState.bid || 0)).toFixed(3);
                const iteration = sState.iteration || 0;
                const rsi = (sState.rsi || 0).toFixed(1);

                currentRow.innerHTML = `
                <td style="padding:10px; font-weight:bold; color:var(--accent2); white-space:nowrap;">
                    <span style="font-size:10px; color:var(--muted); margin-right:4px;">▶</span>${sSym} 🔄
                </td>
                <td style="padding:10px; font-weight:bold; color:var(--green);">
                    <div>$${(sState.bid || sState.last_price || 0).toFixed(2)}</div>
                    <div style="font-size:9px; color:var(--muted); font-weight:normal; display:flex; flex-direction:column; gap:1px;">
                        <span>Spread: ${spread}</span>
                        <span>EMA20: <b style="color:var(--accent2);">${(sState.ema_fast || 0).toFixed(2)}</b></span>
                        <span>EMA200: <b style="color:var(--yellow);">${(sState.ema_200 || 0).toFixed(2)}</b></span>
                    </div>
                </td>
                <td style="padding:10px; text-align:center; font-family:'JetBrains Mono', monospace; font-size:11px;">
                    <span style="color:var(--accent); font-weight:bold;">${iteration.toLocaleString()}</span>
                </td>
                <td style="padding:10px; color:var(--muted);">#${sState.session || 0}</td>
                <td style="padding:10px;">${tradesBadge}</td>
                <td style="padding:10px;">${(sState.win_rate || 0).toFixed(1)}%</td>
                <td style="padding:10px; color:${grossPnlColor}; font-weight:bold;">${grossPnlSign}$${grossPnl.toFixed(2)}</td>
                <td style="padding:10px; color:var(--red);">-$${dynamicFees.toFixed(2)}</td>
                <td style="padding:10px; color:var(--red);">-$${dynamicSlip.toFixed(2)}</td>
                <td style="padding:10px; color:${pnlColor}; font-weight:bold;">${pnlSign}$${pnl.toFixed(2)}</td>
                <td style="padding:10px; text-align:center; font-family:'JetBrains Mono', monospace;">
                    <span style="color:#9c27b0; font-weight:bold;">${sState.total_samples || '─'}</span>
                </td>
                <td style="padding:10px; text-align:center; font-family:'JetBrains Mono', monospace;">
                    <span style="color:#03a9f4; font-weight:bold;">${sState.model_accuracy > 0 ? sState.model_accuracy + '%' : '─'}</span>
                </td>
                <td style="padding:10px; font-size:11px; color:var(--accent); font-style:italic;">
                    <div style="display:flex; flex-direction:column; gap:2px;">
                        <span>RSI: <b>${rsi}</b></span>
                        <div style="display:flex; gap:4px; flex-wrap:wrap; margin-top:2px;">
                            ${sState.model_accuracy > 0 ? '<span style="background:rgba(156,39,176,0.15); color:#ba68c8; padding:1px 4px; border-radius:3px; font-size:9px; font-style:normal; font-weight:bold;">🧠 IA ACTIVE</span>' : ''}
                            ${sState.total_sim_trades > 0 ? '<span style="background:rgba(3,169,244,0.15); color:#4fc3f7; padding:1px 4px; border-radius:3px; font-size:9px; font-style:normal; font-weight:bold;">👻 GHOSTS</span>' : ''}
                        </div>
                        ${(() => {
                            if (sState.is_ml_blocked) return `<span style="color:var(--red); font-weight:bold; font-size:10px;">🔴 BLOQUEO IA</span>`;
                            if (sState.is_quality_blocked) return `<span style="color:var(--orange); font-weight:bold; font-size:10px;">⚠️ FILTRO CALIDAD</span>`;
                            const blockMsg = (sState.blocks && sState.blocks.length > 0) ? sState.blocks[0] : null;
                            if (blockMsg) return `<span style="color:var(--yellow);">${blockMsg}</span>`;
                            
                            // Mostrar última acción (log) si existe
                            if (sState.last_action) {
                                return `<span style="color:#81c784; font-size:10px; opacity:0.9;">💬 ${sState.last_action}</span>`;
                            }
                            
                            if (sState.is_waiting) return `<span style="color:var(--muted);">⏳ (${sState.next_scan_in}s)</span>`;
                            return `<span style="opacity:0.6;">ESPERANDO SEÑAL</span>`;
                        })()}
                    </div>
                </td>
                <td style="padding:10px;">${regimeBadge}</td>
                `;

                // Actualizar sub-fila de trades si ya estaba abierta
                const existingSubRow = document.getElementById(`trades-subrow-${sSym}`);
                if (existingSubRow && existingSubRow.style.display !== 'none') {
                    _renderTradesSubRow(sSym);
                }
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
            const activeSyms = Object.keys(fullState).filter(k => !RESERVED_KEYS.has(k) && fullState[k] && typeof fullState[k] === 'object' && fullState[k].symbol);

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
    }

    // Sidebar estadísticas (esto es global de la cuenta)
    let globalGross = 0;
    let globalFees = 0;
    let globalSlippage = 0;

    // Iteramos por TODOS los símbolos cargados para construir el balance real
    for (const [k, v] of Object.entries(fullState)) {
        if (RESERVED_KEYS.has(k) || !v || typeof v !== 'object') continue;
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

    let init = 10000;
    const firstActiveState = Object.values(fullState).find(v => v && typeof v === 'object' && typeof v.initial_cash === 'number');
    if (firstActiveState) {
        init = firstActiveState.initial_cash;
    }
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

    // Botones de inicio/stop: controlados EXCLUSIVAMENTE por updateBotStatus()
    // updateUI() NO modifica estos botones para evitar conflicto con datos stale del WebSocket

    // Body class para CSS
    document.body.classList.remove('mode-live-paper');
    document.body.classList.add('mode-live-paper');

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

    // Sentimiento de Noticias (desde el estado del bot, se actualiza cada segundo)
    renderSentimentFromState(state);

    // Posición
    renderPosition(state);

    // Trades — Recolectar de TODOS los símbolos activos para no perder trades de símbolos no enfocados
    const allSymbolTrades = [];
    for (const [k, v] of Object.entries(fullState)) {
        if (RESERVED_KEYS.has(k) || !v || typeof v !== 'object' || !v.symbol) continue;
        if (v.trades && v.trades.length > 0) {
            allSymbolTrades.push(...v.trades);
        }
    }
    if (allSymbolTrades.length > 0) renderTrades(allSymbolTrades);

    // Log - Leer de la raíz del mensaje (fullState), no del sub-estado
    if (fullState && fullState.logs) appendLog(fullState.logs);

    // ─── Tab Indicadores (Eliminado a petición del usuario) ───
}

// ═══════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════
function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`);

    ws.onopen = () => {
        // El estado real lo manejará updateBotStatus periódicamente
        updateBotStatus(); 
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
        document.getElementById('status-text').textContent = 'Websocket Desconectado';
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

// ── Nueva función para sincronizar el estado real del bot ──
async function updateBotStatus() {
    try {
        const res = await fetch('/api/live_alpaca_status?mode=live');
        const data = await res.json();
        console.log("Health Check API Response:", data);
        
        const dot = document.getElementById('status-dot');
        const txt = document.getElementById('status-text');
        const btnStop = document.getElementById('btn-stop-paper');
        const btnStart = document.getElementById('btn-start-paper');

        if (data && data.running) {
            if (dot) dot.classList.remove('off');
            if (txt) {
                txt.textContent = 'Bot activo';
                txt.style.color = 'var(--green)';
            }
            if (btnStop) btnStop.style.display = 'flex';
            if (btnStart) btnStart.style.display = 'none';
        } else {
            if (dot) dot.classList.add('off');
            if (txt) {
                txt.textContent = 'Bot detenido';
                txt.style.color = 'var(--muted)';
            }
            if (btnStop) btnStop.style.display = 'none';
            if (btnStart) btnStart.style.display = 'flex';
        }
    } catch (e) {
        console.warn("Error consultando estado del bot:", e);
    }
}

// Iniciar chequeo periódico (cada 5 seg)
setInterval(updateBotStatus, 5000);

_preloadConfigAssets();
connectWS();
