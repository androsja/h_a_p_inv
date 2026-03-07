// ─── GESTOR DE SÍMBOLOS MAESTRO (assets.json) ───
async function openSymbolManager() {
    const wrapper = document.getElementById('sym-manager-wrapper');
    wrapper.style.display = 'flex';
    document.getElementById('sm-restart-warning').style.display = 'none';

    const listContainer = document.getElementById('sym-manager-list');
    listContainer.innerHTML = '<div style="color:var(--muted); text-align:center; grid-column:1/-1; padding:20px;">Cargando catálogo...</div>';

    try {
        const res = await fetch('/api/all_symbols?mode=live');
        const data = await res.json();
        if (data.status === 'success') {
            window.configAssets = data; // Guardar estado global
            let html = '';
            const sortedAssets = data.assets.sort((a, b) => a.symbol.localeCompare(b.symbol));
            sortedAssets.forEach(item => {
                const isEn = item.enabled !== false;
                // Damos ID únicoa cada tarjeta para actualizarlas sin redibujar todo
                html += `
                        <div id="smi-${item.symbol}" class="sm-item ${isEn ? 'enabled' : 'disabled'}">
                            <div id="sms-${item.symbol}" class="sm-sym ${isEn ? 'enabled' : 'disabled'}">${item.symbol}</div>
                            <label class="switch" style="transform: scale(0.7); margin-bottom: 0px;">
                                <input type="checkbox" ${isEn ? 'checked' : ''} onchange="toggleMasterSymbol('${item.symbol}', this.checked)">
                                <span class="slider"></span>
                            </label>
                        </div>
                    `;
            });
            listContainer.innerHTML = html;
            updateLiveSymbolCounter();
        }
    } catch (e) {
        listContainer.innerHTML = '<div style="color:var(--red); grid-column:1/-1;">Error de conexión.</div>';
    }
}

function closeSymbolManager() {
    document.getElementById('sym-manager-wrapper').style.display = 'none';
}

// ── Contador de símbolos habilitados ────────────────────────────────
function updateLiveSymbolCounter() {
    const badge = document.getElementById('live-sym-counter-badge');
    if (!badge) return;
    const checkboxes = document.querySelectorAll('#sym-manager-list input[type="checkbox"]');
    const total = checkboxes.length;
    const enabled = [...checkboxes].filter(c => c.checked).length;
    if (total === 0) { badge.textContent = '─ / ─'; return; }
    badge.textContent = `${enabled} / ${total}`;
    if (enabled === 0) {
        badge.style.background = 'rgba(239,68,68,0.15)';
        badge.style.borderColor = 'rgba(239,68,68,0.4)';
        badge.style.color = '#ef4444';
    } else if (enabled === total) {
        badge.style.background = 'rgba(0,199,129,0.15)';
        badge.style.borderColor = 'rgba(0,199,129,0.35)';
        badge.style.color = '#00c781';
    } else {
        badge.style.background = 'rgba(251,191,36,0.12)';
        badge.style.borderColor = 'rgba(251,191,36,0.35)';
        badge.style.color = '#fbbf24';
    }
}

// ── Calcular HAPI Score (mismo algoritmo que el recomendador de /sim) ─
function scoreSimResults(data) {
    return data.map(row => {
        const gross = row.gross_pnl !== undefined ? row.gross_pnl : (row.pnl || 0);
        const net = gross - (row.total_fees || 0) - (row.slippage_est || 0);
        const wr = parseFloat(row.win_rate) || 0;
        const trades = parseInt(row.total_trades) || 0;
        const gp = parseFloat(row.gross_profit) || 0;
        const gl = Math.abs(parseFloat(row.gross_loss) || 0.01);
        const pf = gp / gl;
        const dd = parseFloat(row.drawdown) || 0;
        const regime = (row.regime || 'NEUTRAL').toUpperCase();

        const scoreWR = wr >= 60 ? 30 : wr >= 50 ? 20 : wr >= 40 ? 10 : 0;
        const scorePF = pf > 2 ? 25 : pf > 1.5 ? 20 : pf > 1.0 ? 12 : 0;
        const npt = trades > 0 ? net / trades : 0;
        const scoreNet = npt > 5 ? 20 : npt > 2 ? 15 : npt > 0 ? 10 : npt > -1 ? 3 : 0;
        const scoreSample = trades >= 5 ? 10 : trades >= 3 ? 7 : trades >= 2 ? 4 : trades >= 1 ? 2 : 0;
        const scoreDD = dd < 0.01 ? 10 : dd < 0.05 ? 8 : dd < 0.1 ? 5 : dd < 0.2 ? 2 : 0;
        const scoreRegime = (regime === 'TREND_UP' || regime === 'MOMENTUM_UP') ? 5 :
            regime === 'NEUTRAL' ? 3 : regime === 'TREND_DOWN' ? 1 : 2;

        return {
            symbol: row.symbol,
            score: Math.round(scoreWR + scorePF + scoreNet + scoreSample + scoreDD + scoreRegime),
            net, wr, trades, pf: Math.round(pf * 100) / 100, dd, regime
        };
    }).sort((a, b) => b.score - a.score);
}

// ── Aplicar Recomendadas de Simulación ──────────────────────────────
async function applySimRecommendations() {
    const btn = document.querySelector('#sym-manager-modal button[onclick="applySimRecommendations()"]');
    const origText = btn ? btn.innerHTML : '';
    if (btn) btn.innerHTML = '⏳ Cargando resultados de simulación...';

    try {
        const res = await fetch('/api/results');
        if (!res.ok) throw new Error('Sin datos');
        const data = await res.json();
        if (!data || data.length === 0) {
            alert('⚠️ No hay resultados de simulación todavía.\n\nCorre al menos un ciclo completo de simulación primero.');
            if (btn) btn.innerHTML = origText;
            return;
        }

        const scored = scoreSimResults(data);
        const fullTop = scored.filter(s => s.score >= 60);
        let top = fullTop.slice(0, 50);
        const mid = scored.filter(s => s.score >= 35 && s.score < 60);

        // Si no llegamos a 50 con las mejores (>60), rellenamos con las intermedias (>35)
        if (top.length < 50) {
            const extra = mid.slice(0, 50 - top.length);
            top = [...top, ...extra];
        }

        const topSyms = top.map(s => s.symbol);

        if (topSyms.length === 0) {
            alert(`⚠️ No hay empresas con score ≥ 35 en la última simulación.`);
            if (btn) btn.innerHTML = origText;
            return;
        }

        // ── Preview modal ─────────────────────────────────────────
        const prev = document.createElement('div');
        prev.id = 'ndr-live-preview';
        prev.style.cssText = 'position:fixed;inset:0;z-index:9999999;background:rgba(0,0,0,0.9);backdrop-filter:blur(8px);display:flex;justify-content:center;align-items:center;';

        const scoreRow = s => `
                    <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;border-radius:6px;
                        background:${s.score >= 60 ? 'rgba(0,199,129,0.08)' : 'rgba(251,191,36,0.08)'};
                        border:1px solid ${s.score >= 60 ? 'rgba(0,199,129,0.25)' : 'rgba(251,191,36,0.25)'};margin-bottom:3px;">
                        <span style="font-weight:700;font-size:13px;min-width:54px;color:white;">${s.symbol}</span>
                        <span style="flex:1;font-size:10px;color:#8b949e;">
                            🎯${s.wr}% · PF:${s.pf} · <span style="color:${s.net >= 0 ? '#00c781' : '#ef4444'}">${s.net >= 0 ? '+' : ''}$${s.net.toFixed(2)}</span> · #${s.trades}T
                        </span>
                        <span style="font-weight:800;font-size:13px;color:${s.score >= 60 ? '#00c781' : '#fbbf24'};">${s.score}</span>
                    </div>`;

        prev.innerHTML = `
                <div style="background:#0d1117;border:1px solid #30363d;border-radius:16px;
                    padding:22px;width:92%;max-width:560px;max-height:85vh;overflow-y:auto;
                    box-shadow:0 24px 80px rgba(0,0,0,0.9);position:relative;">
                    <button onclick="document.getElementById('ndr-live-preview').remove()"
                        style="position:absolute;top:12px;right:14px;background:none;border:none;color:#aaa;font-size:20px;cursor:pointer;">&times;</button>
                    <div style="font-size:16px;font-weight:700;color:#c084fc;margin-bottom:4px;">📅 Recomendadas para el Día Siguiente</div>
                    <div style="font-size:11px;color:#6e7681;margin-bottom:14px;">
                        Basado en ${scored.length} empresas del último ciclo de simulación · HAPI Score ≥ 60<br>
                        <span style="color:#fbbf24; font-weight:bold;">🛡️ Límite de seguridad aplicado: Máximo 50 símbolos en Live</span>
                    </div>

                    <div style="font-size:10px;font-weight:700;color:#00c781;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">
                        ✅ Recomendadas — ${top.length} empresa(s)
                    </div>
                    ${top.map(scoreRow).join('')}

                    ${mid.length > 0 ? `
                    <div style="font-size:10px;font-weight:700;color:#fbbf24;text-transform:uppercase;letter-spacing:1px;margin:12px 0 6px;">
                        🟡 También disponibles (Score 35–59) — ${mid.length}
                    </div>
                    <div style="font-size:10px;color:#6e7681;">
                        ${mid.map(s => `<span style="margin-right:6px;background:#1c2128;padding:2px 6px;border-radius:4px;">${s.symbol}(${s.score})</span>`).join('')}
                    </div>` : ''}

                    <div style="display:flex;gap:8px;margin-top:18px;flex-wrap:wrap;">
                        <button onclick='confirmApplyLiveRec(${JSON.stringify(topSyms)})'
                            style="flex:1;background:linear-gradient(135deg,#166534,#15803d);border:none;color:#bbf7d0;
                                   padding:10px;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;">
                            ✅ Aplicar ${top.length} Recomendadas al Live
                        </button>
                        ${mid.length > 0 ? `<button onclick='confirmApplyLiveRec(${JSON.stringify([...topSyms, ...mid.map(s => s.symbol)])})'
                            style="flex:1;background:linear-gradient(135deg,#7c3aed,#6d28d9);border:none;color:#ede9fe;
                                   padding:10px;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;">
                            🟡 Aplicar + Revisar (${top.length + mid.length})
                        </button>` : ''}
                        <button onclick="document.getElementById('ndr-live-preview').remove()"
                            style="background:#1c2128;border:1px solid #30363d;color:#8b949e;padding:10px 14px;border-radius:8px;font-size:12px;cursor:pointer;">
                            Cancelar
                        </button>
                    </div>
                    <div style="font-size:10px;color:#4d5566;margin-top:10px;text-align:center;">
                        ⚠️ Score = Win Rate×30 + Profit Factor×25 + PnL/T×20 + Muestra×10 + Drawdown×10 + Régimen×5
                    </div>
                </div>`;

        document.body.appendChild(prev);
        prev.onclick = e => { if (e.target === prev) prev.remove(); };

    } catch (e) {
        alert('Error al cargar resultados: ' + e.message);
    } finally {
        if (btn) btn.innerHTML = origText;
    }
}

async function confirmApplyLiveRec(symbols) {
    document.getElementById('ndr-live-preview')?.remove();
    if (!window.configAssets || !window.configAssets.assets) {
        const r = await fetch('/api/all_symbols?mode=live');
        const d = await r.json();
        window.configAssets = d;
    }
    const all = (window.configAssets?.assets || []).map(a => a.symbol);
    const enableSet = new Set(symbols);
    const toEnable = all.filter(s => enableSet.has(s));
    const toDisable = all.filter(s => !enableSet.has(s));

    await applyMassSymbolToggle(toDisable, false);
    await applyMassSymbolToggle(toEnable, true);
    updateLiveSymbolCounter();
}

async function toggleMasterSymbol(symbol, isEnabled) {
    try {
        const res = await fetch('/api/toggle_symbol', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ symbol: symbol, enabled: isEnabled, mode: 'live' })
        });
        const data = await res.json();
        if (res.ok && data.status === 'success') {
            // Actualizar memoria local para que Live Paper lo detecte
            if (window.configAssets && window.configAssets.assets) {
                const asset = window.configAssets.assets.find(a => a.symbol === symbol);
                if (asset) asset.enabled = isEnabled;
            }

            const itemDiv = document.getElementById(`smi-${symbol}`);
            const symDiv = document.getElementById(`sms-${symbol}`);
            if (itemDiv && symDiv) {
                if (isEnabled) {
                    itemDiv.classList.remove('disabled');
                    itemDiv.classList.add('enabled');
                    symDiv.classList.remove('disabled');
                    symDiv.classList.add('enabled');
                } else {
                    itemDiv.classList.remove('enabled');
                    itemDiv.classList.add('disabled');
                    symDiv.classList.remove('enabled');
                    symDiv.classList.add('disabled');
                }
            }
            updateLiveSymbolCounter();
        } else {
            alert("Error guardando el símbolo en el servidor: " + (data.message || "Desconocido"));
            // Revert the toggle visually
            const cb = document.querySelector(`#smi-${symbol} input[type="checkbox"]`);
            if (cb) cb.checked = !isEnabled;
        }
    } catch (e) {
        console.error("Error guardando toggle:", e);
        alert("Error de red conectando con el servidor");
    }
}

async function toggleAllSymbols(enable) {
    if (!window.configAssets || !window.configAssets.assets) return;
    const symbolsToChange = window.configAssets.assets.filter(a => !!a.enabled !== enable).map(a => a.symbol);
    if (symbolsToChange.length === 0) return;
    await applyMassSymbolToggle(symbolsToChange, enable);
}

async function toggleTop50Symbols() {
    if (!window.configAssets || !window.configAssets.assets) return;
    const top50 = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'NFLX', 'AMD', 'DIS',
        'PYPL', 'JPM', 'V', 'WMT', 'KO', 'PEP', 'NKE', 'SBUX', 'COST', 'CRM',
        'INTC', 'ASML', 'AVGO', 'CSCO', 'ADBE', 'ORCL', 'QCOM', 'TXN', 'AMAT', 'MU',
        'ABT', 'BAC', 'BRK.B', 'CMCSA', 'CVX', 'HD', 'LLY', 'MA', 'MRK', 'PFE',
        'PG', 'TMO', 'UNH', 'XOM', 'ACN', 'COST', 'DHR', 'LIN', 'MCD', 'NEE'
    ];

    // Primero deseleccionamos todo
    const allSymbols = window.configAssets.assets.map(a => a.symbol);
    await applyMassSymbolToggle(allSymbols, false);

    // Luego seleccionamos el top 50
    await applyMassSymbolToggle(top50, true);
}

async function applyMassSymbolToggle(symbolsToChange, enable) {
    // Update UI instantly
    symbolsToChange.forEach(sym => {
        const asset = window.configAssets.assets.find(a => a.symbol === sym);
        if (asset) asset.enabled = enable;

        const input = document.querySelector(`#smi-${sym} input[type="checkbox"]`);
        if (input) input.checked = enable;

        const itemDiv = document.getElementById(`smi-${sym}`);
        const symDiv = document.getElementById(`sms-${sym}`);
        if (itemDiv && symDiv) {
            if (enable) {
                itemDiv.classList.remove('disabled'); itemDiv.classList.add('enabled');
                symDiv.classList.remove('disabled'); symDiv.classList.add('enabled');
            } else {
                itemDiv.classList.remove('enabled'); itemDiv.classList.add('disabled');
                symDiv.classList.remove('enabled'); symDiv.classList.add('disabled');
            }
        }
    });

    // Process async sequence to backend
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed; top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999999;display:flex;justify-content:center;align-items:center;color:white;font-weight:bold;text-align:center;padding:20px;';
    overlay.innerHTML = `<div>🚀 Configurando símbolos...<br><span style="font-size:12px; color:var(--muted);">${symbolsToChange.length} activos siendo actualizados</span></div>`;
    document.body.appendChild(overlay);

    try {
        for (let sym of symbolsToChange) {
            await fetch('/api/toggle_symbol', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol: sym, enabled: enable, mode: 'live' })
            });
        }
    } finally {
        document.body.removeChild(overlay);
    }
}

async function resetSymbolManager() {
    if (!confirm("⚠️ ¿Reiniciar el bot ahora?\n\nEsto borrará el historial acumulado y comenzará un nuevo ciclo con los símbolos actualizados."))
        return;
    try {
        const res = await fetch('/api/reset_all', { method: 'POST' });
        if (res.ok) {
            lastResultCount = 0;
            totalNetValue = 0;
            document.getElementById('results-table-body').innerHTML = '';
            closeSymbolManager();
            setTimeout(() => location.reload(), 1000);
        } else {
            alert("Error reiniciando el bot.");
        }
    } catch (e) { console.error(e); }
}

window.openWithdrawalBreakdown = function (totalNet, netToBank) {
    const trm = 3766.30; // TRM Real del 28 de Febrero 2026
    const spread = 80;    // Spread estimado de Hapi/Bancos (~2%)
    const finalRate = trm - spread;
    const hapiFee = 1.99;

    const copProfitOnly = (totalNet - hapiFee) * finalRate;
    const totalBalanceCOP = netToBank * finalRate;

    // Intentar calcular días en Live Mode
    let daysText = "esta simulación";
    try {
        const s1 = document.getElementById('sim-start')?.textContent;
        const s2 = document.getElementById('sim-end')?.textContent;
        if (s1 && s1 !== '─') {
            const d1 = new Date(s1);
            const d2 = (s2 && s2 !== '─') ? new Date(s2) : new Date();
            const diffTime = Math.abs(d2 - d1);
            let diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
            if (diffDays <= 0) diffDays = 1;
            daysText = `tan solo ${diffDays} días`;
        } else {
            const datesEl = document.querySelector('#summary-final-balance')?.parentElement?.parentElement?.querySelector('span[style*="font-size: 10px"]');
            if (datesEl) {
                const text = datesEl.innerText;
                if (text.includes("al")) {
                    const d1 = new Date(text.split(" al ")[0].replace(/[^0-9-]/g, ''));
                    const d2 = new Date(text.split(" al ")[1].replace(/[^0-9-]/g, ''));
                    const diff = Math.ceil(Math.abs(d2 - d1) / (1000 * 60 * 60 * 24));
                    if (!isNaN(diff) && diff > 0) daysText = `tan solo ${diff} días`;
                }
            }
        }
    } catch (e) { }

    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed; top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:9999999;display:flex;justify-content:center;align-items:center;backdrop-filter:blur(6px);';

    const modal = document.createElement('div');
    modal.style.cssText = 'background:#1e212b; padding:30px; border-radius:16px; border:1px solid #3c404b; width:90%; max-width:500px; box-shadow:0 20px 60px rgba(0,0,0,0.8); position:relative;';

    modal.innerHTML = `
                <button onclick="this.parentElement.parentElement.remove()" style="position:absolute; top:20px; right:20px; background:none; border:none; color:white; font-size:24px; cursor:pointer;">&times;</button>
                <h2 style="margin-top:0; color:#4caf50; display:flex; align-items:center; gap:10px;">
                    🇨🇴 Retiro Estimado a Pesos
                </h2>

                <div style="text-align:center; margin:15px 0 25px 0; padding:15px; background:rgba(0,188,212,0.1); border-radius:12px; border:1px dashed #00bcd4;">
                    <p style="margin:0; color:#e0e0e0; font-size:14px; line-height:1.4;">
                        Si esta fuera tu cuenta real, habrías ganado <b style="color:var(--green);">$${totalNet.toFixed(2)} USD</b> 
                        en <span style="color:#00bcd4; font-weight:bold;">${daysText}</span>.
                    </p>
                </div>
                
                <div style="margin:20px 0; background:rgba(255,255,255,0.03); border-radius:12px; padding:15px; border:1px solid rgba(255,255,255,0.05);">
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span style="color:#888;">Utilidad Neta (USD):</span>
                        <span style="font-weight:bold; color:var(--green);">$${totalNet.toFixed(2)}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                        <span style="color:#888;">Costo de Retiro (Hapi):</span>
                        <span style="font-weight:bold; color:#ff9800;">-$${hapiFee.toFixed(2)}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px;">
                        <span style="color:#888;">TRM Actual:</span>
                        <span style="font-weight:bold; color:#03a9f4;">$${trm.toLocaleString('es-CO')} COP</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.85em;">
                        <span style="color:#666;">Tasa Hapi (Spread incl.):</span>
                        <span style="color:#666;">$${finalRate.toLocaleString('es-CO')} COP</span>
                    </div>
                </div>

                <div style="text-align:center; padding:20px; background:rgba(76, 175, 80, 0.1); border-radius:12px; border:1px solid rgba(76, 175, 80, 0.2); margin-bottom:20px;">
                    <div style="font-size:12px; color:#81c784; text-transform:uppercase; letter-spacing:1px; margin-bottom:5px;">Ganancia Real en Pesos</div>
                    <div style="font-size:28px; font-weight:bold; color:#4caf50; font-family:'JetBrains Mono', monospace;">
                        $${copProfitOnly.toLocaleString('es-CO', { maximumFractionDigits: 0 })} COP
                    </div>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center; padding:15px; background:rgba(3, 169, 244, 0.1); border-radius:12px; border:1px solid rgba(3, 169, 244, 0.2);">
                    <span style="font-size:12px; color:#4fc3f7;">BALANCE TOTAL SI RETIRAS TODO:</span>
                    <span style="font-size:18px; font-weight:bold; color:#03a9f4; font-family:'JetBrains Mono', monospace;">
                        $${totalBalanceCOP.toLocaleString('es-CO', { maximumFractionDigits: 0 })} COP
                    </span>
                </div>

                <p style="font-size:10px; color:#555; margin-top:20px; text-align:center;">
                    * Valores aproximados basados en la TRM del día y comisiones estándar de Hapi a bancos colombianos.
                </p>
            `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}
