/**
 * Calendar Visualization Logic for Trading Bot Dashboard
 * Handles the generation and display of daily profit/loss calendars.
 */

window.openCalendarModal = async function (symbol) {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0'; overlay.style.left = '0';
    overlay.style.width = '100%'; overlay.style.height = '100%';
    overlay.style.background = 'rgba(0,0,0,0.85)';
    overlay.style.zIndex = '999999';
    overlay.style.display = 'flex';
    overlay.style.alignItems = 'center';
    overlay.style.justifyContent = 'center';
    overlay.style.backdropFilter = 'blur(6px)';

    const modal = document.createElement('div');
    modal.style.background = '#1e212b';
    modal.style.padding = '30px';
    modal.style.borderRadius = '16px';
    modal.style.border = '1px solid #3c404b';
    modal.style.width = '90%';
    modal.style.maxWidth = '800px';
    modal.style.maxHeight = '90vh';
    modal.style.overflowY = 'auto';
    modal.style.boxShadow = '0 20px 60px rgba(0,0,0,0.8)';
    modal.style.position = 'relative';

    modal.innerHTML = `
        <button id="close-cal" style="position:absolute; top:20px; right:20px; background:none; border:none; color:white; font-size:24px; cursor:pointer;">&times;</button>
        <h2 style="margin-top:0; color:var(--accent); display:flex; align-items:center; gap:10px;">
            📅 Calendario de Trades: <span style="color:white;">${symbol}</span>
        </h2>
        <p style="color:var(--muted); font-size:13px; margin-bottom:25px;">Visualización de ganancias y pérdidas diarias para este símbolo.</p>
        
        <div id="calendar-grid-container" style="display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:20px;">
            <div style="color:var(--muted); text-align:center; padding:40px; grid-column: 1/-1;">Cargando historial de días...</div>
        </div>

        <!-- Leyenda -->
        <div style="margin-top:30px; padding-top:20px; border-top:1px solid rgba(255,255,255,0.05); display:flex; gap:20px; font-size:11px; color:var(--muted);">
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px; background:rgba(0,255,127,0.2); border:1px solid rgba(0,255,127,0.4);"></div>
                Día con Ganancia
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; border-radius:3px; background:rgba(255,61,90,0.2); border:1px solid rgba(255,61,90,0.4);"></div>
                Día con Pérdida
            </div>
            <div style="margin-left:auto; font-style:italic;">
                * Valores en verde/rojo son PnL diario acumulado.
            </div>
        </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    document.getElementById('close-cal').onclick = () => overlay.remove();

    try {
        const res = await fetch(`/api/daily_stats?symbol=${symbol}`);
        const stats = await res.json();
        const container = document.getElementById('calendar-grid-container');

        if (Object.keys(stats).length === 0) {
            container.innerHTML = '<div style="color:var(--muted); text-align:center; padding:40px; grid-column: 1/-1;">No se encontraron trades registrados para este símbolo.</div>';
            return;
        }

        container.innerHTML = '';

        // Agrupar por Mes-Año para mostrar meses separados
        const months = {};
        Object.keys(stats).sort().forEach(dateStr => {
            // Asegurar que solo tomamos la parte YYYY-MM-DD para el agrupamiento
            const cleanDateStr = dateStr.includes(' ') ? dateStr.split(' ')[0] : (dateStr.includes('T') ? dateStr.split('T')[0] : dateStr);
            const date = new Date(cleanDateStr + 'T12:00:00'); // Evitar timezone issues
            const monthKey = date.toLocaleString('default', { month: 'long', year: 'numeric' });
            if (!months[monthKey]) months[monthKey] = [];
            months[monthKey].push(dateStr);
        });

        Object.keys(months).forEach(monthName => {
            const monthDiv = document.createElement('div');
            monthDiv.style.background = 'rgba(255,255,255,0.02)';
            monthDiv.style.borderRadius = '12px';
            monthDiv.style.padding = '15px';
            monthDiv.style.border = '1px solid rgba(255,255,255,0.05)';

            monthDiv.innerHTML = `<h4 style="margin:0 0 15px 0; color:var(--accent2); text-transform:capitalize; font-size:14px;">${monthName}</h4>`;

            const grid = document.createElement('div');
            grid.style.display = 'grid';
            grid.style.gridTemplateColumns = 'repeat(7, 1fr)';
            grid.style.gap = '5px';

            // Dias de la semana headers
            const days = ['D', 'L', 'M', 'M', 'J', 'V', 'S'];
            days.forEach(d => {
                const dEl = document.createElement('div');
                dEl.style.fontSize = '9px';
                dEl.style.color = '#555';
                dEl.style.textAlign = 'center';
                dEl.style.fontWeight = 'bold';
                dEl.textContent = d;
                grid.appendChild(dEl);
            });

            const firstDateInMonthStr = months[monthName][0];
            const parts = firstDateInMonthStr.split('-');
            const y = parseInt(parts[0]), m = parseInt(parts[1]) - 1;
            const firstDayOfMonth = new Date(y, m, 1).getDay();

            for (let i = 0; i < firstDayOfMonth; i++) {
                grid.appendChild(document.createElement('div'));
            }

            // Renderizar días del mes
            const lastDayOfMonth = new Date(y, m + 1, 0).getDate();
            for (let day = 1; day <= lastDayOfMonth; day++) {
                const currentStr = `${y}-${String(m + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
                const dayStat = stats[currentStr];

                const dayEl = document.createElement('div');
                dayEl.style.aspectRatio = '1/1';
                dayEl.style.borderRadius = '4px';
                dayEl.style.display = 'flex';
                dayEl.style.flexDirection = 'column';
                dayEl.style.alignItems = 'center';
                dayEl.style.justifyContent = 'center';
                dayEl.style.fontSize = '10px';
                dayEl.style.position = 'relative';
                dayEl.style.transition = 'all 0.2s';

                if (dayStat) {
                    const isWin = dayStat.pnl >= 0;
                    dayEl.style.background = isWin ? 'rgba(0,255,127,0.15)' : 'rgba(255,61,90,0.15)';
                    dayEl.style.border = `1px solid ${isWin ? 'rgba(0,255,127,0.3)' : 'rgba(255,61,90,0.3)'}`;

                    const winRate = dayStat.trades > 0 ? ((dayStat.wins / dayStat.trades) * 100).toFixed(0) : 0;
                    dayEl.title = `Fecha: ${currentStr}\nTrades: ${dayStat.trades}\nWins: ${dayStat.wins} | Losses: ${dayStat.losses}\nWin Rate: ${winRate}%\nPnL Diario: ${dayStat.pnl >= 0 ? '+' : ''}$${dayStat.pnl.toFixed(2)}`;

                    dayEl.innerHTML = `
                        <span style="font-weight:bold; color:var(--text);">${day}</span>
                        <span style="font-size:9px; color:${isWin ? 'var(--green)' : 'var(--red)'}; font-weight:bold;">
                            ${dayStat.pnl >= 0 ? '+' : ''}$${Math.abs(dayStat.pnl).toFixed(0)}
                        </span>
                        <span style="font-size:7px; color:#555; position:absolute; bottom:2px;">${dayStat.trades}T</span>
                    `;

                    dayEl.onmouseover = () => {
                        dayEl.style.transform = 'scale(1.1)';
                        dayEl.style.zIndex = '10';
                        dayEl.style.background = isWin ? 'rgba(0,255,127,0.3)' : 'rgba(255,61,90,0.3)';
                    };
                    dayEl.onmouseout = () => {
                        dayEl.style.transform = 'scale(1)';
                        dayEl.style.background = isWin ? 'rgba(0,255,127,0.15)' : 'rgba(255,61,90,0.15)';
                    };
                } else {
                    dayEl.style.background = 'rgba(255,255,255,0.02)';
                    dayEl.style.color = 'rgba(255,255,255,0.05)';
                    dayEl.textContent = day;
                }

                grid.appendChild(dayEl);
            }

            monthDiv.appendChild(grid);
            container.appendChild(monthDiv);
        });

    } catch (e) {
        console.error(e);
        document.getElementById('calendar-grid-container').innerHTML = '<div style="color:var(--red); text-align:center; padding:40px; grid-column: 1/-1;">Error al cargar datos del servidor.</div>';
    }
}
