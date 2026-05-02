from flask import Flask, render_template_string, jsonify
import os
import json
import re
from datetime import datetime, timedelta

app = Flask(__name__)

# Configuración
LOG_FILE = "scratch/time_travel.log"
SIM_DETAILS_LOG = "scratch/simulation_details.log"
RESULTS_FILE = "data/backtest_results.json"

def get_metrics():
    # Leer logs principales
    lines = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                # Leer 20,000 líneas para no perder metadatos de misiones largas o cambios de fase
                lines = f.readlines()[-20000:]
        except: pass
        
    # Leer logs detallados (para bloqueos)
    sim_lines = []
    if os.path.exists(SIM_DETAILS_LOG):
        try:
            with open(SIM_DETAILS_LOG, "r") as f:
                # Leer más líneas para no perder el inicio del ciclo (20k líneas)
                sim_lines = f.readlines()[-20000:]
        except: pass
    # 0. Valores por defecto
    current_asset = "..."
    progress = 0
    cycle = "1"
    phase = "ESTUDIO"
    phase_icon = "📚"
    sim_dates = "2024-01-01 a 2024-01-31"
    
    # 1. Analizar LOG PRINCIPAL para estado de misión y símbolos
    symbol_last_event = {}  # sym -> ('START'|'DONE', idx)
    
    # Recorremos 'lines' que viene de LOG_FILE (time_travel.log)
    for line in lines:
        # Detectar Ciclo
        match_cycle = re.search(r"CICLO DE APRENDIZAJE (\d+)", line)
        if match_cycle: cycle = match_cycle.group(1)
        
        # Detectar Fechas (Ej: Fechas: 2025-05-01 a 2025-05-31)
        match_dates = re.search(r"Fechas: ([\d-]+ a [\d-]+)", line)
        if match_dates: 
            sim_dates = match_dates.group(1)
        
        # Patrón alternativo para fechas
        match_alt_dates = re.search(r"HAPI_SIM_START_DATE=([\d-]+)", line)
        if match_alt_dates:
            sim_dates = f"{match_alt_dates.group(1)} a ..." 

        # Rastrear progreso de símbolos (Ej: ▶️ [136/167] ZM en proceso...)
        for m_start in re.finditer(r"▶️ \[(\d+)/167\] (\w+)", line):
            symbol_last_event[m_start.group(2)] = ('START', int(m_start.group(1)))
        
        for m_done in re.finditer(r"[✅❌] \[\d+/167\] (\w+)", line):
            symbol_last_event[m_done.group(1)] = ('DONE', 0)
        
        # Detectar Fase
        if "📚 ENTRENANDO" in line:
            phase = "ESTUDIO"
            phase_icon = "📚"
        elif "🧪 EXAMEN" in line or "🎓 EXAMEN" in line:
            phase = "VALIDACIÓN"
            phase_icon = "🎓"

    # 1.1 Analizar LOG DETALLADO solo para bloqueos de Alpha
    alpha_blocks = sum(1 for l in sim_lines if "🚫 [AlphaOptimizer]" in l)
    total_scans = sum(1 for l in sim_lines if "🎯 GATILLO INTELIGENTE" in l)

    # Derivar activos en vuelo desde LOGS
    active_symbols = {sym for sym, (state, _) in symbol_last_event.items() if state == 'START'}
    
    # 🕵️‍♂️ SENSOR DE PULSO DIRECTO (DOCKER PS)
    # Si los logs fallan o van lentos, Docker nos dirá la verdad
    try:
        import subprocess
        d_ps = subprocess.check_output(["docker", "ps", "--format", "{{.Command}}"], encoding="utf-8")
        for line in d_ps.split("\n"):
            # Buscamos símbolos en los comandos de ejecución de los contenedores
            # El comando suele ser 'python simulation_mode/main_sim.py' pero el TARGET_SYMBOL está en ENV
            # Intentamos detectar por nombres de contenedores o inspección si es necesario
            # Pero por ahora, usemos una técnica más audaz: ver si hay contenedores de hapi-scalping-bot
            pass
        
        # Técnica alternativa: El orquestador ya sabe quién lanzó. 
        # Vamos a confiar en la detección de hilos de sistema.
    except: pass

    progress = max((idx for _, (_, idx) in symbol_last_event.items()), default=0)

    # 2. Procesar Resultados JSON
    home_runs = []
    win_rate_cycle = 0.0
    completed_count = 0
    remaining_count = 167
    current_session_results = []
    total_pnl_global = 0.0
    total_pnl_cycle = 0.0
    
    current_mission_start = sim_dates.split(" a ")[0]
    
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)
                total_pnl_global = sum(float(r.get('pnl', 0)) for r in all_results)
                
                current_session_results = [
                    r for r in all_results 
                    if current_mission_start in str(r.get("sim_start_date",""))
                ]
                
                total_pnl_cycle = sum(float(r.get('pnl', 0)) for r in current_session_results)
                
                finished_syms = {r['symbol'] for r in current_session_results}
                active_symbols -= finished_syms
                
                completed_count = len(finished_syms)
                remaining_count = max(0, 167 - completed_count)
                
                if current_session_results:
                    wins = sum(1 for r in current_session_results if float(r.get('pnl', 0)) > 0)
                    win_rate_cycle = (wins / len(current_session_results)) * 100 if len(current_session_results) > 0 else 0
        except Exception as e:
            print(f"Error JSON: {e}")

    # 3. Identificar activos en paralelo REALES (Sensor de Pulso de Estado)
    try:
        # Usamos state_sim.json que es donde el motor de simulación reporta en vivo
        state_file = "data/state_sim.json"
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                full_state = json.load(f)
                now_ts = time.time()
                for sym, s_data in full_state.items():
                    # Si se actualizó hace menos de 60 segundos, está VIVO
                    if now_ts - s_data.get("last_update", 0) < 60:
                        if sym != "─": # Ignorar separador global
                            active_symbols.add(sym)
    except Exception as e:
        print(f"Error en sensor de pulso: {e}")

    parallel_assets = sorted(list(active_symbols))
    
    # Limitar a los últimos 8 para la visualización compacta
    parallel_assets = parallel_assets[-8:]
    
    current_asset_text = ", ".join(parallel_assets) if parallel_assets else "Sincronizando flota..."
    if len(parallel_assets) > 4:
        current_asset_text = ", ".join(parallel_assets[:4]) + "<br>" + ", ".join(parallel_assets[4:])

    # 4. Construir Radar de Resultados
    for a in active_symbols:
        partial = next((r for r in current_session_results if r.get('symbol') == a), None)
        if partial:
            home_runs.append({
                "symbol": a, "pnl": f"${float(partial.get('pnl',0)):,.2f}",
                "trades": partial.get('total_trades', 0),
                "win_rate": f"{float(partial.get('win_rate',0)):.1f}%",
                "status": "EJECUTANDO", "fees": f"${float(partial.get('total_fees',0)):,.2f}"
            })
        else:
            home_runs.append({
                "symbol": a, "pnl": "...", "trades": "-", "win_rate": "-", "status": "EJECUTANDO", "fees": "-"
            })

    for r in reversed(current_session_results):
        if r['symbol'] not in active_symbols:
            home_runs.append({
                "symbol": r.get('symbol'), "pnl": f"${float(r.get('pnl',0)):,.2f}",
                "trades": r.get('total_trades', 0), "win_rate": f"{float(r.get('win_rate',0)):.1f}%",
                "status": "COMPLETADO", "fees": f"${float(r.get('total_fees',0)):,.2f}"
            })
    # 5. Listado de símbolos con calendario disponible
    calendar_symbols = []
    try:
        cal_path = "data/trade_calendar.json"
        if os.path.exists(cal_path):
            with open(cal_path, "r") as f:
                cal_data = json.load(f)
                all_cal_syms = set()
                for day_trades in cal_data.values():
                    for t in day_trades:
                        sym = t.get("symbol")
                        if sym: all_cal_syms.add(sym.strip())
                calendar_symbols = list(all_cal_syms)
    except: pass

    return {
        "cycle": cycle,
        "total_sessions": total_cycles if 'total_cycles' in locals() else 1,
        "completed_count": completed_count,
        "remaining_count": remaining_count,
        "phase": phase,
        "phase_icon": phase_icon,
        "sim_dates": sim_dates,
        "asset": current_asset_text,
        "total_pnl": round(total_pnl_cycle, 2),
        "total_pnl_global": round(total_pnl_global, 2),
        "total_fees": round(sum(float(r.get('total_fees',0)) for r in current_session_results), 2),
        "win_rate_cycle": round(win_rate_cycle, 1),
        "progress": progress,
        "progress_pct": round((progress / 167) * 100, 1),
        "alpha_blocks": alpha_blocks,
        "selectivity": round((alpha_blocks / total_scans * 100), 1) if total_scans > 0 else 0,
        "home_runs": home_runs,
        "calendar_symbols": calendar_symbols,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "history": get_history()
    }

def get_history():
    history_file = "scratch/TIME_TRAVEL_RESULTS.md"
    if not os.path.exists(history_file): return []
    try:
        with open(history_file, "r") as f:
            lines = f.readlines()
        data = []
        for line in lines:
            if "|" in line and "Ciclo" not in line and "---" not in line and "#" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 6:
                    data.append({
                        "cycle": parts[0],
                        "month": parts[1],
                        "pnl_study": parts[2],
                        "pnl_val": parts[3],
                        "trades": parts[4],
                        "stats": parts[5]
                    })
        return data[::-1] # Invertir para ver lo más reciente primero
    except: return []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alpha Mission Control</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0b10;
            --card-bg: #161b22;
            --neon-cyan: #00f2ff;
            --neon-green: #39ff14;
            --neon-red: #ff3131;
            --text-gray: #8b949e;
            --neon-yellow: #f2ff00;
        }
        .cal-btn {
            background: rgba(0, 242, 255, 0.1);
            border: 1px solid var(--neon-cyan);
            color: var(--neon-cyan);
            cursor: pointer;
            font-size: 0.7rem;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 5px;
            transition: all 0.3s;
            font-family: 'Inter', sans-serif;
        }
        .cal-btn:hover { background: var(--neon-cyan); color: #000; }
        
        body {
            background-color: var(--bg-dark);
            color: white;
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #1f2937;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        h1 {
            font-family: 'Orbitron', sans-serif;
            color: var(--neon-cyan);
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-size: 1.5rem;
        }
        .status-badge {
            background: rgba(57, 255, 20, 0.1);
            color: var(--neon-green);
            padding: 5px 15px;
            border-radius: 20px;
            border: 1px solid var(--neon-green);
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: bold;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #30363d;
            transition: transform 0.3s ease;
        }
        .card:hover { transform: translateY(-5px); border-color: var(--neon-cyan); }
        .card-title {
            color: var(--text-gray);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .card-value {
            font-size: 2rem;
            font-weight: bold;
            font-family: 'Orbitron', sans-serif;
        }
        .progress-container {
            width: 100%;
            background: #21262d;
            height: 10px;
            border-radius: 5px;
            margin-top: 15px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #00f2ff, #39ff14);
            width: 0%;
            transition: width 1s ease-in-out;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th {
            text-align: left;
            color: var(--text-gray);
            font-size: 0.8rem;
            padding: 10px;
            border-bottom: 1px solid #30363d;
        }
        td {
            padding: 15px 10px;
            border-bottom: 1px solid #30363d;
        }
        .pnl-plus { color: var(--neon-green); font-weight: bold; }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.8);
            backdrop-filter: blur(5px);
        }
        .modal-content {
            background-color: var(--card-bg);
            margin: 5% auto;
            padding: 30px;
            border: 1px solid var(--neon-cyan);
            border-radius: 15px;
            width: 80%;
            max-width: 900px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 0 30px rgba(0,242,255,0.2);
        }
        .close-modal {
            color: var(--text-gray);
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }
        .close-modal:hover { color: var(--neon-red); }

        .btn-history {
            background: rgba(0,242,255,0.1);
            border: 1px solid var(--neon-cyan);
            color: var(--neon-cyan);
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.75rem;
            transition: 0.3s;
        }
        .btn-history:hover {
            background: var(--neon-cyan);
            color: black;
            box-shadow: 0 0 15px var(--neon-cyan);
        }
        
        .capital-banner {
            display: flex;
            align-items: center;
            gap: 30px;
            background: linear-gradient(135deg, rgba(0,242,255,0.06), rgba(57,255,20,0.04));
            border: 1px solid rgba(0,242,255,0.2);
            border-radius: 50px;
            padding: 10px 28px;
            margin-bottom: 25px;
            font-size: 0.82rem;
            flex-wrap: wrap;
        }
        .capital-banner .cb-item {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-gray);
        }
        .capital-banner .cb-value {
            color: var(--neon-cyan);
            font-weight: bold;
            font-family: 'Orbitron', sans-serif;
            font-size: 0.9rem;
        }
        .capital-banner .cb-sep {
            width: 1px;
            height: 20px;
            background: rgba(0,242,255,0.2);
        }
        .footer {
            text-align: center;
            color: var(--text-gray);
            font-size: 0.8rem;
            margin-top: 50px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Alpha Mission Control</h1>
                <p style="color: var(--text-gray); margin: 5px 0 0 0;">AI Double-Layer Monitoring System</p>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <button class="btn-history" onclick="toggleModal(true)">📑 Historial de Ciclos</button>
                <div class="status-badge">● System Online</div>
            </div>
        </header>

        <!-- Modal de Historial -->
        <div id="historyModal" class="modal">
            <div class="modal-content">
                <span class="close-modal" onclick="toggleModal(false)">&times;</span>
                <h2 style="font-family: 'Orbitron', sans-serif; color: var(--neon-cyan); margin-bottom: 20px;">📜 Historial de Misiones</h2>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid var(--neon-cyan); color: var(--text-gray); text-align: left;">
                            <th style="padding: 12px;">Ciclo</th>
                            <th>Mes</th>
                            <th>PnL ESTUDIO</th>
                            <th>PnL VALIDACIÓN (2026)</th>
                            <th>Trades</th>
                            <th>Win/Loss</th>
                        </tr>
                    </thead>
                    <tbody id="history-table-body">
                        <!-- Dinámico -->
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Banner de Capital Inicial -->
        <div class="capital-banner">
            <div class="cb-item">
                💰 Capital por empresa:
                <span class="cb-value">$10,000.00</span>
            </div>
            <div class="cb-sep"></div>
            <div class="cb-item">
                🌐 Capital total desplegado:
                <span class="cb-value">$1,670,000</span>
                <span style="color:var(--text-gray); font-size:0.75rem;">(167 empresas)</span>
            </div>
            <div class="cb-sep"></div>
            <div class="cb-item">
                📈 ROI del ciclo:
                <span class="cb-value" id="roi-value">0.00%</span>
            </div>
        </div>

        <div class="grid">
            <div class="card" id="phase-card">
                <div class="card-title">FASE DE MISIÓN</div>
                <div class="card-value" style="color: var(--neon-cyan); font-size: 1.6rem;">
                    <span id="phase-icon">📚</span> <span id="phase-name">ESTUDIO</span>
                </div>
                <div style="margin-top: 8px; color: var(--neon-cyan); font-size: 0.85rem; font-weight: bold;">
                    📅 <span id="sim-dates">Cargando...</span>
                </div>
                <p id="phase-desc" style="color: var(--text-gray); font-size: 0.75rem; margin: 10px 0 0 0;">IA aprendiendo patrones históricos (Cerebro Abierto)</p>
            </div>
            <div class="card">
                <div class="card-title">Evolución IA</div>
                <div class="card-value" style="font-size: 1.5rem; color: var(--neon-cyan);">CICLO <span id="top-cycle-value">1</span></div>
                <p style="color: var(--text-gray); font-size: 0.8rem;">Total histórico: <span id="total-cycles-value">?</span> sesiones</p>
            </div>
            <div class="card" id="pnl-card" style="border-width: 2px;">
                <div class="card-title">💰 TOTAL GANADO (MISIÓN)</div>
                <div class="card-value" id="total-pnl">$0.00</div>
                <p style="color: var(--text-gray); font-size: 0.8rem;" id="win-rate-cycle">Tasa de Éxito Global: 0%</p>
            </div>
            <div class="card">
                <div class="card-title">Selectividad IA</div>
                <div class="card-value" id="selectivity">0%</div>
                <p style="color: var(--text-gray); font-size: 0.8rem;" id="blocks-desc">0 bloqueos</p>
            </div>
            <div class="card">
                <div class="card-title">Activo Actual (Paralelo)</div>
                <div class="card-value" id="current-asset" style="font-size: 1.2rem; word-break: break-word; line-height: 1.4;">...</div>
                <p style="color: var(--text-gray); font-size: 0.8rem; margin-top: 10px; margin-bottom: 5px;" id="progress-text">Avance global: 0%</p>
                <div class="progress-container">
                    <div class="progress-bar" id="progress-bar"></div>
                </div>
            </div>
            <div class="card" style="border: 2px solid var(--neon-cyan); text-align: center;">
                <div class="card-title">⏳ Empresas Faltantes</div>
                <div class="card-value" id="remaining-count" style="font-size: 3rem; color: var(--neon-cyan);">167</div>
                <p style="color: var(--text-gray); font-size: 0.8rem;">de 167 — completadas: <span id="completed-count" style="color: var(--neon-green); font-weight: bold;">0</span></p>
            </div>
        </div>

        <div class="card">
            <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; margin-top: 0;">Radar de Resultados (Ciclo Actual Completo)</h2>
            <div style="max-height: 400px; overflow-y: auto;">
                <table id="results-table">
                    <thead>
                        <tr>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Símbolo</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">PnL Neto</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Comisiones</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Trades</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Win Rate</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Estado</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Dinámico -->
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card" style="margin-top: 20px;">
            <div class="card-title">Línea de Tiempo (Simulación)</div>
            <div class="card-value" style="font-size: 1.2rem; padding-top: 10px;" id="sim-dates-footer">Cargando...</div>
            <p style="color: var(--text-gray); font-size: 0.85rem;" id="cycle-num">Ciclo 1</p>
        </div>

        <!-- Desglose Financiero Real -->
        <div class="card" style="margin-top: 20px; background: linear-gradient(135deg, rgba(0,242,255,0.04), rgba(0,0,0,0));">
            <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1rem; margin-top: 0; color: var(--neon-cyan);">💹 Desglose Financiero Real (Proyección)</h2>
            <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.88rem;">
                <div style="display:flex; justify-content:space-between; padding: 10px 15px; background: rgba(255,255,255,0.03); border-radius: 8px;">
                    <span>📊 Ganancia Bruta (trades)</span>
                    <span style="color:var(--neon-green); font-weight:bold;" id="fis-gross">$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 8px 15px; padding-left:30px; color: var(--text-gray);">
                    <span>│ − Comisiones broker</span>
                    <span style="color:var(--neon-red);" id="fis-fees">-$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 8px 15px; padding-left:30px; color: var(--text-gray);">
                    <span>│ − Slippage de mercado</span>
                    <span style="color:var(--neon-red);" id="fis-slip">-$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 10px 15px; background: rgba(0,242,255,0.06); border-radius: 8px; border-left: 3px solid var(--neon-cyan);">
                    <span style="font-weight:bold;">= PnL Neto (en pantalla)</span>
                    <span style="color:var(--neon-cyan); font-weight:bold;" id="fis-net">$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 8px 15px; padding-left:30px; color: var(--text-gray);">
                    <span>│ − IRS EE.UU. 30% (NRA short-term)</span>
                    <span style="color:var(--neon-red);" id="fis-irs">-$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 10px 15px; background: rgba(255,165,0,0.06); border-radius: 8px; border-left: 3px solid orange;">
                    <span style="font-weight:bold;">= Después de EE.UU.</span>
                    <span style="color:orange; font-weight:bold;" id="fis-post-irs">$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 8px 15px; padding-left:30px; color: var(--text-gray);">
                    <span>│ − GMF Colombia 4x1000 (repatriación)</span>
                    <span style="color:var(--neon-red);" id="fis-gmf">-$0.00</span>
                </div>
                <div style="display:flex; justify-content:space-between; padding: 12px 15px; background: rgba(57,255,20,0.08); border-radius: 8px; border-left: 3px solid var(--neon-green); margin-top: 4px;">
                    <span style="font-weight:bold; font-size:1rem;">🇨🇴 Neto en tu bolsillo (Colombia)</span>
                    <span style="color:var(--neon-green); font-weight:bold; font-size:1.1rem; font-family:'Orbitron',sans-serif;" id="fis-colombia">$0.00</span>
                </div>
                <p style="color:var(--text-gray); font-size:0.72rem; margin-top:8px; padding: 0 5px;">
                    ⚠️ Proyección estimada. IRS 30% aplica como retención NRA sobre ganancias de corto plazo. Colombia no tiene tratado de doble tributación con EE.UU. Consultar con asesor tributario.
                </p>
            </div>
        </div>

        <div class="footer">
            Diseñado por Lumen | Sincronizado con Hapi Core v3.0 | <span id="clock">00:00:00</span>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const response = await fetch('/api/metrics');
                const data = await response.json();

                document.getElementById('selectivity').innerText = data.selectivity + '%';
                document.getElementById('blocks-desc').innerText = data.alpha_blocks + ' trades filtrados por bajo valor';
                
                let progText = document.getElementById('progress-text');
                if(progText) progText.innerText = 'Avance global: ' + data.progress + ' / 167 (' + data.progress_pct + '%)';
                
                document.getElementById('sim-dates-footer').innerText = data.sim_dates;
                document.getElementById('cycle-num').innerText = 'Exploración Ciclo ' + data.cycle;
                
                let topCycle = document.getElementById('top-cycle-value');
                if(topCycle) topCycle.innerText = data.cycle;
                let totalCycles = document.getElementById('total-cycles-value');
                if(totalCycles) {
                    totalCycles.innerText = (data.total_sessions || '1') + ' sesiones';
                }

                // Historial
                const hBody = document.getElementById('history-table-body');
                if (hBody && data.history) {
                    hBody.innerHTML = data.history.map(h => `
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td style="padding: 12px; font-weight:bold; color:var(--neon-cyan)">Ciclo ${h.cycle}</td>
                            <td style="color:var(--text-gray)">${h.month}</td>
                            <td style="opacity:0.8">${h.pnl_study}</td>
                            <td style="font-family:'Orbitron',sans-serif">${h.pnl_val}</td>
                            <td>${h.trades}</td>
                            <td>${h.stats}</td>
                        </tr>
                    `).join('');
                }

                // ROI del ciclo
                const roi = ((data.total_pnl / (167 * 10000)) * 100).toFixed(3);
                const roiEl = document.getElementById('roi-value');
                if(roiEl) {
                    roiEl.innerText = (roi >= 0 ? '+' : '') + roi + '%';
                    roiEl.style.color = roi >= 0 ? 'var(--neon-green)' : 'var(--neon-red)';
                }

                let remaining = document.getElementById('remaining-count');
                let completed = document.getElementById('completed-count');
                if(remaining) {
                    remaining.innerText = data.remaining_count;
                    remaining.style.color = data.remaining_count === 0 ? 'var(--neon-green)' : 'var(--neon-cyan)';
                }
                if(completed) {
                    completed.innerText = data.completed_count;
                }
                if (data.current_asset) {
                    document.getElementById('current-asset').innerHTML = data.current_asset;
                } else {
                    document.getElementById('current-asset').innerText = "Sincronizando flota...";
                }
                
                if (document.getElementById('progress-bar')) {
                    document.getElementById('progress-bar').style.width = data.progress_pct + '%';
                }

                // Desglose Financiero
                const gross = data.total_pnl + (data.total_fees || 0) + (data.total_slippage || 0);
                const irs = data.total_pnl * 0.30;
                const postIrs = data.total_pnl - irs;
                const gmf = postIrs * 0.004;
                const colombia = postIrs - gmf;
                const fmt = v => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
                const setEl = (id, v, force) => { const el = document.getElementById(id); if(el) { el.innerText = force || fmt(v); } };
                setEl('fis-gross', gross);
                setEl('fis-fees', 0, '-$' + (data.total_fees||0).toFixed(2));
                setEl('fis-slip', 0, '-$' + (data.total_slippage||0).toFixed(2));
                setEl('fis-net', data.total_pnl);
                setEl('fis-irs', 0, '-$' + irs.toFixed(2));
                setEl('fis-post-irs', postIrs);
                setEl('fis-gmf', 0, '-$' + gmf.toFixed(2));
                setEl('fis-colombia', colombia);
                const colEl = document.getElementById('fis-colombia');
                if(colEl) colEl.style.color = colombia >= 0 ? 'var(--neon-green)' : 'var(--neon-red)';

                // Actualizar PnL Neto
                const pnlValue = document.getElementById('total-pnl');
                const pnlCard = document.getElementById('pnl-card');
                const winRate = document.getElementById('win-rate-cycle');
                
                pnlValue.innerText = '$' + data.total_pnl.toLocaleString();
                winRate.innerHTML = 'Ganancia Histórica: <b style="color:var(--neon-cyan)">$' + data.total_pnl_global.toLocaleString() + '</b><br>Win Rate Activos: ' + data.win_rate_cycle + '%';
                
                if (data.total_pnl >= 0) {
                    pnlValue.style.color = "var(--neon-green)";
                    pnlCard.style.borderColor = "var(--neon-green)";
                } else {
                    pnlValue.style.color = "var(--neon-red)";
                    pnlCard.style.borderColor = "var(--neon-red)";
                }

                // Actualizar Fase
                const phaseName = document.getElementById('phase-name');
                const phaseIcon = document.getElementById('phase-icon');
                const phaseDesc = document.getElementById('phase-desc');
                const phaseCard = document.getElementById('phase-card');
                const simDates = document.getElementById('sim-dates');
                
                phaseName.innerText = data.phase;
                phaseIcon.innerText = data.phase_icon;
                simDates.innerText = data.sim_dates || '...';
                
                if (data.phase === "ESTUDIO") {
                    phaseName.style.color = "var(--neon-cyan)";
                    phaseDesc.innerHTML = '🧠 <b style="color:var(--neon-green)">Cerebro Abierto</b>: Absorbiendo patrones';
                    phaseCard.style.borderColor = "var(--neon-cyan)";
                } else {
                    phaseName.style.color = "var(--neon-green)";
                    phaseDesc.innerHTML = '🛡️ <b style="color:var(--neon-red)">Cerebro Congelado</b>: Evaluación 2026';
                    phaseCard.style.borderColor = "var(--neon-green)";
                }

                const tbody = document.querySelector('#results-table tbody');
                tbody.innerHTML = '';
                data.home_runs.forEach(r => {
                    let statusBadge = `<span class="status-badge" style="border-color: var(--neon-green); color: var(--neon-green);">COMPLETADO</span>`;
                    let pnlColor = r.pnl.includes('-') ? 'var(--neon-red)' : 'var(--neon-green)';

                    if (r.status === 'EJECUTANDO') {
                        statusBadge = `
                            <div style="display:flex; align-items:center; gap:10px;">
                                <span class="status-badge" style="font-size:0.65rem; border-color: var(--neon-cyan); color: var(--neon-cyan); box-shadow: 0 0 5px var(--neon-cyan);">EN PROCESO ⚙️</span>
                                <button onclick="fetchSymbolUpdate('${r.symbol}')" id="btn-update-${r.symbol}"
                                        style="background:rgba(0,242,255,0.1); border:1px solid var(--neon-cyan); border-radius:6px; color:var(--neon-cyan); cursor:pointer; font-size:0.75rem; padding:4px 10px; font-family:'Orbitron';">
                                    📊 Ver
                                </button>
                            </div>`;
                        pnlColor = 'var(--neon-cyan)';
                    }

                    const hasCalendar = data.calendar_symbols && data.calendar_symbols.includes(r.symbol);
                    const calBtn = hasCalendar ? `<button class="cal-btn" onclick="showTradeCalendar('${r.symbol}')" title="Ver Auditoría">📅</button>` : '';

                    const row = `<tr>
                        <td style="font-weight:bold; color:var(--text-main)">
                            ${r.symbol}
                            ${calBtn}
                        </td>
                        <td style="color: ${pnlColor}">${r.pnl}</td>
                        <td style="color: var(--neon-red); font-size: 0.85rem;">${r.fees || '$0.00'}</td>
                        <td>${r.trades}</td>
                        <td>${r.win_rate}</td>
                        <td>${statusBadge}</td>
                    </tr>`;
                    tbody.innerHTML += row;
                });
            } catch (e) { console.error("Error updating dashboard:", e); }
        }

        // Función para actualizar una fila específica sin salir de la página
        async function fetchSymbolUpdate(symbol) {
            const btn = document.getElementById('btn-update-' + symbol);
            if (btn) { btn.innerText = '⏳'; btn.disabled = true; }
            try {
                const res = await fetch('/api/symbol/' + symbol.replace('.', '-'));
                const d = await res.json();
                
                // Buscar la fila y actualizar celdas
                const rows = document.querySelectorAll('#results-table tbody tr');
                rows.forEach(row => {
                    if (row.cells[0].innerText === symbol) {
                        const pnl = parseFloat(d.pnl || 0);
                        row.cells[1].innerText = '$' + pnl.toFixed(2);
                        row.cells[1].style.color = pnl >= 0 ? 'var(--neon-green)' : 'var(--neon-red)';
                        row.cells[3].innerText = d.trades || '0';
                        row.cells[4].innerText = (parseFloat(d.win_rate || 0)).toFixed(1) + '%';
                        
                        // Si ya terminó, cambiar estado
                        if (d.status === 'COMPLETADO') {
                            row.cells[5].innerHTML = '<span class="status-badge" style="border-color: var(--neon-green); color: var(--neon-green);">COMPLETADO</span>';
                        }
                    }
                });
            } catch(e) { console.error("Error updating symbol row:", e); }
            if (btn) { btn.innerText = '📊 Ver'; btn.disabled = false; }
        }

        async function showTradeCalendar(symbol) {
            const modal = document.getElementById('calendarModal');
            const body = document.getElementById('calendar-body');
            document.getElementById('cal-symbol-title').innerText = 'Auditoría de Trades: ' + symbol;
            body.innerHTML = '<div style="text-align:center; padding:20px;">Cargando bitácora... ⏳</div>';
            modal.style.display = 'block';
            
            try {
                const res = await fetch('/api/calendar/' + symbol);
                const trades = await res.json();
                
                if (trades.error || trades.length === 0) {
                    body.innerHTML = '<div style="text-align:center; padding:20px; color:var(--neon-red)">⚠️ No hay trades detallados para este activo.</div>';
                    return;
                }
                
                let html = '<table style="width:100%; font-size:0.85rem; border-collapse:collapse;">';
                html += '<tr style="color:var(--neon-cyan); border-bottom:1px solid var(--neon-cyan); text-align:left;">';
                html += '<th style="padding:10px;">Fecha</th><th style="padding:10px;">Entrada</th><th style="padding:10px;">Salida</th><th style="padding:10px;">PnL</th><th style="padding:10px;">Estilo</th></tr>';
                
                trades.forEach(t => {
                    const color = t.pnl >= 0 ? 'var(--neon-green)' : 'var(--neon-red)';
                    const styleColor = t.style === 'SWING' ? 'var(--neon-yellow)' : 'var(--neon-cyan)';
                    html += `<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:10px;">${t.date}</td>
                        <td style="padding:10px; color:rgba(255,255,255,0.5);">${t.entry}</td>
                        <td style="padding:10px;">${t.exit}</td>
                        <td style="padding:10px; color:${color}; font-weight:bold;">$${t.pnl.toFixed(2)}</td>
                        <td style="padding:10px;"><span style="color:${styleColor}; font-size:0.7rem; border:1px solid ${styleColor}; padding:2px 4px; border-radius:4px;">${t.style}</span></td>
                    </tr>`;
                });
                html += '</table>';
                body.innerHTML = html;
            } catch (e) {
                body.innerHTML = '<div style="text-align:center; padding:20px; color:var(--neon-red)">❌ Error cargando calendario.</div>';
            }
        }

        setInterval(updateDashboard, 10000);
        updateDashboard();

        function toggleModal(show) {
            document.getElementById('historyModal').style.display = show ? 'block' : 'none';
        }

        function toggleCalModal(show) {
            document.getElementById('calendarModal').style.display = show ? 'block' : 'none';
        }

        // Cerrar al clickear fuera
        window.onclick = function(event) {
            const modal = document.getElementById('historyModal');
            const calModal = document.getElementById('calendarModal');
            if (event.target == modal) toggleModal(false);
            if (event.target == calModal) toggleCalModal(false);
        }
    </script>

    <!-- Modal de Calendario -->
    <div id="calendarModal" class="modal">
        <div class="modal-content" style="max-width: 600px;">
            <div class="modal-header">
                <h2 id="cal-symbol-title">Auditoría de Trades</h2>
                <span class="close-btn" onclick="toggleCalModal(false)">&times;</span>
            </div>
            <div id="calendar-body" class="modal-body">
                <!-- Se llena vía JS -->
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/calendar/<symbol>')
def get_symbol_calendar(symbol):
    try:
        # Saneamiento del símbolo
        search_symbol = symbol.strip().upper()
        
        # Cargar calendario completo
        full_cal = {}
        cal_path = "data/trade_calendar.json"
        if os.path.exists(cal_path):
            with open(cal_path, "r") as f:
                full_cal = json.load(f)
            
        # Filtrar trades para este símbolo
        trades = []
        for date_str, daily_trades in full_cal.items():
            for t in daily_trades:
                # Comparación limpia
                entry_sym = t.get("symbol", "").strip().upper()
                if entry_sym == search_symbol:
                    t_copy = t.copy()
                    t_copy["date"] = date_str
                    trades.append(t_copy)
        
        # Ordenar por fecha y hora de entrada
        trades.sort(key=lambda x: (x["date"], x["entry"]))
        return jsonify(trades)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics')
def api_metrics():
    return jsonify(get_metrics())

@app.route('/api/symbol/<symbol>')
def api_symbol(symbol):
    """Endpoint on-demand: devuelve el progreso de un símbolo en vuelo.
    Lee el archivo temporal results_{symbol}.json si existe.
    Sin polling - solo responde cuando se le consulta."""
    symbol = symbol.upper().replace('-', '.') # normalizar BRK-B -> BRK.B
    res_file = os.path.join("data", f"results_{symbol}.json")
    
    # Buscar en backtest_results.json primero (ya terminó y fue consolidado)
    today_str = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)
            match = next((r for r in reversed(all_results) 
                         if r.get('symbol') == symbol 
                         and r.get('timestamp','').startswith(today_str)), None)
            if match:
                return jsonify({
                    "status": "COMPLETADO",
                    "symbol": symbol,
                    "trades": match.get('total_trades', 0),
                    "pnl": match.get('pnl', 0),
                    "win_rate": match.get('win_rate', 0),
                    "stage": match.get('stage', ''),
                    "sim_start": match.get('sim_start_date', ''),
                    "sim_end": match.get('sim_end_date', ''),
                })
        except: pass
    
    # Buscar en archivo temporal individual (en vuelo)
    if os.path.exists(res_file):
        try:
            with open(res_file, "r") as f:
                data = json.load(f)
            # Puede ser lista o dict
            if isinstance(data, list) and data:
                r = data[-1]
            elif isinstance(data, dict):
                r = data
            else:
                r = {}
            return jsonify({
                "status": "EN_VUELO",
                "symbol": symbol,
                "trades": r.get('total_trades', r.get('trades', '?')),
                "pnl": r.get('pnl', 0),
                "win_rate": r.get('win_rate', 0),
            })
        except: pass

    # No hay datos aún
    return jsonify({"status": "SIN_DATOS", "symbol": symbol, "message": "Aún procesando, sin resultados parciales disponibles."})

if __name__ == '__main__':
    print("🚀 Alpha Web Dashboard corriendo en http://localhost:5001")
    app.run(host='0.0.0.0', port=5001)
