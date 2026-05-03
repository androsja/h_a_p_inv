from flask import Flask, render_template_string, jsonify
import os
import json
import re
import time
from datetime import datetime

app = Flask(__name__)

# Configuración
LOG_FILE = "scratch/orchestrator.log"
SIM_DETAILS_LOG = "scratch/simulation_details.log"
RESULTS_FILE = "data/backtest_results.json"
ASSETS_FILE = "simulation_config.json"
CALENDAR_FILE = "data/trade_calendar.json"

ALGO_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Cerebro HAPI | Inteligencia Artificial</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #05070a;
            --card-bg: rgba(13, 17, 23, 0.95);
            --accent: #00f2ff;
            --neon-green: #39ff14;
            --text: #e6edf3;
            --border: rgba(0, 242, 255, 0.2);
        }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            margin: 0;
            line-height: 1.6;
            background-image: radial-gradient(circle at 50% 0%, #102030 0%, #05070a 100%);
        }
        .container { max-width: 1000px; margin: 0 auto; padding: 40px 20px; }
        .hero { text-align: center; margin-bottom: 60px; }
        h1 { font-family: 'Orbitron', sans-serif; font-size: 2.5rem; color: var(--accent); text-shadow: 0 0 20px rgba(0, 242, 255, 0.5); }
        
        .step-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
        }
        .step-num { 
            position: absolute; top: -10px; left: -10px; 
            background: var(--accent); color: #000; 
            width: 50px; height: 50px; border-radius: 25px; 
            display: flex; align-items: center; justify-content: center; 
            font-family: 'Orbitron'; font-weight: bold; font-size: 1.2rem;
            box-shadow: 0 0 20px var(--accent);
        }

        .visual-flow {
            display: flex; justify-content: space-between; align-items: center;
            margin: 40px 0; padding: 20px; background: rgba(0,0,0,0.3); border-radius: 15px;
        }
        .node { 
            text-align: center; padding: 15px; border: 1px solid var(--accent); 
            border-radius: 10px; width: 180px; font-size: 0.8rem; font-family: 'Orbitron';
            background: rgba(0, 242, 255, 0.05);
        }
        .arrow { color: var(--accent); font-size: 1.5rem; }

        .indicator-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .ind-item { border-left: 4px solid var(--accent); padding-left: 15px; }
        .ind-title { color: var(--accent); font-weight: bold; display: block; margin-bottom: 5px; }
        
        .dimension-pill {
            display: inline-block; background: rgba(0, 242, 255, 0.1);
            color: var(--accent); padding: 5px 12px; border-radius: 15px;
            font-size: 0.75rem; margin: 4px; border: 1px solid var(--border);
        }

        .highlight { color: var(--neon-green); font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1>EL CEREBRO DE HAPI</h1>
            <p>Guía didáctica: ¿Cómo piensa el bot para ganar dinero?</p>
        </div>

        <div class="step-card">
            <div class="step-num">01</div>
            <h2>La "Vista de Águila" (Datos del Mercado)</h2>
            <p>El bot no mira una gráfica confusa. Él divide el tiempo en <span class="highlight">Velas de 1 minuto</span>. Cada minuto, Hapi analiza el precio y se hace una pregunta: <i>"¿Hacia dónde va la energía ahora mismo?"</i></p>
            <div class="visual-flow">
                <div class="node">PRECIO EN VIVO</div>
                <div class="arrow">➜</div>
                <div class="node">EXTRACCIÓN DE DATOS</div>
                <div class="arrow">➜</div>
                <div class="node">DECISIÓN IA</div>
            </div>
        </div>

        <div class="step-card">
            <div class="step-num">02</div>
            <h2>Las 3 Brújulas (Indicadores)</h2>
            <p>Para no perderse, el bot usa tres herramientas que cualquier novato puede entender:</p>
            <div class="indicator-grid">
                <div class="ind-item">
                    <span class="ind-title">🧭 La Brújula de Tendencia (EMAs)</span>
                    Nos dice si el mar está tranquilo o hay tormenta. Solo operamos cuando la tendencia es clara y fuerte.
                </div>
                <div class="ind-item">
                    <span class="ind-title">🌡️ El Termómetro de Miedo (RSI)</span>
                    Nos dice si la gente está comprando por pánico o vendiendo por miedo. Evitamos comprar cuando todos están "eufóricos".
                </div>
            </div>
        </div>

        <div class="step-card">
            <div class="step-num">03</div>
            <h2>El Oráculo: Las 13 Dimensiones</h2>
            <p>Aquí es donde ocurre la magia. Antes de poner tu dinero en riesgo, la IA revisa <span class="highlight">13 factores secretos</span> al mismo tiempo:</p>
            <div>
                <span class="dimension-pill">Fuerza del Precio</span>
                <span class="dimension-pill">Volumen de Dinero</span>
                <span class="dimension-pill">Volatilidad</span>
                <span class="dimension-pill">Sombras de Velas</span>
                <span class="dimension-pill">Hora del Día</span>
                <span class="dimension-pill">Distancia al Promedio</span>
                <span class="dimension-pill">Aceleración</span>
                <span class="dimension-pill">Historial de Éxito</span>
                <span class="dimension-pill">...y 5 más</span>
            </div>
            <p style="margin-top:20px;">Si la IA detecta una probabilidad de éxito mayor al <span class="highlight">51%</span>, da luz verde. Si no, bloquea el trade para proteger tu capital.</p>
        </div>

        <div class="step-card" style="border-color: var(--neon-green);">
            <div class="step-num" style="background: var(--neon-green); box-shadow: 0 0 20px var(--neon-green);">04</div>
            <h2>El Escudo Protector (Gestión)</h2>
            <p>Hapi nunca arriesga todo en una jugada. Usa una regla de oro: <span class="highlight">Ganar más de lo que pierdes.</span></p>
            <p>Por cada $10 que el bot arriesga, él busca ganar $15. Esto significa que aunque se equivoque la mitad de las veces, **tu cuenta sigue creciendo**.</p>
        </div>

        <p style="text-align: center; color: var(--dim); font-size: 0.8rem;">
            HAPI ALPHA MISSION CONTROL | Diseñado para ganar, blindado para durar.
        </p>
    </div>
</body>
</html>
"""

def get_history():
    data = []
    report_file = "scratch/TIME_TRAVEL_RESULTS.md"
    if not os.path.exists(report_file):
        return []
    try:
        with open(report_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "|" in line and "Ciclo" not in line and "---" not in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    # Formato v3 (8 columnas): | Ciclo | Mes Estudio | Mes Validación | PnL Estudio | PnL Validación | Trades Est. | Trades Valid. | Empresas (+/-) |
                    if len(parts) >= 8:
                        data.append({
                            "cycle": parts[0],
                            "month": parts[1],
                            "eval_month": parts[2],
                            "pnl_study": parts[3],
                            "pnl_val": parts[4],
                            "trades_study": parts[5],
                            "trades_val": parts[6],
                            "stats": parts[7]
                        })
                    # Formato v2 (7 columnas): | Ciclo | Mes Estudio | Mes Validación | PnL Estudio | PnL Validación | Trades | Empresas (+/-) |
                    elif len(parts) == 7:
                        data.append({
                            "cycle": parts[0],
                            "month": parts[1],
                            "eval_month": parts[2],
                            "pnl_study": parts[3],
                            "pnl_val": parts[4],
                            "trades_study": "-",
                            "trades_val": parts[5],
                            "stats": parts[6]
                        })
                    # Formato v1 (6 columnas): | Ciclo | Mes Aprendido | PnL ESTUDIO | PnL VALIDACIÓN | Trades | Empresas (+/-) |
                    elif len(parts) == 6:
                        data.append({
                            "cycle": parts[0],
                            "month": parts[1],
                            "eval_month": "-",
                            "pnl_study": parts[2],
                            "pnl_val": parts[3],
                            "trades_study": "-",
                            "trades_val": parts[4],
                            "stats": parts[5]
                        })
        return data[::-1]
    except Exception as e:
        print(f"Error reading history: {e}")
        return []

def get_metrics():
    # Leer logs principales
    lines = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()[-5000:]
        except: pass
        
    # Leer logs detallados
    sim_lines = []
    if os.path.exists(SIM_DETAILS_LOG):
        try:
            with open(SIM_DETAILS_LOG, "r") as f:
                sim_lines = f.readlines()[-20000:]
        except: pass

    # 0. Valores por defecto
    current_asset = "..."
    progress = 0
    cycle = "1"
    display_cycle = "1"
    is_transitioning = False
    phase = "ESTUDIO"
    phase_icon = "📚"
    sim_dates = "2024-01-01 a 2024-01-31"
    target_fleet = []
    
    # 1. Analizar LOG PRINCIPAL para estado de misión y símbolos
    symbol_last_event = {}
    
    for line in lines:
        match_cycle = re.search(r"CICLO DE APRENDIZAJE (\d+)", line)
        if match_cycle: 
            cycle = match_cycle.group(1)
            display_cycle = cycle
        
        match_dates = re.search(r"Fechas: ([\d-]+ a [\d-]+)", line)
        if match_dates: 
            sim_dates = match_dates.group(1)
        
        match_alt_dates = re.search(r"HAPI_SIM_START_DATE=([\d-]+)", line)
        if match_alt_dates:
            sim_dates = f"{match_alt_dates.group(1)} a ..." 

        for m_start in re.finditer(r"▶️ \[(\d+)/(\d+)\] (\w+)", line):
            symbol_last_event[m_start.group(3)] = ('START', int(m_start.group(1)))
        
        for m_done in re.finditer(r"[✅❌] \[\d+/\d+\] (\w+)", line):
            symbol_last_event[m_done.group(1)] = ('DONE', 0)
        
        if "📚 ENTRENANDO" in line:
            phase = "ESTUDIO"
            phase_icon = "📚"
        elif "🧪 EXAMEN" in line or "🎓 EXAMEN" in line:
            phase = "VALIDACIÓN"
            phase_icon = "🎓"

    alpha_blocks = sum(1 for l in sim_lines if "🚫 [AlphaOptimizer]" in l)
    total_scans = sum(1 for l in sim_lines if "🎯 GATILLO INTELIGENTE" in l)

    progress = max((idx for _, (_, idx) in symbol_last_event.items()), default=0)

    # 2. Procesar Resultados JSON
    home_runs = []
    win_rate_cycle = 0.0
    completed_count = 0
    total_pnl_global = 0.0
    total_pnl_cycle = 0.0
    current_session_results = []
    target_fleet = []

    # Detectar flota objetivo dinámicamente
    for config_path in ["simulation_config.json", "assets.json"]:
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    assets_data = json.load(f)
                    if isinstance(assets_data, dict) and "symbols" in assets_data:
                        target_fleet = assets_data["symbols"]
                    elif isinstance(assets_data, list):
                        target_fleet = assets_data
                    elif isinstance(assets_data, dict):
                        target_fleet = list(assets_data.keys())
                    if target_fleet: break # Si encontramos algo, paramos
            except Exception as e:
                print(f"Error loading assets from {config_path}: {e}")
    
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)
                total_pnl_global = sum(float(r.get('pnl', 0)) for r in all_results)
                
                # Priorizar el ciclo detectado en el log para filtrar resultados
                if display_cycle:
                    cycle = display_cycle
                elif all_results:
                    cycle = str(max([int(r.get("session_num", 1)) for r in all_results]))
                
                current_session_results = [r for r in all_results if str(r.get('session_num')) == cycle]

                # Extraer la fecha de inicio actual de sim_dates (ej: "2026-03-01")
                current_start_date = sim_dates.split(" ")[0] if sim_dates else ""

                # Obtener TODOS los resultados del ciclo (Estudio y Validación)
                current_session_results = [
                    r for r in all_results 
                    if str(r.get("session_num")) == str(cycle)
                ]
                
                # Para los totales (PnL, WinRate), usamos SÓLO la fase actual
                current_phase_results = [
                    r for r in current_session_results
                    if current_start_date in str(r.get("sim_start_date", ""))
                ]
                
                total_pnl_cycle = sum(float(r.get('pnl', 0)) for r in current_phase_results)
                completed_count = len({r['symbol'] for r in current_phase_results})
                
                if current_phase_results:
                    wins = sum(1 for r in current_phase_results if float(r.get('pnl', 0)) > 0)
                    win_rate_cycle = (wins / len(current_phase_results)) * 100
        except Exception as e:
            print(f"Error processing results: {e}")

    # 3. Detectar estado de transición y pulso
    active_symbols = set()
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                log_lines = f.readlines()
                for line in log_lines[-20:]:
                    if "REFLEXIONANDO" in line or "🧠 LA IA ESTÁ REFLEXIONANDO" in line:
                        is_transitioning = True
                        break
                
                for line in reversed(log_lines[-10000:]):
                    if "INICIANDO CICLO DE APRENDIZAJE" in line or "CICLO DE APRENDIZAJE" in line:
                        # Limpiar caracteres especiales como 🌀
                        clean_line = line.replace("🌀", "").strip()
                        display_cycle = clean_line.split("APRENDIZAJE")[-1].strip()
                        break
                
                # Fallback: Si no se encuentra en el log, usar la config
                if not display_cycle and os.path.exists("simulation_config.json"):
                    try:
                        with open("simulation_config.json", "r") as f:
                            cfg = json.load(f)
                            display_cycle = str(cfg.get("initial_cycle", "5"))
                    except: pass

                starts, completes = set(), set()
                # Radar de Alta Sensibilidad: Captura el símbolo justo después de los corchetes [x/167]
                sym_pattern = re.compile(r"\[\d+/\d+\]\s+([A-Z\.\-]+)")
                
                for line in log_lines[-20000:]:
                    match = sym_pattern.search(line)
                    if match:
                        s = match.group(1)
                        if s in target_fleet:
                            if "▶️" in line: starts.add(s)
                            if "✅" in line or "❌" in line or "TIMEOUT" in line: completes.add(s)
                
                active_symbols.update(starts - completes)
    except: pass

    # 4. Construir Radar de Resultados DINÁMICO (Separado por estados)
    active_fleet = []
    completed_fleet = []
    
    for sym in target_fleet:
        sym_res = [r for r in current_session_results if r.get('symbol') == sym]
        has_current = len(sym_res) > 0
        
        if has_current:
            sym_res.sort(key=lambda x: str(x.get('sim_start_date', '')))
            sub_results = []
            for r in sym_res:
                # Priorizar el campo 'stage' del JSON para evitar confusiones
                if r.get("stage") == "VALIDATING":
                    fase = "VALIDACIÓN"
                elif r.get("stage") == "TRAINING":
                    fase = "ESTUDIO"
                else:
                    # Fallback por fechas si no hay stage
                    fase = "VALIDACIÓN" if (current_start_date in str(r.get('sim_start_date', '')) and phase == "VALIDACIÓN") else "ESTUDIO"
                sub_results.append({
                    "phase": fase,
                    "pnl": f"${float(r.get('pnl',0)):,.2f}",
                    "trades": r.get('total_trades', 0),
                    "win_rate": f"{float(r.get('win_rate',0)):.1f}%",
                    "fees": f"${float(r.get('total_fees',0)):,.2f}",
                    "raw_pnl": float(r.get('pnl',0))
                })
            
            main_pnl = sub_results[-1]['raw_pnl'] if sub_results else 0
            
            completed_fleet.append({
                "symbol": sym, 
                "sub_results": sub_results,
                "main_pnl": main_pnl
            })
        elif sym in active_symbols:
            active_fleet.append({
                "symbol": sym, "pnl": "OPERANDO...", "trades": "...", "win_rate": "...", "status": "EJECUTANDO", "fees": "..."
            })
    
    # Ordenar por PnL de la última fase evaluada para ver los mejores arriba
    completed_fleet.sort(key=lambda x: x['main_pnl'], reverse=True)

    total_count = len(target_fleet) if target_fleet else 1
    remaining_count = max(0, total_count - completed_count)
    current_asset_text = ", ".join(list(active_symbols)) if active_symbols else "Sincronizando flota..."

    return {
        "cycle": display_cycle,
        "is_transitioning": is_transitioning,
        "total_sessions": len(get_history()),
        "completed_count": completed_count,
        "remaining_count": remaining_count,
        "phase": phase,
        "phase_icon": phase_icon,
        "sim_dates": sim_dates,
        "current_asset": current_asset_text,
        "total_pnl": round(total_pnl_cycle, 2),
        "total_pnl_global": round(total_pnl_global, 2),
        "total_fees": round(sum(float(r.get('total_fees',0)) for r in current_session_results), 2),
        "win_rate_cycle": round(win_rate_cycle, 1),
        "progress": completed_count,
        "progress_pct": round((completed_count / total_count) * 100, 1) if total_count > 0 else 0,
        "total_count": total_count,
        "alpha_blocks": alpha_blocks,
        "selectivity": round((alpha_blocks / total_scans * 100), 1) if total_scans > 0 else 0,
        "active_fleet": active_fleet,
        "completed_fleet": completed_fleet,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "history": get_history()
    }

@app.route('/algoritmo')
def algoritmo():
    return render_template_string(ALGO_TEMPLATE)

@app.route('/proyeccion')
def proyeccion():
    return render_template_string(PROY_TEMPLATE)

PROY_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Plan de Vida Realista | HAPI 2026</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg: #05070a;
            --accent: #00f2ff;
            --neon-green: #39ff14;
            --neon-red: #ff3131;
            --text: #e6edf3;
            --card-bg: rgba(13, 17, 23, 0.95);
        }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            margin: 0;
            background-image: radial-gradient(circle at 50% 0%, #0a1a2a 0%, #05070a 100%);
        }
        .container { max-width: 1100px; margin: 0 auto; padding: 40px 20px; }
        h1 { font-family: 'Orbitron', sans-serif; text-align: center; color: var(--accent); font-size: 2.2rem; margin-bottom: 40px; }
        
        .setup-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; }
        .control-panel { background: var(--card-bg); border: 1px solid var(--accent); border-radius: 20px; padding: 25px; }
        .card { background: var(--card-bg); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 25px; margin-bottom: 20px; }
        
        label { display: block; margin-bottom: 10px; font-size: 0.8rem; color: var(--accent); text-transform: uppercase; }
        input[type="range"] { width: 100%; margin-bottom: 20px; accent-color: var(--accent); }
        .input-val { font-family: 'Orbitron'; color: white; font-size: 1.2rem; float: right; }

        .verdict-box { text-align: center; padding: 20px; border-radius: 15px; margin-top: 20px; font-weight: bold; }
        .sustain-ok { background: rgba(57, 255, 20, 0.1); border: 1px solid var(--neon-green); color: var(--neon-green); }
        .sustain-warn { background: rgba(255, 242, 0, 0.1); border: 1px solid #f2ff00; color: #f2ff00; }
        .sustain-error { background: rgba(255, 49, 49, 0.1); border: 1px solid var(--neon-red); color: var(--neon-red); }

        .stat-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 20px; }
        .stat-card { background: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; text-align: center; }
        .stat-num { display: block; font-family: 'Orbitron'; font-size: 1.2rem; margin-top: 5px; }

        .friction-breakdown { font-size: 0.8rem; margin-top: 20px; color: #8b949e; }
        .friction-item { display: flex; justify-content: space-between; margin-bottom: 5px; }
        .highlight { color: var(--accent); }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 PLAN DE VIDA: VIVIR DE HAPI</h1>
        
        <div class="setup-grid">
            <div class="control-panel">
                <h2 style="font-family:'Orbitron'; font-size:1rem; margin-top:0;">⚙️ CONFIGURACIÓN</h2>
                
                <label>Capital Inicial <span class="input-val" id="cap-val">$10,000</span></label>
                <input type="range" id="capital" min="5000" max="200000" step="5000" value="10000">

                <label>Gasto Mensual (Vida) <span class="input-val" id="cost-val">$1,000</span></label>
                <input type="range" id="expenses" min="0" max="10000" step="100" value="1000">

                <label>Fricción (Broker + Slippage) <span class="input-val" id="friction-val">60%</span></label>
                <input type="range" id="friction" min="10" max="90" step="5" value="60">

                <label>ROI Bruto Esperado <span class="input-val" id="roi-val">10%</span></label>
                <input type="range" id="roi" min="3" max="25" step="1" value="10">

                <div id="verdict" class="verdict-box">Calculando...</div>

                <div class="friction-breakdown">
                    <div class="friction-item"><span>Ganancia Bruta:</span> <span class="highlight" id="bruta-val">$0</span></div>
                    <div class="friction-item"><span>Comisiones + Desliz:</span> <span style="color:var(--neon-red)" id="fric-val">-$0</span></div>
                    <div class="friction-item"><span>Impuestos (Taxes 20%):</span> <span style="color:var(--neon-red)" id="tax-val">-$0</span></div>
                    <div class="friction-item" style="border-top:1px solid #333; padding-top:5px; font-weight:bold;">
                        <span>Ganancia NETA:</span> <span style="color:var(--neon-green)" id="neta-val">$0</span>
                    </div>
                </div>
            </div>

            <div class="main-display">
                <div class="card">
                    <h2 style="font-family:'Orbitron'; font-size:1rem; color:var(--accent); margin-top:0;">📈 PROYECCIÓN DE SUPERVIVENCIA (1 AÑO)</h2>
                    <canvas id="lifeChart" height="150"></canvas>
                </div>

                <div class="stat-grid">
                    <div class="stat-card">
                        <label>Retirado para Vivir</label>
                        <span class="stat-num" id="total-out" style="color:var(--accent)">$0</span>
                    </div>
                    <div class="stat-card">
                        <label>Capital Final</label>
                        <span class="stat-num" id="total-final" style="color:var(--neon-green)">$0</span>
                    </div>
                    <div class="stat-card">
                        <label>Estado de Cuenta</label>
                        <span class="stat-num" id="net-growth" style="color:white">0%</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="card" style="margin-top:30px; border-color: rgba(57, 255, 20, 0.3);">
            <h2 style="font-family:'Orbitron'; font-size:1rem; color:var(--neon-green)">💡 ANÁLISIS DE COMPORTAMIENTO REAL</h2>
            <p id="advice" style="font-size:0.9rem;">Analizando tu situación...</p>
        </div>
    </div>

    <script>
        let chart;
        const ctx = document.getElementById('lifeChart').getContext('2d');

        function updateSim() {
            const initialCap = parseInt(document.getElementById('capital').value);
            const monthlyExp = parseInt(document.getElementById('expenses').value);
            const frictionPct = parseInt(document.getElementById('friction').value) / 100;
            const grossRoi = parseInt(document.getElementById('roi').value) / 100;
            
            document.getElementById('cap-val').innerText = '$' + initialCap.toLocaleString();
            document.getElementById('cost-val').innerText = '$' + monthlyExp.toLocaleString();
            document.getElementById('friction-val').innerText = (frictionPct * 100) + '%';
            document.getElementById('roi-val').innerText = (grossRoi * 100) + '%';

            let currentCap = initialCap;
            const labels = ['Inicio'];
            const capHistory = [initialCap];
            let totalWithdrawn = 0;

            // Stats de un mes ejemplo (el primero)
            const bruteProfit = initialCap * grossRoi;
            const frictionCost = bruteProfit * frictionPct;
            const taxCost = (bruteProfit - frictionCost) * 0.20;
            const netProfit = bruteProfit - frictionCost - taxCost;

            document.getElementById('bruta-val').innerText = '$' + Math.round(bruteProfit).toLocaleString();
            document.getElementById('fric-val').innerText = '-$' + Math.round(frictionCost).toLocaleString();
            document.getElementById('tax-val').innerText = '-$' + Math.round(taxCost).toLocaleString();
            document.getElementById('neta-val').innerText = '$' + Math.round(netProfit).toLocaleString();

            for(let m=1; m<=12; m++) {
                let mProfit = currentCap * grossRoi;
                let mFriction = mProfit * frictionPct;
                let mTax = (mProfit - mFriction) * 0.20;
                let mNet = mProfit - mFriction - mTax;
                
                currentCap = (currentCap + mNet) - monthlyExp;
                
                labels.push('Mes ' + m);
                capHistory.push(Math.round(currentCap));
                totalWithdrawn += monthlyExp;
                
                if(currentCap < 0) {
                    for(let j=m+1; j<=12; j++) { labels.push('Mes ' + j); capHistory.push(0); }
                    break;
                }
            }

            document.getElementById('total-out').innerText = '$' + totalWithdrawn.toLocaleString();
            document.getElementById('total-final').innerText = '$' + Math.max(0, Math.round(currentCap)).toLocaleString();
            const growth = ((currentCap - initialCap) / initialCap * 100).toFixed(1);
            document.getElementById('net-growth').innerText = growth + '%';

            const verdict = document.getElementById('verdict');
            const advice = document.getElementById('advice');
            if(currentCap > initialCap) {
                verdict.className = 'verdict-box sustain-ok';
                verdict.innerText = '✅ LIBERTAD FINANCIERA';
                advice.innerText = `Basado en tu comportamiento real: Con una fricción del ${(frictionPct*100)}%, tu estrategia es ganadora. Puedes vivir con $${monthlyExp.toLocaleString()} y tu capital seguirá creciendo. El bot está venciendo al mercado.`;
            } else if(currentCap > 0) {
                verdict.className = 'verdict-box sustain-warn';
                verdict.innerText = '⚠️ SUPERVIVENCIA FRÁGIL';
                advice.innerText = `Atención: Estás en el "punto de equilibrio". Tu capital se está erosionando lentamente. Para ser sostenible, necesitas o subir el ROI un poco más o reducir la fricción operando activos con menos slippage.`;
            } else {
                verdict.className = 'verdict-box sustain-error';
                verdict.innerText = '❌ QUIEBRA PROYECTADA';
                advice.innerText = `Alerta Crítica: Con este nivel de fricción y gastos, tu cuenta morirá en pocos meses. Necesitas al menos $${Math.round(monthlyExp / (grossRoi * (1-frictionPct) * 0.8)).toLocaleString()} de capital para que este estilo de vida sea posible.`;
            }

            if(chart) chart.destroy();
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Capital Real',
                        data: capHistory,
                        borderColor: '#00f2ff',
                        backgroundColor: 'rgba(0, 242, 255, 0.1)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#8b949e' } },
                        x: { grid: { display: false }, ticks: { color: '#8b949e' } }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        document.getElementById('capital').addEventListener('input', updateSim);
        document.getElementById('expenses').addEventListener('input', updateSim);
        document.getElementById('friction').addEventListener('input', updateSim);
        document.getElementById('roi').addEventListener('input', updateSim);
        updateSim();
    </script>
</body>
</html>
"""

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
        .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 2000; display: flex; align-items: center; justify-content: center; backdrop-filter: blur(8px); }
        .thinking-box { text-align: center; max-width: 500px; padding: 40px; }
        .brain-icon-pulse { font-size: 4rem; animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 0.5; } 50% { opacity: 1; transform: scale(1.1); } 100% { opacity: 0.5; } }
        .loader-bar { width: 100%; height: 6px; background: #333; border-radius: 3px; margin: 20px 0; overflow: hidden; }
        .loader-progress { width: 30%; height: 100%; background: var(--neon-cyan); animation: slide 1.5s infinite linear; }
        @keyframes slide { from { margin-left: -30%; } to { margin-left: 100%; } }
        .modal { display: none; position: fixed; z-index: 3000; left: 0; top: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.8); backdrop-filter: blur(5px); }
        .modal-content { background-color: var(--card-bg); margin: 5% auto; padding: 25px; border: 1px solid var(--neon-cyan); width: 80%; max-width: 900px; border-radius: 15px; box-shadow: 0 0 20px rgba(0,242,255,0.2); }
        .modal-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 15px; margin-bottom: 20px; }
        .close-btn { color: var(--text-gray); font-size: 28px; font-weight: bold; cursor: pointer; }
        .close-btn:hover { color: white; }
        body { background-color: var(--bg-dark); color: white; font-family: 'Inter', sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #1f2937; padding-bottom: 20px; margin-bottom: 30px; }
        h1 { font-family: 'Orbitron', sans-serif; color: var(--neon-cyan); margin: 0; font-size: 1.5rem; }
        .status-badge { background: rgba(57, 255, 20, 0.1); color: var(--neon-green); padding: 5px 15px; border-radius: 20px; border: 1px solid var(--neon-green); font-size: 0.8rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: var(--card-bg); border-radius: 12px; padding: 20px; border: 1px solid #30363d; }
        .card-title { color: var(--text-gray); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 10px; }
        .card-value { font-size: 2rem; font-weight: bold; font-family: 'Orbitron', sans-serif; }
        .progress-container { width: 100%; background: #21262d; height: 10px; border-radius: 5px; margin-top: 15px; overflow: hidden; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #00f2ff, #39ff14); width: 0%; transition: width 1s; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { text-align: left; color: var(--text-gray); font-size: 0.8rem; padding: 10px; border-bottom: 1px solid #30363d; }
        td { padding: 15px 10px; border-bottom: 1px solid #30363d; }
        .capital-banner { display: flex; align-items: center; gap: 30px; background: rgba(0,242,255,0.06); border-radius: 50px; padding: 10px 28px; margin-bottom: 25px; }
        .cb-value { color: var(--neon-cyan); font-weight: bold; font-family: 'Orbitron', sans-serif; }
        .algo-btn { 
            background: rgba(0, 242, 255, 0.1); 
            border: 1px solid var(--neon-cyan); 
            color: var(--neon-cyan); 
            padding: 8px 15px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-family: 'Orbitron'; 
            font-size: 0.8rem; 
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .algo-btn:hover { 
            background: var(--neon-cyan); 
            color: black; 
            box-shadow: 0 0 15px var(--neon-cyan);
        }
        .proy-btn { 
            background: rgba(57, 255, 20, 0.1); 
            border: 1px solid var(--neon-green); 
            color: var(--neon-green); 
            padding: 8px 15px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-family: 'Orbitron'; 
            font-size: 0.8rem; 
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .proy-btn:hover { 
            background: var(--neon-green); 
            color: black; 
            box-shadow: 0 0 15px var(--neon-green);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Alpha Mission Control</h1>
            <div style="display: flex; align-items: center; gap: 15px;">
                <a href="/proyeccion" target="_blank" class="proy-btn">🚀 FUTURO</a>
                <a href="/algoritmo" target="_blank" class="algo-btn">🧠 ALGORITMO</a>
                <button onclick="toggleModal(true)" style="background:rgba(0,242,255,0.1); border:1px solid var(--neon-cyan); color:var(--neon-cyan); padding:8px 15px; border-radius:8px; cursor:pointer; font-family:'Orbitron'; font-size:0.8rem;">📜 HISTORIAL</button>
                <div class="status-badge">● System Online</div>
            </div>
        </header>

        <div id="thinking-modal" class="modal-overlay" style="display: none;">
            <div class="card modal-content thinking-box">
                <div class="brain-icon-pulse">🧠</div>
                <h2>Reflexión de la IA</h2>
                <p id="thinking-msg">Consolidando patrones del Ciclo <span id="current-cycle-modal">?</span>...</p>
                <div class="loader-bar"><div class="loader-progress"></div></div>
                <small>Ajustando redes neuronales para el próximo examen</small>
            </div>
        </div>

        <div class="capital-banner">
            <div class="cb-item">ROI del ciclo: <span class="cb-value" id="roi-value">0.00%</span></div>
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

        <!-- 🛸 NUEVA SECCIÓN: FLOTA EN OPERACIÓN -->
        <div id="active-fleet-card" class="card" style="margin-bottom: 20px; border: 1px solid var(--neon-cyan); display: none;">
            <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; margin-top: 0; color: var(--neon-cyan);">🛸 FLOTA EN OPERACIÓN (EN VIVO)</h2>
            <div style="max-height: 300px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Símbolo</th>
                            <th>Estado</th>
                            <th>Acción</th>
                        </tr>
                    </thead>
                    <tbody id="active-results-table">
                        <!-- Dinámico -->
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2 style="font-family: 'Orbitron', sans-serif; font-size: 1.1rem; margin-top: 0;">📊 HISTORIAL DEL CICLO (COMPLETADOS)</h2>
            <div style="max-height: 400px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Símbolo</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Fase</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">PnL Neto</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Comisiones</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Trades</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Win Rate</th>
                            <th style="position: sticky; top: 0; background: var(--card-bg); z-index: 1;">Estado</th>
                        </tr>
                    </thead>
                    <tbody id="completed-results-table">
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
                if(progText) progText.innerText = 'Avance global: ' + data.progress + ' / ' + data.total_count + ' (' + data.progress_pct + '%)';
                
                document.getElementById('sim-dates-footer').innerText = data.sim_dates;
                // Gestionar Modal de Reflexión
                const modal = document.getElementById('thinking-modal');
                if(data.is_transitioning) {
                    modal.style.display = 'flex';
                    document.getElementById('current-cycle-modal').innerText = data.cycle;
                } else {
                    modal.style.display = 'none';
                }

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
                            <td style="color:var(--neon-yellow); font-size:0.85rem;">${h.eval_month || '-'}</td>
                            <td style="opacity:0.8">${h.pnl_study}</td>
                            <td style="font-family:'Orbitron',sans-serif">${h.pnl_val}</td>
                            <td style="text-align:center">${h.trades_study}</td>
                            <td style="text-align:center">${h.trades_val}</td>
                            <td>${h.stats}</td>
                        </tr>
                    `).join('');
                }

                // ROI del ciclo
                const roi = ((data.total_pnl / (data.total_count * 10000)) * 100).toFixed(3);
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
                
                const remainingText = document.getElementById('remaining-text');
                if(remainingText) remainingText.innerText = ' de ' + data.total_count + ' — completadas:';
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

                // 🛸 Actualizar Flota Activa
                const activeCard = document.getElementById('active-fleet-card');
                const activeTbody = document.getElementById('active-results-table');
                if(data.active_fleet && data.active_fleet.length > 0) {
                    activeCard.style.display = 'block';
                    activeTbody.innerHTML = data.active_fleet.map(r => `
                        <tr>
                            <td style="font-weight:bold; color:var(--text-main)">${r.symbol}</td>
                            <td><span class="status-badge" style="border-color: var(--neon-cyan); color: var(--neon-cyan); box-shadow: 0 0 5px var(--neon-cyan);">OPERANDO... ⚙️</span></td>
                            <td>
                                <button onclick="fetchSymbolUpdate('${r.symbol}')" id="btn-update-${r.symbol}"
                                        style="background:rgba(0,242,255,0.1); border:1px solid var(--neon-cyan); border-radius:6px; color:var(--neon-cyan); cursor:pointer; font-size:0.75rem; padding:4px 10px; font-family:'Orbitron';">
                                    📊 Ver Detalle
                                </button>
                            </td>
                        </tr>
                    `).join('');
                } else {
                    activeCard.style.display = 'none';
                }

                // 📊 Actualizar Historial de Resultados
                const completedTbody = document.getElementById('completed-results-table');
                if (completedTbody && data.completed_fleet) {
                    completedTbody.innerHTML = data.completed_fleet.map(r => {
                        const hasCalendar = data.calendar_symbols && data.calendar_symbols.includes(r.symbol);
                        const calBtn = hasCalendar ? `<button class="cal-btn" onclick="showTradeCalendar('${r.symbol}')" title="Ver Auditoría">📅</button>` : '';
                        
                        let rowsHTML = '';
                        let isFirst = true;
                        let rLen = r.sub_results.length || 1;
                        
                        for (let sub of r.sub_results) {
                            let pnlColor = sub.pnl.includes('-') ? 'var(--neon-red)' : 'var(--neon-green)';
                            let phaseBadge = sub.phase === 'ESTUDIO' 
                                ? `<span style="color:var(--neon-cyan); border:1px solid var(--neon-cyan); padding:2px 6px; border-radius:4px; font-size:0.7rem;">📚 ESTUDIO</span>`
                                : `<span style="color:var(--neon-purple); border:1px solid var(--neon-purple); padding:2px 6px; border-radius:4px; font-size:0.7rem;">🎓 VALIDACIÓN</span>`;
                            
                            let symCol = isFirst ? `<td rowspan="${rLen}" style="font-weight:bold; color:var(--accent); border-right: 1px dashed rgba(0, 242, 255, 0.2); font-size: 1.1rem; vertical-align: middle; background: rgba(0, 242, 255, 0.02);">${r.symbol} ${calBtn}</td>` : '';
                            let statusCol = isFirst ? `<td rowspan="${rLen}" style="border-left: 1px dashed rgba(255,255,255,0.1); vertical-align: middle;"><span class="status-badge" style="border-color: var(--neon-green); color: var(--neon-green); font-weight: bold;">COMPLETADO</span></td>` : '';
                            
                            // Estilo de separación: línea sólida entre empresas, línea suave entre sub-filas
                            let trStyle = isFirst 
                                ? 'border-top: 2px solid rgba(0, 242, 255, 0.1);' 
                                : 'border-top: 1px solid rgba(255, 255, 255, 0.05); background: rgba(255,255,255,0.02);';

                            rowsHTML += `<tr style="${trStyle}">
                                ${symCol}
                                <td style="padding: 12px 8px;">${phaseBadge}</td>
                                <td style="color: ${pnlColor}; font-weight: bold;">${sub.pnl}</td>
                                <td style="color: var(--neon-red); font-size: 0.85rem; opacity: 0.8;">${sub.fees || '$0.00'}</td>
                                <td style="font-family: monospace;">${sub.trades}</td>
                                <td style="font-family: monospace;">${sub.win_rate}</td>
                                ${statusCol}
                            </tr>`;
                            isFirst = false;
                        }
                        return rowsHTML;
                    }).join('');
                }
            } catch (e) { console.error("Error updating dashboard:", e); }
        }

        // Función para actualizar una fila específica sin salir de la página
        async function fetchSymbolUpdate(symbol) {
            const btn = document.getElementById('btn-update-' + symbol);
            if (btn) { 
                btn.innerHTML = '<span style="color:var(--neon-green)">✔️ OK</span>';
                setTimeout(() => { btn.innerHTML = '📊 Ver'; }, 2000);
            }
            try {
                // Normalizar símbolo para la URL (p.ej. BRK.B -> BRK-B)
                const safeSymbol = symbol.replace('.', '-');
                const res = await fetch('/api/symbol/' + safeSymbol);
                const d = await res.json();
                
                // Buscar la fila y actualizar celdas
                const rows = document.querySelectorAll('#results-table tbody tr');
                rows.forEach(row => {
                    // El primer hijo contiene el símbolo y a veces un botón de calendario
                    if (row.cells[0].innerText.includes(symbol)) {
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
            } catch(e) { 
                console.error("Error updating symbol row:", e);
                if (btn) btn.innerHTML = '<span style="color:var(--neon-red)">❌ ERROR</span>';
            }
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

    <!-- Modal de Historial -->
    <div id="historyModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 style="font-family: 'Orbitron', sans-serif; color: var(--neon-cyan); margin:0;">Historial de Ciclos Time Travel</h2>
                <span class="close-btn" onclick="toggleModal(false)">&times;</span>
            </div>
            <div style="max-height: 500px; overflow-y: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="text-align: left; color: var(--text-gray); font-size: 0.8rem; border-bottom: 1px solid #30363d;">
                            <th style="padding: 10px;">Ciclo</th>
                            <th style="padding: 10px;">Mes Estudio</th>
                            <th style="padding: 10px;">Mes Validación</th>
                            <th style="padding: 10px;">PnL Estudio</th>
                            <th style="padding: 10px;">PnL Validación</th>
                            <th style="padding: 10px; text-align:center">Trades Est.</th>
                            <th style="padding: 10px; text-align:center">Trades Valid.</th>
                            <th style="padding: 10px;">Empresas (+/-)</th>
                        </tr>
                    </thead>
                    <tbody id="history-table-body">
                        <!-- Se llena vía JS -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Modal de Calendario -->
    <div id="calendarModal" class="modal">
        <div class="modal-content" style="max-width: 600px;">
            <div class="modal-header">
                <h2 id="cal-symbol-title" style="margin:0;">Auditoría de Trades</h2>
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
    try:
        current_cycle = 1
        report_file = "scratch/TIME_TRAVEL_RESULTS.md"
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                lines = f.readlines()
            for line in reversed(lines):
                if "|" in line and "Ciclo" not in line and "---" not in line:
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    if parts and parts[0].isdigit():
                        current_cycle = int(parts[0])
                        break
    except: pass

    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r") as f:
                all_results = json.load(f)

            match = next((r for r in reversed(all_results) 
                         if r.get('symbol') == symbol 
                         and r.get('session_num') == current_cycle), None)
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
