import os
import json
import time
import random
import subprocess
from pathlib import Path
import calendar
import datetime
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Lock global para evitar corrupción de archivos y logs al escribir en paralelo
file_lock = threading.Lock()
print_lock = threading.Lock()

def generate_random_month(year_choices=[2024, 2025]):
    year = random.choice(year_choices)
    month = random.randint(1, 12)
    
    # Muro de Fuego: Solo permitimos meses de estudio hasta Octubre 2025
    if year == 2025 and month > 10:
        month = random.randint(1, 10)
        
    # Restringir a historia válida (Alpaca IEX data usualmente es buena desde mediados de 2024)
    if year == 2024 and month < 5:
        month = random.randint(5, 12) 
        
    _, last_day = calendar.monthrange(year, month)
    start_date = f"{year}-{month:02d}-01"
    end_date = f"{year}-{month:02d}-{last_day:02d}"
    return start_date, end_date

def run_single_simulation(args):
    i, sym, total, start_date, end_date, stage, env, base_dir, session_num = args
    
    docker_cmd = [
        "docker", "run", "--rm",
        "--network", "trading_bot_default",
        "-v", f"{base_dir}/data:/app/data",
        "-v", f"{base_dir}/logs:/app/logs",
        "-v", f"{base_dir}/shared:/app/shared",
        "-v", f"{base_dir}/simulation_mode:/app/simulation_mode",
        "-e", f"TARGET_SYMBOL={sym}",
        "-e", f"TRADING_MODE=SIMULATED",
        "-e", f"HAPI_IS_TEST_MODE=true",
        "-e", f"HAPI_SIM_START_DATE={start_date}",
        "-e", f"HAPI_SIM_END_DATE={end_date}",
        "-e", f"SIM_STAGE={stage}",
        "-e", f"SESSION_NUM={session_num}",
        "-e", f"FREEZE_LEARNING={env['FREEZE_LEARNING']}",
        "hapi-scalping-bot:latest",
        "python", "simulation_mode/main_sim.py"
    ]
    with print_lock:
        print(f"  ▶️ [{i+1}/{total}] {sym} en proceso...")
    # 📡 Sincronizar inicio con el log para el Dashboard (ANTES de arrancar)
    with file_lock:
        with open(os.path.join(base_dir, "scratch/simulation_details.log"), "a") as f_log:
            f_log.write(f"\n  ▶️ [{i+1}/{total}] {sym} en proceso...\n")
            
    with open(os.path.join(base_dir, "scratch/simulation_details.log"), "a") as f_log:
        try:
            proc = subprocess.run(
                docker_cmd, 
                stdout=f_log, stderr=f_log,
                timeout=900 # 🛡️ Timeout de 15 min para evitar bloqueos infinitos
            )
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            with print_lock:
                print(f"  ⚠️ TIMEOUT: {sym} excedió los 15 minutos. Saltando...")
            return_code = 1
        except Exception as e:
            with print_lock:
                print(f"  ❌ ERROR en {sym}: {e}")
            return_code = 1
    
    if return_code == 0:
        # Consolidar resultado JSON al terminar
        res_file = os.path.join(base_dir, f"data/results_{sym}.json")
        master_file = os.path.join(base_dir, "data/backtest_results.json")
        
        if os.path.exists(res_file):
            with file_lock:
                try:
                    time.sleep(0.5) # ⏳ Pausa para asegurar escritura completa de Docker
                    with open(res_file, "r") as f:
                        new_data = json.load(f)
                    
                    master_data = []
                    if os.path.exists(master_file):
                        with open(master_file, "r") as f:
                            master_data = json.load(f)
                    
                    # Evitar duplicados para ESTA sesión específica y ESTA fase (basado en el inicio de la simulación)
                    new_date = new_data[0].get("sim_start_date", "")[:10] if new_data else ""
                    master_data = [r for r in master_data if not (r.get("symbol") == sym and str(r.get("session_num")) == str(session_num) and r.get("sim_start_date", "").startswith(new_date))]
                    master_data.extend(new_data)
                    
                    with open(master_file, "w") as f:
                        json.dump(master_data, f, indent=2)
                    
                    # 📅 ACTUALIZACIÓN DEL CALENDARIO TÁCTICO
                    calendar_file = os.path.join(base_dir, "data/trade_calendar.json")
                    try:
                        cal_data = {}
                        if os.path.exists(calendar_file):
                            with open(calendar_file, "r") as f:
                                cal_data = json.load(f)
                        
                        # Extraer trades de la bitácora
                        trades = new_data[0].get("trade_log", [])
                        for t in trades:
                            # t format: {"t": exit_ts, "entry_t": entry_ts, "s": "SELL", "r": pnl, ...}
                            if t.get("s") == "SELL":
                                exit_ts = str(t.get("t"))
                                entry_ts = str(t.get("entry_t", exit_ts))
                                exit_date = exit_ts[:10]
                                entry_date = entry_ts[:10]
                                
                                # Calcular estilo (Mismo día vs Swing)
                                style = "INTRAD\u00cdA" if exit_date == entry_date else "SWING"
                                
                                if exit_date not in cal_data: cal_data[exit_date] = []
                                
                                cal_data[exit_date].append({
                                    "symbol": sym,
                                    "entry": entry_ts[11:19],
                                    "exit": exit_ts[11:19],
                                    "pnl": t.get("r", 0),
                                    "style": style,
                                    "reason": t.get("m", ""),
                                    "stage": stage
                                })
                        
                        with open(calendar_file, "w") as f:
                            json.dump(cal_data, f, indent=2)
                    except Exception as cal_err:
                        print(f"  ⚠️ Error actualizando calendario: {cal_err}")

                    os.remove(res_file) # Limpiar temporal
                except Exception as e:
                    print(f"  ⚠️ Error consolidando {sym}: {e}")

        return f"  ✅ [{i+1}/{total}] {sym} completado."
    else:
        return f"  ❌ [{i+1}/{total}] {sym} falló (Code {return_code})."

def run_simulation_batch(base_dir, symbols, start_date, end_date, stage, desc, session_num, max_workers=10):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⏳ {desc} | Fechas: {start_date} a {end_date}")
    
    env = {"FREEZE_LEARNING": "true" if stage == "VALIDATING" else "false"}
    total = len(symbols)
    
    print(f"🚀 Lanzando flota de {total} activos con {max_workers} bots en paralelo...")
    
    tasks = []
    for i, sym in enumerate(symbols):
        tasks.append((i, sym, total, start_date, end_date, stage, env, base_dir, session_num))
        
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            with print_lock:
                print(res)
                # 📡 Sincronizar con el log para que el Dashboard vea el progreso
                with open(os.path.join(base_dir, "scratch/simulation_details.log"), "a") as f_log:
                    f_log.write(f"\n{res}\n")

def evaluate_results(base_dir, eval_start_date, session_num=None):
    res_file = Path(os.path.join(base_dir, "data/backtest_results.json"))
    results = []
    if res_file.exists():
        try:
            with open(res_file, "r") as f:
                results = json.load(f)
        except:
            return 0.0, 0, 0, 0
            
    total_pnl = 0.0
    total_trades = 0
    win_symbols = 0
    loss_symbols = 0
    processed = set()
    
    # Filtrar por sesión Y fecha para no leer datos de ciclos anteriores
    for r in reversed(results):
        sym = r.get("symbol")
        if sym in processed: continue
        
        date_match = eval_start_date in str(r.get("sim_start_date", ""))
        # Si se pasa session_num, filtrar estrictamente por él
        session_match = (session_num is None) or (r.get("session_num") == session_num)
        
        if date_match and session_match:
            pnl = float(r.get("pnl", 0.0))
            total_pnl += pnl
            total_trades += int(r.get("total_trades", 0))
            if pnl > 0: win_symbols += 1
            elif pnl < 0: loss_symbols += 1
            processed.add(sym)
            
    return total_pnl, total_trades, win_symbols, loss_symbols

def get_last_cycle_num(report_file):
    if not report_file.exists():
        return 0
    try:
        with open(report_file, "r") as f:
            lines = f.readlines()
        
        last_cycle = 0
        for line in reversed(lines):
            if "|" in line and "Ciclo" not in line and "---" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts and parts[0].isdigit():
                    last_cycle = max(last_cycle, int(parts[0]))
        return last_cycle
    except:
        return 0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    assets_file = os.path.join(base_dir, 'assets.json')
    config_file = os.path.join(base_dir, 'simulation_config.json')
    report_file = Path(os.path.join(base_dir, "scratch/TIME_TRAVEL_RESULTS.md"))
    
    # Cargar configuración dinámica
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            symbols = config.get("symbols", ["AAPL"])
            cycle_num = config.get("initial_cycle", 1)
    except Exception as e:
        print(f"⚠️ Error cargando configuración: {e}. Usando valores por defecto.")
        symbols = ['AAPL', 'TSLA', 'NVDA']
        cycle_num = 1
    
    while True: # 🔄 Bucle Infinito de Aprendizaje
        print(f"\n\n==============================================")
        print(f"🌀 INICIANDO CICLO DE APRENDIZAJE {cycle_num}")
        print(f"==============================================")
        
        # 1. Determinar el mes de entrenamiento
        if cycle_num == 1:
            train_start, train_end = "2025-05-01", "2025-05-31" # Lo que pediste
        else:
            train_start, train_end = generate_random_month([2024, 2025]) # Meses aleatorios
            
        # 2. Fase de Entrenamiento (La IA APRENDE)
        run_simulation_batch(
            base_dir, symbols, train_start, train_end, "TRAINING", f"📚 ENTRENANDO EN PASADO ({train_start[:7]})", cycle_num
        )
        
        # 🧠 MOMENTO DE REFLEXIÓN (Consolidación de patrones)
        print(f"\n[{(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}] 🧠 LA IA ESTÁ REFLEXIONANDO... (Ajustando redes neuronales para el próximo examen)")
        
        # Calcular resultados del entrenamiento (Estudio)
        pnl_train, trades_train, wins_train, losses_train = evaluate_results(base_dir, train_start, cycle_num)
        
        # if trades_train == 0 and cycle_num > 1:
        #     print("⚠️ Alerta: No se detectaron trades en el entrenamiento. Reintentando con otro mes...")
        #     continue
        
        # 3. Fase de Validación (EXAMEN - MES ALEATORIO DENTRO DEL RANGO VÁLIDO)
        # Rango válido: Nov 2025 - Abr 2026 (6 meses posibles)
        import random
        valid_eval_months = [
            ("2025-11-01", "2025-11-30"),
            ("2025-12-01", "2025-12-31"),
            ("2026-01-01", "2026-01-31"),
            ("2026-02-01", "2026-02-28"),
            ("2026-03-01", "2026-03-31"),
            ("2026-04-01", "2026-04-30"),
        ]
        eval_start, eval_end = random.choice(valid_eval_months)

        run_simulation_batch(
            base_dir, symbols, eval_start, eval_end, "VALIDATING", f"🧪 EXAMEN EN EL PRESENTE ({eval_start[:7]})", cycle_num
        )
        
        # 🧠 MOMENTO DE REFLEXIÓN (Finalizando ciclo)
        print(f"\n[{(datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}] 🧠 LA IA ESTÁ REFLEXIONANDO... (Consolidando métricas finales)")
        
        # 4. Calcular resultados del examen (Validación)
        pnl_eval, trades_eval, wins_eval, losses_eval = evaluate_results(base_dir, eval_start, cycle_num)
        
        # 5. Guardar en reporte (con mes de validación explícito)
        report_file.parent.mkdir(exist_ok=True, parents=True)
        is_new = not report_file.exists()
        with open(report_file, "a") as f:
            if is_new:
                f.write("# 🚀 Resultados del Time Travel Trainer\n\n")
                f.write("| Ciclo | Mes Estudio | Mes Validación | PnL Estudio | PnL Validación | Trades Est. | Trades Valid. | Empresas (+/-) |\n")
                f.write("| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: |\n")
            
            s_train = "🟢" if pnl_train > 0 else "🔴"
            s_eval = "🟢" if pnl_eval > 0 else "🔴"
            f.write(f"| {cycle_num} | {train_start[:7]} | {eval_start[:7]} | {s_train} ${pnl_train:.2f} | {s_eval} ${pnl_eval:.2f} | {trades_train} | {trades_eval} | {wins_eval}/{losses_eval} |\n")
        
        print(f"\n🎉 CICLO {cycle_num} COMPLETADO.")
        print(f"   Resultado del Examen Maestro (Nov-Abr): ${pnl_eval:.2f} Neto.")
        print(f"   Revisa el progreso en: scratch/TIME_TRAVEL_RESULTS.md")
        # --- MOMENTO DE REFLEXIÓN ---
        print("\n🧠 LA IA ESTÁ REFLEXIONANDO: Consolidando aprendizaje y ajustando estrategia...")
        time.sleep(10) # Pausa de 10 segundos para visibilidad del usuario
        
        cycle_num += 1

if __name__ == '__main__':
    main()
