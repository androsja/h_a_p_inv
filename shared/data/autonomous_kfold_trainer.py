import os
import json
import time
import subprocess
from pathlib import Path

def clean_ai_memory(base_dir):
    """Borra la memoria de la IA para iniciar un ciclo de entrenamiento desde cero."""
    files_to_delete = [
        "data/ai_model.joblib",
        "data/neural_model.joblib",
        "data/ml_dataset.csv",
        "data/trade_journal.csv",
        "data/backtest_results.json"
    ]
    for f in files_to_delete:
        p = Path(os.path.join(base_dir, f))
        if p.exists():
            try: p.unlink()
            except: pass

def run_simulation_batch(base_dir, symbols, start_date, end_date, stage, desc):
    """Ejecuta la simulación para todos los símbolos y retorna True si termina bien."""
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Iniciando {desc} | {start_date} a {end_date} | Modo: {stage}")
    env = os.environ.copy()
    env["HAPI_SIM_START_DATE"] = start_date
    env["HAPI_SIM_END_DATE"] = end_date
    env["SIM_STAGE"] = stage
    
    total = len(symbols)
    for i, sym in enumerate(symbols):
        env["TARGET_SYMBOL"] = sym
        proc = subprocess.run(
            ["python3", "-m", "simulation_mode.main_sim"], 
            cwd=base_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if proc.returncode == 0:
            print(f"  [{i+1}/{total}] {sym} completado en {desc}.")
        else:
            print(f"  [{i+1}/{total}] {sym} falló con código {proc.returncode}")

def evaluate_results(base_dir, eval_start_date):
    """Lee los resultados del backtest y calcula el PnL total del periodo de evaluación."""
    res_file = Path(os.path.join(base_dir, "data/backtest_results.json"))
    if not res_file.exists(): return 0.0, 0, 0, 0
        
    with open(res_file, "r") as f:
        try: results = json.load(f)
        except: return 0.0, 0, 0, 0
            
    total_pnl = 0.0
    total_trades = 0
    win_symbols = 0
    loss_symbols = 0
    processed = set()
    
    for r in reversed(results):
        sym = r.get("symbol")
        if sym in processed: continue
            
        if eval_start_date in str(r.get("sim_start_date", "")):
            pnl = float(r.get("pnl", 0.0))
            total_pnl += pnl
            total_trades += int(r.get("total_trades", 0))
            if pnl > 0: win_symbols += 1
            elif pnl < 0: loss_symbols += 1
            processed.add(sym)
            
    return total_pnl, total_trades, win_symbols, loss_symbols

def update_markdown_report(report_file, cycle_num, train_start, train_end, eval_start, eval_end, pnl, trades, wins, losses):
    is_new = not report_file.exists()
    with open(report_file, "a") as f:
        if is_new:
            f.write("# 🧪 Validación Cruzada Autónoma (Walk-Forward Analysis)\n\n")
            f.write("Este documento registra los ciclos de entrenamiento aislados y su evaluación sobre el último mes, para encontrar qué condiciones de mercado generan un modelo más rentable.\n\n")
            f.write("| Ciclo | Entrenamiento (TRAINING) | Evaluación (EVALUATION) | PnL Neto | Trades | Ganadoras | Perdedoras |\n")
            f.write("| :---: | :--- | :--- | :--- | :---: | :---: | :---: |\n")
        
        status = "🟢" if pnl > 0 else "🔴"
        f.write(f"| {cycle_num} | {train_start} a {train_end} | {eval_start} a {eval_end} | **{status} ${pnl:.2f}** | {trades} | {wins} | {losses} |\n")

def main():
    base_dir = "/app"
    assets_file = os.path.join(base_dir, 'assets.json')
    report_file = Path(os.path.join(base_dir, "scratch/VALIDACION_CRUZADA.md"))
    
    with open(assets_file, 'r') as f:
        data = json.load(f)
    symbols = [a['symbol'] for a in data.get('assets', [])]
    
    # El mes fijo de validación donde la IA NO aprende, solo presenta el examen
    eval_start = "2026-03-29"
    eval_end   = "2026-04-28"
    
    # 5 Ciclos de Entrenamiento en meses anteriores
    cycles = [
        {"num": 1, "start": "2025-11-01", "end": "2025-11-30"},
        {"num": 2, "start": "2025-12-01", "end": "2025-12-31"},
        {"num": 3, "start": "2026-01-01", "end": "2026-01-31"},
        {"num": 4, "start": "2026-02-01", "end": "2026-02-28"},
        {"num": 5, "start": "2026-03-01", "end": "2026-03-28"}
    ]
    
    print("=========================================================")
    print(f"🤖 INICIANDO AGENDA AUTÓNOMA: {len(cycles)} CICLOS (WFA)")
    print(f"Total Símbolos: {len(symbols)}")
    print(f"Tiempo estimado: ~{len(cycles) * 16} horas.")
    print("=========================================================\n")
    
    for c in cycles:
        print(f"\n>>>>> INICIANDO CICLO {c['num']}/5 <<<<<")
        
        # 1. Borrar memoria para que el ciclo sea aislado (No arrastrar sesgos)
        clean_ai_memory(base_dir)
        
        # 2. Entrenar en el mes N
        run_simulation_batch(
            base_dir, symbols, c['start'], c['end'], "TRAINING", f"FASE 1: ENTRENAMIENTO (Ciclo {c['num']})"
        )
        
        # 3. Evaluar en el último mes (Examen Final)
        run_simulation_batch(
            base_dir, symbols, eval_start, eval_end, "EVALUATION", f"FASE 2: EXAMEN FINAL (Ciclo {c['num']})"
        )
        
        # 4. Leer resultados y guardar en reporte
        pnl, trades, wins, losses = evaluate_results(base_dir, eval_start)
        update_markdown_report(report_file, c['num'], c['start'], c['end'], eval_start, eval_end, pnl, trades, wins, losses)
        
        print(f"\n✅ CICLO {c['num']} TERMINADO. PnL del examen: ${pnl:.2f}")

    print("\n🎉 AGENDA AUTÓNOMA FINALIZADA AL 100%. Revisa scratch/VALIDACION_CRUZADA.md")

if __name__ == '__main__':
    main()
