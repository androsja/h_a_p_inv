import pandas as pd
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Configuración de rutas
DATA_DIR = Path("data")
JOURNAL_FILE = DATA_DIR / "trade_journal.csv"
MASTERY_FILE = DATA_DIR / "mastery_hub.json"

def get_sector_for(symbol):
    # Diccionario rápido de sectores para la restauración
    sectors = {
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
        "ABBV": "Healthcare", "AMGN": "Healthcare",
        "DIA": "Industrial", "BA": "Industrial",
        "QQQ": "Technology", "SPY": "Finance", "C": "Finance",
    }
    return sectors.get(symbol.upper(), "Other")

def restore():
    print(f"🚀 Iniciando restauración desde {JOURNAL_FILE}...")
    
    if not JOURNAL_FILE.exists():
        print("❌ Error: No se encontró el trade_journal.csv")
        return

    # Leer trades
    df = pd.read_csv(JOURNAL_FILE)
    print(f"📊 Leídos {len(df)} trades históricos.")

    # Agrupar por símbolo
    mastery_data = {}
    
    for symbol, group in df.groupby('symbol'):
        symbol = symbol.upper()
        total_trades = len(group)
        wins = len(group[group['is_win'] == 1])
        win_rate = (wins / total_trades) * 100
        
        net_pnl = group['gross_pnl'].sum()
        gross_profit = group[group['gross_pnl'] > 0]['gross_pnl'].sum()
        gross_loss = abs(group[group['gross_pnl'] < 0]['gross_pnl'].sum())
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0)
        
        # --- FÓRMULA DE MAESTRÍA (Sincronizada con MasteryManager.py) ---
        wr_score = min(win_rate, 100) * 0.40
        pf_score = min(profit_factor * 50, 100) * 0.40
        exp_score = min(total_trades, 100) * 0.20
        rank = round(wr_score + pf_score + exp_score, 1)
        
        # Estabilidad (Aproximación para restauración)
        stability = min(((profit_factor / 1.5) * 40) + (min(total_trades, 50)) + (rank / 10), 100)
        
        status = "LEARNING"
        if total_trades > 30:
            if rank > 75: status = "ELITE"
            elif rank > 50: status = "READY_FOR_LIVE"
            else: status = "NEEDS_IMPROVEMENT"
        elif rank > 50:
            status = "LEARNING" # Sigue en aprendizaje hasta 30 trades

        mastery_data[symbol] = {
            "symbol": symbol,
            "sector": get_sector_for(symbol),
            "rank": rank,
            "raw_rank": rank,
            "status": status,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_trades": total_trades,
            "net_pnl": round(net_pnl, 2),
            "stability": round(stability, 1),
            "confidence_score": rank,
            "risk_of_ruin": 0.0,
            "suggested_capital": 0,
            "audit_drawdown": 0.0,
            "drift_penalty": 0.0,
            "baseline_pf": 0.0,
            "baseline_wr": 0.0,
            "last_updated": datetime.now().isoformat()
        }

    # Guardar resultado
    with open(MASTERY_FILE, 'w', encoding='utf-8') as f:
        json.dump(mastery_data, f, indent=4)
    print(f"✅ Mastery Hub restaurado: {len(mastery_data)} símbolos.")

    # --- RECONSTRUCCIÓN DEL HISTORIAL DE SIMULACIONES (Hitos Docker) ---
    COMPLETED_LOG = Path("data/completed_simulations.json")
    completed_data = []
    if COMPLETED_LOG.exists():
        try:
            completed_data = json.loads(COMPLETED_LOG.read_text())
        except: pass
    
    symbols_done = {b.get("symbol") for b in completed_data}
    added = 0
    
    for symbol, stats in mastery_data.items():
        if symbol not in symbols_done and stats["total_trades"] > 0:
            # Sintetizar hito de SCOUTING para que el AutoTrainer pase a TRAINING
            entry = {
                "run_id": f"{symbol}_RESTORED_{int(time.time())}",
                "symbol": symbol,
                "sim_start_date": "2024-05-01",
                "sim_end_date": "2024-07-31",
                "launched_at": "Restauración",
                "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "pnl": stats["net_pnl"],
                "trades": stats["total_trades"],
                "win_rate": stats["win_rate"],
                "mode": "SIMULATED",
                "mastery_score": stats["rank"],
                "mastery_status": stats["status"],
                "stage": "SCOUTING",
                "freeze_learning": False
            }
            completed_data.insert(0, entry)
            added += 1
    
    with open(COMPLETED_LOG, "w") as f:
        json.dump(completed_data, f, indent=2)

    print(f"✅ Historial de hitos reconstruido: {added} entradas nuevas en {COMPLETED_LOG}.")

if __name__ == "__main__":
    restore()
