"""
simulation_mode/utils.py ─ Auxiliares para el modo simulación y visualización.
"""

import time
import json
import os
import pandas as pd
import threading
from shared import config
from shared.utils.logger import log
from shared.broker.interface import BrokerInterface, Quote
from shared.strategy.models import OpenPosition, AccountState

# Bloqueo para impresiones consistentes en consola
print_lock = threading.Lock()

class SessionInterrupted(Exception):
    """Lanzada cuando llega un comando de reinicio global o cambio de modo."""
    pass

def smart_sleep(seconds: float):
    """Sleep interrumpible que revisa cambios de comando cada segundo."""
    cmd_file = config.COMMAND_FILE
    start_time = time.time()
    
    # Leer estado inicial
    initial_cmds = {}
    try:
        if os.path.exists(cmd_file):
            with open(cmd_file, "r") as f:
                initial_cmds = json.load(f)
    except: pass

    while (time.time() - start_time) < seconds:
        try:
            if os.path.exists(cmd_file):
                with open(cmd_file, "r") as f:
                    cmds = json.load(f)
                if cmds.get("reset_all") or cmds.get("restart_sim"):
                    raise SessionInterrupted("Reinicio solicitado")
                if cmds.get("force_paper_trading") != initial_cmds.get("force_paper_trading"):
                    raise SessionInterrupted("Cambio de modo detectado")
                if cmds.get("reset_neural"):
                    from shared.utils.neural_filter import reset_neural_filter
                    from shared.strategy.ml_predictor import ml_predictor
                    reset_neural_filter()
                    ml_predictor.model = None
                    ml_predictor.is_trained = False
                    cmds["reset_neural"] = False
                    with open(cmd_file, "w") as f_w:
                        json.dump(cmds, f_w)
                    log.info("🧠 [SIM] WIPE detectado — Memoria Neuronal limpiada en caliente.")

        except SessionInterrupted:
            raise
        except: pass
        time.sleep(1)
    return False

def get_candles_json(df: pd.DataFrame, window: int = 60) -> list:
    """Convierte fragmento de DF a lista de dicts para el dashboard asegurando integridad temporal."""
    if df is None or len(df) == 0: return []
    
    # Remover duplicados en el índice de tiempo (evita crash en LightweightCharts)
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep='first')]
    
    df = df.sort_index() # Orden estricto garantizado

    last_window = df.tail(window)
    res = []
    for ts, row in last_window.iterrows():
        try:
            # Priorizar extracción de timezone-aware timestamp
            if 'Datetime' in row.index:
                t_val = int(pd.to_datetime(row['Datetime']).timestamp())
            else:
                t_val = int(pd.to_datetime(ts).timestamp())
        except:
            continue # Omitir vela corrupta en la UI

        # Evitar datos nulos en velas que crashean JSON
        try:
            res.append({
                "time": t_val,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0))
            })
        except: continue
        
    return res

def print_status(
    broker: BrokerInterface,
    symbol: str,
    quote: Quote,
    signal_str: str,
    position: OpenPosition | None,
    account: AccountState,
) -> None:
    """Imprime una línea de estado legible en la consola."""
    mode_label = "🟢 LIVE" if not broker.is_paper_trading else "🔵 SIMULADO"
    pos_str = (
        f"📈 POSICIÓN: {position.qty:.4f} acciones @ ${position.entry_price:.2f} "
        f"[SL=${position.stop_loss:.2f} | TP=${position.take_profit:.2f}]"
        if position else "─ Sin posición abierta"
    )
    
    with print_lock:
        print(
            f"{mode_label} | {symbol} | "
            f"bid=${quote.bid:.2f} ask=${quote.ask:.2f} | "
            f"Señal={signal_str:4s} | "
            f"{pos_str} | "
            f"Cash disponible=${account.available_cash:.2f}",
            flush=True,
        )
