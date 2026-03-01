"""
dashboard/server.py ─ Servidor FastAPI + WebSocket para el dashboard en tiempo real.

Lee el estado del bot desde config.STATE_FILE (volumen compartido con el bot)
y lo empuja a todos los clientes conectados vía WebSocket cada segundo.
También sirve el dashboard HTML estático.
"""

import json
import asyncio
from shared import config
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from shared.config import (
    STATE_FILE, LOG_FILE, ASSETS_FILE, COMMAND_FILE, RESULTS_FILE,
    ML_DATASET_FILE, CHECKPOINT_DB, DATA_DIR, NEURAL_MODEL_FILE,
    STATE_FILE_SIM, STATE_FILE_LIVE, MODEL_SNAPSHOTS_DIR
)

SIM_HTML_FILE  = Path(__file__).parent / "static" / "simulation" / "index.html"
LIVE_HTML_FILE = Path(__file__).parent / "static" / "live" / "index.html"

app = FastAPI(title="Hapi Bot Dashboard")

# ─── Sirve archivos estáticos ─────────────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Manager de conexiones WebSocket ─────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        # self.active ahora es un dict: { "sim": [ws, ...], "live": [ws, ...] }
        self.active: dict[str, list[WebSocket]] = {"sim": [], "live": []}

    async def connect(self, ws: WebSocket, mode: str):
        await ws.accept()
        if mode not in self.active:
            self.active[mode] = []
        self.active[mode].append(ws)

    def disconnect(self, ws: WebSocket):
        for mode in self.active:
            if ws in self.active[mode]:
                self.active[mode].remove(ws)
                break

    async def broadcast(self, data: any, mode: str):
        if mode not in self.active:
            return
            
        dead = []
        for ws in self.active[mode]:
            try:
                if isinstance(data, str):
                    await ws.send_text(data)
                else:
                    await ws.send_json(data)
            except Exception:
                dead.append(ws)
        
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Lectura del estado del bot ───────────────────────────────────────────────
def read_state(mode: str = "sim") -> dict:
    state = {}
    target_file = config.STATE_FILE_SIM if mode == "sim" else config.STATE_FILE_LIVE
    try:
        if target_file.exists():
            with open(target_file) as f:
                state = json.load(f)
    except Exception:
        state = {"status": "starting", "timestamp": datetime.utcnow().isoformat()}

    # Inyectar símbolos forzados (si hay)
    if COMMAND_FILE.exists():
        try:
            with open(COMMAND_FILE) as f:
                cmds = json.load(f)
                state["force_symbols"] = cmds.get("force_symbols", [])
        except: pass

    return state


def read_last_logs(n: int = 50) -> list[str]:
    """Lee las últimas n líneas del log principal y logs de símbolos específicos."""
    all_lines = []
    try:
        # 1. Leer log principal con robustez ante errores de encoding
        if LOG_FILE.exists():
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                all_lines.extend(f.readlines())
        
        # 2. Descubrir logs de hilos específicos (historial_SYMBOL.log)
        log_dir = LOG_FILE.parent
        for symbol_log in log_dir.glob("historial_*.log"):
            try:
                with open(symbol_log, "r", encoding="utf-8", errors="replace") as f:
                    all_lines.extend(f.readlines())
            except: pass
            
        if not all_lines:
            return ["(Esperando logs del bot...)"]

        # Ordenar por tiempo (asumiendo formato ISO al inicio) y tomar las últimas n
        all_lines.sort()
        return [l.strip() for l in all_lines[-n:] if l.strip()]
        
    except Exception as e:
        return [f"Error leyendo logs: {str(e)}"]


# ─── Broadcaster en background ────────────────────────────────────────────────
async def state_broadcaster():
    """Lee periódicamente los estados y los envía a los websockets correspondientes."""
    last_sim_mtime = 0
    last_live_mtime = 0
    
    while True:
        try:
            # 1. Procesar Simulación
            if config.STATE_FILE_SIM.exists():
                mtime = config.STATE_FILE_SIM.stat().st_mtime
                if mtime > last_sim_mtime:
                    last_sim_mtime = mtime
                    content = config.STATE_FILE_SIM.read_text(encoding="utf-8")
                    await manager.broadcast(content, mode="sim")
            elif last_sim_mtime > 0:
                # Si el archivo existía y ya no existe, enviar estado vacío para limpiar UI
                last_sim_mtime = 0
                await manager.broadcast(json.dumps({}), mode="sim")
            
            # 2. Procesar Live
            if config.STATE_FILE_LIVE.exists():
                mtime = config.STATE_FILE_LIVE.stat().st_mtime
                if mtime > last_live_mtime:
                    last_live_mtime = mtime
                    content = config.STATE_FILE_LIVE.read_text(encoding="utf-8")
                    await manager.broadcast(content, mode="live")
            elif last_live_mtime > 0:
                last_live_mtime = 0
                await manager.broadcast(json.dumps({}), mode="live")
                    
        except Exception as e:
            print(f"Error en broadcaster: {e}")
        
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup():
    asyncio.create_task(state_broadcaster())


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def simulation_index():
    if not SIM_HTML_FILE.exists():
        return HTMLResponse("Simulation UI missing", status_code=404)
    content = SIM_HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse(content=content)

@app.get("/live", response_class=HTMLResponse)
async def live_index():
    if not LIVE_HTML_FILE.exists():
        return HTMLResponse("Live UI missing", status_code=404)
    content = LIVE_HTML_FILE.read_text(encoding="utf-8")
    return HTMLResponse(content=content)


@app.websocket("/ws/sim")
async def websocket_sim(websocket: WebSocket):
    await manager.connect(websocket, mode="sim")
    try:
        while True:
            await websocket.receive_text() # Mantener vivo
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await manager.connect(websocket, mode="live")
    try:
        while True:
            await websocket.receive_text() # Mantener vivo
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/state")
async def get_state(mode: str = "sim"):
    return read_state(mode=mode)


@app.post("/api/set_symbol")
async def set_symbol(symbol: str):
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        data["force_symbol"] = symbol
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
        return {"status": "success", "symbol": symbol}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ResetRequest(BaseModel):
    sim_start_date: str | None = None

@app.post("/api/reset_all")
async def reset_all(req: ResetRequest = None):
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        
        data["reset_all"] = True
        data["force_paper_trading"] = False
        data["force_symbols"] = []
        data["force_symbol"] = "AUTO"
        
        # Guardar fecha de inicio personalizada si se proporciona
        if req and req.sim_start_date:
            data["sim_start_date"] = req.sim_start_date
        else:
            data["sim_start_date"] = None # Reset a default
            
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
        
        # ── LIMPIEZA INMEDIATA DE ARCHIVOS DE DATOS (Fix Bug Dashboard Stale) ──
        # Borramos los archivos desde la API para que el Dashboard refleje el reset 
        # al instante, sin esperar a que el bot despierte.
        from shared.config import RESULTS_FILE, TRADE_JOURNAL_FILE
        try:
            if RESULTS_FILE.exists(): RESULTS_FILE.unlink()
            if TRADE_JOURNAL_FILE.exists(): TRADE_JOURNAL_FILE.unlink()
        except Exception as e_files:
            print(f"Error borrando archivos en reset_all: {e_files}")

        # Limpiar logs
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'w') as f:
                    pass # Truncate to 0 bytes
        except: pass
            
        return {"status": "success", "message": "Reseteando bot y limpiando datos..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/restart_sim")
async def restart_sim(req: ResetRequest = None):
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        
        # Este comando le dirá al bot que reinicie la fecha 
        # y la sesión sin purgar los datasets.
        data["restart_sim"] = True

        # Guardar fecha de inicio personalizada si se proporciona
        if req and req.sim_start_date:
            data["sim_start_date"] = req.sim_start_date
        else:
            data["sim_start_date"] = None
        
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        # ── LIMPIEZA INMEDIATA (Fix Bug Dashboard Stale) ──
        from shared.config import RESULTS_FILE, TRADE_JOURNAL_FILE
        try:
            if RESULTS_FILE.exists(): RESULTS_FILE.unlink()
            if TRADE_JOURNAL_FILE.exists(): TRADE_JOURNAL_FILE.unlink()
        except: pass

        return {"status": "success", "message": "Reinicio de simulación en cola y datos limpiados"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/results")
async def get_results():
    import json
    try:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                return json.load(f)
        return []
    except Exception:
        return []

@app.get("/api/daily_stats")
async def get_daily_stats(symbol: str = None):
    """Retorna estadísticas agrupadas por día para un símbolo (o todos)."""
    import csv
    from shared.config import TRADE_JOURNAL_FILE
    stats = {} # {symbol: {date: {pnl, trades, wins, losses}}}
    try:
        if TRADE_JOURNAL_FILE.exists():
            with open(TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    s = r.get("symbol")
                    if symbol and s != symbol:
                        continue
                    
                    # Parsear fecha "2026-02-28T18:11:51..." o "2026-02-28 09:30:00" -> "2026-02-28"
                    ts = r.get("timestamp_close", "")
                    if not ts: continue
                    date_key = ts.split("T")[0].split(" ")[0]
                    
                    if s not in stats: stats[s] = {}
                    if date_key not in stats[s]:
                        stats[s][date_key] = {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
                    
                    pnl = float(r.get("gross_pnl", 0))
                    stats[s][date_key]["pnl"] += pnl
                    stats[s][date_key]["trades"] += 1
                    if r.get("is_win") == "1":
                        stats[s][date_key]["wins"] += 1
                    else:
                        stats[s][date_key]["losses"] += 1
        
        return stats[symbol] if symbol and symbol in stats else stats
    except Exception as e:
        print(f"Error calculando estadísticas diarias: {e}")
        return {}

@app.get("/api/trades_history")
async def get_trades_history(limit: int = 100):
    """Retorna los últimos trades cerrados desde trade_journal.csv."""
    import csv
    from shared.config import TRADE_JOURNAL_FILE
    trades = []
    try:
        if TRADE_JOURNAL_FILE.exists():
            with open(TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
                # Usamos DictReader para manejar las columnas automáticamente
                reader = csv.DictReader(f)
                rows = list(reader)
                # Tomar los últimos 'limit' registros
                for r in rows[-limit:]:
                    # Adaptar nombres de campos si es necesario para el frontend
                    # El frontend espera: { side, price, qty, pnl, symbol }
                    trades.append({
                        "timestamp": r.get("timestamp_close"),
                        "symbol":    r.get("symbol"),
                        "side":      "SELL", # En el journal son cierres (ventas)
                        "price":     float(r.get("exit_price", 0)),
                        "qty":       float(r.get("qty", 0)),
                        "pnl":       float(r.get("gross_pnl", 0))
                    })
        return trades
    except Exception as e:
        print(f"Error leyendo historial de trades: {e}")
        return []

@app.get("/api/neural_stats")
async def get_neural_stats():
    """Estadísticas de la Red Neuronal MLP adaptativa."""
    try:
        import joblib
        import numpy as np
        if not NEURAL_MODEL_FILE.exists():
            return {"status": "ok", "total_samples": 0, "wins": 0, "losses": 0,
                    "win_rate_hist": 0.0, "model_accuracy": 0.0, "mode": "cold-start",
                    "threshold": 0.55}

        bundle = joblib.load(NEURAL_MODEL_FILE)
        X_list = bundle.get("X", [])
        y_list = bundle.get("y", [])
        model  = bundle.get("model")
        n      = len(y_list)
        wins   = sum(y_list)
        acc    = 0.0
        if model is not None and n > 0:
            try:
                acc = float(model.score(np.array(X_list), np.array(y_list)))
            except Exception:
                pass
        return {
            "status":         "ok",
            "total_samples":  n,
            "wins":           wins,
            "losses":         n - wins,
            "win_rate_hist":  round(wins / n * 100, 1) if n > 0 else 0.0,
            "model_accuracy": round(acc * 100, 1),
            "mode":           "MLP" if model is not None else "cold-start",
            "threshold":      0.55,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "total_samples": 0, "mode": "unavailable"}

@app.get("/api/checkpoint_info")
async def get_checkpoint_info():
    """Estado del checkpoint de simulación — lee checkpoint.db."""
    try:
        import sqlite3
        if not CHECKPOINT_DB.exists():
            return {"status": "ok", "last_mode": None, "last_symbol": None,
                    "last_session": 0, "saved_at": None, "live_paper_symbols_count": 0}
        conn = sqlite3.connect(str(CHECKPOINT_DB))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT mode, symbol, session_num, created_at FROM checkpoints ORDER BY id DESC LIMIT 1"
        ).fetchone()
        live_count = conn.execute(
            "SELECT COUNT(*) as n FROM live_paper_symbols WHERE enabled = 1"
        ).fetchone()
        conn.close()
        return {
            "status":       "ok",
            "last_mode":    row["mode"]       if row else None,
            "last_symbol":  row["symbol"]     if row else None,
            "last_session": row["session_num"] if row else 0,
            "saved_at":     row["created_at"] if row else None,
            "live_paper_symbols_count": live_count["n"] if live_count else 0,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/live_paper_symbol")
async def set_live_paper_symbol(symbol: str, enabled: bool):
    """Habilita o deshabilita un símbolo para Live Paper (escribe en checkpoint.db)."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(CHECKPOINT_DB))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS live_paper_symbols (
                symbol TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 0)
        """)
        conn.execute(
            "INSERT INTO live_paper_symbols (symbol, enabled) VALUES (?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET enabled = excluded.enabled",
            (symbol.upper(), 1 if enabled else 0)
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "symbol": symbol.upper(), "enabled": enabled}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/train_ai")
async def train_ai():
    import subprocess
    import os
    try:
        # Resolve train_ai.py relative to project root
        train_script = Path(__file__).resolve().parent.parent / "shared" / "train_ai.py"
        result = subprocess.run(["python3", str(train_script)], capture_output=True, text=True)
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/config")
async def get_config(mode: str = "sim"):
    import json
    path = config.ASSETS_FILE_SIM if mode == "sim" else config.ASSETS_FILE_LIVE
    try:
        total_syms = 0
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                active_symbols = [a for a in data.get('assets', []) if a.get('enabled', True)]
                total_syms = len(active_symbols)
        
        return {
            "total_symbols": total_syms,
            "max_risk_pct": config.MAX_RISK_PCT * 100,
            "stop_loss_pct": config.STOP_LOSS_PCT * 100,
            "take_profit_pct": config.TAKE_PROFIT_PCT * 100,
            "trading_window": f"{config.TRADING_OPEN_HOUR:02d}:{config.TRADING_OPEN_MIN:02d} - {config.TRADING_CLOSE_HOUR:02d}:{config.TRADING_CLOSE_MIN:02d}",
            "interval": config.DATA_INTERVAL,
            "max_pos_usd": config.MAX_POSITION_USD
        }
    except Exception:
        return {"total_symbols": 0, "max_risk_pct": 2.5}

@app.get("/api/active_symbols")
async def get_active_symbols(mode: str = "sim"):
    import json
    path = config.ASSETS_FILE_SIM if mode == "sim" else config.ASSETS_FILE_LIVE
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                active = [a.get('symbol') for a in data.get('assets', []) if a.get('enabled', True)]
                return {"status": "success", "symbols": active}
        return {"status": "error", "symbols": []}
    except Exception as e:
        return {"status": "error", "symbols": []}

@app.get("/api/all_symbols")
async def get_all_symbols(mode: str = "sim"):
    import json
    path = config.ASSETS_FILE_SIM if mode == "sim" else config.ASSETS_FILE_LIVE
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return {"status": "success", "assets": data.get('assets', [])}
        return {"status": "error", "assets": []}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ToggleRequest(BaseModel):
    symbol: str
    enabled: bool
    mode: str = "sim"

@app.post("/api/toggle_symbol")
async def toggle_symbol(req: ToggleRequest):
    import json
    path = config.ASSETS_FILE_SIM if req.mode == "sim" else config.ASSETS_FILE_LIVE
    try:
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            
            found = False
            for asset in data.get("assets", []):
                if asset.get("symbol") == req.symbol:
                    asset["enabled"] = req.enabled
                    found = True
                    break
            
            if found:
                with open(path, "w") as f:
                    json.dump(data, f, indent=4)
                return {"status": "success", "message": f"{req.symbol} updated in {path.name}"}
            else:
                return {"status": "not_found", "message": f"Symbol not found in {path.name}"}
                
        return {"status": "error", "message": f"{path.name} missing"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class DepositRequest(BaseModel):
    amount: float

@app.post("/api/bank/deposit")
async def bank_deposit(req: DepositRequest):
    import json
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            data["available_cash"] = data.get("available_cash", 10000.0) + req.amount
            with open(STATE_FILE, "w") as f:
                json.dump(data, f)
            return {"status": "success", "message": f"Deposited ${req.amount:.2f}", "new_balance": data["available_cash"]}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/bank/withdraw")
async def bank_withdraw(req: DepositRequest):
    import json
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE) as f:
                data = json.load(f)
            # Hapi fixed withdrawal fee $4.99
            withdrawal_fee = 4.99
            total_deduction = req.amount + withdrawal_fee
            
            if data.get("available_cash", 10000.0) >= total_deduction:
                data["available_cash"] -= total_deduction
                with open(STATE_FILE, "w") as f:
                    json.dump(data, f)
                return {"status": "success", "message": f"Withdrew ${req.amount:.2f} (Fee: ${withdrawal_fee:.2f})", "new_balance": data["available_cash"]}
            else:
                return {"status": "error", "message": "Insufficient funds including $4.99 withdrawal fee."}
        return {"status": "error", "message": "State file not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PaperTradeRequest(BaseModel):
    symbols: str  # Comma separated list like "TSLA,AAPL,NVDA"
    mockTime: bool = False  # Para testing nocturno a las 9:30 AM

@app.post("/api/paper_trade_start")
async def paper_trade_start(req: PaperTradeRequest):
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
        if not symbol_list:
            return {"status": "error", "message": "No se proporcionaron símbolos válidos."}

        data["force_paper_trading"] = True
        data["force_symbols"] = symbol_list
        data["force_symbol"] = symbol_list[0] # Start with the first one
        data["reset_all"] = True
        data["mock_time_930"] = req.mockTime
        
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
            
        return {"status": "success", "message": f"Iniciando Live Paper con: {', '.join(symbol_list)}. Reiniciando Bot."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/history")
async def get_history(symbol: str, interval: str = "5m", period: str = "5d"):
    import yfinance as yf
    import pandas as pd
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if df.empty:
            # Try 1m
            df = yf.download(tickers=symbol, period="1d", interval="1m", progress=False)
        
        if df.empty:
            return {"status": "error", "message": "No hay datos para este símbolo."}

        # Convertir MultiIndex a simple si es necesario
        if isinstance(df.columns, pd.MultiIndex):
             df.columns = df.columns.get_level_values(0)

        candles = []
        for ts, row in df.iterrows():
             candles.append({
                 "time": int(ts.timestamp()),
                 "open": float(row["Open"]),
                 "high": float(row["High"]),
                 "low": float(row["Low"]),
                 "close": float(row["Close"]),
                 "volume": int(row["Volume"])
             })
        return {"status": "success", "symbol": symbol, "candles": candles}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class FreezeRequest(BaseModel):
    frozen: bool
    label: str | None = None  # Etiqueta opcional para identificar qué estrategia se congeló

@app.post("/api/freeze")
async def set_freeze(req: FreezeRequest):
    """Congela o descongela la estrategia. Cuando está congelada el bot no acepta reset/wipe."""
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        
        data["strategy_frozen"] = req.frozen
        data["freeze_label"] = req.label or ("Estrategia Congelada" if req.frozen else None)
        
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        action = "🧊 CONGELADA" if req.frozen else "🔥 DESCONGELADA"
        return {"status": "success", "message": f"Estrategia {action}", "frozen": req.frozen}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/freeze_status")
async def get_freeze_status():
    """Retorna el estado actual de congelación de la estrategia."""
    import json
    try:
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
            return {
                "frozen": data.get("strategy_frozen", False),
                "label": data.get("freeze_label", None)
            }
        return {"frozen": False, "label": None}
    except Exception as e:
        return {"frozen": False, "label": None}

@app.post("/api/paper_trade_stop")
async def paper_trade_stop():
    import json
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        
        data["force_paper_trading"] = False
        data["force_symbols"] = []
        data["force_symbol"] = "AUTO"
        data["reset_all"] = True
        
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
            
        return {"status": "success", "message": "Live Paper Trading detenido. Volviendo a la Simulación Global."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ─── 📸 Model Snapshots (Versiones de IA) ─────────────────────────────────────

class SnapshotRequest(BaseModel):
    label: str  # Nombre descriptivo de la versión

@app.post("/api/model_snapshot")
async def save_model_snapshot(req: SnapshotRequest):
    """Guarda una copia de los 2 modelos de IA como versión nombrada."""
    import json, shutil
    try:
        MODEL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

        # Carpeta única por timestamp + label sanitizado
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c if c.isalnum() or c in "-_ " else "_" for c in req.label)[:40]
        snap_dir = MODEL_SNAPSHOTS_DIR / f"{ts}_{safe_label}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Copiar modelos
        files_saved = []
        if NEURAL_MODEL_FILE.exists():
            shutil.copy2(NEURAL_MODEL_FILE, snap_dir / "neural_model.joblib")
            files_saved.append("neural_model.joblib")
        if ML_DATASET_FILE.exists():
            shutil.copy2(ML_DATASET_FILE, snap_dir / "ml_dataset.csv")
            files_saved.append("ml_dataset.csv")

        if not files_saved:
            snap_dir.rmdir()
            return {"status": "error", "message": "No hay modelos entrenados para guardar. Corre al menos una simulación primero."}

        # Guardar metadata
        meta = {
            "id": f"{ts}_{safe_label}",
            "label": req.label,
            "created_at": datetime.utcnow().isoformat(),
            "files": files_saved,
        }
        with open(snap_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        return {"status": "success", "message": f"✅ Versión '{req.label}' guardada", "snapshot_id": meta["id"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/model_snapshots")
async def list_model_snapshots():
    """Lista todas las versiones de IA guardadas, de más reciente a más antigua."""
    import json
    try:
        MODEL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        snapshots = []
        for snap_dir in sorted(MODEL_SNAPSHOTS_DIR.iterdir(), reverse=True):
            if not snap_dir.is_dir():
                continue
            meta_file = snap_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                # Tamaño del modelo MLP si existe
                mlp_path = snap_dir / "neural_model.joblib"
                meta["mlp_size_kb"] = round(mlp_path.stat().st_size / 1024, 1) if mlp_path.exists() else 0
                snapshots.append(meta)
        return {"status": "success", "snapshots": snapshots}
    except Exception as e:
        return {"status": "error", "snapshots": [], "message": str(e)}


class RestoreRequest(BaseModel):
    snapshot_id: str

@app.post("/api/restore_snapshot")
async def restore_model_snapshot(req: RestoreRequest):
    """Restaura una versión guardada de la IA, reemplazando los modelos actuales."""
    import json, shutil
    try:
        snap_dir = MODEL_SNAPSHOTS_DIR / req.snapshot_id
        if not snap_dir.exists():
            return {"status": "error", "message": f"Versión '{req.snapshot_id}' no encontrada."}

        restored = []

        mlp_snap = snap_dir / "neural_model.joblib"
        if mlp_snap.exists():
            shutil.copy2(mlp_snap, NEURAL_MODEL_FILE)
            restored.append("neural_model.joblib")

        csv_snap = snap_dir / "ml_dataset.csv"
        if csv_snap.exists():
            shutil.copy2(csv_snap, ML_DATASET_FILE)
            restored.append("ml_dataset.csv")

        if not restored:
            return {"status": "error", "message": "La versión seleccionada no tiene archivos de modelo."}

        # Señalizar al bot que recargue los modelos
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        data["reload_models"] = True
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        meta_file = snap_dir / "meta.json"
        label = req.snapshot_id
        if meta_file.exists():
            with open(meta_file) as f:
                label = json.load(f).get("label", req.snapshot_id)

        return {"status": "success", "message": f"✅ Versión '{label}' restaurada. El bot recargará la IA en el próximo ciclo.", "restored": restored}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/model_snapshot/{snapshot_id}")
async def delete_model_snapshot(snapshot_id: str):
    """Elimina una versión guardada de la IA."""
    import shutil
    try:
        snap_dir = MODEL_SNAPSHOTS_DIR / snapshot_id
        if not snap_dir.exists():
            return {"status": "error", "message": "Versión no encontrada."}
        shutil.rmtree(snap_dir)
        return {"status": "success", "message": f"Versión '{snapshot_id}' eliminada."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

