"""
routers/simulation_router.py — Endpoints de control de la simulación.

Responsabilidades:
- Control del estado de la simulación (reset, restart)
- Lectura de resultados, trades y estadísticas diarias
- Control de Live Paper Trading
- Control de Freeze de estrategia
- Historial de trades de velas (yfinance)
"""

import json
import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from shared import config
from shared.config import (
    COMMAND_FILE, RESULTS_FILE, LOG_FILE, ML_DATASET_FILE, CHECKPOINT_DB
)

router = APIRouter(prefix="/api", tags=["simulation"])




# ─── Request Models ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    sim_start_date: str | None = None

class MultiplierRequest(BaseModel):
    multiplier: float

class FreezeRequest(BaseModel):
    frozen: bool
    label: str | None = None

class PaperTradeRequest(BaseModel):
    symbols: str  # Comma separated: "TSLA,AAPL,NVDA"
    mockTime: bool = False


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/set_symbol")
async def set_symbol(symbol: str):
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


@router.post("/set_trade_multiplier")
async def set_trade_multiplier(req: MultiplierRequest):
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        data["trade_multiplier"] = max(0.1, min(10.0, req.multiplier))
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
        return {"status": "success", "multiplier": data["trade_multiplier"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/reset_all")
async def reset_all(req: ResetRequest = None):
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)

        data["reset_all"] = True
        data["force_paper_trading"] = False
        data["force_symbols"] = []
        data["force_symbol"] = "AUTO"

        if req and req.sim_start_date:
            data["sim_start_date"] = req.sim_start_date
        else:
            data["sim_start_date"] = None

        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        try:
            if RESULTS_FILE.exists(): RESULTS_FILE.unlink()
            if config.TRADE_JOURNAL_FILE.exists(): config.TRADE_JOURNAL_FILE.unlink()
        except: pass

        try:
            if LOG_FILE.exists():
                LOG_FILE.write_text("")
        except: pass

        return {"status": "success", "message": "Reseteando bot y limpiando datos..."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/restart_sim")
async def restart_sim(req: ResetRequest = None):
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)

        data["restart_sim"] = True

        if req and req.sim_start_date:
            data["sim_start_date"] = req.sim_start_date
        else:
            data["sim_start_date"] = None

        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)

        try:
            if RESULTS_FILE.exists(): RESULTS_FILE.unlink()
            if config.TRADE_JOURNAL_FILE.exists(): config.TRADE_JOURNAL_FILE.unlink()
        except: pass

        return {"status": "success", "message": "Reinicio de simulación en cola y datos limpiados"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/results")
async def get_results():
    try:
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE) as f:
                return json.load(f)
        return []
    except Exception:
        return []


@router.get("/daily_stats")
async def get_daily_stats(symbol: str = None):
    """Estadísticas agrupadas por día para un símbolo (o todos)."""
    import csv
    stats = {}
    try:
        if config.TRADE_JOURNAL_FILE.exists():
            with open(config.TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    s = r.get("symbol")
                    if symbol and s != symbol:
                        continue
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
        return {}


@router.get("/trades_history")
async def get_trades_history(limit: int = 100):
    """Últimos trades cerrados desde trade_journal.csv."""
    import csv
    trades = []
    try:
        if config.TRADE_JOURNAL_FILE.exists():
            with open(config.TRADE_JOURNAL_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for r in rows[-limit:]:
                    ts = r.get("timestamp_close", "")
                    date = ts.split("T")[0] if "T" in ts else ts.split(" ")[0] if ts else ""
                    time_str = ts.split("T")[1][:8] if "T" in ts else (ts.split(" ")[1][:8] if " " in ts else "")
                    trades.append({
                        "timestamp":   ts,
                        "symbol":      r.get("symbol"),
                        "side":        "SELL",
                        "price":       float(r.get("exit_price", 0)),
                        "entry_price": float(r.get("entry_price", 0)),
                        "qty":         float(r.get("qty", 0)),
                        "pnl":         float(r.get("gross_pnl", 0)),
                        "date":        date,
                        "time":        time_str,
                        "reason":      r.get("exit_reason", "")[:50] or "S/E",
                    })
        return trades
    except Exception as e:
        return []


@router.get("/history")
async def get_history(symbol: str, interval: str = "5m", period: str = "5d"):
    """Velas históricas desde yfinance para el gráfico del dashboard."""
    import yfinance as yf
    import pandas as pd
    try:
        df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
        if df.empty:
            df = yf.download(tickers=symbol, period="1d", interval="1m", progress=False)
        if df.empty:
            return {"status": "error", "message": "No hay datos para este símbolo."}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time":   int(ts.timestamp()),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"])
            })
        return {"status": "success", "symbol": symbol, "candles": candles}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/freeze")
async def set_freeze(req: FreezeRequest):
    """Congela o descongela la estrategia."""
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


@router.get("/freeze_status")
async def get_freeze_status():
    try:
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
            return {"frozen": data.get("strategy_frozen", False), "label": data.get("freeze_label")}
        return {"frozen": False, "label": None}
    except Exception:
        return {"frozen": False, "label": None}


@router.post("/paper_trade_start")
async def paper_trade_start(req: PaperTradeRequest):
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        symbol_list = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
        if not symbol_list:
            return {"status": "error", "message": "No se proporcionaron símbolos válidos."}
        
        data["force_paper_trading"] = True
        data["force_symbols"] = symbol_list
        data["mock_time_930"] = req.mockTime
        
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
        return {"status": "success", "message": f"Iniciando Live Paper con: {', '.join(symbol_list)}."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/paper_trade_stop")
async def paper_trade_stop():
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


