"""
routers/assets_router.py — Gestión de activos (símbolos).

Responsabilidades:
- Listar todos los símbolos activos y todos los símbolos
- Habilitar / deshabilitar símbolos individualmente
- Configuración general del bot (config endpoint)
- Reentrenamiento manual de la IA
- Estadísticas de la red neuronal
- Estado del checkpoint
- Control de Live Paper con Alpaca
"""

import json
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from shared import config
from shared.config import (
    COMMAND_FILE, CHECKPOINT_DB, ML_DATASET_FILE, NEURAL_MODEL_FILE,
    MODEL_SNAPSHOTS_DIR
)

router = APIRouter(prefix="/api", tags=["assets"])


# ─── Request Models ──────────────────────────────────────────────────────────

class ToggleRequest(BaseModel):
    symbol: str
    enabled: bool

class LiveAlpacaStartRequest(BaseModel):
    symbols: str


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/config")
async def get_config():
    path = config.ASSETS_FILE
    try:
        total_syms = 0
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                active_symbols = [a for a in data.get('assets', []) if a.get('enabled', True)]
                total_syms = len(active_symbols)

        trade_mult = 1.0
        try:
            if COMMAND_FILE.exists():
                with open(COMMAND_FILE) as f:
                    cdata = json.load(f)
                    trade_mult = float(cdata.get("trade_multiplier", 1.0))
        except:
            pass

        return {
            "total_symbols": total_syms,
            "max_risk_pct": getattr(config, 'MAX_RISK_PCT', 0.025) * 100,
            "stop_loss_pct": getattr(config, 'STOP_LOSS_PCT', 0.01) * 100,
            "take_profit_pct": getattr(config, 'TAKE_PROFIT_PCT', 0.02) * 100,
            "trading_window": f"{config.TRADING_OPEN_HOUR:02d}:{config.TRADING_OPEN_MIN:02d} - {config.TRADING_CLOSE_HOUR:02d}:{config.TRADING_CLOSE_MIN:02d}",
            "interval": config.DATA_INTERVAL,
            "max_pos_usd": config.MAX_POSITION_USD,
            "trade_multiplier": trade_mult
        }
    except Exception as e:
        return {"total_symbols": 0, "max_risk_pct": 2.5}


@router.get("/active_symbols")
async def get_active_symbols():
    path = config.ASSETS_FILE
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                active = [a.get('symbol') for a in data.get('assets', []) if a.get('enabled', True)]
                return {"status": "success", "symbols": active}
        return {"status": "error", "symbols": []}
    except:
        return {"status": "error", "symbols": []}


@router.get("/all_symbols")
async def get_all_symbols(mode: str = "sim"):
    # Determinamos qué archivo usar según el modo
    path = config.ASSETS_FILE
    if mode == "sim" and hasattr(config, "ASSETS_FILE_SIM"):
        path = config.ASSETS_FILE_SIM
    elif mode == "live" and hasattr(config, "ASSETS_FILE_LIVE"):
        path = config.ASSETS_FILE_LIVE
        
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return {"status": "success", "assets": data.get('assets', [])}
        return {"status": "error", "message": f"Archivo {path.name} no encontrado", "assets": []}
    except Exception as e:
        return {"status": "error", "message": f"Error JSON en {path.name}: {str(e)}"}


@router.post("/toggle_symbol")
async def toggle_symbol(req: ToggleRequest):
    path = config.ASSETS_FILE
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
                return {"status": "success", "message": f"{req.symbol} updated"}
            else:
                return {"status": "not_found", "message": f"Symbol not found"}
        return {"status": "error", "message": f"{path.name} missing"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/neural_stats")
async def get_neural_stats():
    """Estadísticas de la Red Neuronal MLP adaptativa."""
    try:
        import joblib
        import numpy as np

        n, wins, losses = 0, 0, 0
        if ML_DATASET_FILE.exists():
            try:
                import csv
                with open(ML_DATASET_FILE, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # La columna objetivo se llama 'is_win' en ml_dataset.csv, no 'label'
                        is_win = int(float(row.get("is_win", 0)))
                        wins += is_win
                        n += 1
                losses = n - wins
            except Exception:
                n = wins = losses = 0

        acc = 0.0
        mode = "cold-start"
        if NEURAL_MODEL_FILE.exists() and n > 0:
            try:
                bundle = joblib.load(NEURAL_MODEL_FILE)
                model = bundle.get("model")
                X_list = bundle.get("X", [])
                y_list = bundle.get("y", [])
                if model is not None and len(X_list) > 0:
                    score_X = X_list[:n]
                    score_y = y_list[:n]
                    if len(score_X) >= n:
                        acc = float(model.score(np.array(score_X), np.array(score_y)))
                    mode = "MLP"
            except Exception:
                pass

        return {
            "status": "ok",
            "total_samples": n,
            "wins": wins,
            "losses": losses,
            "win_rate_hist": round(wins / n * 100, 1) if n > 0 else 0.0,
            "model_accuracy": round(acc * 100, 1),
            "mode": mode,
            "threshold": 0.55,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "total_samples": 0, "mode": "unavailable"}


@router.get("/checkpoint_info")
async def get_checkpoint_info():
    """Estado del checkpoint de simulación."""
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
            "status": "ok",
            "last_mode": row["mode"] if row else None,
            "last_symbol": row["symbol"] if row else None,
            "last_session": row["session_num"] if row else 0,
            "saved_at": row["created_at"] if row else None,
            "live_paper_symbols_count": live_count["n"] if live_count else 0,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/live_paper_symbol")
async def set_live_paper_symbol(symbol: str, enabled: bool):
    """Habilita o deshabilita un símbolo para Live Paper."""
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


@router.post("/train_ai")
async def train_ai():
    import subprocess
    try:
        train_script = Path(__file__).resolve().parent.parent.parent / "shared" / "train_ai.py"
        result = subprocess.run(["python3", str(train_script)], capture_output=True, text=True)
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/live_alpaca_start")
async def live_alpaca_start(req: LiveAlpacaStartRequest):
    """Inicia Live Paper con Alpaca. Solo para /live."""
    try:
        if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
            return {"status": "error", "message": "Configura APCA_API_KEY_ID y APCA_API_SECRET_KEY."}

        symbol_list = [s.strip().upper() for s in req.symbols.split(",") if s.strip()]
        if not symbol_list:
            return {"status": "error", "message": "No se proporcionaron símbolos válidos."}

        base_assets = []
        if config.ASSETS_FILE.exists():
            with open(config.ASSETS_FILE, "r") as f:
                base_data = json.load(f)
                base_assets = base_data.get("assets", [])

        path = config.ASSETS_FILE_LIVE
        data = {"assets": base_assets}
        for asset in data.get("assets", []):
            asset["enabled"] = asset.get("symbol", "").upper() in symbol_list
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        data_cmd = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data_cmd = json.load(f)
        data_cmd["live_start"] = True
        data_cmd["live_stop"] = False
        with open(COMMAND_FILE, "w") as f:
            json.dump(data_cmd, f)

        return {"status": "success", "message": f"Alpaca Paper iniciado con: {', '.join(symbol_list)}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/live_alpaca_stop")
async def live_alpaca_stop():
    try:
        data = {}
        if COMMAND_FILE.exists():
            with open(COMMAND_FILE) as f:
                data = json.load(f)
        data["live_stop"] = True
        data["live_start"] = False
        with open(COMMAND_FILE, "w") as f:
            json.dump(data, f)
        return {"status": "success", "message": "Alpaca Paper detenido."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/live_alpaca_status")
async def live_alpaca_status():
    try:
        import time
        if config.STATE_FILE_LIVE.exists():
            mtime = config.STATE_FILE_LIVE.stat().st_mtime
            if (time.time() - mtime) < 120:
                return {"status": "success", "running": True}
        return {"status": "success", "running": False}
    except Exception:
        return {"status": "success", "running": False}
