"""
utils/checkpoint.py — Checkpoint SQLite para persistir el estado de simulación.

Cuando el usuario cambia entre modos (SIMULATED ↔ LIVE_PAPER), guarda exactamente
en qué símbolo y sesión estaba el bot, para reanudar desde el mismo punto.

SCHEMA:
    checkpoints(
        id          INTEGER PRIMARY KEY,
        mode        TEXT,       -- 'SIMULATED' o 'LIVE_PAPER'
        symbol_idx  INTEGER,    -- índice actual en la lista de símbolos
        symbol      TEXT,       -- nombre del símbolo actual
        session_num INTEGER,    -- número de sesión actual
        created_at  TEXT        -- timestamp ISO
    )
"""
from __future__ import annotations
import sqlite3
import threading
from datetime import datetime, timezone
import config
DB_PATH = config.CHECKPOINT_DB
_lock   = threading.Lock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # escrituras atómicas, sin corrupción
    conn.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            mode        TEXT    NOT NULL,
            symbol_idx  INTEGER NOT NULL DEFAULT 0,
            symbol      TEXT    NOT NULL DEFAULT '',
            session_num INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_paper_symbols (
            symbol      TEXT PRIMARY KEY,
            enabled     INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def save_simulation_checkpoint(symbol_idx: int, symbol: str, session_num: int) -> None:
    """Guarda el progreso actual de la simulación en SQLite."""
    with _lock:
        try:
            conn = _connect()
            conn.execute(
                "INSERT INTO checkpoints (mode, symbol_idx, symbol, session_num, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("SIMULATED", symbol_idx, symbol, session_num,
                 datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
            conn.close()
        except Exception as e:
            import logging
            logging.getLogger("trading_bot").warning(f"Checkpoint save error: {e}")


def load_simulation_checkpoint() -> dict:
    """
    Carga el último checkpoint de simulación guardado.
    Devuelve {'symbol_idx': 0, 'symbol': '', 'session_num': 0} si no hay ninguno.
    """
    with _lock:
        try:
            conn = _connect()
            row = conn.execute(
                "SELECT symbol_idx, symbol, session_num FROM checkpoints "
                "WHERE mode = 'SIMULATED' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if row:
                return {
                    "symbol_idx":  row["symbol_idx"],
                    "symbol":      row["symbol"],
                    "session_num": row["session_num"],
                }
        except Exception as e:
            import logging
            logging.getLogger("trading_bot").warning(f"Checkpoint load error: {e}")
    return {"symbol_idx": 0, "symbol": "", "session_num": 0}


def get_live_paper_symbols() -> list[str]:
    """Devuelve la lista de símbolos habilitados para Live Paper."""
    with _lock:
        try:
            conn = _connect()
            rows = conn.execute(
                "SELECT symbol FROM live_paper_symbols WHERE enabled = 1 ORDER BY symbol"
            ).fetchall()
            conn.close()
            return [r["symbol"] for r in rows]
        except Exception:
            return []


def set_live_paper_symbol(symbol: str, enabled: bool) -> None:
    """Habilita o deshabilita un símbolo para Live Paper."""
    with _lock:
        try:
            conn = _connect()
            conn.execute(
                "INSERT INTO live_paper_symbols (symbol, enabled) VALUES (?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET enabled = excluded.enabled",
                (symbol.upper(), 1 if enabled else 0)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            import logging
            logging.getLogger("trading_bot").warning(f"set_live_paper_symbol error: {e}")


def get_checkpoint_info() -> dict:
    """Para el dashboard — devuelve un resumen del estado guardado."""
    with _lock:
        try:
            conn = _connect()
            row = conn.execute(
                "SELECT mode, symbol, session_num, created_at FROM checkpoints "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            live_count = conn.execute(
                "SELECT COUNT(*) as n FROM live_paper_symbols WHERE enabled = 1"
            ).fetchone()
            conn.close()
            if row:
                return {
                    "last_mode":   row["mode"],
                    "last_symbol": row["symbol"],
                    "last_session": row["session_num"],
                    "saved_at":    row["created_at"],
                    "live_paper_symbols_count": live_count["n"] if live_count else 0,
                }
        except Exception:
            pass
    return {"last_mode": None, "last_symbol": None, "last_session": 0, "saved_at": None, "live_paper_symbols_count": 0}
