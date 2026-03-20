import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

from shared.utils.state_models import TradeRecord, BotState

from shared import config
_state_path = config.STATE_FILE
_state_lock = threading.Lock()

def set_state_file(path: Path) -> None:
    global _state_path
    with _state_lock:
        _state_path = path


# ─── GESTIÓN DE ESTADO ──────────────────────────────────────────────────────

# Almacenamiento global multinivel
_symbol_states: dict[str, BotState] = {}
_trades: list[TradeRecord] = []

# Estadísticas acumulativas globales de la simulación
_global_sim_trades: int = 0
_global_sim_wins:   int = 0
_global_sim_pnl:    float = 0.0
_global_sim_ghosts: int = 0
_global_model_accuracy: float = 0.0
_global_total_samples:  int = 0

def load_global_stats_from_journal() -> None:
    """Carga los totales acumulados y los últimos trades desde el archivo CSV de la bitácora."""
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl, _global_sim_ghosts, _trades
    try:
        journal_path = config.TRADE_JOURNAL_FILE
        if not journal_path.exists():
            return

        import csv
        trades_count = 0
        wins_count = 0
        total_pnl = 0.0
        loaded_trades = []
        
        with open(journal_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades_count += 1
                    pnl = float(row.get("gross_pnl", 0))
                    total_pnl += pnl
                    is_win = int(row.get("is_win", 0))
                    if is_win == 1:
                        wins_count += 1
                    
                    # Cargar también al historial visual (limitado a los últimos 100)
                    ts = row.get("timestamp_close", "")
                    t_str = ts.split('T')[1][:8] if 'T' in ts else ""
                    d_str = ts.split('T')[0] if 'T' in ts else ""
                    
                    # Reconstruir TradeRecord (simplificado para UI)
                    loaded_trades.append(TradeRecord(
                        symbol=row.get("symbol", ""),
                        side="SELL",
                        price=float(row.get("exit_price", 0)),
                        qty=float(row.get("qty", 0)),
                        pnl=pnl,
                        time=t_str,
                        date=d_str,
                        reason=row.get("exit_reason", ""),
                        entry_price=float(row.get("entry_price", 0))
                    ))
                except: continue
        
        with _state_lock:
            _global_sim_trades = trades_count
            _global_sim_wins = wins_count
            _global_sim_pnl = total_pnl
            _trades = loaded_trades[-100:] # Mantener últimos 100 para el dashboard
    except Exception:
        pass

# Inicializar al cargar el módulo
load_global_stats_from_journal()

def clear_state() -> None:
    """Borra todos los estados de símbolos y trades en memoria y en disco."""
    global _symbol_states, _trades, _global_sim_trades, _global_sim_wins, _global_sim_pnl
    with _state_lock:
        _symbol_states.clear()
        _trades.clear()
        _global_sim_trades = 0
        _global_sim_wins = 0
        _global_sim_pnl = 0.0
        try:
            # Eliminar archivos físicos
            if _state_path and _state_path.exists():
                _state_path.unlink()
            
            # También borrar el .tmp si existe
            if _state_path:
                tmp = _state_path.with_suffix(".tmp")
                if tmp.exists():
                    tmp.unlink()
        except Exception:
            pass

def clear_symbol_states() -> None:
    """Borra solo los estados de los símbolos activos, manteniendo el historial de trades."""
    global _symbol_states
    with _state_lock:
        _symbol_states.clear()

def record_trade(symbol: str, side: str, price: float, qty: float, pnl: float, 
                 timestamp: str | None = None, reason: str = "", metadata: dict = None) -> None:
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl
    with _state_lock:
        # Usar timestamp proporcionado o el actual (formato HH:MM:SS para el historial rápido)
        t_str = timestamp.split('T')[1][:8] if (timestamp and 'T' in timestamp) else datetime.now().strftime("%H:%M:%S")
        
        # Extraer campos detallados del metadata si existen
        confirmations = []
        ml_prob = 0.0
        conf_mult = 1.0
        entry_reason = ""
        entry_price = 0.0
        
        if metadata:
            confirmations = metadata.get("confirmations", [])
            ml_prob = metadata.get("ml_prob", 0.0)
            conf_mult = metadata.get("conf_mult", 1.0)
            entry_reason = metadata.get("entry_reason", "")
            entry_price = metadata.get("entry_price", 0.0)
            fees = metadata.get("fees", 0.0)
        else:
            fees = 0.0

        if side == "BUY" and not entry_reason:
            entry_reason = reason
        
        if side == "BUY" and entry_price == 0:
            entry_price = price

        # Fecha completa
        d_str = timestamp.split('T')[0] if (timestamp and 'T' in timestamp) else datetime.now().strftime("%Y-%m-%d")

        _trades.append(TradeRecord(
            symbol=symbol, side=side, price=price, qty=qty, pnl=pnl, 
            time=t_str, date=d_str, reason=reason, confirmations=confirmations, 
            ml_prob=ml_prob, conf_mult=conf_mult, entry_reason=entry_reason,
            entry_price=entry_price, fees=fees
        ))
        if len(_trades) > 100: # Aumentado a 100 para simulaciones largas
            _trades.pop(0)
        
        # Actualizar acumulados globales
        _global_sim_trades += 1
        if pnl > 0:
            _global_sim_wins += 1
        _global_sim_pnl += pnl

def update_state(
    symbol: str,
    mode: str = "SIMULATED",
    session: int = 0,
    iteration: int = 0,
    bid: float = 0.0,
    ask: float = 0.0,
    signal: str = "HOLD",
    rsi: float = 50.0,
    ema_fast: float = 0.0,
    ema_slow: float = 0.0,
    ema_200: float = 0.0,
    macd_hist: float = 0.0,
    vwap: float = 0.0,
    atr: float = 0.0,
    confirmations: list = None,
    initial_cash: float = 10000.0,
    available_cash: float = 10000.0,
    settlement: float = 0.0,
    win_rate: float = 0.0,
    total_trades: int = 0,
    winning_trades: int = 0,
    gross_profit: float = 0.0,
    gross_loss: float = 0.0,
    total_fees: float = 0.0,
    total_slippage: float = 0.0,
    total_sim_trades: int = None,
    total_sim_wins: int = None,
    total_sim_pnl: float = None,
    total_sim_ghosts: int = None,
    total_ghosts: int = 0,
    ghost_trades_count: int = 0,
    position: dict | None = None,
    candles: list | None = None,
    timestamp: str | None = None,
    regime: str = "NEUTRAL",
    mock_time_930: bool = False,
    blocks: list[str] | None = None,
    sim_start: str = "",
    sim_end: str = "",
    status: str = "running",
    ai_win_prob: float = 0.0,
    ai_recommendation: str = "",
    ai_expected_up: float = 0.0,
    ai_expected_down: float = 0.0,
    model_accuracy: float = 0.0,
    total_samples: int = 0,
    **kwargs
) -> None:
    global _symbol_states, _global_sim_trades, _global_sim_wins, _global_sim_pnl, _global_sim_ghosts, _global_model_accuracy, _global_total_samples
    
    if total_sim_trades is not None: _global_sim_trades = total_sim_trades
    if total_sim_wins   is not None: _global_sim_wins   = total_sim_wins
    if total_sim_pnl    is not None: _global_sim_pnl    = total_sim_pnl
    if total_sim_ghosts is not None: _global_sim_ghosts = total_sim_ghosts
    
    # Persistir métricas de IA
    if model_accuracy is not None: _global_model_accuracy = model_accuracy
    if total_samples is not None: _global_total_samples = total_samples
    if total_sim_ghosts is not None: _global_sim_ghosts = total_sim_ghosts

    ts_trades = _global_sim_trades
    ts_wins   = _global_sim_wins
    ts_pnl    = _global_sim_pnl
    ts_ghosts = _global_sim_ghosts

    new_s = BotState(
        mode=mode,
        symbol=symbol,
        session=session,
        iteration=iteration,
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        bid=bid,
        ask=ask,
        signal=signal,
        regime=regime,
        rsi=rsi,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_200=ema_200,
        macd_hist=macd_hist,
        vwap=vwap,
        atr=atr,
        confirmations=confirmations,
        initial_cash=initial_cash,
        available_cash=available_cash,
        settlement=settlement,
        win_rate=win_rate,
        total_trades=total_trades,
        winning_trades=winning_trades,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_fees=total_fees,
        total_slippage=total_slippage,
        total_sim_trades=ts_trades,
        total_sim_wins=ts_wins,
        total_sim_pnl=ts_pnl,
        total_sim_ghosts=ts_ghosts,
        total_ghosts=total_ghosts,
        ghost_trades_count=ghost_trades_count,
        position=position,
        trades=[asdict(t) for t in _trades if t.symbol == symbol],
        candles=candles or [],
        status=status,
        next_scan_in=kwargs.get("next_scan_in", 0),
        is_waiting=kwargs.get("is_waiting", False),
        mock_time_930=mock_time_930,
        blocks=blocks or [],
        sim_start=sim_start,
        sim_end=sim_end,
        ai_win_prob=ai_win_prob,
        ai_recommendation=ai_recommendation,
        ai_expected_up=ai_expected_up,
        ai_expected_down=ai_expected_down,
        model_accuracy=_global_model_accuracy,
        total_samples=_global_total_samples,
    )
    
    with _state_lock:
        _symbol_states[symbol] = new_s
        _flush()

def _flush() -> None:
    try:
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        # El estado final es un diccionario de símbolos
        output = {}
        for sym, st in _symbol_states.items():
            d = asdict(st)
            # Forzar campos críticos si no están (redundancia de seguridad)
            d["model_accuracy"] = getattr(st, "model_accuracy", 0.0)
            d["total_samples"] = getattr(st, "total_samples", 0)
            output[sym] = d
        
        # Inyectar una vista "global" (el primer símbolo o una mezcla relevante)
        # para compatibilidad parcial o para que el UI sepa qué símbolos hay
        if _symbol_states:
            # Seleccionamos el ÚLTIMO símbolo actualizado para ser el 'main'
            keys = list(_symbol_states.keys())
            if keys:
                last_sym = keys[-1]
                st_main = _symbol_states[last_sym]
                d_main = asdict(st_main)
                d_main["model_accuracy"] = getattr(st_main, "model_accuracy", 0.0)
                d_main["total_samples"] = getattr(st_main, "total_samples", 0)
                output["_main"] = d_main
        
        tmp = _state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2, default=str))
        tmp.replace(_state_path)
    except Exception:
        pass
