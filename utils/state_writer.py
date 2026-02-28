import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

import config
STATE_PATH = config.STATE_FILE
_state_lock = threading.Lock()

@dataclass
class TradeRecord:
    symbol: str
    side:  str
    price: float
    qty:   float
    pnl:   float
    time:  str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

@dataclass
class BotState:
    # Identificación
    mode:    str = "SIMULATED"
    symbol:  str = "─"
    session: int = 1
    iteration: int = 0
    timestamp: str = ""

    # Precios
    bid: float = 0.0
    ask: float = 0.0

    # Señal
    signal:   str   = "HOLD"
    regime:   str   = "NEUTRAL"  # Régimen de mercado detectado
    rsi:      float = 50.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    ema_200:  float = 0.0
    macd_hist: float = 0.0
    vwap:     float = 0.0
    atr:      float = 0.0
    confirmations: list = field(default_factory=list)

    # Cuenta
    initial_cash:   float = 10_000.0
    available_cash: float = 10_000.0
    settlement:     float = 0.0

    # Estadísticas
    win_rate:       float = 0.0
    total_trades:   int   = 0
    winning_trades: int   = 0
    gross_profit:   float = 0.0
    gross_loss:     float = 0.0

    # Posición abierta
    position: dict | None = None
    trades: list = field(default_factory=list)
    candles: list = field(default_factory=list)

    status: str = "running"
    next_scan_in: int = 0
    is_waiting: bool = False
    mock_time_930: bool = False
    blocks: list[str] = None

# Almacenamiento global multinivel
_symbol_states: dict[str, BotState] = {}
_trades: list[TradeRecord] = []

def clear_state() -> None:
    """Borra todos los estados de símbolos y trades en memoria y en disco."""
    global _symbol_states, _trades
    with _state_lock:
        _symbol_states.clear()
        _trades.clear()
        try:
            if STATE_PATH.exists():
                STATE_PATH.unlink()
            # También borrar el .tmp si existe
            tmp = STATE_PATH.with_suffix(".tmp")
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass

def record_trade(symbol: str, side: str, price: float, qty: float, pnl: float) -> None:
    with _state_lock:
        _trades.append(TradeRecord(symbol=symbol, side=side, price=price, qty=qty, pnl=pnl))
        if len(_trades) > 50:
            _trades.pop(0)

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
    position: dict | None = None,
    candles: list | None = None,
    timestamp: str | None = None,
    regime: str = "NEUTRAL",
    mock_time_930: bool = False,
    blocks: list[str] | None = None,
    status: str = "running",
    **kwargs
) -> None:
    global _symbol_states
    
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
        position=position,
        trades=[asdict(t) for t in _trades],
        candles=candles or [],
        status=status,
        next_scan_in=kwargs.get("next_scan_in", 0),
        is_waiting=kwargs.get("is_waiting", False),
        mock_time_930=mock_time_930,
        blocks=blocks or []
    )
    
    with _state_lock:
        _symbol_states[symbol] = new_s
        _flush()

def _flush() -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # El estado final es un diccionario de símbolos
        output = {sym: asdict(st) for sym, st in _symbol_states.items()}
        
        # Inyectar una vista "global" (el primer símbolo o una mezcla relevante)
        # para compatibilidad parcial o para que el UI sepa qué símbolos hay
        if _symbol_states:
            # Seleccionamos el ÚLTIMO símbolo actualizado para ser el 'main'
            last_sym = list(_symbol_states.keys())[-1]
            output["_main"] = asdict(_symbol_states[last_sym])
        
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2, default=str))
        tmp.replace(STATE_PATH)
    except Exception:
        pass
