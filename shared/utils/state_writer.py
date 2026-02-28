import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

from shared import config
_state_path = config.STATE_FILE
_state_lock = threading.Lock()

def set_state_file(path: Path) -> None:
    global _state_path
    with _state_lock:
        _state_path = path

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
    # Estadísticas Acumuladas (para simulaciones de múltiples activos)
    total_sim_trades: int = 0
    total_sim_wins:   int = 0
    total_sim_pnl:    float = 0.0

    # Posición abierta
    position: dict | None = None
    trades: list = field(default_factory=list)
    candles: list = field(default_factory=list)

    status: str = "running"
    next_scan_in: int = 0
    is_waiting: bool = False
    mock_time_930: bool = False
    blocks: list[str] = None
    sim_start: str = ""   # Fecha de inicio de los datos
    sim_end:   str = ""   # Fecha de fin de los datos

# Almacenamiento global multinivel
_symbol_states: dict[str, BotState] = {}
_trades: list[TradeRecord] = []

# Estadísticas acumulativas globales de la simulación
_global_sim_trades: int = 0
_global_sim_wins:   int = 0
_global_sim_pnl:    float = 0.0

def load_global_stats_from_journal() -> None:
    """Carga los totales acumulados desde el archivo CSV de la bitácora."""
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl
    try:
        journal_path = config.TRADE_JOURNAL_FILE
        if not journal_path.exists():
            return

        import csv
        trades_count = 0
        wins_count = 0
        total_pnl = 0.0
        
        with open(journal_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades_count += 1
                    pnl = float(row.get("gross_pnl", 0))
                    total_pnl += pnl
                    if int(row.get("is_win", 0)) == 1:
                        wins_count += 1
                except: continue
        
        with _state_lock:
            _global_sim_trades = trades_count
            _global_sim_wins = wins_count
            _global_sim_pnl = total_pnl
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

def record_trade(symbol: str, side: str, price: float, qty: float, pnl: float) -> None:
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl
    with _state_lock:
        _trades.append(TradeRecord(symbol=symbol, side=side, price=price, qty=qty, pnl=pnl))
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
    total_sim_trades: int = None,
    total_sim_wins: int = None,
    total_sim_pnl: float = None,
    position: dict | None = None,
    candles: list | None = None,
    timestamp: str | None = None,
    regime: str = "NEUTRAL",
    mock_time_930: bool = False,
    blocks: list[str] | None = None,
    sim_start: str = "",
    sim_end: str = "",
    status: str = "running",
    **kwargs
) -> None:
    global _symbol_states, _global_sim_trades, _global_sim_wins, _global_sim_pnl
    
    # Si no se pasan acumulados específicos, usar los de la memoria global
    ts_trades = total_sim_trades if total_sim_trades is not None else _global_sim_trades
    ts_wins   = total_sim_wins   if total_sim_wins   is not None else _global_sim_wins
    ts_pnl    = total_sim_pnl    if total_sim_pnl    is not None else _global_sim_pnl

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
        total_sim_trades=ts_trades,
        total_sim_wins=ts_wins,
        total_sim_pnl=ts_pnl,
        position=position,
        trades=[asdict(t) for t in _trades],
        candles=candles or [],
        status=status,
        next_scan_in=kwargs.get("next_scan_in", 0),
        is_waiting=kwargs.get("is_waiting", False),
        mock_time_930=mock_time_930,
        blocks=blocks or [],
        sim_start=sim_start,
        sim_end=sim_end
    )
    
    with _state_lock:
        _symbol_states[symbol] = new_s
        _flush()

def _flush() -> None:
    try:
        _state_path.parent.mkdir(parents=True, exist_ok=True)
        # El estado final es un diccionario de símbolos
        output = {sym: asdict(st) for sym, st in _symbol_states.items()}
        
        # Inyectar una vista "global" (el primer símbolo o una mezcla relevante)
        # para compatibilidad parcial o para que el UI sepa qué símbolos hay
        if _symbol_states:
            # Seleccionamos el ÚLTIMO símbolo actualizado para ser el 'main'
            last_sym = list(_symbol_states.keys())[-1]
            output["_main"] = asdict(_symbol_states[last_sym])
        
        tmp = _state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2, default=str))
        tmp.replace(_state_path)
    except Exception:
        pass
