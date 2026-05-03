import json
import os
import requests
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Dict, List

from shared.utils.state_models import TradeRecord, BotState

from shared import config
_state_path = config.STATE_FILE
_state_lock = threading.Lock()
_last_flush_time = 0.0
_flush_throttle_ms = 1000 # Solo escribir en disco cada 1 segundo máximo

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
_global_sim_fees:         float = 0.0
_global_sim_slippage:     float = 0.0
_global_sim_gross_profit: float = 0.0
_global_sim_gross_loss:   float = 0.0
_global_model_accuracy: float = 0.0
_global_total_samples:  int = 0
# Per-symbol IA stats (accuracy and samples tracked independently per symbol)
_symbol_model_stats: dict[str, dict] = {}  # {symbol: {"model_accuracy": float, "total_samples": int}}

# 🏆 Métricas de Maestría (Persistentes en sesión)
_session_max_drawdown: float = 0.0
_session_peak_pnl:     float = 0.0

# Per-symbol state persistent (acumulado de sesiones previas en esta simulación)
_symbol_accumulated_stats: dict[str, dict] = {} # {symbol: {trades, wins, pnl, fees, slip, gross_p, gross_l}}

def load_global_stats_from_journal() -> None:
    """Carga los totales acumulados y los últimos trades desde el archivo CSV de la bitácora."""
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl, _global_sim_ghosts, _trades
    global _global_sim_fees, _global_sim_slippage, _global_sim_gross_profit, _global_sim_gross_loss
    
    try:
        journal_path = config.TRADE_JOURNAL_FILE
        if not journal_path.exists():
            return

        import csv
        trades_count = 0
        wins_count = 0
        total_pnl = 0.0
        total_fees = 0.0
        total_slip = 0.0
        total_gross_prof = 0.0
        total_gross_loss = 0.0
        
        loaded_trades = []
        
        with open(journal_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    trades_count += 1
                    
                    # Extraer pnl neto/bruto, fees, slippage de la fila
                    gp = float(row.get("gross_pnl", 0))
                    f_val = float(row.get("fees", 0))
                    s_val = float(row.get("slippage", 0))
                    
                    total_fees += f_val
                    total_slip += s_val
                    
                    if gp > 0:
                        total_gross_prof += gp
                    elif gp < 0:
                        total_gross_loss += abs(gp)
                        
                    # El PnL neto real del CSV o aproximado
                    npnl = gp + f_val + s_val # fees/slippage son negativos o los restamos si son positivos
                    if "net_pnl" in row and row["net_pnl"]:
                        npnl = float(row["net_pnl"])
                    total_pnl += npnl
                        
                    is_win = int(row.get("is_win", 0))
                    if is_win == 1:
                        wins_count += 1
                    
                    # Cargar también al historial visual (limitado a los últimos 100)
                    ts = row.get("timestamp_close", "")
                    t_str = ts.split('T')[1][:8] if 'T' in ts else ""
                    d_str = ts.split('T')[0] if 'T' in ts else ""
                    
                    loaded_trades.append(TradeRecord(
                        symbol=row.get("symbol", ""),
                        side="SELL",
                        price=float(row.get("exit_price", 0)),
                        qty=float(row.get("qty", 0)),
                        pnl=npnl,
                        time=t_str,
                        date=d_str,
                        reason=row.get("exit_reason", ""),
                        entry_price=float(row.get("entry_price", 0))
                    ))
                except Exception as ex: 
                    # Ignorar filas corruptas
                    continue
        
        with _state_lock:
            _global_sim_trades = trades_count
            _global_sim_wins = wins_count
            _global_sim_pnl = total_pnl
            _global_sim_fees = total_fees
            _global_sim_slippage = total_slip
            _global_sim_gross_profit = total_gross_prof
            _global_sim_gross_loss = total_gross_loss
            # Se omite cargar loaded_trades a _trades para que el dashboard muestre una pizarra limpia en cada nueva sesión
    except Exception:
        pass

# Inicializar al cargar el módulo
load_global_stats_from_journal()

def clear_state() -> None:
    """Borra todos los estados de símbolos y trades en memoria y en disco."""
    global _symbol_states, _trades, _global_sim_trades, _global_sim_wins, _global_sim_pnl, _symbol_model_stats
    global _global_sim_fees, _global_sim_slippage, _global_sim_gross_profit, _global_sim_gross_loss, _global_sim_ghosts
    global _session_max_drawdown, _session_peak_pnl  # ← FIX: declarar global para no crear var local
    with _state_lock:
        _symbol_states.clear()
        _symbol_model_stats.clear()  # Limpiar métricas IA por símbolo
        _trades.clear()
        _global_sim_trades = 0
        _global_sim_wins = 0
        _global_sim_pnl = 0.0
        _global_sim_fees = 0.0
        _global_sim_slippage = 0.0
        _global_sim_gross_profit = 0.0
        _global_sim_gross_loss = 0.0
        _global_sim_ghosts = 0
        _session_max_drawdown = 0.0
        _session_peak_pnl = 0.0
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
    global _symbol_states, _symbol_model_stats
    with _state_lock:
        _symbol_states.clear()
        _symbol_model_stats.clear()  # Limpiar métricas IA por símbolo

def record_trade(symbol: str, side: str, price: float, qty: float, pnl: float, 
                 timestamp: Optional[str] = None, reason: str = "", metadata: Optional[dict] = None) -> None:
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl, _global_sim_fees, _global_sim_slippage
    global _global_sim_gross_profit, _global_sim_gross_loss
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
            xai_metrics = metadata.get("xai_metrics", {})
        else:
            fees = 0.0
            xai_metrics = {}

        if side == "BUY" and entry_price == 0:
            entry_price = price
 
        # Fecha completa
        d_str = timestamp.split('T')[0] if (timestamp and 'T' in timestamp) else datetime.now().strftime("%Y-%m-%d")
 
        # ─── CONTABILIDAD TRANSPARENTE (GROSS vs NET) ───
        slippage = metadata.get("slippage", 0.0) if metadata else 0.0
        # pnl recibido de hapi_mock ya es nominal (Gross) como per plan
        gross_pnl = pnl
        net_pnl   = round(gross_pnl - fees - slippage, 2)

        _trades.append(TradeRecord(
            symbol=symbol, side=side, price=price, qty=qty, 
            pnl=gross_pnl, # Priorizamos Gross en PnL principal para la paridad de suma manual
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            slippage=slippage,
            fees=fees,
            time=t_str, date=d_str, reason=reason, confirmations=confirmations, 
            ml_prob=ml_prob, conf_mult=conf_mult, entry_reason=entry_reason,
            entry_price=entry_price, xai_metrics=xai_metrics
        ))
        if len(_trades) > 100: # Aumentado a 100 para simulaciones largas
            _trades.pop(0)
        
        # Actualizar acumulados globales
        _global_sim_trades += 1
        if gross_pnl >= 0:
            _global_sim_wins += 1
            _global_sim_gross_profit += gross_pnl
        else:
            _global_sim_gross_loss += gross_pnl # Es negativo
            
        _global_sim_pnl += net_pnl # El PnL total acumulado sigue siendo Neto
        _global_sim_fees += fees
        _global_sim_slippage += slippage

def update_state(
    symbol: str,
    mode: Optional[str] = None,
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
    sim_progress_pct: float = 0.0,
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
    total_sim_trades: Optional[int] = None,
    total_sim_wins: Optional[int] = None,
    total_sim_pnl: Optional[float] = None,
    total_sim_ghosts: Optional[int] = None,
    total_sim_fees: Optional[float] = None,
    total_sim_slippage: Optional[float] = None,
    total_sim_gross_profit: Optional[float] = None,
    total_sim_gross_loss: Optional[float] = None,
    total_ghosts: int = 0,
    ghost_trades_count: int = 0,
    position: Optional[dict] = None,
    candles: Optional[list] = None,
    timestamp: Optional[str] = None,
    regime: str = "NEUTRAL",
    mock_time_930: bool = False,
    blocks: Optional[List[str]] = None,
    blocking_summary: Optional[dict] = None,
    sim_start: str = "",
    sim_end: str = "",
    sim_duration: float = 0.0,
    status: str = "running",
    ai_win_prob: float = 0.0,
    ai_recommendation: str = "",
    ai_expected_up: float = 0.0,
    ai_expected_down: float = 0.0,
    model_accuracy: Optional[float] = None,
    total_samples: Optional[int] = None,
    is_ml_blocked: bool = False,
    is_quality_blocked: bool = False,
    last_action: str = "",
    mock_time: str = "",  # Reloj simulado para modo Replay
    stage: str = "TRAINING",
    oracle_stats: dict = {},
    effective_threshold: float = 0.60,
    equity_history: Optional[List[float]] = None,
    trades: Optional[List] = None,
    trade_log: Optional[List] = None,
    **kwargs
) -> None:

    global _symbol_states, _trades
    global _global_sim_trades, _global_sim_wins, _global_sim_pnl, _global_sim_ghosts
    global _global_sim_fees, _global_sim_slippage, _global_sim_gross_profit, _global_sim_gross_loss
    global _global_model_accuracy, _global_total_samples, _symbol_model_stats
    
    # Actualizar acumulados globales solo si se proporcionan
    if total_sim_trades is not None: _global_sim_trades = total_sim_trades
    if total_sim_wins   is not None: _global_sim_wins   = total_sim_wins
    if total_sim_pnl    is not None: _global_sim_pnl    = total_sim_pnl
    if total_sim_ghosts is not None: _global_sim_ghosts = total_sim_ghosts
    if total_sim_fees   is not None: _global_sim_fees   = total_sim_fees
    if total_sim_slippage is not None: _global_sim_slippage = total_sim_slippage
    if total_sim_gross_profit is not None: _global_sim_gross_profit = total_sim_gross_profit
    if total_sim_gross_loss is not None: _global_sim_gross_loss = total_sim_gross_loss

    # Guardar métricas IA por símbolo (no compartidas entre símbolos)
    if symbol not in _symbol_model_stats:
        _symbol_model_stats[symbol] = {"model_accuracy": 0.0, "total_samples": 0}
    if model_accuracy is not None:
        _global_model_accuracy = model_accuracy  # Mantener global para compatibilidad
        _symbol_model_stats[symbol]["model_accuracy"] = model_accuracy
    if total_samples is not None:
        _global_total_samples = total_samples  # Mantener global para compatibilidad
        _symbol_model_stats[symbol]["total_samples"] = total_samples
    # Determinar el modo: Preservar el existente si el nuevo es None
    # Si es la primera vez que vemos el símbolo y estamos en live, default a LIVE_PAPER
    current_mode = "SIMULATED"
    if symbol in _symbol_states:
        current_mode = _symbol_states[symbol].mode
    elif _state_path and "state_live.json" in str(_state_path):
        current_mode = "LIVE_PAPER"
    
    final_mode = mode if mode is not None else current_mode

    # ─── CÁLCULO DE TOTALES VISUALES (Base Acumulada o Inyectada del Runner) ───
    # Si el motor ya provee los totales acumulados (total_sim_*), usamos esos directamente.
    # Si son None (caso Live o Cold Start), recalculamos sumando la base del diario + memoria.
    
    use_manual_sum = (total_sim_pnl is None)

    if not use_manual_sum:
        # El Runner ya nos mandó los totales absolutos, ya guardados en las variables _global_sim_*
        ts_trades       = _global_sim_trades
        ts_wins         = _global_sim_wins
        ts_fees         = round(_global_sim_fees, 2)
        ts_slippage     = round(_global_sim_slippage, 2)
        # 🏆 VOLVEMOS A BRUTO: Para que la suma manual de trades coincida con la tarjeta, 
        # y luego se resten los costos (Fees/Slip) en sus propias tarjetas.
        ts_gross_profit = round(kwargs.get("total_sim_gross_profit", _global_sim_gross_profit), 2)
        ts_gross_loss   = round(kwargs.get("total_sim_gross_loss", _global_sim_gross_loss), 2)
        ts_ghosts       = _global_sim_ghosts
    else:
        # Caso Fallback (Live): Sumatoria de lo que hay en memoria actualmente
        ts_trades  = sum(s.total_trades for s in _symbol_states.values() if s.symbol != symbol) + total_trades
        ts_wins    = sum(s.winning_trades for s in _symbol_states.values() if s.symbol != symbol) + winning_trades
        ts_fees    = round(sum(s.total_fees for s in _symbol_states.values() if s.symbol != symbol) + total_fees, 2)
        ts_slippage= round(sum(s.total_slippage for s in _symbol_states.values() if s.symbol != symbol) + total_slippage, 2)
        
        ts_gross_profit = round(sum(s.gross_profit for s in _symbol_states.values() if s.symbol != symbol) + gross_profit, 2)
        ts_gross_loss   = round(-sum(abs(s.gross_loss) for s in _symbol_states.values() if s.symbol != symbol) - abs(gross_loss), 2)
        ts_ghosts  = sum(s.total_ghosts for s in _symbol_states.values() if s.symbol != symbol) + total_ghosts

    # ─── CÁLCULO DE PNL NETO TOTAL ───
    # ts_pnl = (Beneficio Bruto - Pérdida Bruta) - Fees - Slippage
    ts_gross_pnl = round(ts_gross_profit - abs(ts_gross_loss), 2)
    ts_pnl = round(ts_gross_pnl - ts_fees - ts_slippage, 2)

    new_s = BotState(
        mode=final_mode,
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
        total_sim_fees=ts_fees,
        total_sim_slippage=ts_slippage,
        total_sim_gross_profit=ts_gross_profit,
        total_sim_gross_loss=ts_gross_loss,
        total_sim_gross_pnl=ts_gross_pnl,
        sim_progress_pct=sim_progress_pct,
        total_ghosts=total_ghosts,
        ghost_trades_count=ghost_trades_count,
        position=position,
        trades=trades if trades is not None else [asdict(t) for t in _trades if t.symbol == symbol],
        trade_log=trade_log or [],
        candles=candles or [],
        status=status,
        next_scan_in=kwargs.get("next_scan_in", 0),
        is_waiting=kwargs.get("is_waiting", False),
        mock_time_930=mock_time_930,
        blocks=blocks or [],
        blocking_summary=blocking_summary or {},
        sim_start=sim_start,
        sim_end=sim_end,
        sim_duration=sim_duration,
        mock_time=mock_time,
        ai_win_prob=ai_win_prob,
        ai_recommendation=ai_recommendation,
        ai_expected_up=ai_expected_up,
        ai_expected_down=ai_expected_down,
        stage=stage,
        model_accuracy=_symbol_model_stats.get(symbol, {}).get("model_accuracy", 0.0),
        total_samples=_symbol_model_stats.get(symbol, {}).get("total_samples", 0),
        is_quality_blocked=is_quality_blocked,
        last_action=last_action,
        oracle_stats=oracle_stats,
        effective_threshold=effective_threshold,
        equity_history=equity_history if equity_history is not None else (
            _symbol_states[symbol].equity_history if symbol in _symbol_states else []
        ),

        **(_calculate_mastery_cert(symbol, ts_trades, ts_gross_profit, ts_gross_loss, ts_pnl))
    )
    
    with _state_lock:
        _symbol_states[symbol] = new_s
        _flush()
        
    # 📡 Reporte al Orquestador (para Dashboard en tiempo real)
    orchestrator_url = os.getenv("ORCHESTRATOR_URL")
    if orchestrator_url:
        try:
            # Enviar solo el fragmento de este símbolo para eficiencia
            payload = asdict(new_s)
            requests.post(
                f"{orchestrator_url}/api/state/update",
                json={"symbol": symbol, "data": payload},
                timeout=1
            )
        except Exception:
            # Silencioso: no queremos tumbar el bot si el orquestador tiene lag
            pass

def _flush(blocking: bool = False) -> None:
    """Vuelca el estado a disco con protección de concurrencia y throttling."""
    global _last_flush_time
    import time
    now = time.time()
    
    # Throttle: No escribir más de una vez por segundo para ahorrar IO
    if not blocking and (now - _last_flush_time) < (_flush_throttle_ms / 1000.0):
        return

    lock_path = _state_path.with_suffix(".lock")
    try:
        # 1. Preparar el fragmento de estado de ESTE bot
        local_output = {}
        for sym, st in _symbol_states.items():
            d = asdict(st)
            sym_stats = _symbol_model_stats.get(sym, {})
            d["model_accuracy"] = sym_stats.get("model_accuracy", 0.0)
            d["total_samples"] = sym_stats.get("total_samples", 0)
            local_output[sym] = d

        # 2. Lógica de Fusión Protegida (Merge-on-Write)
        # Usamos un sistema de bloqueo simple para coordinación multi-contenedor
        final_output = {}
        
        # Esperar turno si hay bloqueo (máximo 2 segundos)
        for _ in range(20):
            if not lock_path.exists():
                try:
                    lock_path.write_text(str(os.getpid()))
                    break
                except: pass
            time.sleep(0.1)
            
        try:
            if _state_path.exists():
                try:
                    content = _state_path.read_text(encoding="utf-8")
                    if content.strip():
                        final_output = json.loads(content)
                except: pass

            # 3. Fusionar
            final_output.update(local_output)
            if local_output:
                last_sym = list(local_output.keys())[-1]
                final_output["_main"] = local_output[last_sym]

            # 4. Escritura robusta y atómica
            json_str = json.dumps(final_output, indent=2, default=str)
            _state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = _state_path.with_suffix(".tmp")
            tmp.write_text(json_str, encoding="utf-8")
            
            if tmp.exists():
                tmp.replace(_state_path)
            else:
                _state_path.write_text(json_str, encoding="utf-8")
                
        finally:
            # Liberar bloqueo
            if lock_path.exists():
                try: lock_path.unlink()
                except: pass
            
        _last_flush_time = now
    except Exception as e:
        from shared.utils.logger import log
        log.error(f"❌ [StateWriter] Error Crítico al fusionar estado: {e}")

def force_flush() -> None:
    """Fuerza un volcado inmediato y síncrono del estado."""
    with _state_lock:
        _flush(blocking=True)

def _calculate_mastery_cert(symbol: str, n_trades: int, gp: float, gl: float, pnl: float) -> dict:
    """Calcula métricas de maestría para decidir si el usuario puede ir en vivo."""
    global _session_max_drawdown, _session_peak_pnl
    
    # 1. Drawdown dinámico
    if pnl > _session_peak_pnl:
        _session_peak_pnl = pnl
    current_dd = _session_peak_pnl - pnl
    if current_dd > _session_max_drawdown:
        _session_max_drawdown = current_dd
        
    # 2. Métricas Expertas
    pf = gp / abs(gl) if gl != 0 else (gp if gp > 0 else 0)
    
    # 3. Score de Maestría (0-100)
    # 30% Volumen (meta 50) | 40% Rentabilidad (meta PF 1.4) | 30% Estabilidad (Drawdown < 1500)
    score_vol = min(1.0, n_trades / 50) * 30
    score_pf  = min(1.0, pf / 1.4) * 40
    score_dd  = max(0.0, 1.0 - (_session_max_drawdown / 1500.0)) * 30.0
    
    total_score = round(score_vol + score_pf + score_dd, 1)
    # Si no hay trades ni drawdown, dar base de estabilidad (30%)
    if n_trades == 0 and _session_max_drawdown == 0:
        total_score = 30.0
    
    # 4. Métricas de Analista (Profesional)
    expectancy = pnl / n_trades if n_trades > 0 else 0.0
    
    # Eficiencia: Qué tanto del lucro bruto queda tras costos
    # (Si gross_profit es 100 y fees son 20, eficiencia es 80%)
    efficiency = 0.0
    if gp > 0:
        total_costs = abs(gl) if gl < 0 else 0 # En este modelo GL es la pérdida bruta
        # Pero aquí queremos medir el impacto de las COMISIONES si estuvieran separadas.
        # Como en este simulador GP/GL ya incluyen costos usualmente, usaremos un proxy o
        # si runner_sim pasa los fees por separado.
        # Por ahora, usaremos: (Net PnL / Gross Profit) si GP > 0
        efficiency = max(0.0, min(1.0, pnl / gp)) if pnl > 0 else 0.0

    # Score de Estabilidad: Premia consistencia (muchos trades) y bajo drawdown
    stability = (pf * (n_trades / 50)) / (1 + (_session_max_drawdown / 1000))
    stability = round(min(100, stability * 10), 1) # Normalizado a 0-100 aprox

    # 5. Capital recomendado
    # Multiplicador según comando del usuario (1x, 2x, 3x)
    try:
        if config.COMMAND_FILE.exists():
            with open(config.COMMAND_FILE, "r") as cf:
                cdata = json.load(cf)
                mult = float(cdata.get("trade_multiplier", 1.0))
                tier = cdata.get("risk_tier", "Normal (1x)")
        else:
            mult = 1.0
            tier = "Normal (1x)"
    except:
        mult, tier = 1.0, "Normal (1x)"
        
    # El capital sugerido es el Drawdown * 2.5 + el capital por operación (base 500)
    rec_cash = (500 * mult) + (_session_max_drawdown * 2.5)
    
    status = "APRENDIENDO"
    checklist = []
    
    # Evaluar checklist
    if n_trades < 50: checklist.append(f"📦 Muestras: {n_trades}/50 trades")
    if pf < 1.4: checklist.append(f"📈 Profit Factor: {pf:.1f}/1.4")
    if _session_max_drawdown > 1500: checklist.append("🛡️ Drawdown: Reducir riesgo")
    
    if n_trades >= 50 and pf >= 1.4 and _session_max_drawdown <= 1500: 
        status = "LISTO PARA VIVO"
    elif n_trades >= 20: 
        status = "VALIDANDO"
    
    return {
        "mastery_score": total_score,
        "mastery_status": status,
        "recommended_cash": round(rec_cash, 2),
        "risk_tier": tier,
        "actual_profit_factor": round(pf, 2),
        "actual_max_drawdown": round(_session_max_drawdown, 2),
        "mastery_checklist": checklist,
        # --- Analyst Upgrade ---
        "expectancy": round(expectancy, 2),
        "efficiency": round(efficiency * 100, 1),
        "stability_score": stability
    }
