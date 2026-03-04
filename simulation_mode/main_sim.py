"""
main.py ─ Motor principal del Bot de Trading Automatizado para Hapi Trade.

╔═══════════════════════════════════════════════════════════╗
║  HAPI TRADE SCALPING BOT  •  Colombia Edition             ║
║  Modos: LIVE (cuenta real) | SIMULATED (yfinance replay)  ║
╚═══════════════════════════════════════════════════════════╝

Flujo principal:
  1. Al iniciar, detecta el modo (LIVE o SIMULATED) desde .env o args.
  2. Si es LIVE: solicita credenciales y conecta con HapiLive.
  3. Si es SIMULATED: carga datos históricos y crea HapiMock.
  4. Cada iteración:
      a. Verifica horario de mercado NYSE (ajustado a Colombia).
      b. Obtiene la cotización actual.
      c. Calcula indicadores técnicos (EMA + RSI).
      d. Si hay señal de ENTRADA y no hay posición abierta → BUY limit.
      e. Si hay posición abierta → verifica SL/TP y emite SELL limit si aplica.
      f. Loguea el estado del bot en tiempo real.
  5. Al finalizar (Ctrl+C o datos agotados): muestra estadísticas.

Uso:
  python main.py                   → Usa el modo definido en .env
  python main.py --mode SIMULATED  → Fuerza modo simulado
  python main.py --mode LIVE       → Fuerza modo live
  python main.py --symbol AAPL     → Símbolo específico (solo simulado)
"""

import sys
import time
import argparse
import getpass
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

import importlib
from shared.data import market_data as market_data
from shared import config
from shared.utils.logger import log, log_order_attempt, log_market_closed, set_symbol_log
from shared.utils.market_hours import is_market_open, market_status_str, next_open_str
from shared.utils import state_writer, market_hours, logger, trade_journal, checkpoint
from shared.utils.state_writer import set_state_file
from shared.data.market_data import set_assets_file, get_symbols
from shared.broker.interface import BrokerInterface, Quote, OrderResponse
from shared.broker.hapi_live import HapiLive
from shared.broker.hapi_mock import HapiMock
from shared.strategy.indicators import analyze, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from shared.strategy.risk_manager import RiskManager, AccountState, OpenPosition
from shared.strategy.ml_predictor import ml_predictor
from shared.utils.trade_journal import record_trade as journal_record_trade
from shared.utils.trade_journal import record_trade as journal_record_trade
from dataclasses import asdict
import threading


# ═══════════════════════════════════════════════════════════════════════════════
#  BANNER DE INICIO
# ═══════════════════════════════════════════════════════════════════════════════
BANNER = """
╔══════════════════════════════════════════════════════╗
║     🇨🇴 HAPI SCALPING BOT  ─  Colombia Edition       ║
║     Estrategia: EMA Crossover + RSI Filter           ║
║     Gestión de riesgo: SL 1%  •  TP 2%              ║
╚══════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  EXCEPCIONES DE CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
class SessionInterrupted(Exception):
    """Lanzada cuando llega un comando de reinicio global o cambio de modo."""
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS PARA RESPONSIVIDAD
# ═══════════════════════════════════════════════════════════════════════════════
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
        except SessionInterrupted:
            raise
        except: pass
        time.sleep(1)
    return False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE ARGUMENTOS CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot de Trading - Modo SIMULACIÓN"
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Símbolo específico para simular (ej. AAPL).",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="Capital inicial para la simulación (default: $10,000)",
    )
    p = parser.parse_args()
    p.mode = "SIMULATED"
    
    import os
    fixed = os.getenv("FIXED_SYMBOL", "").strip()
    if fixed and not p.symbol:
        p.symbol = fixed
        log.info(f"🔒 Símbolo fijo (Env): {fixed}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN DEL BRÓKER
# ═══════════════════════════════════════════════════════════════════════════════
def init_broker(args: argparse.Namespace, is_live_paper_override: bool = False) -> BrokerInterface:
    """Inicializa el bróker de simulación (siempre Mock)."""
    is_live_paper = is_live_paper_override
    force_symbols = []
    sim_start_date = None
    try:
        import json, os
        cmd_file = config.COMMAND_FILE
        if cmd_file.exists():
            with open(cmd_file) as f:
                cmds = json.load(f)
            
            if cmds.get("force_paper_trading", False):
                is_live_paper = True
                
            force_symbols = cmds.get("force_symbols", [])
            force_symbol_val = cmds.get("force_symbol", "")
            
            if is_live_paper:
                if force_symbols and not (len(force_symbols) == 1 and force_symbols[0] == "AUTO"):
                    if not args.symbol or args.symbol not in force_symbols:
                        args.symbol = force_symbols[0]
                elif force_symbol_val and force_symbol_val != "AUTO":
                    args.symbol = force_symbol_val
                
            sim_start_date = cmds.get("sim_start_date")
    except Exception:
        pass

    return HapiMock(symbol=args.symbol, initial_cash=args.cash, live_paper=is_live_paper, start_date=sim_start_date)


# ── Auxiliar para formatear velas ──────────────────────────────────────────────
def get_candles_json(df: pd.DataFrame, window: int = 60) -> list:
    """Convierte fragmento de DF a lista de dicts para el dashboard."""
    if df is None or len(df) == 0: return []
    last_60 = df.tail(window)
    res = []
    import pandas as pd
    for ts, row in last_60.iterrows():
        try:
            # Si el índice es DatetimeIndex
            t_val = int(ts.timestamp())
        except:
            t_val = str(ts)
        res.append({
            "time": t_val,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row.get("Volume", 0))
        })
    return res


# ═══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN DE ESTADO DEL BOT (DISPLAY EN CONSOLA)
# ═══════════════════════════════════════════════════════════════════════════════
print_lock = threading.Lock()
results_lock = threading.Lock()

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


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL DEL BOT
# ═══════════════════════════════════════════════════════════════════════════════
def run_bot(broker: BrokerInterface, args: argparse.Namespace, session_num: int = 1, stop_event: threading.Event = None, asset_type: str = "normal") -> None:
    """
    Bucle principal del bot de trading para un símbolo específico.
    """
    is_mock = isinstance(broker, HapiMock)
    symbol  = broker.symbol if is_mock else (args.symbol or "AAPL")
    
    # Activar el log específico para este símbolo para análisis forense
    set_symbol_log(symbol)

    # ── Inicializar estado de cuenta ─────────────────────────────────────────
    acct_info = broker.get_account_info()
    account   = AccountState(total_cash=acct_info.total_cash)
    risk_mgr  = RiskManager(account)
    position: OpenPosition | None = None

    log.info(f"{'─'*60}")
    log.info(f"Bot iniciado | Símbolo: {symbol} | Modo: {broker.name}")
    log.info(f"Capital inicial: ${acct_info.total_cash:,.2f}")
    log.info(f"Estrategia: EMA({config.EMA_FAST}/{config.EMA_SLOW}) + RSI({config.RSI_PERIOD})")
    log.info(f"SL: {config.STOP_LOSS_PCT*100:.0f}%  TP: {config.TAKE_PROFIT_PCT*100:.0f}%  MaxPos: ${config.MAX_POSITION_USD}")
    log.info(f"{'─'*60}")

    sim_start_date_str = ""
    sim_end_date_str   = ""
    if is_mock:
        try:
            # Capturar el DF real tras el filtrado para informar al dashboard
            _replay = getattr(broker, '_replay', None)
            if _replay:
                _df = getattr(_replay, 'df', getattr(_replay, 'full_df', None))
                if _df is not None and not _df.empty:
                    sim_start_date_str = _df.index[0].strftime("%Y-%m-%d")
                    sim_end_date_str   = _df.index[-1].strftime("%Y-%m-%d")
                    log.info(f"📊 Ventana de simulación detectada: {sim_start_date_str} a {sim_end_date_str}")
        except Exception as e_dates:
            log.warning(f"No se pudo extraer fechas del replay: {e_dates}")

    # Enviar un primer pulso de estado para limpiar la UI y mostrar las fechas reales
    from shared.utils.state_writer import update_state
    update_state(
        mode=args.mode,
        symbol=symbol,
        status="running",
        sim_start=sim_start_date_str,
        sim_end=sim_end_date_str,
        available_cash=account.available_cash,
        total_trades=0,
        win_rate=0
    )

    # 📊 SEGUIMIENTO DE BLOQUEOS (Para análisis de estrategia)
    import collections
    blocking_history = collections.Counter()
    ml_predictor.blocking_count = 0  # Reiniciar contador para esta sesión

    iteration = 0
    try:
        while not (stop_event and stop_event.is_set()):
            iteration += 1

            # ── 1. Verificar horario de mercado (solo en modo LIVE) ───────────
            if not is_mock and not is_market_open():
                log_market_closed(next_open_str())
                print(f"\n{market_status_str()}")
                time.sleep(config.REST_INTERVAL_SEC)
                continue

            # ── 2. Obtener cotización ────────────────────────────────────────
            try:
                quote = broker.get_quote(symbol)
                last_price_str = f"${quote.last:.2f}"
                current_sim_time = quote.timestamp # Capturar tiempo lógico
            except StopIteration:
                log.info("▶ Datos de simulación agotados. Finalizando sesión.")
                break
            except ConnectionError as exc:
                log.error(f"Connection error: {exc}. Reintentando en 5s…")
                time.sleep(5)
                continue

            # ── 3. Obtener datos históricos para indicadores ─────────────────
            # 220 velas = EMA 200 + 20 de margen para estabilización
            if is_mock:
                df = broker.get_history_slice(window=220)
                
                # REVISAR COMANDOS DESDE EL DASHBOARD (Instántaneo)
                try:
                    cmd_file = config.COMMAND_FILE
                    if os.path.exists(cmd_file):
                        with open(cmd_file, "r") as f:
                            cmds = json.load(f)
                            
                        if cmds.get("reset_all") or cmds.get("restart_sim") or cmds.get("force_paper_trading") is False:
                            log.info(f"🔄 Interrumpiendo simulación ({symbol}): Señal de RESET/RESTART recibida.")
                            return # Salir de run_bot para procesar en main()

                        new_sym = cmds.get("force_symbol")
                        if new_sym and new_sym != "AUTO" and new_sym != symbol:
                            log.info(f"🔄 Interrumpiendo simulación. Usuario solicita cambiar a {new_sym}")
                            break # Salir de este loop para reiniciar sesión con el nuevo
                except: pass
            else:
                from shared.data.market_data import download_bars
                df = download_bars(symbol)

            signal = analyze(df, symbol=symbol, asset_type=asset_type)

            # 📊 REGISTRAR BLOQUEOS: Si no hay señal y hay razones de bloqueo
            if signal.signal == "HOLD" and signal.blocks:
                for b in signal.blocks:
                    # Limpiar el mensaje para agrupar mejor (ej: "ADX: 15 < 18" -> "ADX")
                    clean_b = b.split(":")[0].strip()
                    blocking_history[clean_b] += 1

            # ── 4. Gestionar SALIDA (si hay posición abierta) ────────────────
            if position is not None:
                position.hold_bars = getattr(position, "hold_bars", 0) + 1
                should_exit, exit_reason = risk_mgr.should_exit(position, quote.bid)

                # Eliminamos la regla de "panic sell" del indicador, dejamos que el Trailing Stop decida
                if should_exit:
                    reason = exit_reason
                    sell_plan = risk_mgr.calculate_sell_order(position, quote.bid)

                    log_order_attempt(symbol, "SELL", sell_plan.limit_price, sell_plan.qty, reason)

                    # ⚡ Stop Loss / Trailing Stop / Time Exit → MARKET SELL inmediato al bid actual.
                    # Un limit SELL solo se llena cuando bar_high >= precio objetivo.
                    # Si el precio cae a gap (muy común en acciones), la orden nunca se llenaría
                    # hasta que el precio subiera de vuelta — causando pérdidas masivas más allá del SL.
                    is_protective_exit = any(k in reason for k in ("STOP_LOSS", "TRAILING_STOP", "TIEMPO", "TIME"))
                    is_take_profit     = "TAKE_PROFIT" in reason or "🎯" in reason

                    if is_protective_exit and is_mock and hasattr(broker, "immediate_market_sell"):
                        # Usar market sell al bid actual — garantiza salida al precio real
                        # Consolidar metadata de salida con la de entrada para historial
                        exit_metadata = {
                            "exit_reason": reason,
                            "entry_reason": getattr(position, 'entry_reason', ""),
                            "entry_metadata": getattr(position, 'entry_metadata', {}),
                            "ml_prob": position.entry_metadata.get('ml_prob', 0),
                            "conf_mult": position.entry_metadata.get('conf_mult', 1.0),
                            "confirmations": position.entry_metadata.get('confirmations', []),
                            "entry_price": position.entry_price
                        }
                        actual_fill = broker.immediate_market_sell(
                            symbol, position.qty, quote.bid, 
                            reason=reason, metadata=exit_metadata
                        )
                        resp_status = "FILLED"
                        sell_price  = actual_fill
                    else:
                        # Take Profit → limit order (busca el precio objetivo hacia arriba)
                        sell_price = sell_plan.limit_price
                        # Consolidar metadata de salida con la de entrada para historial
                        exit_metadata = {
                            "exit_reason": reason,
                            "entry_reason": getattr(position, 'entry_reason', ""),
                            "entry_metadata": getattr(position, 'entry_metadata', {}),
                            "ml_prob": position.entry_metadata.get('ml_prob', 0),
                            "conf_mult": position.entry_metadata.get('conf_mult', 1.0),
                            "confirmations": position.entry_metadata.get('confirmations', []),
                            "entry_price": position.entry_price
                        }
                        resp = broker.place_limit_order(
                            symbol=symbol, side="SELL",
                            limit_price=sell_price, qty=sell_plan.qty,
                            reason=reason, metadata=exit_metadata
                        )
                        resp_status = resp.status

                    if resp_status not in ("REJECTED",):
                        pnl = (sell_price - position.entry_price) * position.qty

                        log.info(
                            f"🔴 SELL | {symbol} → ${sell_price:.2f} | "
                            f"PnL estimado: ${pnl:+.2f} | Motivo: {reason}"
                        )
                        
                        # 🧠 Alimentar al modelo de ML con el resultado real
                        if hasattr(position, 'ml_features') and position.ml_features:
                            log.info(f"[{symbol}] SAVING TRADE: Pnl={pnl} Features={position.ml_features}")
                            ml_predictor.save_trade(symbol, position.ml_features, pnl)
                            # 📓 Bitácora completa de trading (para análisis y mejora de algoritmos)
                            journal_record_trade(
                                symbol=symbol,
                                session=session_num,
                                entry_price=position.entry_price,
                            exit_price=sell_price,
                                qty=position.qty,
                                stop_loss=position.initial_stop,
                                take_profit=position.take_profit,
                                hold_bars=getattr(position, 'hold_bars', 0),
                                exit_reason=reason.replace('🔒','TRAILING_STOP').replace('🛑','STOP_LOSS').replace('🎯','TAKE_PROFIT').replace('⏰','TIME_EXIT').replace('🛡️','FORCED_SELL'),
                                ml_features=position.ml_features,
                                timestamp=current_sim_time
                            )
                            # 🧠 Alimentar la Red Neural MLP con el resultado real del trade
                            try:
                                from shared.utils.neural_filter import get_neural_filter
                                from shared.utils.neural_filter import NeuralTradeFilter
                                nf = get_neural_filter()
                                ml_f = position.ml_features
                                features_vec = nf.build_features(
                                    rsi=ml_f.get('rsi', 50.0),
                                    macd_hist=ml_f.get('macd_hist', 0.0),
                                    atr_pct=ml_f.get('atr_pct', 0.0),
                                    vol_ratio=ml_f.get('vol_ratio', 1.0),
                                    ema_fast=ml_f.get('ema_fast', ml_f.get('ema_diff_pct', 0.0) + 100),
                                    ema_slow=ml_f.get('ema_slow', 100.0),
                                    zscore_vwap=ml_f.get('zscore_vwap', ml_f.get('vwap_dist_pct', 0.0)),
                                    regime=ml_f.get('regime', 'NEUTRAL'),
                                    num_confirmations=int(ml_f.get('num_confirmations', 2)),
                                )
                                nf.fit(features_vec, won=(pnl > 0))
                            except Exception as _e:
                                log.warning(f"No se pudo entrenar red neuronal: {_e}")
                        # En LIVE registrar el capital como T+1 pendiente
                        if not is_mock:
                            account.record_sale(sell_price * position.qty)
                        position = None

            # ── 5. Gestionar ENTRADA (si no hay posición y hay señal BUY) ────
            elif signal.signal == SIGNAL_BUY:
                # ── REGLA: Evitar el Sesgo de Look-Ahead (Solo operar cuando la vela cierre exactamente) ──
                # En simulación, simulamos que el tiempo ocurre en saltos por lo que las velas "cierran".
                # No podemos bloquear por segundos ya que mock data viaja en chunks, 
                # pero aplicaremos un castigo/simulación del tiempo si fuera necesario, 
                # o dejaremos constancia de que la vela está "completada" en el mock.

                # 🧠 CONSULTAR AL MODELO DE MACHINE LEARNING ANTES DE ENTRAR
                is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
                
                # 📊 CALCULAR FACTOR DE CONFIANZA (Sistematización de aumento de compra)
                conf_mult = 1.0
                if is_win:
                    if prob_win > 0.85: conf_mult *= 1.5  # IA muy segura (+50%)
                    elif prob_win > 0.70: conf_mult *= 1.2 # IA segura (+20%)
                
                if len(signal.confirmations) >= 4:
                    conf_mult *= 1.2 # Muchas confluencias técnicas (+20%)
                
                # Cap de seguridad: nunca duplicar más del 2x el riesgo base
                conf_mult = min(conf_mult, 2.0)

                # En modo simulado, sincronizar el cash real desde el MockBroker
                if is_mock:
                    live_info = broker.get_account_info()
                    account.total_cash = live_info.total_cash

                # ── REGLA: Liquidez / SLIPPAGE PREVENTIVO ──
                spread = quote.ask - quote.bid
                max_spread_allowed = quote.bid * 0.0020 # 0.2% max spread
                if spread > max_spread_allowed:
                    msg = f"📉 Spread Tóxico ({spread:.2f} > {max_spread_allowed:.2f}) | Slippage risk"
                    signal.blocks = [msg]
                    signal.signal = SIGNAL_HOLD
                    log.info(f"[{symbol}] {msg}")
                    continue

                buy_plan = risk_mgr.calculate_buy_order(
                    symbol, quote.ask,
                    atr_value=signal.atr_value,
                    confidence_multiplier=conf_mult
                )

                if not is_win and prob_win < 0.48 and buy_plan.is_viable:
                    buy_plan.is_viable = False
                    buy_plan.block_reason = f"ML_REJECTION: El robot recuerda haber perdido con estos parámetros (Probabilidad de Ganar: {prob_win*100:.1f}%)"
                    log.info(f"[{symbol}] 🧠 ML EVITÓ PERDER DINERO | Bloqueó entrada | Prob={prob_win*100:.1f}%")

                # ─ Consultar Neural Filter (solo predict, igual que Live) ─
                if buy_plan.is_viable:
                    try:
                        from shared.utils.neural_filter import get_neural_filter
                        nf = get_neural_filter()
                        ml_f = signal.ml_features
                        fv = nf.build_features(
                            rsi=ml_f.get("rsi", 50.0),
                            macd_hist=ml_f.get("macd_hist", 0.0),
                            atr_pct=ml_f.get("atr_pct", 0.0),
                            vol_ratio=ml_f.get("vol_ratio", 1.0),
                            ema_fast=ml_f.get("ema_diff_pct", 0.0) + 100,
                            ema_slow=100.0,
                            zscore_vwap=ml_f.get("vwap_dist_pct", 0.0),
                            regime=ml_f.get("regime", "NEUTRAL"),
                            num_confirmations=int(ml_f.get("num_confirmations", 2)),
                        )
                        # ─ ML Tolerance Threshold ─
                        if not is_win and prob_win < 0.48 and buy_plan.is_viable:
                            buy_plan.is_viable = False
                            msg = f"🧠 AI bloqueó entrada | MLP P={prob_win*100:.1f}%"
                            buy_plan.block_reason = msg
                            signal.signal = SIGNAL_HOLD
                            signal.blocks = [msg]
                            log.info(f"[LIVE:{symbol}] {msg}")
                        nf_win, nf_prob = nf.predict(fv)
                        if not nf_win:
                            buy_plan.is_viable = False
                            msg = f"🧠 Red Neuronal bloqueó | P={nf_prob*100:.1f}%"
                            buy_plan.block_reason = msg
                            signal.signal = SIGNAL_HOLD
                            log.info(f"[{symbol}] {msg}")
                    except Exception as nfe:
                        log.debug(f"[{symbol}] NF error: {nfe}")

                if buy_plan.is_viable:
                    if conf_mult > 1.0:
                        log.info(f"[{symbol}] 🔥 COMPRA AGRESIVA: Aumentando tamaño x{conf_mult:.2f} por alta confianza (IA={prob_win*100:.0f}%, Confluencias={len(signal.confirmations)})")
                    
                    reason_str = f"BUY: {len(signal.confirmations)} confluencias"
                    if conf_mult > 1.0:
                        reason_str += f" | Aumento x{conf_mult:.1f}"

                    metadata = {
                        "confirmations": signal.confirmations,
                        "ml_prob": prob_win,
                        "conf_mult": conf_mult
                    }

                    log_order_attempt(
                        symbol, "BUY",
                        buy_plan.limit_price, buy_plan.qty,
                        f"confluencias={len(signal.confirmations)} | "
                        f"RSI={signal.rsi_value:.1f} | MACD={signal.macd_hist:.4f} | "
                        f"IA={prob_win*100:.0f}% | Aumento=x{conf_mult:.2f}"
                    )
                    resp = broker.place_limit_order(
                        symbol=symbol,
                        side="BUY",
                        limit_price=buy_plan.limit_price,
                        qty=buy_plan.qty,
                        reason=reason_str,
                        metadata=metadata
                    )
                    if resp.status not in ("REJECTED",):
                        position = OpenPosition(
                            symbol=symbol,
                            entry_price=buy_plan.limit_price,
                            qty=buy_plan.qty,
                            stop_loss=buy_plan.stop_loss,
                            take_profit=buy_plan.take_profit,
                            initial_stop=buy_plan.stop_loss,
                            ml_features=signal.ml_features,
                            entry_reason=reason_str,
                            entry_metadata=metadata
                        )
                        if not is_mock:
                            account.total_cash -= buy_plan.limit_price * buy_plan.qty
                        log.info(
                            f"🟢 BUY  | {symbol} @ ${buy_plan.limit_price:.2f} | "
                            f"qty={buy_plan.qty:.4f} | "
                            f"SL=${buy_plan.stop_loss:.2f} (ATR×1.5=${buy_plan.atr_stop:.2f}) | "
                            f"TP=${buy_plan.take_profit:.2f}"
                        )
                else:
                    log.debug(f"HOLD | Orden bloqueada: {buy_plan.block_reason}")

            # ── 6. Escribir estado para el dashboard ─────────────────────────
            pos_dict = None
            if position:
                pos_dict = {
                    "symbol":      position.symbol,
                    "entry_price": position.entry_price,
                    "qty":         position.qty,
                    "stop_loss":   position.stop_loss,
                    "take_profit": position.take_profit,
                }
            stats = broker.stats if is_mock else type('S', (), {
                'win_rate': 0, 'total_trades': 0, 'winning_trades': 0,
                'gross_profit': 0, 'gross_loss': 0})()

            # ── Serializar últimas 60 velas para el gráfico de velas japonesas ─
            candles_data = []
            if is_mock:
                hist = broker.get_history_slice(window=60)
                for ts, row in hist.iterrows():
                    try:
                        import calendar
                        t = int(calendar.timegm(ts.utctimetuple()))
                        candles_data.append({
                            "time":  t,
                            "open":  round(float(row["Open"]),  4),
                            "high":  round(float(row["High"]),  4),
                            "low":   round(float(row["Low"]),   4),
                            "close": round(float(row["Close"]), 4),
                            "volume": int(row["Volume"]),
                        })
                    except Exception:
                        pass

            # Determine UI mode label
            ui_mode = args.mode
            if is_mock and getattr(broker, '_live_paper', False):
                ui_mode = "LIVE_PAPER"

            from shared.utils.market_hours import _is_mock_time_active
            is_mock_active = _is_mock_time_active()

            state_writer.update_state(
                mode=ui_mode,
                symbol=symbol,
                session=session_num,
                iteration=iteration,
                bid=quote.bid,
                ask=quote.ask,
                signal=signal.signal,
                rsi=signal.rsi_value,
                ema_fast=signal.ema_fast,
                ema_slow=signal.ema_slow,
                ema_200=signal.ema_200,
                macd_hist=signal.macd_hist,
                vwap=signal.vwap_value,
                atr=signal.atr_value,
                confirmations=signal.confirmations,
                initial_cash=acct_info.total_cash,
                available_cash=account.available_cash,
                settlement=account.pending_settlement_total,
                win_rate=stats.win_rate,
                total_trades=stats.total_trades,
                winning_trades=stats.winning_trades,
                gross_profit=stats.gross_profit,
                gross_loss=stats.gross_loss,
                position=pos_dict,
                candles=candles_data,
                timestamp=signal.timestamp.isoformat() if hasattr(signal, 'timestamp') and signal.timestamp else None,
                regime=getattr(signal, 'regime', 'NEUTRAL'),
                mock_time_930=is_mock_active,
                blocks=signal.blocks,
                sim_start=sim_start_date_str,
                sim_end=sim_end_date_str
            )

            # ── 7. Display de estado (cada 10 iteraciones en simulación) ─────
            if iteration % 10 == 0 or not is_mock:
                print_status(broker, symbol, quote, signal.signal, position, account)

            # ── 8. Pausa entre escaneos (Live y Live Paper) ──────────────────
            is_live_paper_b = getattr(broker, '_live_paper', False)
            if not is_mock or is_live_paper_b:
                from shared.utils.market_hours import _is_mock_time_active
                if is_live_paper_b and _is_mock_time_active():
                    pause_time = 1 # Aceleración x60 en Test Nocturno
                else:
                    pause_time = 60 if is_live_paper_b else config.SCAN_INTERVAL_SEC
                    
                log.info(f"[{symbol}] ⏳ Esperando pausa de {pause_time}s (interrumpible)...")
                
                for i in range(pause_time, 0, -1):
                    # Actualizar estado con la cuenta regresiva
                    from shared.utils.state_writer import update_state
                    from shared.utils.market_hours import _is_mock_time_active
                    is_mock_active = _is_mock_time_active()

                    update_state(
                        mode=args.mode if not is_live_paper_b else "LIVE_PAPER",
                        symbol=symbol,
                        session=session_num,
                        iteration=iteration,
                        bid=quote.bid if quote else 0,
                        ask=quote.ask if quote else 0,
                        signal=signal.signal,
                        rsi=signal.rsi_value,
                        ema_fast=signal.ema_fast,
                        ema_slow=signal.ema_slow,
                        ema_200=signal.ema_200,
                        macd_hist=signal.macd_hist,
                        vwap=signal.vwap_value,
                        atr=signal.atr_value,
                        confirmations=signal.confirmations,
                        initial_cash=account.total_cash,
                        available_cash=account.available_cash,
                        settlement=account.pending_settlement_total,
                        win_rate=broker.stats.win_rate,
                        total_trades=broker.stats.total_trades,
                        winning_trades=broker.stats.winning_trades,
                        gross_profit=broker.stats.gross_profit,
                        gross_loss=broker.stats.gross_loss,
                        position=asdict(position) if position else None,
                        candles=get_candles_json(df, 60),
                        next_scan_in=i,
                        is_waiting=True,
                        mock_time_930=is_mock_active,
                        blocks=signal.blocks,
                        sim_start=sim_start_date_str,
                        sim_end=sim_end_date_str
                    )
                    smart_sleep(1)
                
        if not (stop_event and stop_event.is_set()):
            smart_sleep(1)
            if not is_live_paper_b:
                try:
                    if config.COMMAND_FILE.exists():
                        with open(config.COMMAND_FILE, "r") as f:
                            c = json.load(f)
                        if c.get("reset_all") or c.get("restart_sim") or c.get("force_paper_trading") is False:
                            log.info(f"🛑 Interrupción detectada para {symbol} (Reset/Restart).")
                            return
                except: pass

    except KeyboardInterrupt:
        print("\n")
        log.info("🛑 Bot detenido manualmente (KeyboardInterrupt).")

    finally:
        final_info = None
        # ── Cerrar posición abierta al salir (seguridad) ─────────────────────
        if position is not None:
            log.warning(
                f"⚠️  Quedó una posición abierta en {position.symbol} "
                f"({position.qty:.4f} acciones @ ${position.entry_price:.2f}). "
                f"Revisa tu cuenta manualmente en Hapi."
            )

        # ── Mostrar y Guardar estadísticas finales ───────────────────────────
        if is_mock:
            broker.print_stats()
            
            # Si se interrumpió por reset_all, no guardamos estadística corrupta
            should_save = True
            try:
                cmd_file = config.COMMAND_FILE
                if os.path.exists(config.COMMAND_FILE):
                    with open(config.COMMAND_FILE, "r") as f:
                        cmds = json.load(f)
                        if cmds.get("reset_all") or cmds.get("restart_sim"):
                            log.info("Cancelando guardado de sesión debido a comando de reinicio/wipe.")
                            should_save = False
            except Exception:
                pass
                
            # Guardar en Bitácora para ML y Análisis de Resultados
            if should_save:
                try:
                    import json
                    from datetime import datetime
                    from pathlib import Path
                    res_file = config.RESULTS_FILE
                    final_info = broker.get_account_info()
                    
                    # ── Generar un Diagnóstico Rápido de la Estrategia (Insight) ──
                    pnl_val = getattr(broker.stats, 'total_pnl', 0.0)
                    trades_val = getattr(broker.stats, 'total_trades', 0)
                    winrate_val = getattr(broker.stats, 'win_rate', 0.0)

                    if not trades_val and not pnl_val: pass
                    
                    insight = "Sin actividad."
                    if trades_val == 0:
                        insight = "No hubo entradas. La estrategia filtró el ruido (bueno) o las condiciones de indicadores fueron demasiado estrictas para el volumen de hoy."
                    elif pnl_val > 0 and winrate_val >= 50:
                        insight = "ESTRATEGIA EXITOSA. Altamente efectiva, detectó bien la tendencia y el ratio de StopLoss/TakeProfit es óptimo."
                    elif pnl_val > 0 and winrate_val < 50:
                        insight = "RENTABLE POR GESTIÓN. Hubo bastantes señales falsas, pero la gestión de riesgo (ganar mucho, perder poco) salvó el balance."
                    elif pnl_val < 0 and winrate_val >= 50:
                        insight = "ERROR DE GESTIÓN DE RIESGO. Se gana frecuentemente pero las comisiones/spreads o los Stop Loss muy anchos destrozaron las pequeñas ganancias."
                    else:
                        insight = "ESTRATEGIA FALLIDA. Constantes señales engañosas (whipsaws). Sugiere añadir filtro de tendencia mayor (ej. ADX) o descartar este símbolo por volatilidad impredecible."

                    from datetime import timezone
                    # Datos estructurados para futuro Data Science / ML
                    total_fees = round(getattr(broker.stats, 'total_fees', 0.0), 4)
                    gross_pnl  = round(pnl_val + total_fees, 2)  # PnL antes de comisiones
                    # Slippage estimado: 0.05% por lado (compra + venta) × trades
                    slippage_est = round(trades_val * 0.0005 * 200, 4)  # ~$0.10 por trade
                    
                    from shared.utils.state_writer import _symbol_states
                    regime_val = _symbol_states.get(symbol).regime if symbol in _symbol_states else getattr(broker, 'final_regime', 'NEUTRAL')
                    
                    session_result = {
                        "timestamp":      datetime.now(timezone.utc).isoformat(),
                        "symbol":         symbol,
                        "session_num":    session_num,
                        "total_trades":   trades_val,
                        "winning_trades": getattr(broker.stats, 'winning_trades', 0),
                        "losing_trades":  trades_val - getattr(broker.stats, 'winning_trades', 0),
                        "win_rate":       round(winrate_val, 2),
                        "pnl":            round(pnl_val, 2),
                        "gross_pnl":      gross_pnl,
                        "total_fees":     total_fees,
                        "slippage_est":   slippage_est,
                        "gross_profit":   round(getattr(broker.stats, 'gross_profit', 0.0), 2),
                        "gross_loss":     round(getattr(broker.stats, 'gross_loss', 0.0), 2),
                        "profit_factor":  round(getattr(broker.stats, 'profit_factor', 0.0), 2),
                        "drawdown":       round(getattr(broker.stats, 'max_drawdown', 0.0), 2),
                        "last_price":     round(getattr(final_info, 'last_price', 0.0), 2),
                        "insight":        insight,
                        "regime":         regime_val or "NEUTRAL",
                        "blocking_summary": dict(blocking_history),
                        "ml_blocked_count": getattr(ml_predictor, 'blocking_count', 0)
                    }
                    
                    with results_lock:
                        res_data = []
                        if res_file.exists():
                            with open(res_file, "r") as f:
                                res_data = json.load(f)
                                
                        # Deduplicar: Solo si coincide símbolo Y sesión (evita duplicados exactos en re-arranques)
                        res_data = [r for r in res_data if not (r.get("symbol") == symbol and r.get("session_num") == session_num)]
                        res_data.append(session_result)
                            
                        with open(res_file, "w") as f:
                            json.dump(res_data, f, indent=2)
                        
                    log.info(f"[{symbol}] 📝 ESTRATEGIA RESULTADO | {insight}")
                    # Limpiar estado activo para este símbolo
                    from shared.utils.state_writer import update_state
                    update_state(
                        mode=args.mode,
                        symbol=symbol,
                        status="completed",
                        session=session_num,
                        iteration=0,
                        bid=getattr(quote, 'bid', 0.0), 
                        ask=getattr(quote, 'ask', 0.0), 
                        signal="HOLD", 
                        rsi=50, 
                        ema_fast=0, ema_slow=0, ema_200=0, macd_hist=0, vwap=0, atr=0, confirmations=[],
                        initial_cash=account.total_cash, available_cash=account.available_cash, settlement=0,
                        win_rate=broker.stats.win_rate, total_trades=broker.stats.total_trades, winning_trades=broker.stats.winning_trades,
                        gross_profit=broker.stats.gross_profit, gross_loss=broker.stats.gross_loss,
                        sim_start=sim_start_date_str,
                        sim_end=sim_end_date_str
                    )
                except Exception as e:
                    log.error(f"Error guardando bitácora: {e}")
        else:
            final_info = broker.get_account_info()
            log.info(f"Estado final de cuenta: ${final_info.total_cash:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(BANNER)
    args = parse_args()
    set_state_file(config.STATE_FILE_SIM)
    set_assets_file(config.ASSETS_FILE_SIM)
    
    log.info(f"Modo seleccionado: {args.mode}")
    log.info(market_status_str())

    from shared.utils.state_writer import update_state
    update_state(mode=args.mode, status="initializing", symbol="─", session=0, iteration=0)

    is_simulated = args.mode == "SIMULATED"
    SESSION_PAUSE = 10   # segundos entre sesiones simuladas
    session_num   = 0

    import os, json
    all_symbols = []
    symbol_idx = 0

    # ── Restaurar checkpoint de simulación (si existe) ────────────────────────
    try:
        from shared.utils.checkpoint import load_simulation_checkpoint, save_simulation_checkpoint
        _ckpt = load_simulation_checkpoint()
        if _ckpt["symbol_idx"] > 0 and is_simulated:
            symbol_idx  = _ckpt["symbol_idx"]
            session_num = _ckpt["session_num"]
            log.info(
                f"💾 CHECKPOINT restaurado: reanudando desde '{_ckpt['symbol']}' (idx={symbol_idx}, sesión #{session_num})"
            )
        _checkpoint_fn = save_simulation_checkpoint
    except Exception as _ce:
        log.warning(f"No se pudo cargar checkpoint: {_ce}")
        _checkpoint_fn = None

    # Inicializar el primer símbolo secuencial si no se especificó uno fijo ni en args
    log.info("🚀 SISTEMA INICIADO: Preparando motores de trading...")

    while True:
        # ── 1. Recarga Dinámica de Símbolos ──────────────────────────────────
        try:
            importlib.reload(market_data)
            # Volver a setear el archivo correcto tras el reload
            set_assets_file(config.ASSETS_FILE_SIM)
            
            # Obtener objetos completos de activos para conocer su 'type'
            with open(config.ASSETS_FILE_SIM, "r") as f:
                _assets_data = json.load(f)
                all_assets = [a for a in _assets_data.get("assets", []) if a.get("enabled", True)]
                all_symbols = [a["symbol"] for a in all_assets]
        except Exception as e_reload:
            log.warning(f"⚠️ Error recargando símbolos: {e_reload}")
            all_symbols = []
            all_assets  = []

        # ── 2. Procesar Comandos Globales (Reinicios/Wipe) ────────────────────
        cmd_file = config.COMMAND_FILE
        if os.path.exists(cmd_file):
            try:
                import json
                with open(cmd_file, "r") as f:
                    cmds = json.load(f)
                
                # ── Sincronizar estado de congelación de IA ───────────────────────
                _ai_frozen = cmds.get("strategy_frozen", False)
                try:
                    from shared.utils.neural_filter import get_neural_filter
                    _nf = get_neural_filter()
                    if _ai_frozen and not _nf.is_frozen:
                        _nf.freeze()
                    elif not _ai_frozen and _nf.is_frozen:
                        _nf.unfreeze()
                except Exception: pass
                try:
                    from shared.strategy.ml_predictor import ml_predictor as _ml
                    if _ai_frozen and not _ml.is_frozen:
                        _ml.freeze()
                    elif not _ai_frozen and _ml.is_frozen:
                        _ml.unfreeze()
                except Exception: pass

                # ── Recargar modelos si se restauró un snapshot ───────────────────
                if cmds.get("reload_models"):
                    log.info("📸 Recargando modelos de IA desde snapshot restaurado...")
                    try:
                        from shared.utils.neural_filter import get_neural_filter
                        _nf = get_neural_filter()
                        _nf._load()   # Fuerza recarga desde disco
                        log.info("✅ NeuralFilter (MLP) recargado desde snapshot")
                    except Exception as e_nf:
                        log.warning(f"⚠️ No se pudo recargar NeuralFilter: {e_nf}")
                    try:
                        from shared.strategy.ml_predictor import ml_predictor as _ml
                        _ml._load()   # Fuerza recarga desde disco
                        log.info("✅ MLPredictor (RF) recargado desde snapshot")
                    except Exception as e_ml:
                        log.warning(f"⚠️ No se pudo recargar MLPredictor: {e_ml}")
                    # Limpiar la señal
                    cmds["reload_models"] = False
                    with open(cmd_file, "w") as f:
                        json.dump(cmds, f)
                if cmds.get("reset_all") or cmds.get("restart_sim"):
                    is_purgue = cmds.get("reset_all", False)
                    log.info(f"🔄 {'PURGANDO MEMORIA TOTAL' if is_purgue else 'REINICIANDO SIMULACIÓN'} por orden del usuario...")
                    
                    # Limpiar flags
                    cmds["reset_all"] = False
                    cmds["restart_sim"] = False
                    with open(cmd_file, "w") as f:
                        json.dump(cmds, f)
                    
                    from shared.utils.state_writer import clear_state
                    clear_state()
                    
                    # ── LIMPIEZA DE RESULTADOS (Siempre en ambos casos) ───────────
                    if config.RESULTS_FILE.exists(): 
                        log.info(f"🗑️ Eliminando resultados previos: {config.RESULTS_FILE.name}")
                        config.RESULTS_FILE.unlink()
                    if config.TRADE_JOURNAL_FILE.exists(): 
                        log.info(f"🗑️ Eliminando diario de trades: {config.TRADE_JOURNAL_FILE.name}")
                        config.TRADE_JOURNAL_FILE.unlink()
                    
                    # ── LIMPIEZA ADICIONAL (Solo en Reset Total / Wipe) ──────────
                    if is_purgue:
                        log.info("🧹 Limpieza profunda (Checkpoints + ML Model)...")
                        from shared.utils.checkpoint import clear_simulation_checkpoints
                        clear_simulation_checkpoints()
                        if config.NEURAL_MODEL_FILE.exists():
                            config.NEURAL_MODEL_FILE.unlink()
                    importlib.reload(market_data)
                    set_assets_file(config.ASSETS_FILE_SIM)
                    all_symbols = market_data.get_symbols()
                    symbol_idx = 0
                    session_num = 0
                    
                    # Limpiar memoria interna de estados de símbolos
                    try:
                        from shared.utils.state_writer import _symbol_states
                        _symbol_states.clear()
                    except: pass

                    from shared.utils.state_writer import update_state
                    from shared.utils.market_hours import _is_mock_time_active
                    update_state(mode="SIMULATED", status="restarting", symbol="─", session=0, iteration=0, mock_time_930=_is_mock_time_active())
                    continue # Saltar al inicio con el nuevo estado
            except Exception as e_cmd:
                log.error(f"Error procesando comandos: {e_cmd}")

        # ── 3. Verificar Fin de Exploración ──────────────────────────────────
        if is_simulated and all_symbols and symbol_idx >= len(all_symbols):
            # Verificar si se han añadido nuevos símbolos desde el Dashboard
            importlib.reload(market_data)
            set_assets_file(config.ASSETS_FILE_SIM)
            with open(config.ASSETS_FILE_SIM, "r") as f:
                _new_data = json.load(f)
                current_assets = [a for a in _new_data.get("assets", []) if a.get("enabled", True)]
                current_list = [a["symbol"] for a in current_assets]

            if len(current_list) > len(all_symbols):
                log.info(f"✨ ¡Nuevos símbolos detectados ({len(current_list)})! Reanudando...")
                all_assets = current_assets
                all_symbols = current_list
            else:
                from shared.utils.state_writer import update_state
                update_state(mode="SIMULATED", status="completed", symbol="─", session=session_num, iteration=0)
                smart_sleep(5)
                continue # Volver a chequear comandos/símbolos

        # ── 4. Preparar Símbolo para esta Sesión ───────────────────────────────
        if is_simulated and all_symbols:
            args.symbol = all_symbols[symbol_idx]
        
        session_num += 1

        # ── Detectar si hay Live Paper ANTES de inicializar broker ──────────
        # Esto evita crear un broker de MarketReplay innecesario cuando
        # los hilos paralelos crearán sus propios brokers con LivePaperReplay.
        force_symbols = []
        is_lp = False
        try:
            if config.COMMAND_FILE.exists():
                with open(config.COMMAND_FILE) as f:
                    cmds_pre = json.load(f)
                    force_symbols = cmds_pre.get("force_symbols", [])
                    is_lp = cmds_pre.get("force_paper_trading", False)
        except: pass

        if is_lp and len(force_symbols) >= 1:
            log.info(f"🚀 Modo Live Paper detectado para {len(force_symbols)} símbolo(s). Lanzando hilos paralelos…")
            from shared.utils.live_paper_launcher import launch_parallel_bots
            launch_parallel_bots(args, force_symbols, session_num, run_bot, init_broker, _checkpoint_fn)
        else:
            # ── FLUJO SINGLE-THREAD NORMAL (Simulación) ──────────────────────
            if is_simulated:
                log.info(f"{'═'*60}")
                log.info(f"  🔁 SESIÓN DE SIMULACIÓN #{session_num} | Activo: {args.symbol}")
                log.info(f"{'═'*60}")

            try:
                # Limpiar estados previos para evitar "fantasmas" paralelos en el dashboard (solo en serial)
                try:
                    from shared.utils.state_writer import clear_symbol_states
                    clear_symbol_states()
                except: pass

                broker = init_broker(args)
                
                # Obtener tipo de activo (normal por defecto)
                current_asset_type = "normal"
                if all_assets and symbol_idx < len(all_assets):
                    current_asset_type = all_assets[symbol_idx].get("type", "normal")
                
                run_bot(broker, args, session_num, asset_type=current_asset_type)
            except SessionInterrupted:
                log.info("📢 Sesión interrumpida por comando superior. Reiniciando bucle...")
                continue # Re-evaluar comandos globales de inmediato
            except Exception as e:
                log.error(f"❌ Error en sesión para {args.symbol}: {e}")
                try:
                    from shared.utils.state_writer import update_state
                    update_state(mode=args.mode, status="error", symbol=args.symbol, session=session_num, insight=f"Error: {e}")
                except: pass

        # Ssalir si no es simulación (Live Mode finalizado)
        if not is_simulated:
            break

        # ── 5. Post-Sesión: Checkpoint y Siguiente Activo ────────────────────
        if is_simulated and all_symbols:
            symbol_idx += 1
            # Guardar checkpoint para reanudar si el bot se apaga
            if _checkpoint_fn:
                try:
                    _checkpoint_fn(
                        symbol_idx, 
                        all_symbols[symbol_idx] if symbol_idx < len(all_symbols) else "─", 
                        session_num
                    )
                except: pass
            
            if symbol_idx < len(all_symbols):
                log.info(f"⏸ Sesión #{session_num} finalizada. Siguiente: {all_symbols[symbol_idx]} en {SESSION_PAUSE}s...")
            else:
                log.info(f"🏁 Simulación COMPLETA. Sesión #{session_num} cerrada.")
        
        smart_sleep(SESSION_PAUSE)


if __name__ == "__main__":
    main()
