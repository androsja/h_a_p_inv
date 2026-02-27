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

import config
from utils.logger import log, log_order_attempt, log_market_closed, set_symbol_log
from utils.market_hours import is_market_open, market_status_str, next_open_str
from utils import state_writer
from broker.interface import BrokerInterface, Quote, OrderResponse
from broker.hapi_live import HapiLive
from broker.hapi_mock import HapiMock
from strategy.indicators import analyze, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from strategy.risk_manager import RiskManager, AccountState, OpenPosition
from strategy.ml_predictor import ml_predictor
from utils.trade_journal import record_trade as journal_record_trade
from utils.trade_journal import record_trade as journal_record_trade
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
#  HELPERS PARA RESPONSIVIDAD
# ═══════════════════════════════════════════════════════════════════════════════
def smart_sleep(seconds: float):
    """Sleep interrumpible que revisa cambios de comando cada segundo."""
    cmd_file = "/app/data/command.json"
    start_time = time.time()
    while (time.time() - start_time) < seconds:
        try:
            if os.path.exists(cmd_file):
                with open(cmd_file, "r") as f:
                    cmds = json.load(f)
                if cmds.get("reset_all") or cmds.get("force_paper_trading") is not None:
                    # Si hubo un cambio de modo o reset, salimos del sleep para procesar
                    break
        except: pass
        time.sleep(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE ARGUMENTOS CLI
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bot de trading algorítmico para Hapi Trade"
    )
    parser.add_argument(
        "--mode",
        choices=["LIVE", "SIMULATED"],
        default=config.TRADING_MODE,
        help="Modo de operación: LIVE (dinero real) o SIMULATED (yfinance)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Símbolo específico para simular (ej. AAPL). También se lee de FIXED_SYMBOL en .env.",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="Capital inicial para el modo SIMULATED (default: $10,000)",
    )
    p = parser.parse_args()
    # FIXED_SYMBOL en .env tiene prioridad sobre el flag --symbol
    import os
    fixed = os.getenv("FIXED_SYMBOL", "").strip()
    if fixed and not p.symbol:
        p.symbol = fixed
        log.info(f"🔒 Símbolo fijo configurado: {fixed} (FIXED_SYMBOL en .env)")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN DEL BRÓKER
# ═══════════════════════════════════════════════════════════════════════════════
def init_broker(args: argparse.Namespace) -> BrokerInterface:
    """
    Crea e inicializa el bróker correcto según el modo de operación.
    Si las credenciales de Hapi están vacías, cambia automáticamente
    al modo simulado con una advertencia.
    """
    if args.mode == "LIVE":
        # Verificar credenciales en .env
        api_key    = config.HAPI_API_KEY
        client_id  = config.HAPI_CLIENT_ID
        user_token = config.HAPI_USER_TOKEN

        if not all([api_key, client_id, user_token]):
            log.warning(
                "⚠️  Credenciales de Hapi no encontradas en .env. "
                "Solicitando interactivamente…"
            )
            print("\n📋 Ingresa tus credenciales de Hapi Trade:")
            api_key    = input("   API_KEY    : ").strip()
            client_id  = input("   CLIENT_ID  : ").strip()
            user_token = getpass.getpass("   USER_TOKEN : ").strip()

        # Si aún están vacías, caer a modo simulado
        if not all([api_key, client_id, user_token]):
            log.warning(
                "❌ Credenciales vacías. Cambiando automáticamente a modo SIMULATED."
            )
            return HapiMock(symbol=args.symbol, initial_cash=args.cash)

        return HapiLive(api_key=api_key, client_id=client_id, user_token=user_token)

    # ── Modo SIMULATED / LIVE PAPER ──────────────────────────────────────────
    is_live_paper = False
    force_symbols = []
    try:
        import json, os
        cmd_file = "/app/data/command.json"
        if os.path.exists(cmd_file):
            with open(cmd_file) as f:
                cmds = json.load(f)
            is_live_paper = cmds.get("force_paper_trading", False)
            force_symbols = cmds.get("force_symbols", [])
            force_symbol_val = cmds.get("force_symbol", "")
            
            if is_live_paper:
                if force_symbols and not (len(force_symbols) == 1 and force_symbols[0] == "AUTO"):
                    # If we have a list, use the current active one from args or fallback to first
                    if not args.symbol or args.symbol not in force_symbols:
                        args.symbol = force_symbols[0]
                elif force_symbol_val and force_symbol_val != "AUTO":
                    args.symbol = force_symbol_val
    except Exception:
        pass

    return HapiMock(symbol=args.symbol, initial_cash=args.cash, live_paper=is_live_paper)


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
    print(
        f"\r{mode_label} | {symbol} | "
        f"bid=${quote.bid:.2f} ask=${quote.ask:.2f} | "
        f"Señal={signal_str:4s} | "
        f"{pos_str} | "
        f"Cash disponible=${account.available_cash:.2f}",
        end="",
        flush=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL DEL BOT
# ═══════════════════════════════════════════════════════════════════════════════
def run_bot(broker: BrokerInterface, args: argparse.Namespace, session_num: int = 1, stop_event: threading.Event = None) -> None:
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
            except StopIteration:
                log.info("▶ Datos de simulación agotados. Finalizando sesión.")
                break
            except ConnectionError as exc:
                log.error(f"Connection error: {exc}. Reintentando en 5s…")
                time.sleep(5)
                continue

            # ── 3. Obtener datos históricos para indicadores ─────────────────
            # 220 velas = EMA 200 + 20 de margen para estabilización
            # ── 3. Obtener datos históricos para indicadores ─────────────────
            # 220 velas = EMA 200 + 20 de margen para estabilización
            if is_mock:
                df = broker.get_history_slice(window=220)
                
                # REVISAR COMANDOS DESDE EL DASHBOARD (Instántaneo)
                try:
                    cmd_file = "/app/data/command.json"
                    if os.path.exists(cmd_file):
                        with open(cmd_file, "r") as f:
                            cmds = json.load(f)
                            
                        if cmds.get("reset_all") or cmds.get("force_paper_trading") is False:
                            log.info("🔄 Interrumpiendo simulación: Señal de RESET o STOP recibida.")
                            return # Salir de run_bot para procesar en main()

                        new_sym = cmds.get("force_symbol")
                        if new_sym and new_sym != "AUTO" and new_sym != symbol:
                            log.info(f"🔄 Interrumpiendo simulación. Usuario solicita cambiar a {new_sym}")
                            break # Salir de este loop para reiniciar sesión con el nuevo
                except: pass
            else:
                from data.market_data import download_bars
                df = download_bars(symbol)

            signal = analyze(df)

            # ── 4. Gestionar SALIDA (si hay posición abierta) ────────────────
            if position is not None:
                position.hold_bars = getattr(position, "hold_bars", 0) + 1
                should_exit, exit_reason = risk_mgr.should_exit(position, quote.bid)

                # Eliminamos la regla de "panic sell" del indicador, dejamos que el Trailing Stop decida
                if should_exit:
                    reason = exit_reason
                    sell_plan = risk_mgr.calculate_sell_order(position, quote.bid)

                    log_order_attempt(
                        symbol, "SELL",
                        sell_plan.limit_price, sell_plan.qty, reason
                    )
                    resp = broker.place_limit_order(
                        symbol=symbol,
                        side="SELL",
                        limit_price=sell_plan.limit_price,
                        qty=sell_plan.qty,
                    )
                    if resp.status not in ("REJECTED",):
                        pnl = (sell_plan.limit_price - position.entry_price) * position.qty
                        log.info(
                            f"🔴 SELL | {symbol} → ${sell_plan.limit_price:.2f} | "
                            f"PnL estimado: ${pnl:+.2f} | Motivo: {reason}"
                        )
                        
                        # 🧠 Alimentar al modelo de ML con el resultado real
                        if hasattr(position, 'ml_features') and position.ml_features:
                            log.info(f"SAVING TRADE: {symbol} Pnl={pnl} Features={position.ml_features}")
                            ml_predictor.save_trade(symbol, position.ml_features, pnl)
                            # 📓 Bitácora completa de trading (para análisis y mejora de algoritmos)
                            journal_record_trade(
                                symbol=symbol,
                                session=session_num,
                                entry_price=position.entry_price,
                                exit_price=sell_plan.limit_price,
                                qty=position.qty,
                                stop_loss=position.initial_stop,
                                take_profit=position.take_profit,
                                hold_bars=getattr(position, 'hold_bars', 0),
                                exit_reason=reason.split(' ')[0].replace('🔒','TRAILING_STOP').replace('��','STOP_LOSS').replace('🎯','TAKE_PROFIT').replace('⏰','TIME_EXIT').replace('🛡️','FORCED_SELL'),
                                ml_features=position.ml_features,
                            )
                            
                        # En LIVE registrar el capital como T+1 pendiente
                        if not is_mock:
                            account.record_sale(sell_plan.limit_price * sell_plan.qty)
                        position = None

            # ── 5. Gestionar ENTRADA (si no hay posición y hay señal BUY) ────
            elif signal.signal == SIGNAL_BUY:
                # En modo simulado, sincronizar el cash real desde el MockBroker
                if is_mock:
                    live_info = broker.get_account_info()
                    account.total_cash = live_info.total_cash

                buy_plan = risk_mgr.calculate_buy_order(
                    symbol, quote.ask,
                    atr_value=signal.atr_value,   # SL dinámico con ATR
                )

                # 🧠 CONSULTAR AL MODELO DE MACHINE LEARNING ANTES DE ENTRAR
                if buy_plan.is_viable:
                    is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
                    if not is_win:
                        buy_plan.is_viable = False
                        buy_plan.block_reason = f"ML_REJECTION: El robot recuerda haber perdido con estos parámetros (Probabilidad de Ganar: {prob_win*100:.1f}%)"
                        log.info(f"🧠 ML EVITÓ PERDER DINERO | Bloqueó entrada en {symbol} | Prob={prob_win*100:.1f}%")

                if buy_plan.is_viable:
                    log_order_attempt(
                        symbol, "BUY",
                        buy_plan.limit_price, buy_plan.qty,
                        f"confluencias={len(signal.confirmations)} | "
                        f"RSI={signal.rsi_value:.1f} | MACD={signal.macd_hist:.4f}"
                    )
                    resp = broker.place_limit_order(
                        symbol=symbol,
                        side="BUY",
                        limit_price=buy_plan.limit_price,
                        qty=buy_plan.qty,
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
            )

            # ── 7. Display de estado (cada 10 iteraciones en simulación) ─────
            if iteration % 10 == 0 or not is_mock:
                print_status(broker, symbol, quote, signal.signal, position, account)

            # ── 8. Pausa entre escaneos (Live y Live Paper) ──────────────────
            is_live_paper_b = getattr(broker, '_live_paper', False)
            if not is_mock or is_live_paper_b:
                pause_time = 60 if is_live_paper_b else config.SCAN_INTERVAL_SEC
                log.info(f"⏳ Esperando pausa de {pause_time}s (interrumpible)...")
                
                for i in range(pause_time, 0, -1):
                    # Actualizar estado con la cuenta regresiva
                    from utils.state_writer import update_state
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
                        is_waiting=True
                    )
                    
        if not (stop_event and stop_event.is_set()):
            smart_sleep(1)
            # Verificar si llegó una orden de parar o resetear durante la pausa
            try:
                if os.path.exists("/app/data/command.json"):
                    with open("/app/data/command.json", "r") as f:
                        c = json.load(f)
                    if c.get("reset_all") or c.get("force_paper_trading") is False:
                        log.info(f"🛑 Interrupción detectada para {symbol}.")
                        return
            except: pass

    except KeyboardInterrupt:
        print("\n")
        log.info("🛑 Bot detenido manualmente (KeyboardInterrupt).")

    finally:
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
                cmd_file = "/app/data/command.json"
                if os.path.exists(cmd_file):
                    with open(cmd_file, "r") as f:
                        if json.load(f).get("reset_all"):
                            log.info("Cancelando guardado de sesión debido a Wipe Memory.")
                            should_save = False
            except Exception:
                pass
                
            # Guardar en Bitácora para ML y Análisis de Resultados
            if should_save:
                try:
                    import json
                    from datetime import datetime
                    from pathlib import Path
                    res_file = Path("/app/data/backtest_results.json")
                    if not getattr(broker, 'total_trades', 0): pass
                    res_data = []
                    if res_file.exists():
                        with open(res_file, "r") as f:
                            res_data = json.load(f)
                    
                    # ── Generar un Diagnóstico Rápido de la Estrategia (Insight) ──
                    pnl_val = getattr(broker.stats, 'total_pnl', 0.0)
                    trades_val = getattr(broker.stats, 'total_trades', 0)
                    winrate_val = getattr(broker.stats, 'win_rate', 0.0)
                    
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
                    
                    from utils.state_writer import _symbol_states
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
                        "insight":        insight,
                        "regime":         regime_val
                    }
                    
                    # Deduplicar: Filtramos y removemos TODOS los registros anteriores de este símbolo
                    res_data = [r for r in res_data if r.get("symbol") != symbol]
                    res_data.append(session_result)
                        
                    with open(res_file, "w") as f:
                        json.dump(res_data, f, indent=2)
                        
                    log.info(f"📝 ESTRATEGIA RESULTADO | {symbol}: {insight}")
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

    log.info(f"Modo seleccionado: {args.mode}")
    log.info(market_status_str())

    is_simulated = args.mode == "SIMULATED"
    SESSION_PAUSE = 10   # segundos entre sesiones simuladas
    session_num   = 0

    import os, json
    try:
        from data.market_data import get_symbols
        all_symbols = get_symbols()
    except Exception:
        all_symbols = []
    symbol_idx = 0

    # Inicializar el primer símbolo secuencial si no se especificó uno fijo ni en args
    if is_simulated and not args.symbol and not os.getenv("FIXED_SYMBOL", "").strip():
        if all_symbols:
            args.symbol = all_symbols[symbol_idx]

    while True:
        session_num += 1
        if is_simulated:
            log.info(f"{'═'*60}")
            log.info(f"  🔁 SESIÓN DE SIMULACIÓN #{session_num}")
            log.info(f"{'═'*60}")

        try:
            broker = init_broker(args)
            log.info(
                f"Bróker inicializado: {broker.name} | "
                f"Paper trading: {broker.is_paper_trading}"
            )
        except Exception as e:
            log.error(f"❌ Error al inicializar bróker para {args.symbol}: {e}. Saltando de activo.")
            
            # Registrar en dashboard que este activo se evaluó pero falló/no tiene datos
            try:
                from utils.state_writer import update_state
                from datetime import datetime, timezone
                update_state(
                    mode=args.mode,
                    status="error_skipping",
                    symbol=args.symbol or "UNKNOWN",
                    session=session_num,
                    iteration=0,
                    available_cash=getattr(broker, 'initial_cash', 10000.0) if 'broker' in locals() else 10000.0,
                    pnl=0.0,
                    win_rate=0.0,
                    total_trades=0,
                    insight="Sin datos suficientes de mercado."
                )

                res_file = Path("/app/data/backtest_results.json")
                res_data = []
                if res_file.exists():
                    with open(res_file, "r") as f:
                        res_data = json.load(f)
                
                # Filtrar deduplicado
                res_data = [r for r in res_data if r.get("symbol") != args.symbol]
                res_data.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "symbol": args.symbol,
                    "session_num": session_num,
                    "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                    "win_rate": 0.0, "pnl": 0.0, "gross_pnl": 0.0, "total_fees": 0.0,
                    "slippage_est": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
                    "profit_factor": 0.0, "drawdown": 0.0,
                    "insight": "Sin datos suficientes (Omitido)."
                })
                with open(res_file, "w") as f:
                    json.dump(res_data, f, indent=2)
            except Exception as e_dash:
                log.error(f"Error actualizando dashboard para símbolo omitido: {e_dash}")

            if is_simulated and all_symbols:
                symbol_idx += 1
                if symbol_idx >= len(all_symbols):
                    log.info("🎯 Exploración de Símbolos COMPLETA. Esperando nueva orden (WIPE MEMORY)...")
                    symbol_idx = len(all_symbols) - 1 # Mantenerse al final
                    smart_sleep(SESSION_PAUSE)
                    continue
                args.symbol = all_symbols[symbol_idx]
            else:
                 smart_sleep(SESSION_PAUSE)
            continue

        # Detectamos si hay múltiples símbolos para Live Paper
        force_symbols = []
        is_lp = False
        try:
            if os.path.exists("/app/data/command.json"):
                with open("/app/data/command.json") as f:
                    cmds = json.load(f)
                    force_symbols = cmds.get("force_symbols", [])
                    is_lp = cmds.get("force_paper_trading", False)
        except: pass

        if is_lp and len(force_symbols) > 1:
            log.info(f"🚀 Iniciando monitoreo PARALELO para {len(force_symbols)} símbolos...")
            threads = []
            stop_event = threading.Event()
            
            for sym in force_symbols:
                # Crear una copia de args para cada hilo con su símbolo
                thread_args = argparse.Namespace(**vars(args))
                thread_args.symbol = sym
                # Cada hilo necesita su propia instancia de broker (HapiMock es ligero)
                thread_broker = init_broker(thread_args)
                
                t = threading.Thread(
                    target=run_bot, 
                    args=(thread_broker, thread_args, session_num, stop_event),
                    name=f"Worker-{sym}"
                )
                t.daemon = True
                t.start()
                threads.append(t)
            
            # El hilo principal espera y vigila command.json
            try:
                while True:
                    time.sleep(2)
                    if os.path.exists("/app/data/command.json"):
                        with open("/app/data/command.json") as f:
                            c = json.load(f)
                        if c.get("reset_all") or c.get("force_paper_trading") is False:
                            log.info("🛑 Deteniendo todos los hilos paralelos...")
                            stop_event.set()
                            break
            except KeyboardInterrupt:
                stop_event.set()
            
            for t in threads:
                t.join(timeout=5)
        else:
            run_bot(broker, args, session_num)

        # En modo LIVE el bot solo llega aquí por KeyboardInterrupt o señal externa → salir
        if not is_simulated:
            break

        # En modo SIMULATED: descansar y reiniciar
        fixed = os.getenv("FIXED_SYMBOL", "").strip()
        
        cmd_file = "/app/data/command.json"
        if os.path.exists(cmd_file):
            try:
                with open(cmd_file, "r") as f:
                    cmds = json.load(f)
                
                # REINICIO GLOBAL (RESET ALL MEMORY)
                if cmds.get("reset_all"):
                    log.info("💥 PURGANDO MEMORIA COMPLETA del bot por orden del usuario...")
                    cmds["reset_all"] = False # Consume flag
                    with open(cmd_file, "w") as f:
                        json.dump(cmds, f)
                    
                    # Wipe global state for fresh start 
                    global _trades, final_pnl_global, win_pct, num_trades
                    try:
                        from utils.state_writer import _trades as state_trades
                        state_trades.clear()
                    except: pass
                    
                    try:
                        import os
                        for wipe_target in ["/app/data/backtest_results.json", "/app/data/state.json", "/app/data/ml_dataset.csv"]:
                            if os.path.exists(wipe_target):
                                os.remove(wipe_target)
                    except: pass
                    
                    symbol_idx = 0
                    session_num = 0
                    fixed = ""
                    log.info("✅ Memoria limpiada y archivos borrados de disco. Arrancando nuevo set desde 0%.")
                    # Update state immediately to reflect the reset
                    from utils.state_writer import update_state
                    update_state(mode="SIMULATED", status="restarting", symbol="─", session=0, iteration=0)

                new_sym = cmds.get("force_symbol")
                if new_sym and new_sym != "AUTO":
                    fixed = new_sym
                elif new_sym == "AUTO":
                    fixed = "" # rotar
            except: pass

        if fixed:
            # Overwrite arguments flag to ensure it passes the overridden fixed symbol directly into init_broker next loop
            args.symbol = fixed
            log.info(
                f"⏸  Sesión #{session_num} finalizada. "
                f"Reiniciando en {SESSION_PAUSE}s con {fixed} (símbolo fijo manual/env)…"
            )
        else:
            # Check for force_symbols for multi-symbol live paper
            try:
                cmd_file = "/app/data/command.json"
                if os.path.exists(cmd_file):
                    with open(cmd_file, "r") as f:
                        cmds = json.load(f)
                    force_symbols = cmds.get("force_symbols", [])
                    if force_symbols:
                        is_lp = cmds.get("force_paper_trading", False)
                        # Find current symbol index in force_symbols and pick next
                        try:
                            curr_idx = force_symbols.index(args.symbol)
                            next_idx = (curr_idx + 1) % len(force_symbols)
                            args.symbol = force_symbols[next_idx]
                            
                            # Si es Live Paper, rotación rápida entre símbolos, pero pausa al final de la vuelta
                            if is_lp:
                                if next_idx == 0:
                                    log.info(f"🏁 Vuelta completa de escaneo finalizada ({len(force_symbols)} símbolos).")
                                    smart_sleep(60)
                                else:
                                    smart_sleep(1) # Rotación rápida
                                continue
                        except ValueError:
                            args.symbol = force_symbols[0]
                        
                        log.info(f"⏸  Sesión #{session_num} finalizada. Rotando a {args.symbol} del set Live Paper…")
                        smart_sleep(SESSION_PAUSE)
                        continue
            except Exception as e: 
                log.debug(f"Error rotation: {e}")
                pass

            if all_symbols:
                symbol_idx += 1
                if symbol_idx >= len(all_symbols):
                    log.info("🎯 Exploración de Símbolos COMPLETA. Esperando nueva orden (WIPE MEMORY)...")
                    while True:
                        time.sleep(10)
                        
                        # Monitor command json to allow escape
                        try:
                            if os.path.exists("/app/data/command.json"):
                                with open("/app/data/command.json") as f:
                                    if json.load(f).get("reset_all"):
                                        break # escape and let the main loop wipe
                        except: pass
                        
                    symbol_idx = len(all_symbols) - 1 # Mantenerse al final
                    continue

                args.symbol = all_symbols[symbol_idx]
                log.info(
                    f"⏸  Sesión #{session_num} finalizada. "
                    f"Reiniciando en {SESSION_PAUSE}s con el siguiente símbolo secuencial: {args.symbol}…"
                )
            else:
                args.symbol = None # auto-rotate (fallback a random)
                log.info(
                    f"⏸  Sesión #{session_num} finalizada. "
                    f"Reiniciando en {SESSION_PAUSE}s con un símbolo aleatorio nuevo…"
                )
        smart_sleep(SESSION_PAUSE)


if __name__ == "__main__":
    main()
