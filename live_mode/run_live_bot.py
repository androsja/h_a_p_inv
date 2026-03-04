"""
live_mode/run_live_bot.py ─ Motor de trading en vivo con Alpaca Paper.

Este archivo es el ÚNICO lugar donde corre el bot de Live/Paper.
NO modifica, NO importa, NO toca nada de simulation_mode/.

Reutiliza (solo lectura) los módulos shared/:
  - shared/strategy/indicators.py  → señales técnicas
  - shared/strategy/risk_manager.py → SL/TP/tamaño de posición
  - shared/strategy/ml_predictor.py → predicción RF
  - shared/utils/neural_filter.py  → filtro MLP
  - shared/utils/state_writer.py   → escribe en state_live.json ÚNICAMENTE

Cada símbolo corre en su propio thread.
"""

import time
import json
import threading
import collections
from datetime import datetime, timezone
from pathlib import Path

from shared import config
from shared.utils.logger import log
from shared.utils.market_hours import is_market_open, market_status_str, next_open_str
from shared.strategy.indicators import analyze, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from shared.strategy.risk_manager import RiskManager, AccountState, OpenPosition
from shared.strategy.ml_predictor import ml_predictor
from shared.utils.trade_journal import record_trade as journal_record_trade
from shared.utils import state_writer


# ─── Lock global para escritura de estado del live ────────────────────────────
_live_state_lock = threading.Lock()
_live_results: dict = {}   # {symbol: {pnl, trades, ...}}


def _read_live_cmd(key: str, default=None):
    """Lee un comando del command.json ─ solo claves prefijadas con 'live_'."""
    try:
        if config.COMMAND_FILE.exists():
            with open(config.COMMAND_FILE) as f:
                data = json.load(f)
            return data.get(key, default)
    except Exception:
        pass
    return default


def run_symbol_live(
    symbol: str,
    api_key: str,
    secret_key: str,
    initial_cash: float = 10_000.0,
    stop_event: threading.Event = None,
    session_num: int = 1,
):
    """
    Bucle principal del bot en vivo para UN símbolo.
    Diseñado para correr como hilo independiente.

    Flujo por iteración:
      1. Verificar horario NYSE
      2. Obtener barras reales de Alpaca (últimas 220 velas de 5m)
      3. analyze() → señal técnica (IGUAL que simulación, sin tocarla)
      4. ml_predictor.predict_win() → filtro ML (solo lectura, sin fit)
      5. neural_filter.predict() → filtro MLP (solo lectura)
      6. Si BUY → place_limit_order() en Alpaca Paper
      7. Si posición abierta → verificar SL/TP → SELL
      8. Actualizar state_live.json (NUNCA state_sim.json)
      9. Dormir 60s
    """
    from shared.broker.alpaca_paper import AlpacaPaperBroker
    from shared.utils.neural_filter import get_neural_filter

    log.info(f"[LIVE:{symbol}] 🚀 Hilo iniciado")

    broker = AlpacaPaperBroker(
        symbol=symbol,
        api_key=api_key,
        secret_key=secret_key,
    )

    # ── Inicializar cuenta y gestión de riesgo ────────────────────────────────
    # Usamos el balance forzado configurado por el usuario ($25,000)
    initial_cash = config.INITIAL_CASH_LIVE
    
    # Comentamos la sincronización con el bróker para permitir al usuario simular el balance que desee
    # try:
    #     acct_info = broker.get_account_info()
    #     initial_cash = acct_info.total_cash
    # except Exception:
    #     pass

    account  = AccountState(total_cash=initial_cash)
    risk_mgr = RiskManager(account)
    position: OpenPosition | None = None
    nf = get_neural_filter()

    blocking_history = collections.Counter()
    iteration = 0

    log.info(f"[LIVE:{symbol}] 💰 Capital inicial: ${initial_cash:,.2f}")
    log.info(f"[LIVE:{symbol}] EMA({config.EMA_FAST}/{config.EMA_SLOW}) | RSI({config.RSI_PERIOD})")

    try:
        while not (stop_event and stop_event.is_set()):
            iteration += 1

            # ── 1. Verificar horario de mercado ───────────────────────────────
            if not is_market_open():
                log.debug(f"[LIVE:{symbol}] 🔴 Mercado cerrado. {next_open_str()}")
                # Escribir estado de espera
                _write_live_state(symbol, session_num, iteration, account,
                                  broker, position, None, "MARKET_CLOSED")
                _smart_sleep(60, stop_event)
                continue

            # ── 2. Obtener barras reales desde Alpaca ────────────────────────
            try:
                df = broker.get_bars_df(symbol, limit=220)
            except Exception as e:
                log.warning(f"[LIVE:{symbol}] ⚠️ Error obteniendo barras: {e}. Reintentando en 30s.")
                _smart_sleep(30, stop_event)
                continue

            # ── 3. Analizar señal técnica (módulo shared compartido) ──────────
            signal = analyze(df, symbol=symbol)

            if signal.signal == SIGNAL_HOLD and signal.blocks:
                for b in signal.blocks:
                    blocking_history[b.split(":")[0].strip()] += 1

            # ── 4. Obtener cotización actual (REAL TIME) ──────────────────────
            try:
                # Usamos exclusivamente Alpaca para garantizar TIEMPO REAL (0 retraso)
                quote = broker.get_quote(symbol)
            except Exception as e:
                log.error(f"[LIVE:{symbol}] ❌ Error obteniendo precio en tiempo real (Alpaca): {e}")
                _smart_sleep(20, stop_event)
                continue

            # ── 5. Gestionar SALIDA (posición abierta) ────────────────────────
            if position is not None:
                position.hold_bars = getattr(position, "hold_bars", 0) + 1
                should_exit, exit_reason = risk_mgr.should_exit(position, quote.bid)

                if should_exit:
                    sell_price = quote.bid
                    resp = broker.place_market_order(
                        symbol=symbol, side="SELL",
                        qty=position.qty, reason=exit_reason,
                    )

                    if resp.status not in ("REJECTED",):
                        actual_sell = resp.fill_price or sell_price
                        
                        gross_sale = actual_sell * position.qty
                        # --- Hapi / IBKR Fees (simulados para ser realistas en el Live) ---
                        comm = max(0.35, min(0.01 * gross_sale, 0.0035 * position.qty))
                        sec_fee = max(0.01, round(gross_sale * 0.0000278, 2))
                        taf_fee = min(5.95, round(position.qty * 0.000166, 2))
                        if taf_fee < 0.01: taf_fee = 0.01
                        total_fees = comm + sec_fee + taf_fee
                        
                        expected_sell = quote.bid
                        slippage_cost = max(0.0, (expected_sell - actual_sell) * position.qty)
                        
                        raw_pnl = (actual_sell - position.entry_price) * position.qty
                        net_pnl = raw_pnl - total_fees
                        
                        # Guardar estadísticas en la memoria temporal del Broker
                        if hasattr(broker, 'update_stats'): # Compatibilidad segura
                            broker.update_stats(pnl=net_pnl, fees=total_fees, slippage=slippage_cost)
                        
                        # Registro en la UI (JSON state)
                        from shared.utils.state_writer import record_trade
                        record_trade(
                            symbol=symbol,
                            side="SELL",
                            price=actual_sell,
                            qty=position.qty,
                            pnl=net_pnl,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            reason=exit_reason,
                            metadata={
                                "entry_price": position.entry_price,
                                "confirmations": [], # Podríamos pasar las originales
                                "conf_mult": 1.0,
                                "fees": total_fees
                            }
                        )

                        # Actualizar el capital real de la cuenta para que el Dashboard lo refleje
                        account.total_cash += net_pnl

                        # Registro en bitácora (CSV persistente)
                        try:
                            journal_record_trade(
                                symbol=symbol,
                                session=session_num,
                                entry_price=position.entry_price,
                                exit_price=actual_sell,
                                qty=position.qty,
                                stop_loss=position.initial_stop,
                                take_profit=position.take_profit,
                                hold_bars=getattr(position, "hold_bars", 0),
                                exit_reason=exit_reason.split()[0],
                                ml_features=position.ml_features or {},
                            )
                        except Exception as je:
                            log.warning(f"[LIVE:{symbol}] journal error: {je}")

                        log.info(
                            f"[LIVE:{symbol}] 🔴 SELL @ ${actual_sell:.2f} | "
                            f"PnL: ${net_pnl:+.2f} | Fees: ${total_fees:.2f} | "
                            f"Slippage: ${slippage_cost:.2f} | "
                            f"Motivo: {exit_reason}"
                        )

                        position = None
                        account.open_position = None

            # ── 6. Gestionar ENTRADA ──────────────────────────────────────────
            elif signal.signal == SIGNAL_BUY:
                
                # REGLA 1: Evitar el Sesgo de Look-Ahead (Solo operar cuando la vela cierre exactamente)
                now = datetime.now(config.NY_TZ)
                if now.minute % 5 != 0 or now.second > 15:
                    msg = "⏱️ Esperando cierre de vela (Look-Ahead Prevention)"
                    signal.blocks = [msg]
                    signal.signal = SIGNAL_HOLD
                    blocking_history[msg.split("(")[0].strip()] += 1
                    continue
                
                # REGLA 2: Liquidez / SLIPPAGE PREVENTIVO
                spread = quote.ask - quote.bid
                max_spread_allowed = quote.bid * 0.0020 # 0.2% max spread
                if spread > max_spread_allowed:
                    msg = f"📉 Spread Tóxico ({spread:.2f} > {max_spread_allowed:.2f}) | Slippage risk"
                    signal.blocks = [msg]
                    signal.signal = SIGNAL_HOLD
                    blocking_history[msg.split("(")[0].strip()] += 1
                    continue

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

                buy_plan = risk_mgr.calculate_buy_order(
                    symbol, quote.ask,
                    atr_value=signal.atr_value,
                    confidence_multiplier=conf_mult
                )

                # ─ ML Tolerance Threshold ─
                if not is_win and prob_win < 0.48 and buy_plan.is_viable:
                    buy_plan.is_viable = False
                    msg = f"🧠 AI bloqueó entrada | MLP P={prob_win*100:.1f}%"
                    buy_plan.block_reason = msg
                    signal.signal = SIGNAL_HOLD
                    signal.blocks = [msg]
                    log.info(f"[LIVE:{symbol}] {msg}")


                # ─ Consultar Neural Filter (solo predict) ─
                if buy_plan.is_viable:
                    try:
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
                        nf_win, nf_prob = nf.predict(fv)
                        if not nf_win:
                            buy_plan.is_viable = False
                            msg = f"🧠 Red Neuronal bloqueó | P={nf_prob*100:.1f}%"
                            buy_plan.block_reason = msg
                            signal.signal = SIGNAL_HOLD
                            signal.blocks = [msg]
                            log.info(f"[LIVE:{symbol}] {msg}")
                    except Exception as nfe:
                        log.debug(f"[LIVE:{symbol}] NF error: {nfe}")

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

                    resp = broker.place_limit_order(
                        symbol=symbol, side="BUY",
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
                        )
                        account.open_position = position
                        log.info(
                            f"[LIVE:{symbol}] 🟢 BUY @ ${buy_plan.limit_price:.2f} | "
                            f"qty={buy_plan.qty:.4f} | SL=${buy_plan.stop_loss:.2f} | "
                            f"TP=${buy_plan.take_profit:.2f}"
                        )
                        from shared.utils.state_writer import record_trade
                        record_trade(
                            symbol=symbol,
                            side="BUY",
                            price=buy_plan.limit_price,
                            qty=buy_plan.qty,
                            pnl=0.0,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            reason=f"confluencias={len(signal.confirmations)} RSI={signal.rsi_value:.1f}",
                            metadata={
                                "entry_price": buy_plan.limit_price,
                                "fees": 0.0 # BUY fees se aplican al cerrar en simulación simplificada
                            }
                        )

            # ── 7. Escribir estado para el dashboard /live ────────────────────
            _write_live_state(
                symbol, session_num, iteration, account,
                broker, position, signal, "running",
            )

            # ── 8. Dormir 60s entre scans (mercado real) ──────────────────────
            log.info(f"[LIVE:{symbol}] ⏳ Próximo scan en 60s...")
            _smart_sleep(60, stop_event)

    except Exception as e:
        log.error(f"[LIVE:{symbol}] ❌ Error inesperado en hilo: {e}")
    finally:
        log.info(f"[LIVE:{symbol}] 🛑 Hilo finalizado.")
        if position:
            log.warning(
                f"[LIVE:{symbol}] ⚠️ Posición abierta al cerrar: "
                f"{position.qty:.4f} @ ${position.entry_price:.2f}. "
                f"¡Verifica tu cuenta Alpaca Paper manualmente!"
            )


def _smart_sleep(seconds: int, stop_event: threading.Event = None):
    """Sleep interrumpible: sale si stop_event es seteado."""
    for _ in range(seconds):
        if stop_event and stop_event.is_set():
            return
        time.sleep(1)


def _write_live_state(
    symbol: str,
    session: int,
    iteration: int,
    account: AccountState,
    broker,
    position,
    signal,
    status: str = "running",
):
    """Escribe en state_live.json sin tocar state_sim.json."""
    try:
        pos_dict = None
        if position:
            pos_dict = {
                "symbol":      position.symbol,
                "entry_price": position.entry_price,
                "qty":         position.qty,
                "stop_loss":   position.stop_loss,
                "take_profit": position.take_profit,
            }

        stats = broker.stats

        state_writer.update_state(
            mode="LIVE_PAPER",
            symbol=symbol,
            session=session,
            iteration=iteration,
            bid=signal.close if signal and hasattr(signal, "close") else 0,
            ask=signal.close if signal and hasattr(signal, "close") else 0,
            signal=signal.signal if signal else "HOLD",
            rsi=signal.rsi_value if signal else 50,
            ema_fast=signal.ema_fast if signal else 0,
            ema_slow=signal.ema_slow if signal else 0,
            ema_200=signal.ema_200 if signal else 0,
            macd_hist=signal.macd_hist if signal else 0,
            vwap=signal.vwap_value if signal else 0,
            atr=signal.atr_value if signal else 0,
            confirmations=signal.confirmations if signal else [],
            initial_cash=account.total_cash,
            available_cash=account.available_cash,
            settlement=account.pending_settlement_total,
            win_rate=stats.win_rate,
            total_trades=stats.total_trades,
            winning_trades=stats.winning_trades,
            gross_profit=stats.gross_profit,
            gross_loss=stats.gross_loss,
            total_fees=getattr(stats, 'total_fees', 0.0),
            total_slippage=getattr(stats, 'total_slippage', 0.0),
            position=pos_dict,
            status=status,
            regime=getattr(signal, "regime", "NEUTRAL") if signal else "NEUTRAL",
            blocks=signal.blocks if signal else [],
        )
    except Exception as e:
        log.debug(f"[LIVE:{symbol}] _write_live_state error: {e}")
