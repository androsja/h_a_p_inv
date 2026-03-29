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
from shared.utils.market_hours import is_market_open, market_status_str, next_open_str, TZ_NYC, now_nyc
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
    live_mode_type: str = "PAPER",
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
    from shared.broker.hapi_mock import HapiMock
    from shared.utils.neural_filter import get_neural_filter
    # from shared.utils.market_hours import now_nyc # Moved to top

    log.info(f"[LIVE:{symbol}] 🚀 Hilo iniciado con modo {live_mode_type}")

    if live_mode_type == "DEMO":
        broker = HapiMock(
            symbol=symbol,
            initial_cash=initial_cash,
            live_paper=True
        )
    else:
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
    shadow_positions: list[OpenPosition] = []
    nf = get_neural_filter(symbol)

    blocking_history = collections.Counter()
    iteration = 0

    log.info(f"[LIVE:{symbol}] 💰 Capital inicial: ${initial_cash:,.2f}")
    log.info(f"[LIVE:{symbol}] EMA({config.EMA_FAST}/{config.EMA_SLOW}) | RSI({config.RSI_PERIOD})")

    try:
        while not (stop_event and stop_event.is_set()):
            iteration += 1

            # ── 1. Verificar horario de mercado y comandos de parada ──────────
            if _read_live_cmd("live_stop", False):
                log.info(f"[LIVE:{symbol}] 🛑 PARADA SOLICITADA desde el Dashboard. Finalizando hilo.")
                break

            if not is_market_open():
                log.info(f"[LIVE:{symbol}] 🔴 Fuera de horario. Próxima apertura: {next_open_str()}")
                last_action = f"MARKET_CLOSED: {next_open_str()}"
                _write_live_state(symbol, session_num, iteration, account,
                                  broker, position, None, "MARKET_CLOSED", nf.get_stats(),
                                  last_action=last_action)
                _smart_sleep(120, stop_event) # Aumentado a 120s fuera de horario para reducir I/O
                continue

            log.info(f"[LIVE:{symbol}] 🔍 Iteración #{iteration} | Escaneando mercado...")
            if iteration % 10 == 0:
                log.info(f"💓 HEARTBEAT | [LIVE:{symbol}] | Bot activo y escaneando indicadores...")

            # ── 2. Obtener barras reales desde Alpaca ────────────────────────
            try:
                df = broker.get_bars_df(symbol, limit=220)
            except Exception as e:
                log.warning(f"[LIVE:{symbol}] ⚠️ Error obteniendo barras: {e}. Reintentando en 30s.")
                # Escribir estado incluso en error para actualizar el reloj mock en el dashboard
                _write_live_state(symbol, session_num, iteration, account,
                                  broker, position, None, f"ERROR_DATA:{str(e)[:20]}", nf.get_stats())
                _smart_sleep(30, stop_event)
                continue

            # ── 3. Analizar señal técnica (módulo shared compartido) ──────────
            last_action = "Escaneando indicadores..."
            asset_type = "inverted" if symbol in ["SQQQ", "SOXS", "SARK"] else "normal"
            signal = analyze(df, symbol=symbol, asset_type=asset_type)
            
            # LOG DETALLADO DE CADA SCAN
            log.info(f"[LIVE:{symbol}] 📊 RSI: {signal.rsi_value:.1f} | Régimen: {signal.regime} | Precio: ${signal.close:.2f}")

            if signal.signal == SIGNAL_HOLD and signal.blocks:
                for b in signal.blocks:
                    blocking_history[b.split(":")[0].strip()] += 1
                block_str = ", ".join(signal.blocks)
                last_action = f"Esperando señal. Bloqueos: {block_str}"
                log.info(f"[LIVE:{symbol}] 🕒 Sin señal clara: {block_str}")
            elif signal.signal == SIGNAL_BUY:
                last_action = "¡MOMENTO DE COMPRA! Validando filtros..."
                log.info(f"[LIVE:{symbol}] 🎯 SEÑAL DE COMPRA detectada. Validando filtros IA/Riesgo...")

            # ── 4. Obtener cotización actual (REAL TIME) ──────────────────────
            try:
                # Usamos exclusivamente Alpaca para garantizar TIEMPO REAL (0 retraso)
                quote = broker.get_quote(symbol)
            except Exception as e:
                log.error(f"[LIVE:{symbol}] ❌ Error obteniendo precio en tiempo real (Alpaca): {e}")
                _smart_sleep(20, stop_event)
                continue

            # ── 5. Gestionar SALIDA (posición abierta SOCIAL/REAL) ──────────
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

                        # 🧠 Alimentar al modelo de ML con el resultado real
                        if hasattr(position, 'ml_features') and position.ml_features:
                            log.info(f"[{symbol}] 🧠 SAVING REAL TRADE: Pnl=${net_pnl:.2f} Features={position.ml_features}")
                            ml_predictor.save_trade(symbol, position.ml_features, net_pnl)
                            
                            try:
                                ml_f = position.ml_features
                                fv = nf.build_features(
                                    symbol=ml_f.get("symbol", symbol),
                                    hour_of_day=ml_f.get("hour_of_day", 10.0),
                                    vwap_dist_pct=ml_f.get("vwap_dist_pct", 0.0),
                                    rsi=ml_f.get("rsi", 50.0),
                                    macd_hist=ml_f.get("macd_hist", 0.0),
                                    atr_pct=ml_f.get("atr_pct", 0.0),
                                    vol_ratio=ml_f.get("vol_ratio", 1.0),
                                    ema_fast=ml_f.get("ema_fast", 100.0),
                                    ema_slow=ml_f.get("ema_slow", 100.0),
                                    zscore_vwap=ml_f.get("zscore_vwap", 0.0),
                                    regime=ml_f.get("regime", "NEUTRAL"),
                                    num_confirmations=int(ml_f.get("num_confirmations", 2)),
                                    adx=ml_f.get("adx", 20.0),
                                    has_pattern=ml_f.get("has_pattern", False),
                                    is_adx_rising=ml_f.get("is_adx_rising", False),
                                )
                                s_won = net_pnl > 0
                                nf.fit(fv, s_won)
                                log.info(f"[LIVE:{symbol}] 🧠 AI MLP aprendiendo del trade real (Ganado={s_won})")
                            except Exception as _e:
                                log.warning(f"[LIVE:{symbol}] No se pudo entrenar red neuronal: {_e}")

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
            if position is None and signal.signal == SIGNAL_BUY:
                
                # REGLA 1: Evitar el Sesgo de Look-Ahead
                now = datetime.now(TZ_NYC)
                if now.minute % 5 != 0:
                    msg = "⏱️ Esperando inicio de nueva vela de 5m (Look-Ahead Prevention)"
                    signal.blocks = [msg]
                    signal.signal = SIGNAL_HOLD
                    blocking_history[msg.split("(")[0].strip()] += 1
                    # Eliminado 'continue' para proteger el pipeline de sleep/estado
                
                # REGLA 2: Liquidez / SLIPPAGE PREVENTIVO
                spread = quote.ask - quote.bid
                max_spread_allowed = quote.bid * 0.0020 # 0.2% max spread
                if spread > max_spread_allowed:
                    msg = f"📉 Spread Tóxico ({spread:.2f} > {max_spread_allowed:.2f}) | Slippage risk"
                    signal.blocks = [msg]
                    signal.signal = SIGNAL_HOLD
                    blocking_history[msg.split("(")[0].strip()] += 1
                    log.warning(f"[LIVE:{symbol}] 🎯🛑 OPORTUNIDAD EVITADA: {msg}")
                    # Eliminado 'continue' para proteger el pipeline de sleep/estado

                # 🧠 CONSULTAR AL MODELO DE AI
                is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
                
                log.info(f"[LIVE:{symbol}] 🧠 IA PREDICTION | Probabilidad: {prob_win:.1%} (Score: {prob_win:.2f})")
                
                conf_mult = 1.0
                if is_win:
                    if prob_win > 0.85: 
                        conf_mult *= 1.5 
                        log.info(f"[LIVE:{symbol}] 🚀 IA BOOST | Confianza ALTA (x1.5)")
                    elif prob_win > 0.70: 
                        conf_mult *= 1.2
                        log.info(f"[LIVE:{symbol}] 📈 IA BOOST | Confianza media (x1.2)")
                
                if len(signal.confirmations) >= 4:
                    conf_mult *= 1.2 

                conf_mult = min(conf_mult, 2.0)

                buy_plan = risk_mgr.calculate_buy_order(
                    symbol, quote.ask,
                    atr_value=signal.atr_value,
                    confidence_multiplier=conf_mult
                )

                # ─ ML Tolerance Threshold ─
                ai_blocked = False
                if not is_win and prob_win < 0.48 and buy_plan.is_viable:
                    buy_plan.is_viable = False
                    ai_blocked = True
                    msg = f"🧠 AI bloqueó entrada | MLP P={prob_win*100:.1f}%"
                    buy_plan.block_reason = msg
                    signal.signal = SIGNAL_HOLD
                    signal.blocks = [msg]
                    log.warning(f"[LIVE:{symbol}] 🎯🛑 OPORTUNIDAD EVITADA: {msg}")

                # ─ Consultar Neural Filter (solo predict) ─
                nf_blocked = False
                if buy_plan.is_viable:
                    try:
                        ml_f = signal.ml_features
                        fv = nf.build_features(
                            symbol=ml_f.get("symbol", symbol),
                            hour_of_day=ml_f.get("hour_of_day", 10.0),
                            vwap_dist_pct=ml_f.get("vwap_dist_pct", 0.0),
                            rsi=ml_f.get("rsi", 50.0),
                            macd_hist=ml_f.get("macd_hist", 0.0),
                            atr_pct=ml_f.get("atr_pct", 0.0),
                            vol_ratio=ml_f.get("vol_ratio", 1.0),
                            ema_fast=ml_f.get("ema_fast", 100.0), # Fallback simplificado
                            ema_slow=ml_f.get("ema_slow", 100.0),
                            zscore_vwap=ml_f.get("zscore_vwap", 0.0),
                            regime=ml_f.get("regime", "NEUTRAL"),
                            num_confirmations=int(ml_f.get("num_confirmations", 2)),
                            adx=ml_f.get("adx", 20.0),
                            has_pattern=ml_f.get("has_pattern", False),
                            is_adx_rising=ml_f.get("is_adx_rising", False),
                        )
                        proba, reason = nf.predict(fv)
                        from shared.config import CONFIDENCE_THRESHOLD
                        if proba < CONFIDENCE_THRESHOLD:
                            buy_plan.is_viable = False
                            nf_blocked = True
                            msg = f"🧠 Red Neuronal bloqueó | P={proba*100:.1f}%"
                            buy_plan.block_reason = msg
                            signal.signal = SIGNAL_HOLD
                            signal.blocks = [msg]
                            log.warning(f"[LIVE:{symbol}] 🎯🛑 OPORTUNIDAD EVITADA: {msg}")
                    except Exception as nfe:
                        log.debug(f"[LIVE:{symbol}] NF error: {nfe}")

                # 🕵️ SHADOW TRADING: Si fue bloqueado por IA o Smart Filter (pero es viable por riesgo/capital)
                # O si estamos en SIGNAL_HOLD pero por filtros de indicadores (Smart Filter)
                technical_buy = (signal.signal == SIGNAL_BUY) or (ai_blocked or nf_blocked)
                
                if technical_buy and not buy_plan.is_viable and position is None:
                    # Verificar si la razón de no viabilidad es AI o si es un Smart Filter de indicators.py
                    is_smart_filtered = any("SMART FILTER" in str(b) or "RANGE:" in str(b) for b in signal.blocks)
                    
                    if ai_blocked or nf_blocked or is_smart_filtered:
                        # Si no hay posición real, crear una virtual para aprender
                        shadow_pos = OpenPosition(
                            symbol=symbol,
                            entry_price=quote.ask,
                            qty=1.0, # Tamaño simbólico para shadow
                            stop_loss=quote.ask * 0.985, # 1.5% fixed fallback
                            take_profit=quote.ask * 1.03, # 3.0% fixed fallback
                            ml_features=signal.ml_features,
                            entry_metadata={"ml_features": signal.ml_features, "reason": "shadow"}
                        )
                        # Intentar usar el plan de riesgo si existía pero fue bloqueado solo por IA
                        # (Si fue bloqueado por capital, shadow_pos usa los default arriba)
                        if buy_plan.qty > 0:
                            shadow_pos.qty = buy_plan.qty
                            shadow_pos.stop_loss = buy_plan.stop_loss
                            shadow_pos.take_profit = buy_plan.take_profit
                            shadow_pos.initial_stop = buy_plan.stop_loss

                        shadow_positions.append(shadow_pos)
                        log.info(f"[LIVE:{symbol}] 🕵️ Shadow Trade iniciado para aprender de esta omisión (SL=${shadow_pos.stop_loss:.2f} TP=${shadow_pos.take_profit:.2f})")

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
            if signal.signal == SIGNAL_BUY:
                if signal.is_ml_blocked: last_action = "Bloqueado por IA (RF/Neural)"
                elif signal.is_quality_blocked: last_action = "Filtro de Calidad"
                else: last_action = "EJECUTANDO ÓRDEN"

            _write_live_state(
                symbol, session_num, iteration, account,
                broker, position, signal, "running", nf.get_stats(),
                last_action=last_action
            )

            # ── 8. Dormir 60s entre scans (mercado real) ──────────────────────
            log.info(f"[LIVE:{symbol}] ⏳ Próximo scan en 60s...")
            _smart_sleep(60, stop_event)

            # ── 8.1 Gestionar SHADOW POSITIONS (Aprendizaje de Omisiones) ──
            completed_shadows = []
            for sp in shadow_positions:
                sp.hold_bars += 1
                s_exit, s_reason = risk_mgr.should_exit(sp, quote.bid)
                if s_exit:
                    s_net_pnl = (quote.bid - sp.entry_price) * sp.qty
                    s_won = s_net_pnl > 0
                    log.info(f"[LIVE:{symbol}] 🧠 Shadow Trade Cerrado: {s_reason} | PnL=${s_net_pnl:+.2f} | AI aprendiendo de esta omisión...")
                    # Alimentar a la Red Neuronal
                    try:
                        # Extraer features originales
                        ml_f = sp.entry_metadata.get("ml_features", {})
                        if ml_f:
                            fv = nf.build_features(
                                symbol=ml_f.get("symbol", symbol),
                                hour_of_day=ml_f.get("hour_of_day", 10.0),
                                vwap_dist_pct=ml_f.get("vwap_dist_pct", 0.0),
                                rsi=ml_f.get("rsi", 50.0),
                                macd_hist=ml_f.get("macd_hist", 0.0),
                                atr_pct=ml_f.get("atr_pct", 0.0),
                                vol_ratio=ml_f.get("vol_ratio", 1.0),
                                ema_fast=ml_f.get("ema_fast", 100.0),
                                ema_slow=ml_f.get("ema_slow", 100.0),
                                zscore_vwap=ml_f.get("zscore_vwap", 0.0),
                                regime=ml_f.get("regime", "NEUTRAL"),
                                num_confirmations=int(ml_f.get("num_confirmations", 2)),
                                adx=ml_f.get("adx", 20.0),
                                has_pattern=ml_f.get("has_pattern", False),
                                is_adx_rising=ml_f.get("is_adx_rising", False),
                            )
                            nf.fit(fv, s_won)
                    except Exception as fe:
                        log.debug(f"[LIVE:{symbol}] Fit error en shadow position: {fe}")
                    completed_shadows.append(sp)
            
            for cs in completed_shadows:
                shadow_positions.remove(cs)

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
    """Sleep interrumpible: sale si stop_event es seteado o si se recibe live_stop."""
    from shared.utils.market_hours import _is_mock_time_active
    
    # ACELERACIÓN MOCK: Si es un Replay/Test Nocturno (1s real = 1m clock), dormimos solo 1s
    if _is_mock_time_active() and seconds > 1:
        seconds = 1
        
    for i in range(seconds):
        if stop_event and stop_event.is_set():
            return
        # Verificar comandos cada 5 segundos para no saturar I/O
        if i % 5 == 0:
            if _read_live_cmd("live_stop", False):
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
    nf_stats: dict | None = None,
    last_action: str = ""
):
    """Escribe en state_live.json sin tocar state_sim.json."""
    from shared.utils.market_hours import now_nyc
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
            model_accuracy=nf_stats.get("model_accuracy") if nf_stats else None,
            total_samples=nf_stats.get("total_samples") if nf_stats else None,
            regime=getattr(signal, "regime", "NEUTRAL") if signal else "NEUTRAL",
            blocks=signal.blocks if signal else [],
            is_ml_blocked=getattr(signal, "is_ml_blocked", False) if signal else False,
            is_quality_blocked=getattr(signal, "is_quality_blocked", False) if signal else False,
            last_action=last_action,
            mock_time=now_nyc().strftime("%Y-%m-%d %H:%M:%S") if 'now_nyc' in locals() else "SCOPING_ERROR"
        )
    except Exception as e:
        log.error(f"[LIVE:{symbol}] _write_live_state error: {e}")
        import traceback
        log.error(traceback.format_exc())
