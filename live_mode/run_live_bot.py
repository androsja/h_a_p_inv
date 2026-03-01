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
    try:
        acct_info = broker.get_account_info()
        initial_cash = acct_info.total_cash
    except Exception:
        pass

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

            # ── 4. Obtener cotización actual ──────────────────────────────────
            try:
                quote = broker.get_quote(symbol)
            except Exception as e:
                log.warning(f"[LIVE:{symbol}] ⚠️ Error obteniendo quote: {e}")
                _smart_sleep(10, stop_event)
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
                        pnl = (actual_sell - position.entry_price) * position.qty
                        broker.update_stats(pnl)

                        log.info(
                            f"[LIVE:{symbol}] 🔴 SELL @ ${actual_sell:.2f} | "
                            f"PnL: ${pnl:+.2f} | {exit_reason}"
                        )

                        # ⚠️ SOLO LECTURA de los modelos en Live — NO se hace fit()
                        # El aprendizaje ocurre SoloP en simulación.
                        # Los modelos ya están entrenados y congelados.

                        # Registro en bitácora (carpeta live separada)
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

                        position = None

            # ── 6. Gestionar ENTRADA ──────────────────────────────────────────
            elif signal.signal == SIGNAL_BUY:
                # Sincronizar cash con cuenta real Alpaca
                try:
                    live_info = broker.get_account_info()
                    account.total_cash = live_info.total_cash
                except Exception:
                    pass

                buy_plan = risk_mgr.calculate_buy_order(
                    symbol, quote.ask,
                    atr_value=signal.atr_value,
                )

                # ─ Consultar ML (solo predict, sin fit) ─
                if buy_plan.is_viable:
                    is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
                    if not is_win:
                        buy_plan.is_viable = False
                        buy_plan.block_reason = (
                            f"ML_REJECTION: P(win)={prob_win*100:.1f}%"
                        )
                        log.info(f"[LIVE:{symbol}] 🧠 ML bloqueó entrada | P={prob_win*100:.1f}%")

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
                            buy_plan.block_reason = f"NF_REJECTION: P(win)={nf_prob*100:.1f}%"
                            log.info(f"[LIVE:{symbol}] 🧠 NeuralFilter bloqueó | P={nf_prob*100:.1f}%")
                    except Exception as nfe:
                        log.debug(f"[LIVE:{symbol}] NF error: {nfe}")

                if buy_plan.is_viable:
                    resp = broker.place_limit_order(
                        symbol=symbol, side="BUY",
                        limit_price=buy_plan.limit_price,
                        qty=buy_plan.qty,
                        reason=f"confluencias={len(signal.confirmations)} RSI={signal.rsi_value:.1f}",
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
                        log.info(
                            f"[LIVE:{symbol}] 🟢 BUY @ ${buy_plan.limit_price:.2f} | "
                            f"qty={buy_plan.qty:.4f} | SL=${buy_plan.stop_loss:.2f} | "
                            f"TP=${buy_plan.take_profit:.2f}"
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
            mode="LIVE_ALPACA",
            symbol=symbol,
            session=session,
            iteration=iteration,
            bid=signal.current_price if signal and hasattr(signal, "current_price") else 0,
            ask=signal.current_price if signal and hasattr(signal, "current_price") else 0,
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
            position=pos_dict,
            status=status,
            regime=getattr(signal, "regime", "NEUTRAL") if signal else "NEUTRAL",
            blocks=signal.blocks if signal else [],
        )
    except Exception as e:
        log.debug(f"[LIVE:{symbol}] _write_live_state error: {e}")
