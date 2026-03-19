"""
shared/engine/trading_engine.py ─ Motor de ejecución de trading compartido.
"""

import time
import json
import os
import collections
import threading
import argparse
from shared import config
from shared.utils.logger import log, log_order_attempt, log_market_closed, set_symbol_log
from shared.utils.market_hours import is_market_open, market_status_str, next_open_str
from shared.broker.interface import BrokerInterface
from shared.broker.hapi_mock import HapiMock
from shared.strategy.indicators import analyze, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from shared.strategy.risk_manager import RiskManager, AccountState, OpenPosition
from shared.strategy.ml_predictor import ml_predictor
from shared.utils.trade_journal import record_trade as journal_record_trade
from shared.utils.state_writer import update_state

from .utils import SessionInterrupted

class TradingEngine:
    """
    Gestiona el ciclo de vida de una sesión de trading (un símbolo).
    """
    def __init__(self, broker: BrokerInterface, args: argparse.Namespace):
        self.broker = broker
        self.args = args
        self.is_mock = isinstance(broker, HapiMock)
        self.symbol = broker.symbol if self.is_mock else (args.symbol or "AAPL")
        
        # Estado de la sesión
        self.account = None
        self.risk_mgr = None
        self.position = None
        self.session_num = 1
        self.stop_event = None
        self.asset_type = "normal"
        self.blocking_history = collections.Counter()
        self.sim_start_date = None
        self.sim_end_date = None
        self.investment_style = "Normal"
        self.ghost_positions: list[OpenPosition] = [] # 👻 Rastreo de señales bloqueadas

    def run(self, session_num: int = 1, stop_event: threading.Event = None, asset_type: str = "normal"):
        self.session_num = session_num
        self.stop_event = stop_event
        self.asset_type = asset_type
        
        set_symbol_log(self.symbol)
        
        # Inicializar componentes
        acct_info = self.broker.get_account_info()
        self.account = AccountState(total_cash=acct_info.total_cash)
        self.risk_mgr = RiskManager(self.account)
        self.position = None
        
        log.info(f"{'─'*60}")
        log.info(f"Engine iniciado | Símbolo: {self.symbol} | Modo: {self.broker.name}")
        log.info(f"Capital inicial: ${acct_info.total_cash:,.2f}")
        log.info(f"{'─'*60}")

        self._send_initial_state()
        
        ml_predictor.blocking_count = 0 
        iteration = 0
        try:
            while not (self.stop_event and self.stop_event.is_set()):
                iteration += 1
                
                # 1. Mercado Abierto?
                if not self.is_mock and not is_market_open():
                    log_market_closed(next_open_str())
                    time.sleep(config.REST_INTERVAL_SEC)
                    continue
                
                # Detectar Estilo de Inversión (UI multiplier)
                try:
                    mult = self.risk_mgr._read_user_multiplier()
                    self.investment_style = "Agresivo 🔥" if mult > 1.1 else "Normal"
                except: pass

                # 2. Obtener Precio (o Fin de Simulación)
                try:
                    quote = self.broker.get_quote(self.symbol)
                    if not self.sim_start_date: self.sim_start_date = quote.timestamp
                    self.sim_end_date = quote.timestamp
                except StopIteration:
                    log.info("▶ Datos agotados. Finalizando sesión.")
                    break

                # 3. Análisis Técnico
                df = self._get_data_slice()
                if df is None: continue
                
                # Chequeo de comandos rápidos
                self._check_interrupts()

                signal = analyze(df, symbol=self.symbol, asset_type=self.asset_type)
                self._record_blocks(signal)

                # 4. Gestión de Posición
                if self.position:
                    self._handle_exit(quote, signal)
                elif signal.signal == SIGNAL_BUY:
                    self._handle_entry(quote, signal)

                # 4.1 Gestión de Posiciones Fantasma
                self._handle_ghost_exits(quote, signal)

                # 5. UI Update per iteration
                if iteration % 2 == 0 or signal.signal != SIGNAL_HOLD:
                    from .utils import get_candles_json
                    update_state(
                        mode=self.args.mode, symbol=self.symbol, session=self.session_num,
                        iteration=iteration, bid=quote.bid, ask=quote.ask, signal=signal.signal,
                        rsi=signal.rsi_value, ema_fast=signal.ema_fast, ema_slow=signal.ema_slow,
                        ema_200=signal.ema_200, macd_hist=signal.macd_hist, vwap=signal.vwap_value,
                        atr=signal.atr_value, confirmations=signal.confirmations,
                        available_cash=self.account.available_cash,
                        win_rate=self.broker.stats.win_rate,
                        total_trades=self.broker.stats.total_trades,
                        winning_trades=self.broker.stats.winning_trades,
                        gross_profit=self.broker.stats.gross_profit,
                        gross_loss=self.broker.stats.gross_loss,
                        position=self.position.__dict__ if self.position else None,
                        candles=get_candles_json(df), timestamp=quote.timestamp,
                        regime=signal.regime, blocks=signal.blocks,
                        investment_style=self.investment_style
                    )

        except SessionInterrupted:
            raise
        except Exception as e:
            log.error(f"Error crítico en Engine ({self.symbol}): {e}")
            raise

    def _get_data_slice(self):
        if self.is_mock:
            return self.broker.get_history_slice(window=220)
        else:
            from shared.data.market_data import download_bars
            return download_bars(self.symbol)



    def _check_interrupts(self):
        """Verifica señales de interrupción desde el dashboard."""
        if not self.is_mock: return
        try:
            cmd_file = config.COMMAND_FILE
            if os.path.exists(cmd_file):
                with open(cmd_file, "r") as f:
                    cmds = json.load(f)
                if cmds.get("reset_all") or cmds.get("restart_sim"):
                    raise SessionInterrupted("Reinicio solicitado")
                
                new_sym = cmds.get("force_symbol")
                if new_sym and new_sym != "AUTO" and new_sym != self.symbol:
                    raise SessionInterrupted("Cambio de símbolo solicitado")
        except SessionInterrupted: raise
        except: pass

    def _record_blocks(self, signal):
        if signal.signal == "HOLD" and signal.blocks:
            for b in signal.blocks:
                clean_b = b.split(":")[0].strip()
                self.blocking_history[clean_b] += 1

    def _handle_exit(self, quote, signal):
        self.position.hold_bars = getattr(self.position, "hold_bars", 0) + 1
        should_exit, reason = self.risk_mgr.should_exit(self.position, quote.bid)
        
        if should_exit:
            plan = self.risk_mgr.calculate_sell_order(self.position, quote.bid)
            log_order_attempt(self.symbol, "SELL", plan.limit_price, plan.qty, reason)
            
            # Ejecutar venta
            metadata = self._build_exit_metadata(reason)
            resp_status, fill_price = self._execute_sell(quote, plan, reason, metadata)

            if resp_status not in ("REJECTED",):
                self._process_closed_trade(fill_price, reason, quote.timestamp)

    def _execute_sell(self, quote, plan, reason, metadata):
        is_protective = any(k in reason for k in ("STOP_LOSS", "TRAILING_STOP", "TIEMPO", "TIME"))
        if is_protective and self.is_mock and hasattr(self.broker, "immediate_market_sell"):
            price = self.broker.immediate_market_sell(self.symbol, self.position.qty, quote.bid, reason, metadata)
            return "FILLED", price
        else:
            resp = self.broker.place_limit_order(self.symbol, "SELL", plan.limit_price, plan.qty, reason, metadata)
            return resp.status, plan.limit_price

    def _handle_entry(self, quote, signal):
        # Lógica de ML y Neural Filter integrada
        conf_mult = self._calculate_confidence(signal)
        
        # Sincronizar cash en mock
        if self.is_mock:
            self.account.total_cash = self.broker.get_account_info().total_cash

        # Spread check
        spread = quote.ask - quote.bid
        if spread > (quote.bid * 0.0020):
            return

        buy_plan = self.risk_mgr.calculate_buy_order(
            self.symbol, quote.ask, signal.atr_value, confidence_multiplier=conf_mult
        )

        if buy_plan.is_viable:
            metadata = {
                "confirmations": signal.confirmations,
                "ml_prob": signal.ml_features.get('ml_prob', 0.5),
                "conf_mult": conf_mult
            }
            
            log_order_attempt(self.symbol, "BUY", buy_plan.limit_price, buy_plan.qty, f"conf={conf_mult:.2f}")
            resp = self.broker.place_limit_order(
                self.symbol, "BUY", buy_plan.limit_price, buy_plan.qty, 
                f"BUY: {len(signal.confirmations)} confs", metadata
            )
            
            if resp.status not in ("REJECTED",):
                self.position = OpenPosition(
                    symbol=self.symbol, entry_price=buy_plan.limit_price, qty=buy_plan.qty,
                    stop_loss=buy_plan.stop_loss, take_profit=buy_plan.take_profit,
                    initial_stop=buy_plan.stop_loss, entry_reason="Technical Signal",
                    entry_metadata=metadata, ml_features=signal.ml_features,
                    is_ghost=False
                )
        else:
            # 👻 No es viable para trade real (riesgo/capital/comisión) -> Crear Ghost Trade para aprender
            if len(self.ghost_positions) < 5:
                ghost_metadata = {
                    "confirmations": signal.confirmations,
                    "ml_prob": signal.ml_features.get('ml_prob', 0.5),
                    "conf_mult": conf_mult,
                    "block_reason": buy_plan.block_reason
                }
                ghost = OpenPosition(
                    symbol=self.symbol, entry_price=buy_plan.limit_price, qty=buy_plan.qty or 1.0,
                    stop_loss=buy_plan.stop_loss, take_profit=buy_plan.take_profit,
                    initial_stop=buy_plan.stop_loss, entry_reason=f"Ghost: {buy_plan.block_reason}",
                    entry_metadata=ghost_metadata, ml_features=signal.ml_features,
                    is_ghost=True
                )
                self.ghost_positions.append(ghost)
                log.info(f"👻 GHOST ENTRY | {self.symbol} | Price={buy_plan.limit_price} | Reason: {buy_plan.block_reason}")

    def _calculate_confidence(self, signal):
        try:
            is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
            signal.ml_features['ml_prob'] = prob_win
        except: prob_win = 0.5

        mult = 1.0
        if prob_win > 0.85: mult *= 1.5
        elif prob_win > 0.70: mult *= 1.2
        elif prob_win < 0.40: mult *= 0.7
        
        if len(signal.confirmations) >= 4: mult *= 1.2
        return min(mult, 2.0)

    def _send_initial_state(self):
        update_state(
            mode=self.args.mode, symbol=self.symbol, status="running",
            available_cash=self.account.available_cash, total_trades=0, win_rate=0
        )

    def _build_exit_metadata(self, reason):
        return {
            "exit_reason": reason,
            "entry_reason": getattr(self.position, 'entry_reason', ""),
            "ml_prob": self.position.entry_metadata.get('ml_prob', 0.5),
            "entry_price": self.position.entry_price
        }

    def _process_closed_trade(self, sell_price, reason, timestamp):
        pnl = (sell_price - self.position.entry_price) * self.position.qty
        log.info(f"🔴 SELL | {self.symbol} | PnL: ${pnl:+.2f} | Reason: {reason}")
        
        # ML feedback
        if hasattr(self.position, 'ml_features'):
            ml_predictor.save_trade(self.symbol, self.position.ml_features, pnl)
            self._train_neural_filter(pnl)
            
            journal_record_trade(
                symbol=self.symbol, session=self.session_num,
                entry_price=self.position.entry_price, exit_price=sell_price,
                qty=self.position.qty, stop_loss=self.position.initial_stop,
                take_profit=self.position.take_profit, hold_bars=self.position.hold_bars,
                exit_reason=reason, ml_features=self.position.ml_features, 
                timestamp=timestamp
            )
        
        if not self.is_mock:
            self.account.record_sale(sell_price * self.position.qty)
        
        self.position = None

    def _train_neural_filter(self, pnl):
        try:
            from shared.utils.neural_filter import get_neural_filter
            nf = get_neural_filter()
            ml_f = self.position.ml_features
            f_vec = nf.build_features(
                symbol=self.symbol,
                hour_of_day=ml_f.get('hour_of_day', 10.0),
                vwap_dist_pct=ml_f.get('vwap_dist_pct', 0.0),
                rsi=ml_f.get('rsi', 50.0), macd_hist=ml_f.get('macd_hist', 0.0),
                atr_pct=ml_f.get('atr_pct', 0.0), vol_ratio=ml_f.get('vol_ratio', 1.0),
                ema_fast=ml_f.get('ema_fast', 0.0), 
                ema_slow=ml_f.get('ema_slow', 0.0),
                zscore_vwap=ml_f.get('zscore_vwap', 0.0), regime=ml_f.get('regime', 'NEUTRAL'),
                num_confirmations=ml_f.get('num_confirmations', 2),
                adx=ml_f.get('adx', 20.0),
                has_pattern=ml_f.get('has_pattern', False),
                is_adx_rising=ml_f.get('is_adx_rising', False)
            )
            nf.fit(f_vec, won=(pnl > 0))
        except Exception as e:
            log.warning(f"NeuralFit Error: {e}")

    def _handle_ghost_exits(self, quote, signal):
        """Rastrea la evolución de los trades fantasmas y aprende de ellos al salir."""
        if not self.risk_mgr: return
        active_ghosts = []
        for gp in self.ghost_positions:
            gp.hold_bars += 1
            should_exit, reason = self.risk_mgr.should_exit(gp, quote.bid)
            
            if should_exit:
                pnl = gp.pnl(quote.bid)
                log.info(f"👻 GHOST EXIT | {self.symbol} | PnL: ${pnl:+.2f} | Reason: {reason}")
                
                # Aprender del trade fantasma
                if hasattr(gp, 'ml_features'):
                    ml_predictor.save_trade(self.symbol, gp.ml_features, pnl)
                    # No entrenamos el filtro neuronal con fantasmas para no sesgar el riesgo real,
                    # pero sí alimentamos el dataset del predictor para que sepa qué señales eran buenas.
            else:
                active_ghosts.append(gp)
        
        self.ghost_positions = active_ghosts
