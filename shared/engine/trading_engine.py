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
from shared.engine.utils import get_candles_json

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
        self.is_live_paper = getattr(broker, "_live_paper", False)
        
        # Estado de la sesión
        self.account = None
        self.risk_mgr = None
        self.position = None
        self.session_num = 1
        self.stop_event = None
        self.asset_type = "normal"
        self.blocking_history = collections.Counter()
        self.sim_start_date = getattr(args, 'start_date', None) or os.getenv("HAPI_SIM_START_DATE")
        self.sim_end_date = getattr(args, 'end_date', None) or os.getenv("HAPI_SIM_END_DATE")
        self.investment_style = "Normal"
        self.ghost_positions: list[OpenPosition] = [] # 👻 Rastreo de señales bloqueadas activas
        self.ghost_history: list[dict] = []          # 📜 Registro de todos los ghosts cerrados
        self.stage = os.getenv("SIM_STAGE", "TRAINING")  # 🏷️ Etapa del curriculum PVC

    @property
    def total_ghosts(self) -> int:
        return len(self.ghost_history) + len(self.ghost_positions)

    @property
    def active_ghosts(self) -> list:
        return self.ghost_positions

    def run(self, session_num: int = 1, stop_event: threading.Event = None, asset_type: str = "normal"):
        self.session_num = session_num
        self.stop_event = stop_event
        self.asset_type = asset_type
        
        set_symbol_log(self.symbol)
        
        self._engine_start_time = time.time()
        
        # Inicializar componentes
        acct_info = self.broker.get_account_info()
        self.account = AccountState(total_cash=acct_info.total_cash)
        self.risk_mgr = RiskManager(self.account)
        self.position = None
        
        log.info(f"{'─'*60}")
        log.info(f"Engine iniciado | Símbolo: {self.symbol} | Modo: {self.broker.name} | LivePaper: {self.is_live_paper}")
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

                from shared.utils.neural_filter import get_neural_filter
                nf = get_neural_filter(self.symbol)

                # 4. Gestión de Posición
                if self.position:
                    self._handle_exit(quote, signal)
                elif signal.signal == SIGNAL_BUY:
                    self._handle_entry(quote, signal, df)

                # 4.1 Gestión de Posiciones Fantasma (Ghost Trades)
                if self.ghost_positions:
                    n_ghosts = len(self.ghost_positions)
                    # Solo loguear si hay cambio o cada 10 iteraciones para no saturar
                    if iteration % 10 == 0:
                        log.info(f"👻 GHOST MONITOR: Analizando {n_ghosts} 'fantasmas' activos en {self.symbol}...")
                
                if iteration % 20 == 0 and not self.position and not self.ghost_positions:
                    log.info(f"💓 HEARTBEAT | {self.symbol} | Bot activo y escaneando... | RSI={signal.rsi_value:.1f}")

                self._handle_ghost_exits(quote, signal)

                # 5. UI Update per iteration
                # En modo LIVE priorizamos la actualización constante para evitar el look "congelado"
                _has_activity = (
                    signal.signal != SIGNAL_HOLD
                    or self.position is not None
                    or len(self.ghost_positions) > 0
                )
                
                # Para Live Paper, actualizamos cada 2 iteraciones (aprox 2s)
                # Para MOCK (simulación), actualizamos cada 10 para mayor fluidez del progreso
                _ui_interval = 10 if self.is_mock else 2
                
                if _has_activity or iteration % _ui_interval == 0:
                    update_state(
                        mode=self.args.mode, symbol=self.symbol, session=self.session_num,
                        iteration=iteration, bid=quote.bid, ask=quote.ask, signal=signal.signal,
                        status="running", # Aseguramos que siempre reporte running una vez superada la carga inicial
                        rsi=signal.rsi_value, ema_fast=signal.ema_fast, ema_slow=signal.ema_slow,
                        ema_200=signal.ema_200, macd_hist=signal.macd_hist, vwap=signal.vwap_value,
                        atr=signal.atr_value, confirmations=signal.confirmations,
                        available_cash=round(self.account.available_cash, 2),
                        win_rate=self.broker.stats.win_rate,
                        total_trades=self.broker.stats.total_trades,
                        winning_trades=self.broker.stats.winning_trades,
                        gross_profit=round(self.broker.stats.gross_profit, 2),
                        gross_loss=round(self.broker.stats.gross_loss, 2),
                        total_fees=round(getattr(self.broker.stats, 'total_fees', 0.0), 2),
                        total_slippage=round(self.broker.stats.total_trades * 0.10, 2),
                        position=self.position.__dict__ if self.position else None,
                        trades=self.broker.stats.trade_history,
                        total_ghosts=self.total_ghosts,
                        ghost_trades_count=len(self.active_ghosts),
                        candles=[], timestamp=quote.timestamp,
                        regime=signal.regime, blocks=signal.blocks,
                        blocking_summary=dict(self.blocking_history),
                        # Reportar estadísticas de aprendizaje de la IA 
                        **(nf.get_stats() if 'nf' in locals() else {"total_samples": self.args.get('total_samples', 0) if hasattr(self.args, 'get') else 0, "model_accuracy": 0.0}),
                        ai_recommendation=getattr(signal, 'ai_recommendation', 'Analizando...'),
                        ai_win_prob=signal.ai_win_prob if hasattr(signal, 'ai_win_prob') else 0.5,
                        sim_duration=time.time() - getattr(self, '_engine_start_time', time.time()),
                        sim_start=str(self.sim_start_date) if self.sim_start_date else "",
                        sim_end=quote.timestamp if self.is_mock else (str(self.sim_end_date) if self.sim_end_date else ""),
                        sim_progress_pct=self._calculate_progress(quote.timestamp),
                        is_live_paper=self.is_live_paper,
                        stage=self.stage
                    )

                # 6. Throttling for Live Paper
                if self.is_live_paper:
                    # En modo Live Paper (Mock o Real), pausamos para no saturar CPU y API
                    time.sleep(config.SCAN_INTERVAL_SEC)

            # --- Forzar liquidación al terminar la simulación (End of Data) ---
            if self.position and self.is_mock and 'quote' in locals() and quote:
                log.info(f"▶ Forzando liquidación de posición abierta al final de la simulación en {self.symbol}.")
                try:
                    reason = "SIMULATION_END"
                    plan = self.risk_mgr.calculate_sell_order(self.position, quote.bid)
                    metadata = self._build_exit_metadata(reason)
                    resp_status, fill_price = self._execute_sell(quote, plan, reason, metadata)
                    if resp_status not in ("REJECTED",):
                        self._process_closed_trade(fill_price, reason, quote.timestamp)
                except Exception as e:
                    log.warning(f"Error forzando venta al final de la simulación: {e}")

        except SessionInterrupted:
            raise
        except Exception:
            log.exception(f"Error crítico en Engine ({self.symbol})")
            raise

    def _calculate_progress(self, current_ts: str) -> float:
        """Calcula el porcentaje (0-100) de avance de la simulación."""
        if not self.is_mock or not self.sim_start_date or not current_ts:
            return 0.0
        
        try:
            from datetime import datetime
            # Normalizar formatos de fecha (YYYY-MM-DD vs ISO)
            def parse_dt(dt_str):
                if not dt_str: return None
                # Tomar solo los primeros 10 caracteres (YYYY-MM-DD) y normalizar
                clean_date = str(dt_str)[:10].replace("/", "-")
                try:
                    return datetime.strptime(clean_date, "%Y-%m-%d")
                except:
                    return None

            start = parse_dt(self.sim_start_date)
            # Si no hay sim_end_date, usamos hoy como referencia para el 100%
            end = parse_dt(self.sim_end_date) or datetime.now()
            curr = parse_dt(current_ts)

            if not start or not end or not curr:
                return 0.0
            
            total_delta = (end - start).total_seconds()
            curr_delta = (curr - start).total_seconds()

            if total_delta <= 0:
                return 0.0 
            
            pct = (curr_delta / total_delta) * 100.0
            return round(max(0.0, min(100.0, pct)), 1)
        except Exception as e:
            log.error(f"Error en calculo de progreso para {self.symbol}: {e}")
            return 0.0

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
        if signal.blocks:
            for b in signal.blocks:
                if "⏳ Cargando datos" in b:
                    continue
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

    def _handle_entry(self, quote, signal, df):
        if signal.signal == SIGNAL_BUY:
            log.info(f"[{self.symbol}] 🎯 SEÑAL TÉCNICA detectada | RSI: {signal.rsi_value:.1f} | Conf: {len(signal.confirmations)}")
        
        # Lógica de ML y Neural Filter integrada
        conf_mult = self._calculate_confidence(signal)
        
        # 2. Sincronizar cash en mock
        if self.is_mock:
            self.account.total_cash = self.broker.get_account_info().total_cash

        # 3. Filtro de Spread
        spread = quote.ask - quote.bid
        if spread > (quote.bid * 0.0020):
            return

        # 4. Plan de Compra (Risk Manager)
        # Calculamos el Swing Low de las últimas 3 velas para protección extra
        swing_low_val = float(df["Low"].iloc[-3:].min()) if len(df) >= 3 else quote.bid * 0.98

        buy_plan = self.risk_mgr.calculate_buy_order(
            self.symbol, quote.ask, signal.atr_value, 
            swing_low=swing_low_val,
            confidence_multiplier=conf_mult
        )

        # 🚀 Determinar si es un trade Real o Ghost
        is_ghost_force = getattr(signal, "is_quality_blocked", False) or getattr(signal, "is_ml_blocked", False)

        # 6. Ejecución (Real o Ghost)
        if buy_plan.is_viable and not is_ghost_force:
            who_decided = "AI Oportunity" if getattr(signal, "ai_win_prob", getattr(signal, "ml_features", {}).get("ml_prob", 0)) > 0.8 else "Technical Indicators"
            metadata = {
                "confirmations": signal.confirmations,
                "ml_prob": signal.ml_features.get('ml_prob', 0.5),
                "conf_mult": conf_mult,
                "xai_metrics": {
                    "rsi": signal.rsi_value,
                    "zscore": getattr(signal.ml_features, 'get', lambda k,d: d)('zscore_vwap', 0.0),
                    "regime": signal.regime,
                    "ema_fast": signal.ema_fast,
                    "ema_slow": signal.ema_slow,
                    "ema_200": signal.ema_200,
                    "who_decided": who_decided,
                    "blocks_ignored": signal.blocks
                }
            }
            
            log_order_attempt(self.symbol, "BUY", buy_plan.limit_price, buy_plan.qty, f"conf={conf_mult:.2f}")
            resp = self.broker.place_limit_order(
                self.symbol, "BUY", buy_plan.limit_price, buy_plan.qty, 
                f"BUY: {len(signal.confirmations)} confs", metadata
            )
            
            if resp.status not in ("REJECTED",):
                log.info(f"✅ ORDEN EJECUTADA: {buy_plan.qty} @ ${buy_plan.limit_price}")
                self.position = OpenPosition(
                    symbol=self.symbol, entry_price=buy_plan.limit_price, qty=buy_plan.qty,
                    stop_loss=buy_plan.stop_loss, take_profit=buy_plan.take_profit,
                    initial_stop=buy_plan.stop_loss, entry_reason="Technical Signal",
                    entry_metadata=metadata, ml_features=signal.ml_features,
                    is_ghost=False
                )
        else:
            if buy_plan.block_reason:
                if "Ganancia esperada" in buy_plan.block_reason:
                    clean_b = "Filtro Rentabilidad Mínima"
                elif "Sin capital" in buy_plan.block_reason:
                    clean_b = "Sin capital suficiente"
                else:
                    clean_b = buy_plan.block_reason.split(":")[0].strip()
                self.blocking_history[clean_b] += 1

            # 👻 No es viable para trade real (riesgo/capital/comisión) -> Crear Ghost Trade para aprender
            if len(self.ghost_positions) < 5:
                # Log extra si el bloqueo fue por profit insuficiente
                if "Ganancia esperada" in (buy_plan.block_reason or ""):
                    log.info(f"📉 BLOQUEO RENTABILIDAD: {buy_plan.block_reason}")

                ghost_metadata = {
                    "confirmations": signal.confirmations,
                    "ml_prob": signal.ml_features.get('ml_prob', 0.5),
                    "conf_mult": conf_mult,
                    "block_reason": buy_plan.block_reason or "Filtro de Calidad",
                    "xai_metrics": {
                        "rsi": signal.rsi_value,
                        "zscore": signal.ml_features.get('zscore_vwap', 0.0),
                        "regime": signal.regime,
                        "ema_fast": signal.ema_fast,
                        "ema_slow": signal.ema_slow,
                        "ema_200": signal.ema_200,
                        "who_decided": "Ghost Analyst",
                        "blocks_ignored": signal.blocks
                    }
                }
                ghost_reason = buy_plan.block_reason or ("ML Blocked" if getattr(signal, 'is_ml_blocked', False) else "Quality Blocked")
                ghost = OpenPosition(
                    symbol=self.symbol, entry_price=buy_plan.limit_price, qty=buy_plan.qty or 1.0,
                    stop_loss=buy_plan.stop_loss, take_profit=buy_plan.take_profit,
                    initial_stop=buy_plan.stop_loss, entry_reason=f"Ghost: {ghost_reason}",
                    entry_metadata=ghost_metadata, ml_features=signal.ml_features,
                    is_ghost=True
                )
                self.ghost_positions.append(ghost)
                log.info(f"👻 GHOST ENTRY | {self.symbol} | Price={buy_plan.limit_price} | Reason: {ghost_reason}")
            else:
                log.debug(f"[{self.symbol}] Límite de fantasmas alcanzado, ignorando señal.")

    def _calculate_confidence(self, signal):
        try:
            from shared.strategy.ml_predictor import ml_predictor
            is_win, prob_win = ml_predictor.predict_win(signal.ml_features)
            signal.ml_features['ml_prob'] = prob_win
        except: prob_win = 0.5

        mult = 1.0
        log.info(f"🧠 IA PREDICTION | {self.symbol} | Probabilidad de éxito: {prob_win:.1%} (Score: {prob_win:.2f})")
        
        if prob_win > 0.85: 
            mult *= 1.5
            log.info(f"🚀 IA BOOST | {self.symbol} | Confianza ALTA detected (x1.5)")
        elif prob_win > 0.70: 
            mult *= 1.2
            log.info(f"📈 IA BOOST | {self.symbol} | Confianza media detected (x1.2)")
        elif prob_win < 0.40: 
            mult *= 0.7
            log.warning(f"⚠️ IA CAUTION | {self.symbol} | Probabilidad baja, reduciendo exposición (x0.7)")
        
        return mult
        
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

    def _get_net_pnl(self, gross_pnl: float, qty: float, entry_price: float) -> float:
        """Calcula el PnL neto aproximado descontando comisiones y slippage para el aprendizaje de la IA."""
        # Costo mínimo Hapi (Tiered IBKR aprox $0.15 entrada + $0.15 salida + comisiones SEC/FINRA)
        fees = 0.35
        # Slippage estimado: 0.1% del valor nocional total (ida y vuelta)
        slippage = (qty * entry_price) * 0.001
        return gross_pnl - fees - slippage

    def _process_closed_trade(self, sell_price, reason, timestamp):
        pnl = (sell_price - self.position.entry_price) * self.position.qty
        log.info(f"🔴 SELL | {self.symbol} | PnL: ${pnl:+.2f} | Reason: {reason}")
        
        # ML feedback (Aprender del PnL Neto, no Bruto, para no engañar a la IA con micro-movimientos)
        # Ignorar "SIMULATION_END" para no ensuciar el entrenamiento de la IA con salidas no naturales
        if hasattr(self.position, 'ml_features') and reason != "SIMULATION_END":
            net_pnl = self._get_net_pnl(pnl, self.position.qty, self.position.entry_price)
            ml_predictor.save_trade(self.symbol, self.position.ml_features, net_pnl)
            self._train_neural_filter(net_pnl)
            
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

    def _train_neural_filter(self, pnl, pos=None):
        position_to_train = pos if pos else self.position
        if not position_to_train: return
        
        try:
            from shared.utils.neural_filter import get_neural_filter
            nf = get_neural_filter(self.symbol)
            ml_f = position_to_train.ml_features
            f_vec = nf.build_features(
                symbol=self.symbol,
                hour_of_day=ml_f.get('hour_of_day', 10.0),
                vwap_dist_pct=ml_f.get('vwap_dist_pct', 0.0),
                rsi=ml_f.get('rsi', 50.0), macd_hist=ml_f.get('macd_hist', 0.0),
                atr_pct=ml_f.get('atr_pct', 0.0), vol_ratio=ml_f.get('vol_ratio', 1.0),
                ema_spread_pct=ml_f.get('ema_diff_pct', 0.0),
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
                
                # Aprender del trade fantasma (Ambos modelos usan el PnL Neto)
                if hasattr(gp, 'ml_features'):
                    net_pnl = self._get_net_pnl(pnl, gp.qty, gp.entry_price)
                    ml_predictor.save_trade(self.symbol, gp.ml_features, net_pnl)
                    # AHORA SÍ entrenamos el NeuralFilter con fantasmas
                    # De lo contrario nunca saldrá de su fase Cold-Start
                    self._train_neural_filter(net_pnl, pos=gp)
                
                self.ghost_history.append({
                    "pnl": pnl,
                    "reason": reason,
                    "entry_price": gp.entry_price,
                    "exit_price": quote.bid,
                    "hold_bars": gp.hold_bars
                })
            else:
                active_ghosts.append(gp)
        
        self.ghost_positions = active_ghosts
