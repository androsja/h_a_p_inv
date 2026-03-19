"""
shared/engine/runner.py ─ Orquestador de trading compartido (SIM/LIVE).
"""

import os
import json
import importlib
import argparse
import threading
from pathlib import Path
from shared import config
from shared.utils.logger import log
from shared.data import market_data
from shared.utils.state_writer import update_state, clear_state
from shared.utils.market_hours import _is_mock_time_active
from shared.utils.checkpoint import load_simulation_checkpoint, save_simulation_checkpoint, clear_simulation_checkpoints

from .utils import SessionInterrupted, smart_sleep
from .trading_engine import TradingEngine

class SimulationRunner:
    """
    Gestiona la ejecución secuencial o paralela de múltiples sesiones de trading.
    """
    def __init__(self, init_broker_fn, run_bot_fn):
        self.init_broker = init_broker_fn
        self.run_bot_logic = run_bot_fn # Para compatibilidad con launch_parallel_bots
        self.session_num = 0
        self.symbol_idx = 0
        self.all_symbols = []
        self.all_assets = []
        self.is_simulated = True
        self.session_pause = config.SESSION_PAUSE
        
    def main_loop(self, args: argparse.Namespace):
        log.info("🚀 SISTEMA INICIADO: Preparando motores de trading...")
        self.is_simulated = (args.mode == "SIMULATED")
        
        # Restore checkpoint
        self._restore_checkpoint()
        
        while True:
            # 1. Recarga Dinámica
            self._reload_symbols()
            
            # 2. Comandos Globales
            if self._process_global_commands():
                continue # Restart loop if requested
            
            # 3. Fin de lista?
            if self.is_simulated and self.all_symbols and self.symbol_idx >= len(self.all_symbols):
                if not self._handle_completion():
                    continue
            
            # 4. Siguiente Sesión
            if self.is_simulated and self.all_symbols:
                args.symbol = self.all_symbols[self.symbol_idx]
            
            self.session_num += 1
            
            # 5. Live Paper Parallel?
            if self._check_live_paper(args):
                continue
                
            # 6. Ejecución Serial (Simulación)
            try:
                self._run_single_session(args)
            except SessionInterrupted:
                log.info("📢 Sesión interrumpida. Reiniciando bucle...")
                continue
            except Exception as e:
                log.error(f"❌ Error en sesión: {e}")
                update_state(mode=args.mode, status="error", symbol=args.symbol, session=self.session_num)

            if not self.is_simulated: break
            
            # 7. Siguiente Activo
            self._advance_to_next()
            smart_sleep(self.session_pause)

    def _restore_checkpoint(self):
        try:
            ckpt = load_simulation_checkpoint()
            if ckpt["symbol_idx"] > 0 and self.is_simulated:
                self.symbol_idx = ckpt["symbol_idx"]
                self.session_num = ckpt["session_num"]
                log.info(f"💾 CHECKPOINT restaurado: reanudando desde {ckpt['symbol']} (idx={self.symbol_idx})")
        except: pass

    def _reload_symbols(self):
        try:
            importlib.reload(market_data)
            from shared.data.market_data import set_assets_file
            set_assets_file(config.ASSETS_FILE_SIM)
            with open(config.ASSETS_FILE_SIM, "r") as f:
                data = json.load(f)
                self.all_assets = [a for a in data.get("assets", []) if a.get("enabled", True)]
                self.all_symbols = [a["symbol"] for a in self.all_assets]
                
                # Report total to dashboard
                update_state(
                    mode="SIMULATED", status="starting", symbol="─", 
                    total_sim_trades=0, total_sim_pnl=0,
                    # HACK: Using a kwarg to pass the total count to index.html logic if needed
                    # but index.html usually derives it from assets.json or history.
                    # We ensure we have symbols to start.
                )
                # Solo loguear si es la primera vez o si cambió la lista
                if not getattr(self, '_last_symbols_count', None) == len(self.all_symbols):
                    log.info(f"📋 Cargadas {len(self.all_symbols)} empresas para simular.")
                    self._last_symbols_count = len(self.all_symbols)
        except Exception as e:
            log.warning(f"⚠️ Error recargando símbolos: {e}")

    def _process_global_commands(self):
        cmd_file = config.COMMAND_FILE
        if not cmd_file.exists(): return False
        
        try:
            with open(cmd_file, "r") as f: cmds = json.load(f)
            
            # Snapshot reload / Freeze IA
            self._sync_ia_status(cmds)
            
            if cmds.get("reset_all") or cmds.get("restart_sim"):
                is_purgue = cmds.get("reset_all", False)
                log.info(f"🔄 {'PURGANDO' if is_purgue else 'REINICIANDO'} simulación...")
                
                cmds["reset_all"] = False
                cmds["restart_sim"] = False
                with open(cmd_file, "w") as f: json.dump(cmds, f)
                
                clear_state()
                self._cleanup_files(is_purgue)
                
                self.symbol_idx = 0
                self.session_num = 0
                update_state(mode="SIMULATED", status="restarting", symbol="─", mock_time_930=_is_mock_time_active())
                return True
        except: pass
        return False

    def _sync_ia_status(self, cmds):
        frozen = cmds.get("strategy_frozen", False)
        # Sincronizar NeuralFilter y MLPredictor
        try:
            from shared.utils.neural_filter import get_neural_filter
            nf = get_neural_filter()
            if frozen and not nf.is_frozen: nf.freeze()
            elif not frozen and nf.is_frozen: nf.unfreeze()
        except: pass

        if cmds.get("reload_models"):
            # Lógica de recarga...
            cmds["reload_models"] = False
            with open(config.COMMAND_FILE, "w") as f: json.dump(cmds, f)

    def _cleanup_files(self, is_purgue):
        for f in [config.RESULTS_FILE, config.TRADE_JOURNAL_FILE]:
            if f.exists(): f.unlink()
        if is_purgue:
            clear_simulation_checkpoints()
            if config.NEURAL_MODEL_FILE.exists(): config.NEURAL_MODEL_FILE.unlink()

    def _handle_completion(self):
        # Re-chequear si hay nuevos
        update_state(mode="SIMULATED", status="completed", symbol="─", session=self.session_num)
        smart_sleep(1)
        return False

    def _check_live_paper(self, args):
        is_lp = False
        force_symbols = []
        try:
            if config.COMMAND_FILE.exists():
                with open(config.COMMAND_FILE) as f:
                    c = json.load(f)
                    is_lp = c.get("force_paper_trading", False)
                    force_symbols = c.get("force_symbols", [])
        except: pass
        
        if is_lp and force_symbols:
            log.info(f"🚀 Modo Live Paper detectado ({len(force_symbols)} símbolos).")
            from shared.utils.live_paper_launcher import launch_parallel_bots
            try:
                launch_parallel_bots(args, force_symbols, self.session_num, self.run_bot_logic, self.init_broker, save_simulation_checkpoint)
            except SessionInterrupted:
                log.info("📢 Live Paper interrumpido por comando global.")
            return True
        return False

    def _run_single_session(self, args):
        broker = self.init_broker(args)
        asset_type = "normal"
        if self.all_assets and self.symbol_idx < len(self.all_assets):
            asset_type = self.all_assets[self.symbol_idx].get("type", "normal")
            
        engine = TradingEngine(broker, args)
        engine.run(session_num=self.session_num, asset_type=asset_type)
        
        # 8. Record Results
        self._record_session_results(engine, broker)

    def _record_session_results(self, engine, broker):
        try:
            stats = broker.stats
            # Only record if it's simulated and has some data
            if not self.is_simulated: return
            
            pnl_val = getattr(stats, 'total_pnl', 0.0)
            trades_val = getattr(stats, 'total_trades', 0)
            winrate_val = getattr(stats, 'win_rate', 0.0)
            
            insight = "Sin actividad."
            if trades_val == 0:
                insight = "No hubo entradas. Filtro de calidad o condiciones estrictas."
            elif pnl_val > 0:
                insight = "ESTRATEGIA EXITOSA. Rentabilidad positiva en este activo."
            else:
                insight = "ESTRATEGIA FALLIDA. Pérdidas en este escenario."

            from datetime import datetime, timezone
            total_fees = round(getattr(stats, 'total_fees', 0.0), 4)

            # Get IA accuracy if possible
            accuracy_ia = 0.0
            try:
                from shared.utils.neural_filter import get_neural_filter
                accuracy_ia = get_neural_filter().get_stats().get('model_accuracy', 0.0)
            except: pass
            
            res_file = config.RESULTS_FILE
            
            # Extract last trade data from broker history
            last_price = None
            last_order_date = None
            try:
                broker_trades = getattr(stats, 'trade_history', None) or getattr(broker, 'trade_history', None) or []
                if broker_trades:
                    last_t = broker_trades[-1]
                    last_price = last_t.get('price') or last_t.get('fill_price') or last_t.get('exit_price')
                    last_order_date = last_t.get('timestamp') or last_t.get('timestamp_close') or datetime.now(timezone.utc).isoformat()
            except Exception:
                pass

            session_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": engine.symbol,
                "session_num": self.session_num,
                "total_trades": trades_val,
                "winning_trades": getattr(stats, 'winning_trades', 0),
                "losing_trades": trades_val - getattr(stats, 'winning_trades', 0),
                "win_rate": round(winrate_val, 2),
                "accuracy": accuracy_ia,
                "pnl": round(pnl_val, 2),
                "gross_pnl": round(pnl_val + total_fees, 2),
                "total_fees": total_fees,
                "slippage_est": round(trades_val * 0.10, 2),
                "gross_profit": round(getattr(stats, 'gross_profit', 0.0), 2),
                "gross_loss": round(getattr(stats, 'gross_loss', 0.0), 2),
                "drawdown": round(getattr(stats, 'max_drawdown', 0.0), 2),
                "insight": insight,
                "sim_start": engine.sim_start_date,
                "sim_end": engine.sim_end_date,
                "investment_style": engine.investment_style,
                "blocking_summary": dict(engine.blocking_history or {}),
                "last_price": round(float(last_price), 2) if last_price else None,
                "last_order_date": last_order_date,
            }

            results = []
            if res_file.exists():
                with open(res_file, "r") as f:
                    try: results = json.load(f)
                    except: results = []
            
            # De-duplicate
            results = [r for r in results if not (r.get("symbol") == engine.symbol and r.get("session_num") == self.session_num)]
            results.append(session_result)
            
            with open(res_file, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            log.warning(f"Error recording session results: {e}")

    def _advance_to_next(self):
        if self.is_simulated and self.all_symbols:
            self.symbol_idx += 1
            save_simulation_checkpoint(
                self.symbol_idx, 
                self.all_symbols[self.symbol_idx] if self.symbol_idx < len(self.all_symbols) else "─",
                self.session_num
            )
            if self.symbol_idx < len(self.all_symbols):
                log.info(f"⏸ Sesión finalizada. Siguiente: {self.all_symbols[self.symbol_idx]}...")
